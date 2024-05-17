import os
import json
import tqdm
import numpy as np
import torch
from torch import Tensor

import argparse

from datasets import Dataset
from typing import List, Dict
from functools import partial
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
    DataCollatorWithPadding,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mteb import MTEB, AbsTaskRetrieval, DRESModel

from utils import (
    logger,
    move_to_cuda,
    get_detailed_instruct,
    get_task_def_by_task_name_and_type,
    input_transform_func,
)

parser = argparse.ArgumentParser(description="evaluation for BEIR benchmark")
parser.add_argument(
    "--model-name-or-path",
    default="bert-base-uncased",
    type=str,
    metavar="N",
    help="which model to use",
)
parser.add_argument(
    "--output-dir",
    default="tmp-outputs/",
    type=str,
    metavar="N",
    help="output directory",
)
parser.add_argument(
    "--task",
    type=str,
    help="specify one task for MTEB",
)
parser.add_argument("--extend-pe", default=False, type=bool)
parser.add_argument(
    "--doc-as-query", action="store_true", help="use query prefix for passages"
)
parser.add_argument("--pool-type", default="avg", help="pool type")
parser.add_argument("--max-length", default=512, help="max length")
parser.add_argument("--prefix-type", default="instruction", help="prefix type")
parser.add_argument(
    "--dry-run", action="store_true", help="whether to run the script in dry run mode"
)
parser.add_argument("--batch-size", default=24, help="batch size for MTEB")
args = parser.parse_args()
base_name: str = args.model_name_or_path.split("/")[-1]
args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)
logger.info("Args: {}".format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ["cls", "avg", "last"], "pool_type should be cls / avg / last"
assert args.prefix_type in [
    "query_or_passage",
    "instruction",
], "prefix_type should be query_or_passage / instruction"
os.makedirs(args.output_dir, exist_ok=True)
print("config_args", args)


def last_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    last_hidden = hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        emb = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        emb = last_hidden[torch.arange(batch_size), sequence_lengths]

    return emb


def input_transform_func(
    tokenizer: PreTrainedTokenizerFast, examples: Dict[str, List]
) -> BatchEncoding:
    max_length = 512
    return tokenizer(
        examples["contents"],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        truncation=True,
        return_tensors="pt",
    )


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.encoder = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        # self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        if args.prefix_type == "query_or_passage":
            input_texts = [f"{q}" for q in queries]
        else:
            input_texts = [self.prompt + q for q in queries]

        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d["text"] for d in corpus], **kwargs)

        input_texts = [
            "{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus
        ]
        # no need to add prefix for instruct models
        if args.prefix_type == "query_or_passage":
            input_texts = ["{}".format(t) for t in input_texts]

        return self._do_encode(input_texts)

    @torch.no_grad()
    def encode(self, input_texts: List[str], batch_size=1, **kwargs) -> np.ndarray:
        return self._do_encode(input_texts, batch_size, **kwargs)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], batch_size=1, **kwargs) -> np.ndarray:
        dataset: Dataset = Dataset.from_dict({"contents": input_texts})
        dataset.set_transform(
            partial(
                input_transform_func,
                self.tokenizer,
            )
        )

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True,
        )

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc="encoding", mininterval=10):
            batch_dict = move_to_cuda(batch_dict)
            batch_dict["output_hidden_states"] = True
            with torch.cuda.amp.autocast():

                outputs: BaseModelOutput = self.encoder(**batch_dict)
                last_hidden_state = outputs["hidden_states"][-1]
                embeds = last_pooling(last_hidden_state, batch_dict["attention_mask"])
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


def main():
    model = RetrievalModel()
    task_types = [
        "PairClassification", 
        "Classification", 
        "Retrieval", 
        "Reranking", 
        "Clustering",
        "STS"
        ]
    if args.task is not None:
        evaluation = [t for t in MTEB(tasks=[args.task]).tasks]
    else:
        evaluation = [t for t in MTEB(task_types=task_types, task_langs=['zh', 'zh-CN']).tasks]

    for task_cls in evaluation:
        task_name: str = task_cls.description["name"]
        if task_name == 'STS22':
            continue
        task_type: str = task_cls.description["type"]
        print('task_cls.description["path"]', task_cls.description)
        if args.prefix_type == "query_or_passage":
            prompt: str = "query: "
        else:
            task_def: str = get_task_def_by_task_name_and_type(
                task_name=task_name, task_type=task_type
            )
            prompt: str = get_detailed_instruct(task_def)
        model.set_prompt(prompt=prompt)
        logger.info("Set prompt: {}".format(prompt))

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == "Classification":
            logger.info("Set l2_normalize to False for classification task")
            model.l2_normalize = False
        else:
            model.l2_normalize = True
            logger.info("Set l2_normalize to {}".format(model.l2_normalize))

        sub_eval = MTEB(tasks=[task_name], task_langs=['zh'])
        logger.info(
            "Running evaluation for task: {}, type: {}".format(task_name, task_type)
        )
        eval_splits = (
            ["test"]
            if "test" in task_cls.description["eval_splits"]
            else task_cls.description["eval_splits"]
        )
        sub_eval.run(model, eval_splits=eval_splits, output_folder=args.output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
