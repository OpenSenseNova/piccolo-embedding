import os
import yaml
import torch
from pathlib import Path
from optimum.bettertransformer import BetterTransformer  # optional
from transformers.trainer import logger, Optional

import datasets
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, load_from_disk
from dataclasses import asdict
from transformers import AutoTokenizer, HfArgumentParser, TrainerCallback  # type: ignore
from transformers.trainer import Trainer


from uniem.arguments import ModelArguments, DataArguments, STETrainingArguments
from uniem.data import (
    UniCollator, 
    UniDataset,
    M3EDataset,
    M3EDatasetHardNeg,
    M3EHfDatsetWithInfo,
)
from uniem.model import (
    STEmbedder,
)
from uniem.trainer_ste import STETrainer
from tqdm import tqdm


def load_all_datasets(meta_paths, root_dirs, query_prefix, doc_prefix) -> list[M3EHfDatsetWithInfo]:
    all_datasets = []
    for meta_path, root_dir in zip(meta_paths, root_dirs):
        if root_dir.startswith("SDC-OSS-2"): # load from ceph
            from petrel_client.client import Client # sensecore上暂时不支持ceph
            client = Client('~/petreloss.conf')
            CNT = 0
            meta_file = open(meta_path, 'r')
            for line in tqdm(meta_file.readlines()):
                dataset_name, repeat_num = line.strip().split(' ')
                data_lists = []
                for fileitem in client.get_file_iterator(os.path.join(root_dir, dataset_name)):
                    if fileitem[0].endswith('.arrow'):
                        dataset_path = 'SDC-OSS-2:s3://' + fileitem[0]
                        data = client.get(dataset_path)
                        data = datasets.Dataset.from_buffer(data)
                        data_lists.append(data)
                dataset = concatenate_datasets(data_lists)
                for idx in range(int(repeat_num)):
                    all_datasets.append(
                        M3EHfDatsetWithInfo(hf_dataset=dataset, name=dataset_name + '_{}'.format(idx),
                            query_prefix=query_prefix, passage_prefix=doc_prefix)
                    )
                CNT += 1
        else:  # load from disk
            CNT = 0
            meta_file = open(meta_path, 'r')
            for line in tqdm(meta_file.readlines()):
                dataset_name, repeat_num = line.strip().split(' ')
                dataset_dict = load_from_disk(str(os.path.join(root_dir, dataset_name)))
                # domain, instruction, is_symmetric = dataset_dict.info.description.split('\n')[:3]
                if isinstance(dataset_dict, dict):
                    dataset: HfDataset = concatenate_datasets(list(dataset_dict.values()))
                else:
                    dataset = dataset_dict
                for idx in range(int(repeat_num)):
                    all_datasets.append(
                        M3EHfDatsetWithInfo(hf_dataset=dataset, name=dataset_name + '_{}'.format(idx),
                            query_prefix=query_prefix, passage_prefix=doc_prefix)
                    )
                CNT += 1
        print('loading {} datasets from path: {}'.format(CNT, meta_path))
    return all_datasets

class MyCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, train_dataloader, **kwargs):
        train_dataloader.dataset.create_or_refresh_data()

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, STETrainingArguments))
    parser.parse_args()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: STETrainingArguments

    # DataLoader
    if "Qwen-7B" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            pad_token="<|endoftext|>",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            # padding_side="right",
            # use_fast=False,
            trust_remote_code=True,
        )
        # tokenizer.pad_token_id = tokenizer.eod_id
    all_datasets = load_all_datasets(data_args.meta_paths, data_args.root_dirs, data_args.query_prefix, data_args.doc_prefix)
    train_dataset = UniDataset(all_datasets, batch_size=data_args.batch_size, with_instruction=data_args.with_instruction, 
                               neg_num=data_args.neg_num, drop_last=data_args.drop_last)
    data_collator = UniCollator(tokenizer=tokenizer, max_length=model_args.max_length)
    loss_kwargs = {'loss_type': model_args.loss_type, 'temperature': model_args.temperature,
                   'neg_num': data_args.neg_num, 'use_all_pair': data_args.use_all_pair}
 
    # Model
    if model_args.use_rope:
        model = EmbedderForRelativePE(
            model_name_or_path=model_args.model_name_or_path,
            temperature=model_args.temperature,
            loss_type=model_args.loss_type,
            embedding_strategy=model_args.embedding_strategy,
        )
    elif model_args.expand_pe:
        model = EmbedderForFTExtendPE(
            model_name_or_path=model_args.model_name_or_path,
            loss_kwargs=loss_kwargs,
            embedding_strategy=model_args.embedding_strategy,
            max_length=model_args.max_length,
        )
    else :
        model = STEmbedder(
            model_name_or_path=model_args.model_name_or_path,
            loss_kwargs = loss_kwargs,
            embedding_strategy=model_args.embedding_strategy,
            freeze_pos_emb=False, # TODO hard code here, 谁pretrain放开PE啊
            add_scaling_layer=model_args.use_scaling_layer,
            use_mrl=model_args.use_mrl,
            add_cls_head=model_args.add_cls_head
        )
    model.embedder.encoder.config.pad_token_id = tokenizer.pad_token_id
    
    # 如果在A100上，试一试
    if training_args.use_optimum:
        from optimum.bettertransformer import BetterTransformer # optional
        model.embedder.encoder = BetterTransformer.transform(model.embedder.encoder) # optimum better transformer
        # model.embedder.encoder = torch.compile(model.embedder.encoder)

    # Trainer
    trainer = STETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[MyCallback],
        use_optimum=training_args.use_optimum,
        efficient_save=training_args.efficient_save,
    )

    # save parameter and model at the end
    if trainer.is_world_process_zero():
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(training_args.output_dir, 'parameters')).mkdir(parents=True, exist_ok=True)
        ## save data list info
        meta_paths = data_args.meta_paths
        with open(os.path.join(training_args.output_dir, 'parameters','data.list'), 'w') as f:
            for meta_path in meta_paths:
                f.writelines(f'list_name: {meta_path} \n') 
                f.writelines(open(meta_path, 'r').readlines())
                f.writelines('\n\n')        ## todo write down data list

    # run training
    if training_args.use_optimum:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            trainer.train()
    else:
        trainer.train()

    # save parameter and model at the end
    if trainer.is_world_process_zero():
        trainer.save_model(training_args.output_dir, _internal_call=True)
        ## save parameter
        parameter_dict = {'model_args': asdict(model_args), 'data_args': asdict(data_args), 'train_args': asdict(training_args)}
        Path(os.path.join(training_args.output_dir, 'parameters')).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'parameters', 'param.yaml'), 'w') as yaml_file:
            yaml.dump(parameter_dict, yaml_file)



if __name__ == "__main__":
    main()
