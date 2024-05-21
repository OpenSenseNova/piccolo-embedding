import os
import yaml
import torch
from pathlib import Path
from optimum.bettertransformer import BetterTransformer

from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, load_from_disk
from dataclasses import asdict
from transformers import AutoTokenizer, HfArgumentParser, TrainerCallback
from transformers.trainer import Trainer
from transformers.trainer import logger, Optional


from piccolo.arguments import ModelArguments, DataArguments, STETrainingArguments
from piccolo.data import (
    UniCollator,
    UniDataset,
    DatsetWithInfo,
)
from piccolo.model import GPTEmbedder
from tqdm import tqdm


def load_all_datasets(
    meta_paths, root_dirs, query_prefix, doc_prefix
) -> list[DatsetWithInfo]:
    all_datasets = []
    for meta_path, root_dir in zip(meta_paths, root_dirs):
        CNT = 0
        meta_file = open(meta_path, "r")
        for line in tqdm(meta_file.readlines()):
            dataset_name, repeat_num = line.strip().split(" ")
            dataset_dict = load_from_disk(str(os.path.join(root_dir, dataset_name)))
            if isinstance(dataset_dict, dict):
                dataset: HfDataset = concatenate_datasets(list(dataset_dict.values()))
            else:
                dataset = dataset_dict
            for idx in range(int(repeat_num)):
                all_datasets.append(
                    DatsetWithInfo(
                        hf_dataset=dataset,
                        name=dataset_name + "_{}".format(idx),
                        query_prefix=query_prefix,
                        passage_prefix=doc_prefix,
                    )
                )
            CNT += 1
        print("loading {} datasets from path: {}".format(CNT, meta_path))
    return all_datasets


class MyCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, train_dataloader, **kwargs):
        train_dataloader.dataset.create_or_refresh_data()


class GPTTrainer(Trainer):
    def __init__(self, use_optimum, efficient_save, **kwargs):
        super().__init__(**kwargs)
        self.use_optimum = use_optimum
        self.efficient_save = efficient_save

    def _save(self, output_dir: Optional[str] = None, **kwargs):
        """save the unwrap model, bcz we use better transformer"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_dir)
        if self.use_optimum:
            from optimum.bettertransformer import BetterTransformer

            unwrap_model = BetterTransformer.reverse(self.model.embedder.encoder)
        else:
            unwrap_model = self.model.embedder.encoder
        if self.is_world_process_zero():
            unwrap_model.save_pretrained(
                output_dir, safe_serialization=self.args.save_safetensors
            )
            self.tokenizer.save_pretrained(output_dir)
            if hasattr(self.model, "scaling_layer"):
                scaling_layer_sd_st = {
                    "linear.weight": self.model.scaling_layer.state_dict()[
                        "linear.weight"
                    ].data.cpu(),
                    "linear.bias": self.model.scaling_layer.state_dict()[
                        "linear.bias"
                    ].data.cpu(),
                }
                torch.save(
                    scaling_layer_sd_st,
                    os.path.join(output_dir, "scaling_layer_st.bin"),
                )

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.efficient_save:
            """only save the model ckpt weights to save disk mem"""
            from transformers.trainer import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
        else:
            super()._save_checkpoint(model, trial, metrics)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, STETrainingArguments))
    parser.parse_args()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: STETrainingArguments

    # DataLoader and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    all_datasets = load_all_datasets(
        data_args.meta_paths,
        data_args.root_dirs,
        data_args.query_prefix,
        data_args.doc_prefix,
    )
    train_dataset = UniDataset(
        all_datasets,
        batch_size=data_args.batch_size,
        with_instruction=data_args.with_instruction,
        neg_num=data_args.neg_num,
        drop_last=data_args.drop_last,
    )
    data_collator = UniCollator(tokenizer=tokenizer, max_length=model_args.max_length)
    loss_kwargs = {
        "loss_type": model_args.loss_type,
        "temperature": model_args.temperature,
        "neg_num": data_args.neg_num,
        "use_all_pair": data_args.use_all_pair,
    }

    # Model
    model = GPTEmbedder(
        model_name_or_path=model_args.model_name_or_path,
        loss_kwargs=loss_kwargs,
        embedding_strategy=model_args.embedding_strategy,
        freeze_pos_emb=False,
        add_scaling_layer=model_args.use_scaling_layer,
        use_mrl=model_args.use_mrl,
        add_cls_head=model_args.add_cls_head,
    )
    model.embedder.encoder.config.pad_token_id = tokenizer.pad_token_id

    # If on A100 GPU, try this.
    if training_args.use_optimum:
        from optimum.bettertransformer import BetterTransformer

        model.embedder.encoder = BetterTransformer.transform(
            model.embedder.encoder
        )  # optimum better transformer

    # Trainer
    trainer = GPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[MyCallback],
        use_optimum=training_args.use_optimum,
        efficient_save=training_args.efficient_save,
    )

    # Save parameter model at the end
    if trainer.is_world_process_zero():
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(training_args.output_dir, "parameters")).mkdir(
            parents=True, exist_ok=True
        )
        # Save data list info
        meta_paths = data_args.meta_paths
        with open(
            os.path.join(training_args.output_dir, "parameters", "data.list"), "w"
        ) as f:
            for meta_path in meta_paths:
                f.writelines(f"list_name: {meta_path} \n")
                f.writelines(open(meta_path, "r").readlines())
                f.writelines("\n\n")

    # Run training
    if training_args.use_optimum:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            trainer.train()
    else:
        trainer.train()

    # Save parameter and model at the end
    if trainer.is_world_process_zero():
        trainer.save_model(training_args.output_dir, _internal_call=True)
        # Save parameter
        parameter_dict = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "train_args": asdict(training_args),
        }
        Path(os.path.join(training_args.output_dir, "parameters")).mkdir(
            parents=True, exist_ok=True
        )
        with open(
            os.path.join(training_args.output_dir, "parameters", "param.yaml"), "w"
        ) as yaml_file:
            yaml.dump(parameter_dict, yaml_file)


if __name__ == "__main__":
    main()
