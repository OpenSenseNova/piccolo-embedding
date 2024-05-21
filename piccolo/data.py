import random
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast, TypeAlias
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, RandomSampler

from piccolo.data_structures import (
    PairRetriContrastRecord,
    PairNegRecord,
    PairScoredRecord,
    PairClsContrastRecord,
    PairCLSRecord,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

Tokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


class UniCollator:
    """
    Uni Data Collator

    for retrieval, sts, pair classification, the query max length is 64, doc max length is 512
    for clustering and classification, we specially set the query max length to 512,
    bcz for clustering and classification task, query('text') is ofent much longer than the pos/neg ('label')
    """

    def __init__(
        self, tokenizer: Tokenizer, max_length: int, q_max_length: int = 64
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.q_max_length = q_max_length

    def __call__(self, records: list) -> dict[str, torch.Tensor]:
        records = records[0]
        if isinstance(records[0], PairClsContrastRecord):
            texts = [record.text for record in records]
            texts_pos = [record.text_pos for record in records]
            texts_neg = []
            for i, record in enumerate(records):
                for neg in record.text_neg:
                    texts_neg.append(neg)
            text_ids = self.tokenizer(
                texts,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            text_pos_ids = self.tokenizer(
                texts_pos,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            text_neg_ids = self.tokenizer(
                texts_neg,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

            return {
                "text_ids": cast(torch.Tensor, text_ids),
                "text_pos_ids": cast(torch.Tensor, text_pos_ids),
                "text_neg_ids": cast(torch.Tensor, text_neg_ids),
                "type": "cls_contrast",
            }
        elif isinstance(records[0], PairRetriContrastRecord):
            texts = [record.text for record in records]
            texts_pos = [record.text_pos for record in records]
            texts_neg = []
            texts_neg_index = (
                []
            )  # index indictates for which text the negative sample belongs
            for i, record in enumerate(records):
                for neg in record.text_neg:
                    texts_neg.append(neg)
                    texts_neg_index.append(i)

            text_ids = self.tokenizer(
                texts,
                padding=True,
                max_length=self.q_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            text_pos_ids = self.tokenizer(
                texts_pos,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            if len(texts_neg) > 0:
                text_neg_ids = self.tokenizer(
                    texts_neg,
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
            else:
                text_neg_ids = None
            return {
                "text_ids": cast(torch.Tensor, text_ids),
                "text_pos_ids": cast(torch.Tensor, text_pos_ids),
                "text_neg_ids": cast(torch.Tensor, text_neg_ids),
                "text_neg_index": cast(torch.Tensor, texts_neg_index),
                "type": "retri_contrast",
            }
        elif isinstance(records[0], PairNegRecord):
            texts = [record.text for record in records]
            texts_pos = [record.text_pos for record in records]
            texts_neg = []
            texts_neg_index = []
            for i, record in enumerate(records):
                for neg in record.text_neg:
                    texts_neg.append(neg)
                    texts_neg_index.append(i)

            text_ids = self.tokenizer(
                texts,
                padding=True,
                max_length=self.q_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            text_pos_ids = self.tokenizer(
                texts_pos,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            if len(texts_neg) > 0:
                text_neg_ids = self.tokenizer(
                    texts_neg,
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
            else:
                text_neg_ids = None
            return {
                "text_ids": cast(torch.Tensor, text_ids),
                "text_pos_ids": cast(torch.Tensor, text_pos_ids),
                "text_neg_ids": cast(torch.Tensor, text_neg_ids),
                "text_neg_index": cast(torch.Tensor, texts_neg_index),
            }
        elif isinstance(records[0], PairScoredRecord):
            texts = [record.text for record in records]
            texts_pair = [record.text_pair for record in records]
            labels = [record.label for record in records]
            labels = torch.tensor(labels, dtype=torch.float32)

            text_ids = self.tokenizer(
                texts,
                padding=True,
                max_length=self.q_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            text_pair_ids = self.tokenizer(
                texts_pair,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

            return {
                "text_ids": cast(torch.Tensor, text_ids),
                "text_pair_ids": cast(torch.Tensor, text_pair_ids),
                "labels": labels,
                "type": "cosent",
            }
        elif isinstance(records[0], PairCLSRecord):
            texts = [record.text for record in records]
            text_labels = [record.text_label for record in records]
            text_ids = self.tokenizer(
                texts,
                padding=True,
                max_length=self.q_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            return {
                "text_ids": cast(torch.Tensor, text_ids),
                "text_labels": torch.tensor(text_labels, dtype=torch.float32),
            }
        else:
            raise NotImplementedError("only support pairscored and pairneg records")


@dataclass
class TaskBatchIndex:
    name: str
    batch_index: list[int]


@dataclass
class DatsetWithInfo:
    hf_dataset: HfDataset
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""


class UniDataset(Dataset):
    """
    Task-Homogenous Dataset

    Code is modified from M3E: https://github.com/wangyuxinwhy/uniem
    It's also adopted in SFR Embedding: https://blog.salesforceairesearch.com/sfr-embedded-mistral

    This technique can ensure that the in-batch-negative samples come from the same data set,
    and it is generally believed that this can improve the quality of the in batch negatives.
    """

    def __init__(
        self,
        hf_datasets: list[DatsetWithInfo],
        neg_num: int = 1,
        batch_size: int = 32,
        with_instruction: bool = False,
        drop_last: bool = True,
        max_samples: int | None = None,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hf_datasets = hf_datasets
        self.max_samples = max_samples
        self.name_dataset_map = {
            dataset.name: dataset.hf_dataset for dataset in hf_datasets
        }
        self.with_instruction = with_instruction
        self.neg_num = neg_num
        if with_instruction:
            self.query_prefix_map = {
                dataset.name: dataset.query_prefix for dataset in hf_datasets
            }
            self.passage_prefix_map = {
                dataset.name: dataset.passage_prefix for dataset in hf_datasets
            }
        else:
            self.query_prefix_map, self.passage_prefix_map = None, None
        self.create_or_refresh_data()

    def __len__(self):
        return len(self.task_batch_index_list)

    @staticmethod
    def is_valid_text(text: Any) -> bool:
        return isinstance(text, str) and bool(text.strip())

    def create_or_refresh_data(self):
        self.task_batch_index_list: list[TaskBatchIndex] = []
        for dataset in self.hf_datasets:
            max_samples = self.max_samples or len(dataset.hf_dataset)
            batch_size = self.batch_size
            num_samples = (max_samples // batch_size) * batch_size
            buffer = []
            for i in RandomSampler(dataset.hf_dataset, num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == batch_size:
                    self.task_batch_index_list.append(
                        TaskBatchIndex(name=dataset.name, batch_index=buffer)
                    )
                    buffer = []
        self.random_index_list = list(RandomSampler(self.task_batch_index_list))

    def get_clsf_records(self, records, task_name):
        cls_records = []
        for record in records:
            text = record["text"]
            text_label = record["label"]
            if not self.is_valid_text(text):
                continue
            cls_records.append(PairCLSRecord(text=text, text_label=text_label))
        return cls_records

    def get_pair_scored_records(self, records, task_name):
        pair_records = []
        for record in records:
            text = record["text"]
            text_pair = record["text_pair"]
            label = record["label"]
            if not (self.is_valid_text(text) and self.is_valid_text(text_pair)):
                continue
            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pair = self.passage_prefix_map[task_name] + text_pair
            pair_records.append(
                PairScoredRecord(text=text, text_pair=text_pair, label=label)
            )
        return pair_records

    def get_pair_contrast_records(self, records, task_name, hf_dataset, batch_index):

        def process_records(record):
            text = record["text"]
            if isinstance(record["text_pos"], list):  # random sample a positive
                if len(record["text_pos"]) == 0:
                    text_pos = text
                else:
                    text_pos = random.sample(record["text_pos"], 1)[0]
            else:
                text_pos = record["text_pos"]

            if not (self.is_valid_text(text) and self.is_valid_text(text_pos)):
                # skip current sample and random sample an index, ensure right batch size
                random_index = random.sample(range(len(hf_dataset)), k=1)[0]
                while random_index in batch_index:
                    random_index = random.sample(range(len(hf_dataset)), k=1)[0]
                return process_records(hf_dataset[random_index])
            # append neg list to const length
            text_neg = random.sample(
                record["text_neg"], min(self.neg_num, len(record["text_neg"]))
            )

            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pos = self.passage_prefix_map[task_name] + text_pos
                text_neg = [
                    self.passage_prefix_map[task_name] + neg for neg in text_neg
                ]
            return text, text_pos, text_neg

        pair_records = []
        for record in records:
            text, text_pos, text_neg = process_records(record)
            pair_records.append(
                PairNegRecord(text=text, text_pos=text_pos, text_neg=text_neg)
            )
        assert (
            len(pair_records) == self.batch_size
        ), "error, current batch size not match !!!"
        return pair_records

    def get_pair_retri_contrast_records(
        self, records, task_name, hf_dataset, batch_index
    ):

        def process_records(record):
            text = record["text"]
            if isinstance(record["text_pos"], list):  # random sample a positive
                assert len(record["text_pos"]) >= 1, "text pos num should be at least 1"
                text_pos = random.sample(record["text_pos"], 1)[0]
            else:
                text_pos = record["text_pos"]

            if not (self.is_valid_text(text) and self.is_valid_text(text_pos)):
                # skip current sample and random sample an index
                random_index = random.sample(range(len(hf_dataset)), k=1)[0]
                while random_index in batch_index:
                    random_index = random.sample(range(len(hf_dataset)), k=1)[0]
                return process_records(hf_dataset[random_index])

            text_neg = random.sample(
                record["text_neg"], min(self.neg_num, len(record["text_neg"]))
            )
            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pos = self.passage_prefix_map[task_name] + text_pos
                text_neg = [self.passage_prefix_map[task_name] + neg for neg in text_neg]
            return text, text_pos, text_neg

        pair_records = []
        for record in records:
            text, text_pos, text_neg = process_records(record)
            pair_records.append(
                PairRetriContrastRecord(text=text, text_pos=text_pos, text_neg=text_neg)
            )
        assert (
            len(pair_records) == self.batch_size
        ), "error, current batch size not match !!!"
        return pair_records

    def get_pair_cls_contrast_records(self, records, task_name):
        pair_records = []
        for record in records:
            text, text_pos, text_neg = (
                record["text"],
                record["text_pos"],
                record["text_neg"],
            )
            if isinstance(record["text_pos"], list):
                text_pos = random.sample(record["text_pos"], 1)[0]
            elif isinstance(record["text_pos"], str):
                text_pos = record["text_pos"]
            else:
                assert False, "type error"
            text_neg = random.sample(
                record["text_neg"], min(10, len(record["text_neg"]))
            )  # TODO， hard code the neg num to 10
            if self.is_valid_text(text) and self.is_valid_text(text_pos):
                pair_records.append(
                    PairClsContrastRecord(
                        text=text, text_pos=text_pos, text_neg=text_neg
                    )
                )
        return pair_records

    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index

        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        if hf_dataset[0]["type"] == "cls_contrast":
            pair_records = self.get_pair_cls_contrast_records(records, task_name)
        elif hf_dataset[0]["type"] == "retri_contrast":
            pair_records = self.get_pair_retri_contrast_records(
                records, task_name, hf_dataset, batch_index
            )
        elif hf_dataset[0]["type"] == "cosent":
            pair_records = self.get_pair_scored_records(records, task_name)
        else:
            raise NotImplementedError("only support pair contrast and pair scored")

        if not pair_records:
            print(f"records is empty", records)
            return self.__getitem__(index + 1)
        return pair_records
