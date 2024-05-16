import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast
from datasets import load_dataset

import torch
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, RandomSampler

from uniem.data_structures import (
    PairRecord,
    PairNegRecord,
    PairScoredRecord,
    PairClsContrastRecord,
    PairCLSRecord,
    RecordType,
    ScoredPairRecord,
    TripletRecord,
    infer_record_type,
    record_type_cls_map,
)
from uniem.types import Tokenizer


class PairCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int, q_max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.q_max_length = q_max_length

    def __call__(self, records: list[PairRecord]) -> dict[str, torch.Tensor]:
        records = records[0]
        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.q_max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_ids = cast(torch.Tensor, text_ids)

        text_pos_ids = self.tokenizer(
            texts_pos,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pos_ids = cast(torch.Tensor, text_pos_ids)

        return {
            'text_ids': text_ids,
            'text_pos_ids': text_pos_ids,
        }

@DeprecationWarning
class PairNegCollator(PairCollator):
    def __call__(self, records: list[PairNegRecord]) -> dict[str, torch.Tensor]:
        records = records[0]
        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]
        texts_neg = []
        texts_neg_index = [] # 每个neg属于哪个text
        for i, record in enumerate(records):
            for neg in record.text_neg:
                texts_neg.append(neg)
                texts_neg_index.append(i)
        
        text_ids = self.tokenizer(texts, padding=True, max_length=self.q_max_length, truncation=True, return_tensors='pt',)['input_ids']
        text_pos_ids = self.tokenizer(texts_pos, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
        if len(texts_neg) > 0:
            text_neg_ids = self.tokenizer(texts_neg, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
        else:
            text_neg_ids = None
        return {
            'text_ids': cast(torch.Tensor, text_ids),
            'text_pos_ids': cast(torch.Tensor, text_pos_ids),
            'text_neg_ids': cast(torch.Tensor, text_neg_ids),
            'text_neg_index': cast(torch.Tensor, texts_neg_index)
        }

@DeprecationWarning
class PairScoredCollator(PairCollator):

    def __call__(self, records: list[PairScoredRecord]) -> dict[str, torch.Tensor]:
        records = records[0] # 由于目前使用的这种提前组batch的方法, 必须有这一步
        texts = [record.text for record in records]
        texts_pair = [record.text_pair for record in records]
        labels = [record.label for record in records]
        labels = torch.tensor(labels, dtype=torch.float32)

        text_ids = self.tokenizer(texts, padding=True, max_length=self.q_max_length, truncation=True, return_tensors='pt',)['input_ids']
        text_pair_ids = self.tokenizer(texts_pair, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']

        return {
            'text_ids': cast(torch.Tensor, text_ids),
            'text_pair_ids': cast(torch.Tensor, text_pair_ids),
            'labels': labels,
        }

class UniCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int, q_max_length: int = 64) -> None:
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
            # 对于cluster或者classification任务来说, pos和neg的长度一般都比较短, 而query的长度会更长一些
            text_ids = self.tokenizer(texts, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
            text_pos_ids = self.tokenizer(texts_pos, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
            text_neg_ids = self.tokenizer(texts_neg, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
            
            return {
                'text_ids': cast(torch.Tensor, text_ids),
                'text_pos_ids': cast(torch.Tensor, text_pos_ids),
                'text_neg_ids': cast(torch.Tensor, text_neg_ids), 
                'type': 'cls_contrast',
            }
        elif isinstance(records[0], PairNegRecord):
            texts = [record.text for record in records]
            texts_pos = [record.text_pos for record in records]
            texts_neg = []
            texts_neg_index = [] # 每个neg属于哪个text
            for i, record in enumerate(records):
                for neg in record.text_neg:
                    texts_neg.append(neg)
                    texts_neg_index.append(i)
        
            text_ids = self.tokenizer(texts, padding=True, max_length=self.q_max_length, truncation=True, return_tensors='pt',)['input_ids']
            text_pos_ids = self.tokenizer(texts_pos, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
            if len(texts_neg) > 0:
                text_neg_ids = self.tokenizer(texts_neg, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']
            else:
                text_neg_ids = None
            return {
                'text_ids': cast(torch.Tensor, text_ids),
                'text_pos_ids': cast(torch.Tensor, text_pos_ids),
                'text_neg_ids': cast(torch.Tensor, text_neg_ids),
                'text_neg_index': cast(torch.Tensor, texts_neg_index)
            }
        elif isinstance(records[0], PairScoredRecord):
            texts = [record.text for record in records]
            texts_pair = [record.text_pair for record in records]
            labels = [record.label for record in records]
            labels = torch.tensor(labels, dtype=torch.float32)

            text_ids = self.tokenizer(texts, padding=True, max_length=self.q_max_length, truncation=True, return_tensors='pt',)['input_ids']
            text_pair_ids = self.tokenizer(texts_pair, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt',)['input_ids']

            return {
                'text_ids': cast(torch.Tensor, text_ids),
                'text_pair_ids': cast(torch.Tensor, text_pair_ids),
                'labels': labels,
            }
        elif isinstance(records[0], PairCLSRecord):
            texts = [record.text for record in records]
            text_labels = [record.text_label for record in records]
            text_ids = self.tokenizer(texts, padding=True, max_length=self.q_max_length, truncation=True, return_tensors='pt',)['input_ids']
            return {
                'text_ids': cast(torch.Tensor, text_ids),
                'text_labels': torch.tensor(text_labels, dtype=torch.float32)
            }
        else:
            raise NotImplementedError("only support pairscored and pairneg records")

@DeprecationWarning
class TripletCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records: list[TripletRecord]) -> dict[str, torch.Tensor]:
        records = records[0]
        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]
        texts_neg = [record.text_neg for record in records]

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pos_ids = self.tokenizer(
            texts_pos,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_neg_ids = self.tokenizer(
            texts_neg,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']

        text_ids = cast(torch.Tensor, text_ids)
        text_pos_ids = cast(torch.Tensor, text_pos_ids)
        text_neg_ids = cast(torch.Tensor, text_neg_ids)
        return {
            'text_ids': text_ids,
            'text_pos_ids': text_pos_ids,
            'text_neg_ids': text_neg_ids,
        }

@DeprecationWarning
class ScoredPairCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records: list[ScoredPairRecord]) -> dict[str, torch.Tensor]:
        texts = [record.sentence1 for record in records]
        texts_pair = [record.sentence2 for record in records]
        labels = [record.label for record in records]
        labels = torch.tensor(labels, dtype=torch.float32)

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_ids = cast(torch.Tensor, text_ids)

        text_pair_ids = self.tokenizer(
            texts_pair,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pair_ids = cast(torch.Tensor, text_pair_ids)

        return {
            'text_ids': text_ids,
            'text_pair_ids': text_pair_ids,
            'labels': labels,
        }


class FinetuneDataset(Dataset):
    def __init__(
        self,
        dataset: HfDataset | Sequence[dict],
        record_type: RecordType | str | None = None,
    ) -> None:
        self.dataset = dataset
        if record_type:
            self.record_type = RecordType(record_type)
        else:
            self.record_type = infer_record_type(dataset[0])
        self.record_cls = record_type_cls_map[self.record_type]

    def __getitem__(self, index: int):
        record = self.dataset[index]
        return self.record_cls(**record)

    def __len__(self):
        return len(self.dataset)

class MixFinetuneDataset(Dataset):
    def __init__(
        self,
        datasets: list[HfDataset],
        record_type: RecordType | str,
        split: str = 'train',
    ) -> None:
        self.record_type = record_type
        self.record_cls = record_type_cls_map[self.record_type]
        self.dataset = []
        for name, dataset in datasets.items():
            for data in dataset[split]:
                if 'STS_B' == name:
                    label = int(data['label'] > 2.5)
                    self.dataset.append({'sentence1': data['sentence1'], 'sentence2': data['sentence2'], 'label': label})
                else:
                    self.dataset.append(data)
                    
    def __getitem__(self, index: int):
        record = self.dataset[index]
        return self.record_cls(**record)

    def __len__(self):
        return len(self.dataset)

class PrefixFinetuneDataset(FinetuneDataset):
    def __init__(
        self,
        dataset: HfDataset | Sequence[dict],
        prefix: str,
        record_type: RecordType | str | None = None,
    ) -> None:
        super().__init__(dataset=dataset, record_type=record_type)
        self.prefix = prefix

    def __getitem__(self, index: int):
        record = self.dataset[index]
        match self.record_type:
            case RecordType.PAIR:
                record['text'] = self.prefix + record['text']
            case RecordType.TRIPLET:
                record['text'] = self.prefix + record['text']
            case RecordType.SCORED_PAIR:
                record['sentence1'] = self.prefix + record['sentence1']
        return self.record_cls(**record)


class MediDataset(Dataset):
    def __init__(
        self,
        medi_data_file: str | Path,
        batch_size: int = 32,
        pair_or_triplet: str = 'triplet',
        with_prompt: bool = True,
        join_with: str = '\n',
        drop_last: bool = True,
    ):
        medi_data = json.load(fp=Path(medi_data_file).open())
        self.batch_size = batch_size
        self.join_with = join_with
        self.drop_last = drop_last
        assert pair_or_triplet in ('pair', 'triplet')

        self._task_records_map: dict[str, list[TripletRecord | PairRecord]] = defaultdict(list)
        for record in medi_data:
            taks_name = record['task_name']
            if with_prompt:
                if pair_or_triplet == 'triplet':
                    record = TripletRecord(
                        text=join_with.join(record['query']),
                        text_pos=join_with.join(record['pos']),
                        text_neg=join_with.join(record['neg']),
                    )
                else:
                    record = PairRecord(
                        text=join_with.join(record['query']),
                        text_pos=join_with.join(record['pos']),
                    )
            else:
                if pair_or_triplet == 'triplet':
                    record = TripletRecord(
                        text=record['query'][1],
                        text_pos=record['pos'][1],
                        text_neg=record['neg'][1],
                    )
                else:
                    record = PairRecord(
                        text=record['query'][1],
                        text_pos=record['pos'][1],
                    )
            self._task_records_map[taks_name].append(record)
        self.create_or_refresh_data()

    def create_or_refresh_data(self):
        batch_size = self.batch_size
        self.batched_records = []
        for _, records in self._task_records_map.items():
            buffer = []

            num_samples = (len(records) // batch_size) * batch_size
            if not self.drop_last and len(records) % batch_size != 0:
                num_samples += batch_size

            if not num_samples:
                self.batched_records.append(records)
                continue

            for i in RandomSampler(records, num_samples=num_samples):
                buffer.append(records[i])
                if len(buffer) == batch_size:
                    self.batched_records.append(buffer)
                    buffer = []
        self.random_index_list = list(RandomSampler(self.batched_records))

    def __getitem__(self, index):
        index = self.random_index_list[index]
        return self.batched_records[index]

    def __len__(self):
        return len(self.batched_records)


@dataclass
class TaskBatchIndex:
    name: str
    batch_index: list[int]


@dataclass
class M3EHfDatsetWithInfo:
    hf_dataset: HfDataset
    name: str
    query_prefix: str = ''
    passage_prefix: str = ''


# Moka Massive Mixed Embedding Dataset
@DeprecationWarning
class M3EDataset(Dataset):
    def __init__(
        self,
        m3e_hf_datasets: list[M3EHfDatsetWithInfo],
        batch_size: int = 32,
        with_instruction: bool = False,
        drop_last: bool = True,
        max_samples: int | None = None,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.m3e_hf_datasets = m3e_hf_datasets
        self.max_samples = max_samples
        self.name_dataset_map = {dataset.name: dataset.hf_dataset for dataset in m3e_hf_datasets}
        self.with_instruction = with_instruction
        if with_instruction:
            self.query_prefix_map = {dataset.name: dataset.query_prefix for dataset in m3e_hf_datasets}
            self.passage_prefix_map = {dataset.name: dataset.passage_prefix for dataset in m3e_hf_datasets}
        else:
            self.query_prefix_map, self.passage_prefix_map = None, None
        self.create_or_refresh_data()

    @staticmethod
    def is_valid_text(text: Any) -> bool:
        return isinstance(text, str) and bool(text.strip())

    def create_or_refresh_data(self):
        self.task_batch_index_list: list[TaskBatchIndex] = []
        for dataset in self.m3e_hf_datasets:
            max_samples = self.max_samples or len(dataset.hf_dataset)
            num_samples = (max_samples // self.batch_size) * self.batch_size
            buffer = []
            for i in RandomSampler(dataset.hf_dataset, num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == self.batch_size:
                    self.task_batch_index_list.append(TaskBatchIndex(name=dataset.name, batch_index=buffer))
                    buffer = []
        self.random_index_list = list(RandomSampler(self.task_batch_index_list))

    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index
        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        pair_records = []
        for record in records:
            text = record['text']
            if isinstance(record['text_pos'], list): # random sample a positive
                text_pos = random.sample(record['text_pos'], 1)[0]
            else:
                text_pos = record['text_pos']
            if not (self.is_valid_text(text) and self.is_valid_text(text_pos)):
                continue
            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pos = self.passage_prefix_map[task_name] + text_pos
            pair_records.append(PairRecord(text=text, text_pos=text_pos))
        if not pair_records:
            print(f'records is empty', records)
            return self.__getitem__(index + 1)
        return pair_records

    def __len__(self):
        return len(self.task_batch_index_list)

@DeprecationWarning
class M3EDatasetHardNeg(M3EDataset):
    '''
    构造数据集的时候就要保证,必须有neg_num个负样本,挖掘不出来可以随机sample些easy neg放进去
    '''
    def __init__(self, all_datasets, neg_num, **kwargs):
        super().__init__(all_datasets, **kwargs)
        
        # def flaten_2d_list(a):
        #     neg_list = [element for sublist in a for element in sublist]
        #     MAX_NUM = len(neg_list) / 5  # set MAX NUM for memory efficient
        #     return neg_list[ : int(MAX_NUM)]
        self.neg_num = neg_num
        # self.name_dataset_neg_map = {dataset.name: flaten_2d_list(dataset.hf_dataset['text_neg']) for dataset in self.m3e_hf_datasets}
    
    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index
        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        pair_records = []
        for record in records:
            text = record['text']
            if isinstance(record['text_pos'], list): # random sample a positive
                if len(record['text_pos']) == 0:
                    text_pos = text # 以自己为positive
                else:
                    text_pos = random.sample(record['text_pos'], 1)[0]
            else:
                text_pos = record['text_pos']
            
            if not (self.is_valid_text(text) and self.is_valid_text(text_pos)):
                continue
            
            # append neg list to const length
            text_neg = random.sample(record['text_neg'], min(self.neg_num, len(record['text_neg'])))
            # PADDING_NUM = self.neg_num - len(text_neg)
            # text_neg.extend(random.sample(self.name_dataset_neg_map[task_name], PADDING_NUM))

            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pos = self.passage_prefix_map[task_name] + text_pos
                text_neg = [self.passage_prefix_map[task_name] + neg for neg in text_neg]
            pair_records.append(PairNegRecord(text=text, text_pos=text_pos, text_neg=text_neg))
        if not pair_records:
            print(f'records is empty', records)
            return self.__getitem__(index + 1)
        return pair_records

@DeprecationWarning
class M3EDatasetWithScorePair(M3EDataset):

    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index
        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        pair_records = []
        for record in records:
            text = record['text']
            text_pair = record['text_pair']
            label = record['label']
            
            if not (self.is_valid_text(text) and self.is_valid_text(text_pair)):
                continue
            
            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pair = self.passage_prefix_map[task_name] + text_pair
            pair_records.append(PairScoredRecord(text=text, text_pair=text_pair, label=label))
        if not pair_records:
            print(f'records is empty', records)
            return self.__getitem__(index + 1)
        return pair_records

class UniDataset(Dataset):
    def __init__(
        self,
        m3e_hf_datasets: list[M3EHfDatsetWithInfo],
        neg_num: int = 1,
        batch_size: int = 32,
        with_instruction: bool = False,
        drop_last: bool = True,
        max_samples: int | None = None,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.m3e_hf_datasets = m3e_hf_datasets
        self.max_samples = max_samples
        self.name_dataset_map = {dataset.name: dataset.hf_dataset for dataset in m3e_hf_datasets}
        self.with_instruction = with_instruction
        self.neg_num = neg_num
        if with_instruction:
            self.query_prefix_map = {dataset.name: dataset.query_prefix for dataset in m3e_hf_datasets}
            self.passage_prefix_map = {dataset.name: dataset.passage_prefix for dataset in m3e_hf_datasets}
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
        for dataset in self.m3e_hf_datasets:
            max_samples = self.max_samples or len(dataset.hf_dataset)
            if 'type' in dataset.hf_dataset[0] and dataset.hf_dataset[0]['type'] == 'cls_contrast':
                batch_size = 32 # 对于分类/聚类数据,不能用InbatchNegative,而HardNeg较多(=类别数), 也许bs应该设置得小一些?
            else:
                batch_size = self.batch_size
            num_samples = (max_samples // batch_size) * batch_size
            buffer = []
            for i in RandomSampler(dataset.hf_dataset, num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == batch_size:
                    self.task_batch_index_list.append(TaskBatchIndex(name=dataset.name, batch_index=buffer))
                    buffer = []
        self.random_index_list = list(RandomSampler(self.task_batch_index_list))

    def get_clsf_records(self, records, task_name):
        cls_records = []
        for record in records:
            text = record['text']
            text_label = record['label']
            if not self.is_valid_text(text):
                continue
            cls_records.append(PairCLSRecord(text=text, text_label=text_label))
        return cls_records

    def get_pair_scored_records(self, records, task_name):
        pair_records = []
        for record in records:
            text = record['text']
            text_pair = record['text_pair']
            label = record['label']
            if not (self.is_valid_text(text) and self.is_valid_text(text_pair)):
                continue
            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pair = self.passage_prefix_map[task_name] + text_pair
            pair_records.append(PairScoredRecord(text=text, text_pair=text_pair, label=label))
        return pair_records

    def get_pair_contrast_records(self, records, task_name, hf_dataset, batch_index):

        def process_records(record):
            text = record['text']
            if isinstance(record['text_pos'], list): # random sample a positive
                if len(record['text_pos']) == 0:
                    text_pos = text # 以自己为positive
                else:
                    text_pos = random.sample(record['text_pos'], 1)[0]
            else:
                text_pos = record['text_pos']
 
            if not (self.is_valid_text(text) and self.is_valid_text(text_pos)):
                # skip current sample and random sample an index, 这里的目的是要保证batch size, 不然gather时会有问题 
                random_index = random.sample(range(len(hf_dataset)), k=1)[0]
                while random_index in batch_index:
                    random_index = random.sample(range(len(hf_dataset)), k=1)[0]
                return process_records(hf_dataset[random_index])
            # append neg list to const length
            text_neg = random.sample(record['text_neg'], min(self.neg_num, len(record['text_neg'])))
            # PADDING_NUM = self.neg_num - len(text_neg)
            # text_neg.extend(random.sample(self.name_dataset_neg_map[task_name], PADDING_NUM))

            if self.with_instruction:
                text = self.query_prefix_map[task_name] + text
                text_pos = self.passage_prefix_map[task_name] + text_pos
                text_neg = [self.passage_prefix_map[task_name] + neg for neg in text_neg]
            return text, text_pos, text_neg

        pair_records = []
        for record in records:
            text, text_pos, text_neg = process_records(record)
            pair_records.append(PairNegRecord(text=text, text_pos=text_pos, text_neg=text_neg))
        assert len(pair_records) == self.batch_size, 'error, current batch size not match !!!'
        return pair_records

    def get_pair_cls_contrast_records(self, records, task_name):
        pair_records = []
        for record in records:
            text, text_pos, text_neg = record['text'], record['text_pos'], record['text_neg']
            # 这里hard code了neg num的数目, 最大为10
            text_neg = random.sample(record['text_neg'], min(10, len(record['text_neg'])))
            if self.is_valid_text(text) and self.is_valid_text(text_pos):
                pair_records.append(PairClsContrastRecord(text=text, text_pos=text_pos, text_neg=text_neg))
        return pair_records

    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index
       
        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        # 判断当前的数据集类型, 不同类型有不同的算loss方法
        if 'type' in hf_dataset[0] and hf_dataset[0]['type'] == 'cls_contrast':
            pair_records = self.get_pair_cls_contrast_records(records, task_name)
        elif 'text' in hf_dataset[0] and 'text_pos' in hf_dataset[0]: # pair contrast dataset
            pair_records = self.get_pair_contrast_records(records, task_name, hf_dataset, batch_index) 
        elif 'text' in hf_dataset[0] and 'text_pair' in hf_dataset[0] and 'label' in hf_dataset[0]: # pair scored dataset
            pair_records = self.get_pair_scored_records(records, task_name)
        elif 'text' in hf_dataset[0] and 'text_label' in hf_dataset[0]:
            pair_records = self.get_clsf_records(records, task_name)
        else:
            raise NotImplementedError('only support pair contrast and pair scored')

        if not pair_records:
            print(f'records is empty', records)
            return self.__getitem__(index + 1)
        return pair_records

class WudaoCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int, q_max_length: int = 32) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records) -> dict[str, torch.Tensor]:
        texts = ['查询: ' + record['title'] for record in records]
        texts_pos = ['结果: ' + record['content'] for record in records]
        # iterative dataset在accelerator中默认使用 dispatch loader，由main worker统一fetch数据然后dispatch，所以每张卡的数据的长度要一致，因此这里设置padding=max_length
        text_ids = self.tokenizer(texts, padding="max_length", max_length=self.q_max_length, truncation=True, return_tensors='pt')['input_ids']
        text_pos_ids = self.tokenizer(texts_pos, padding="max_length", max_length=self.max_length, truncation=True, return_tensors='pt')['input_ids']

        text_ids = cast(torch.Tensor, text_ids)
        text_pos_ids = cast(torch.Tensor, text_pos_ids)
        return {'text_ids': text_ids, 'text_pos_ids': text_pos_ids}
