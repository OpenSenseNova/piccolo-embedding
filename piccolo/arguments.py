from dataclasses import dataclass, field
from transformers import TrainingArguments
from piccolo.model import PoolingStrategy

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field() # must require
    embedding_strategy: PoolingStrategy = field(default=PoolingStrategy.mean)
    extend_pe: bool = field(default=False)
    max_length: int = field(default=512)
    # scaling layer and mrl Training
    use_scaling_layer: bool = field(default=False)
    use_mrl: bool = field(default=False)

@dataclass
class DataArguments:
    # train data
    meta_paths: list[str] = field() # must require
    root_dirs: list[str] = field() # must require
    batch_size: int = field(default=16)
    query_prefix: str = field(default='')
    doc_prefix: str = field(default='')
    # hard neg
    neg_num: int = field(default=1) # only affects retri_contrast_loss

@dataclass
class STETrainingArguments(TrainingArguments):
    efficient_save: bool = field(default=True)
