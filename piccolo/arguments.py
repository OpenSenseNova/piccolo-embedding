from dataclasses import dataclass, field
from transformers import TrainingArguments
from piccolo.model import (
    InBatchNegLossType,
    PoolingStrategy,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field()  # must require
    temperature: float = field(default=0.01)
    loss_type: InBatchNegLossType = field(default=InBatchNegLossType.softmax)
    embedding_strategy: PoolingStrategy = field(default=PoolingStrategy.last_mean)
    extend_pe: bool = field(default=False)
    use_rope: bool = field(default=False)
    max_length: int = field(default=512)
    # scaling layer and mrl Training
    use_scaling_layer: bool = field(default=False)
    use_mrl: bool = field(default=False)
    add_cls_head: bool = field(default=False)


@dataclass
class DataArguments:
    # train data
    meta_paths: list[str] = field()  # must require
    root_dirs: list[str] = field()  # must require
    batch_size: int = field(default=16)
    with_instruction: bool = field(default=False)
    drop_last: bool = field(default=True)
    query_prefix: str = field(default="")
    doc_prefix: str = field(default="")
    # hard neg
    neg_num: int = field(default=1)  # only affects retri_contrast_loss
    use_all_pair: bool = field(default=False)


@dataclass
class STETrainingArguments(TrainingArguments):
    use_optimum: bool = field(default=False)
    efficient_save: bool = field(default=True)
