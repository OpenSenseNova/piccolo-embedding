import functools
import gc
import logging
from functools import wraps
from itertools import islice
from typing import Callable, Generator, Iterable, TypeVar

import torch
from accelerate.utils.memory import should_reduce_batch_size
import torch.distributed as dist
from typing import List, Union, Optional, Tuple, Mapping, Dict

T = TypeVar('T')
logger = logging.getLogger(__name__)


def create_adamw_optimizer(model: torch.nn.Module, lr: float, weight_decay=1e-3):
    parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'layernorm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def split_dataset_dict(dataset_dict: dict[str, T]) -> tuple[T, T | None]:
    if isinstance(dataset_dict, dict):
        train_dataset = dataset_dict['train']
        if 'dev' in dataset_dict:
            validation_dataset = dataset_dict['dev']
        elif 'validation' in dataset_dict:
            validation_dataset = dataset_dict['validation']
        else:
            logger.warning(
                'No validation dataset found in dataset_dict, validation dataset key should be either "dev" or "validation"'
            )
            validation_dataset = None
    else:
        train_dataset = dataset_dict
        validation_dataset = None
    return train_dataset, validation_dataset


def find_executable_batch_size(function: Callable | None = None, starting_batch_size: int = 128):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    @wraps(function)
    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        is_manually_passed_batch_size = 'batch_size' in kwargs

        if is_manually_passed_batch_size:
            return function(*args, **kwargs)
        else:
            while True:
                if batch_size == 0:
                    raise RuntimeError('No executable batch size found, reached zero.')
                try:
                    kwargs['batch_size'] = batch_size
                    return function(*args, **kwargs)
                except Exception as e:
                    if should_reduce_batch_size(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_size //= 2
                        print('Reducing batch size to', batch_size)
                    else:
                        raise

    return decorator

import os
from accelerate import Accelerator
class AllGather_multi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, accelerator: Accelerator):
        ctx.rank = int(os.environ['RANK'])
        ctx.batch_size = tensor.shape[0]
        tensor = accelerator.gather(tensor)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )

def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

def write_out_param(output_dir, meta_paths, **kwargs):
    import yaml
    os.makedirs(os.path.join(output_dir, 'parameters'), exist_ok=True)
    # 将字典写入到 config.yaml 文件中
    with open(os.path.join(output_dir, 'parameters','config.yaml'), 'w') as f:
        yaml.dump(kwargs, f)
    
    meta_paths = meta_paths.strip().split(' ')
    with open(os.path.join(output_dir, 'parameters','data.txt'), 'w') as f:
        for meta_path in meta_paths:
            f.writelines(open(meta_path, 'r').readlines())
            f.writelines('\n\n')