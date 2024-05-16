from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, Type, TypeVar, cast

import numpy as np
import torch
import tqdm
import os

from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, AutoModelForCausalLM  # type: ignore
from accelerate import Accelerator

from uniem.criteria import (
    CoSentLoss,
    ClsContrastLoss,
    PairInBatchNegCoSentLoss,
    PairInBatchNegSigmoidContrastLoss,
    PairInBatchNegSoftmaxContrastLoss,
    ALLNegContrastLoss,
    PairInBatchHardNegSoftmaxContrastLoss,
    PairInBatchHardNegSoftmaxContrastLossV2,
    # ALLNegContrastLossLocal,
    TripletInBatchNegCoSentLoss,
    TripletInBatchNegSigmoidContrastLoss,
    TripletInBatchNegSoftmaxContrastLoss,
)
from uniem.types import Tokenizer
from uniem.utils import generate_batch
from .utils import AllGather_multi

T = TypeVar('T')


class PoolingStrategy(str, Enum):
    cls = 'cls'
    last_mean = 'last_mean'
    last_mean_dropout = 'last_mean_dropout'
    first_last_mean = 'first_last_mean'
    embedding_last_mean = 'embedding_last_mean'
    last_weighted = 'last_weighted'


class InBatchNegLossType(str, Enum):
    sigmoid = 'sigmoid'
    softmax = 'softmax'
    cosent = 'cosent'
    allneg_softmax = 'allneg_softmax'
    allneg_softmax_local = 'allneg_softmax_local'
    hardneg_softmax = 'hardneg_softmax'
    hardneg_softmax_v2 = 'hardneg_softmax_v2'
    cls_contrast = 'cls_contrast'

def build_loss(loss_type, temperature, **kwargs):
    loss_type = InBatchNegLossType(loss_type)
    match loss_type:
        case InBatchNegLossType.sigmoid:
            return PairInBatchNegSigmoidContrastLoss(temperature)
        case InBatchNegLossType.softmax:
            return PairInBatchNegSoftmaxContrastLoss(temperature)
        case InBatchNegLossType.cosent:
            return CoSentLoss(temperature)
        case InBatchNegLossType.cls_contrast:
            return ClsContrastLoss(temperature)
        case InBatchNegLossType.allneg_softmax:
            return ALLNegContrastLoss(temperature, **kwargs)
        case InBatchNegLossType.allneg_softmax_local:
            '''
            这个一直不太work, 不知道为啥
            '''
            return ALLNegContrastLossLocal(temperature)
        case InBatchNegLossType.hardneg_softmax:
            return PairInBatchHardNegSoftmaxContrastLoss(temperature, **kwargs)
        case InBatchNegLossType.hardneg_softmax_v2:
            return PairInBatchHardNegSoftmaxContrastLossV2(temperature)

def creat_attention_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return input_ids != pad_token_id


def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    return torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1, keepdim=True)


def last_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    last_hidden = hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        emb = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        emb = last_hidden[torch.arange(batch_size), sequence_lengths]

    return emb
    
def mean_pooling_with_dropout(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    vector_dropout = torch.nn.Dropout1d(0.3)
    if attention_mask is None:
        hidden_state = vector_dropout(hidden_state)
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    hidden_state = hidden_state * attention_mask.unsqueeze(-1)
    hidden_state = vector_dropout(hidden_state)
    return torch.sum(hidden_state, dim=1) / torch.sum(attention_mask, dim=-1, keepdim=True)


def load_hf_pretrained_model(model_name_or_path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.model_type == "t5":
        from transformers import T5EncoderModel  # type: ignore

        pretrained_model = T5EncoderModel.from_pretrained(model_name_or_path)
    elif config.model_type == "mistral":
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    return pretrained_model  # type: ignore


StrategyEmbedderClsMap: dict[PoolingStrategy, Type['Embedder']] = {}


class Embedder(torch.nn.Module):
    pooling_strategy: ClassVar[PoolingStrategy]

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.encoder.config.uniem_pooling_strategy = str(self.pooling_strategy.value)

        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id

    def __init_subclass__(cls) -> None:
        StrategyEmbedderClsMap[cls.pooling_strategy] = cls

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        encoder = load_hf_pretrained_model(model_name_or_path)
        return cls(encoder)

    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings


class LastMeanEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_mean

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = embeddings.hidden_states[-1]
        embeddings=last_pooling(last_hidden_state,attention_mask)
        return embeddings

class LastMeanEmbedderDropout(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_mean_dropout

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = mean_pooling_with_dropout(embeddings, attention_mask)
        return embeddings

class ClsEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.cls

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return embeddings


class FirstLastEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.first_last_mean

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        first_embeddings = mean_pooling(embeddings[0], attention_mask)
        last_embeddings = mean_pooling(embeddings[-1], attention_mask)
        embeddings = (first_embeddings + last_embeddings) / 2
        return embeddings


class EmbeddingLastEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.embedding_last_mean

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__(encoder, pad_token_id)
        self.embedding_layer = self.encoder.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        static_embeddings = self.embedding_layer(input_ids)
        mean_last_embeddings = mean_pooling(
            self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state, attention_mask
        )
        mean_static_embeddings = mean_pooling(static_embeddings, attention_mask)
        return (mean_last_embeddings + mean_static_embeddings) / 2


class LastWeightedEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_weighted

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__(encoder, pad_token_id)
        self.embedding_layer = self.encoder.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        weights = (torch.arange(input_ids.shape[1], device=input_ids.device) + 1).float()
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = embeddings * attention_mask.unsqueeze(-1).float() * weights.unsqueeze(0).unsqueeze(-1)
        embeddings = torch.sum(embeddings, dim=1) / torch.sum(weights * attention_mask, dim=-1, keepdim=True)
        return embeddings


class AutoEmbedder:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path):
        encoder = load_hf_pretrained_model(str(model_name_or_path))
        if hasattr(encoder.config, 'uniem_pooling_strategy'):
            strategy_string = encoder.config.uniem_pooling_strategy
        elif hasattr(encoder.config, 'uniem_embedding_strategy'):
            strategy_string = encoder.config.uniem_embedding_strategy
        else:
            raise ValueError('Can not find uniem pooling strategy in config, Model is not trained by UniEmbedder.')
        embedder_cls = StrategyEmbedderClsMap[PoolingStrategy(strategy_string)]
        return embedder_cls(encoder)


class EmbedderForTrain(torch.nn.Module):
    embedder: Embedder

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.embedder.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

class EmbedderForPairInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        loss_kwargs: dict, 
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
        freeze_pos_emb: bool = False,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        self.criterion = build_loss(**loss_kwargs)
        if freeze_pos_emb:
            for name, param in self.embedder.encoder.embeddings.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, text_neg_ids: torch.Tensor = None, **kwargs) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        if text_neg_ids is not None:
            text_neg_embeddings = self.embedder(text_neg_ids)
        else:
            text_neg_embeddings = None
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings, **kwargs)
        return {'loss': loss}

# class EmbedderForPairAllNegTrain(EmbedderForPairInBatchNegTrain):
#     def __init__(
#         self,
#         accelerator: Accelerator,
#         *args, **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.accelrator = accelerator

#     def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
#         text_embeddings = self.embedder(text_ids)
#         text_pos_embeddings = self.embedder(text_pos_ids)
#         text_embeddings = AllGather_multi.apply(text_embeddings, self.accelrator)
#         text_pos_embeddings = AllGather_multi.apply(text_pos_embeddings, self.accelrator)
#         loss = self.criterion(text_embeddings, text_pos_embeddings)
#         return {'loss': loss}

class EmbedderForTripletInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float | None = None,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
        add_swap_loss: bool = False,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        temperature = temperature or 0.05
        self.loss_type = InBatchNegLossType(loss_type)
        match self.loss_type:
            case InBatchNegLossType.sigmoid:
                self.criterion = TripletInBatchNegSigmoidContrastLoss(temperature, add_swap_loss)
            case InBatchNegLossType.softmax:
                self.criterion = TripletInBatchNegSoftmaxContrastLoss(temperature, add_swap_loss)
            case InBatchNegLossType.cosent:
                self.criterion = TripletInBatchNegCoSentLoss(temperature, add_swap_loss)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        text_neg_embeddings = self.embedder(text_neg_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}


class EmbedderForScoredPairTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float | None = None,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        temperature = temperature or 0.05
        self.criterion = CoSentLoss(temperature)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_pair_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pair_ids)
        predict_labels = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        loss = self.criterion(predict_labels, labels)
        return {'loss': loss, 'predict_labels': predict_labels}


class UniEmbedder:
    PROGRESS_BAR_THRESHOLD = 1000

    def __init__(
        self,
        embedder: Embedder,
        tokenizer: Tokenizer,
        normalize: bool = True,
        max_length: int | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.embedder = embedder.eval()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = self.embedder.to(device)
        self.tokenizer = tokenizer
        self.normalize = normalize
        self.max_length = (
            max_length or self.embedder.encoder.config.max_length or self.embedder.encoder.config.max_position_embeddings
        )

    def __call__(self, sentences: list[str], batch_size: int = 32):
        return self.encode(sentences, batch_size)

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        progress_bar: Literal['auto'] | bool = 'auto',
    ):
        embeddings: list[np.ndarray] = []
        if progress_bar == 'auto':
            progress_bar = len(sentences) > self.PROGRESS_BAR_THRESHOLD

        for batch in tqdm.tqdm(
            generate_batch(sentences, batch_size),
            disable=not progress_bar,
            total=len(sentences) // batch_size,
            unit='batch',
            desc='Encoding',
        ):
            encodes = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
                max_length=self.max_length,
            )

            input_ids = encodes['input_ids']
            input_ids = cast(torch.Tensor, input_ids)
            input_ids = input_ids.to(self.embedder.encoder.device)

            attention_mask = encodes['attention_mask']
            attention_mask = cast(torch.Tensor, attention_mask)
            attention_mask = attention_mask.to(self.embedder.encoder.device)

            with torch.inference_mode():
                batch_embeddings = self.embedder(input_ids, attention_mask=attention_mask)
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                batch_embeddings = cast(torch.Tensor, batch_embeddings)
            embeddings.extend([i.cpu().numpy() for i in batch_embeddings])
        return embeddings

    def encode_single(self, sentence: str):
        return self.encode([sentence])[0]

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        encoder = AutoEmbedder.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(encoder, tokenizer, **kwargs)

    def save_pretrained(self, ouptut_dir: str):
        self.embedder.save_pretrained(ouptut_dir)
        self.tokenizer.save_pretrained(ouptut_dir)

class ScalingLayer(torch.nn.Module):
    '''为了适配sentence transfomer里的dense layer'''
    def __init__(self, origin_dim: int = 1024, scaling_dim: int = 1792):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=origin_dim, out_features=scaling_dim, bias=True)
    def forward(self, input):
        return self.linear(input)


class STEmbedder(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        loss_kwargs: dict, 
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
        freeze_pos_emb: bool = False,
        add_scaling_layer: bool = False,
        use_mrl: bool = False,
        add_cls_head: bool = False
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        self.criterion = build_loss(**loss_kwargs)
        self.cosent_loss = build_loss('cosent', temperature=0.05)
        self.cls_contrast_loss = build_loss('cls_contrast', temperature=0.05)
        self.use_mrl = use_mrl
        self.add_scaling_layer = add_scaling_layer

        if add_scaling_layer:
            scaling_layer_state_dict = torch.load(os.path.join(model_name_or_path, '2_Dense/pytorch_model.bin'))
            self.scaling_layer = ScalingLayer(origin_dim=1024, scaling_dim=1792) # hard code here
            self.scaling_layer.load_state_dict(scaling_layer_state_dict, strict=True)

        if use_mrl:
            self.mrl_nesting_list = [256, 512, 768, 1024, 1280, 1536, 1792] # hard code here

        if freeze_pos_emb:
            for name, param in self.embedder.encoder.embeddings.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False

        if add_cls_head:
            self.cls_head = torch.nn.Linear(1024, 2) # hard code here

    def get_embedding(self, text_ids):
        if text_ids is None:
            return None
        text_embeddings = self.embedder(text_ids)
        if self.add_scaling_layer:
            text_embeddings = self.scaling_layer(text_embeddings.half()).float()
        return text_embeddings
    
    def compute_cls_loss(self, text_ids: torch.Tensor, text_labels: torch.tensor):
        text_embeddings = self.get_embedding(text_ids)
        pred_cls = self.cls_head(text_embeddings.half())
        loss = torch.nn.functional.cross_entropy(pred_cls, text_labels)
        return {'loss': loss}

    def compute_cls_contrast_loss(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, 
                                  text_neg_ids: torch.Tensor = None, type: str = 'cls_contrast') -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = text_embeddings[..., :num_feat], text_pos_embeddings[..., :num_feat], text_neg_embeddings[..., :num_feat]
                loss += self.cls_contrast_loss(emb, pos_emb, neg_emb) / len(self.mrl_nesting_list)
        else:
            loss = self.cls_contrast_loss(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        print('cls contrast loss: ', loss)
        return {'loss': loss}

    def compute_triplet_loss(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, text_neg_ids: torch.Tensor = None, **kwargs) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = text_embeddings[..., :num_feat], text_pos_embeddings[..., :num_feat], text_neg_embeddings[..., :num_feat]
                loss += self.criterion(emb, pos_emb, neg_emb, **kwargs) / len(self.mrl_nesting_list)
        else:
            loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings, **kwargs)
        print('triplet loss: ', loss)
        return {'loss': loss}

    def compute_scored_pair_loss(self, text_ids: torch.Tensor, text_pair_ids: torch.Tensor, labels: torch.Tensor):
        text_embeddings = self.get_embedding(text_ids)
        text_pair_embeddings = self.get_embedding(text_pair_ids)
        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, emb_pair = text_embeddings[..., :num_feat], text_pair_embeddings[..., :num_feat]
                predict_labels = torch.cosine_similarity(emb, emb_pair, dim=-1)
                loss += self.cosent_loss(predict_labels, labels) / len(self.mrl_nesting_list)
        else:
            predict_labels = torch.cosine_similarity(text_embeddings, text_pair_embeddings, dim=-1)
            loss = self.cosent_loss(predict_labels, labels)
        print('cosent loss: ', loss)
        return {'loss': loss, 'predict_labels': predict_labels}

    def forward(self, **kwargs):
        if 'type' in kwargs and 'cls_contrast' == kwargs['type']:
            return self.compute_cls_contrast_loss(**kwargs)
        elif 'text_ids' in kwargs and 'text_pos_ids' in kwargs:
            return self.compute_triplet_loss(**kwargs)
        elif 'text_ids' in kwargs and 'text_pair_ids' in kwargs and 'labels' in kwargs:
            return self.compute_scored_pair_loss(**kwargs)
        elif 'text_ids' in kwargs and 'text_labels' in kwargs:
            return self.compute_cls_loss(**kwargs)
        else:
            raise NotImplementedError('not suuport current input kwargs')

class EmbedderForFTExtendPE(EmbedderForPairInBatchNegTrain):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        loss_kwargs: dict, 
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
    ):
        super().__init__(model_name_or_path=model_name_or_path,
                         loss_kwargs=loss_kwargs,embedding_strategy=embedding_strategy, 
                         freeze_pos_emb=False)

        # re-init the position embeddings
        org_pe = self.embedder.encoder.embeddings.position_embeddings
        pad_idx = self.embedder.encoder.embeddings.position_embeddings.padding_idx
        extended_pe = torch.nn.Embedding(max_length + 2, org_pe.embedding_dim, padding_idx=pad_idx) # 这里加个2，follow xlm-roberta
        for start_idx in range(0, max_length + 2, org_pe.num_embeddings): # 迭代式地去复制,从而扩增embedding
            end_idx = min(start_idx + org_pe.num_embeddings, max_length + 2)
            extended_pe.weight.data[start_idx : end_idx] = org_pe.weight.data[:end_idx - start_idx].clone()
        self.embedder.encoder.embeddings.position_embeddings = extended_pe
        self.embedder.encoder.embeddings.position_ids = torch.arange(max_length + 2).expand((1, -1))
        self.embedder.encoder.embeddings.token_type_ids = \
            torch.zeros(self.embedder.encoder.embeddings.position_ids.size(), dtype=torch.long)
        self.embedder.encoder.config.max_position_embeddings = max_length + 2


class EmbedderForRelativePE(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.05,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
    ):
        from third_model.xlm_roberta_rope import XLMRobertaModelROPE
        config = AutoConfig.from_pretrained(model_name_or_path)
        pretrained_model = XLMRobertaModelROPE.from_pretrained(model_name_or_path, config=config)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        self.criterion = build_loss(loss_type, temperature)
        
    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}
