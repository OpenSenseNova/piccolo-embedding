import numpy as np
import torch
import os

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
)
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, Type, TypeVar, cast

# type: ignore
from piccolo.criteria import (
    CoSentLoss,
    ClsContrastLoss,
    RetriContrastLoss,
    PairInBatchNegSoftmaxContrastLoss,
    PairInBatchHardNegSoftmaxContrastLoss,
)


class PoolingStrategy(str, Enum):
    cls = "cls"
    mean = "mean"
    last_mean = "last_mean"
    last_mean_dropout = "last_mean_dropout"


class InBatchNegLossType(str, Enum):
    cosent = "cosent"
    retri_contrast = "retri_contrast"
    softmax = "softmax"
    hardneg_softmax = "hardneg_softmax"
    cls_contrast = "cls_contrast"


def build_loss(loss_type, temperature, **kwargs):
    loss_type = InBatchNegLossType(loss_type)
    match loss_type:
        case InBatchNegLossType.cosent:
            return CoSentLoss(temperature)
        case InBatchNegLossType.cls_contrast:
            return ClsContrastLoss(temperature)
        case InBatchNegLossType.softmax:
            return PairInBatchNegSoftmaxContrastLoss(temperature)
        case InBatchNegLossType.hardneg_softmax:
            return PairInBatchHardNegSoftmaxContrastLoss(temperature, **kwargs)
        case InBatchNegLossType.retri_contrast:
            return RetriContrastLoss(temperature, **kwargs)


def creat_attention_mask_from_input_ids(
    input_ids: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    return input_ids != pad_token_id


def mean_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if attention_mask is None:
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    return torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(
        attention_mask, dim=-1, keepdim=True
    )


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


def load_hf_pretrained_model(model_name_or_path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.model_type == "t5":
        from transformers import T5EncoderModel  # type: ignore

        pretrained_model = T5EncoderModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    else:
        pretrained_model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    return pretrained_model  # type: ignore


StrategyEmbedderClsMap: dict[PoolingStrategy, Type["Embedder"]] = {}


class Embedder(torch.nn.Module):
    pooling_strategy: ClassVar[PoolingStrategy]

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.encoder.config.piccolo_pooling_strategy = str(self.pooling_strategy.value)

        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id

    def __init_subclass__(cls) -> None:
        StrategyEmbedderClsMap[cls.pooling_strategy] = cls

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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


class LastEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_mean

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(
                input_ids, self.pad_token_id
            )
        embeddings = self.encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_state = embeddings.hidden_states[-1]
        embeddings = last_pooling(last_hidden_state, attention_mask)
        return embeddings


class LastMeanEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.mean

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(
                input_ids, self.pad_token_id
            )
        embeddings = self.encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
        embeddings = mean_pooling(embeddings, attention_mask)
        return embeddings


class ClsEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.cls

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(
                input_ids, self.pad_token_id
            )
        embeddings = self.encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return embeddings


class EmbedderForTrain(torch.nn.Module):
    embedder: Embedder

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.embedder.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )


class ScalingLayer(torch.nn.Module):
    def __init__(self, origin_dim: int = 1024, scaling_dim: int = 1792):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=origin_dim, out_features=scaling_dim, bias=True
        )

    def forward(self, input):
        return self.linear(input)


class STEmbedder(EmbedderForTrain):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model_name_or_path (`str` ):
            the path of the pretrain model
        embedding_strategy (`PoolingStrategy`):
            only support 'mean' and 'cls'
            for 'mean', we use the average embedding across all valid tokens embeddings of BERT's last layer.
            for 'cls', we use the first token embedding of the BERT's last layer
        freeze_pos_emb (`bool`):
            whether fix the position embedding or not, default is True
        add_scaling_layer (`bool`):
            add a scaling layer after the last layer of BERT, it is used for dimension scaling.
        use_mrl (`bool`)
            employ the Matryoshka Representation Learning algorithm to support flexible dimension
        extend_pe (`bool`)
            extend position embedding to longer length, here we adopt a very simple method from:
            https://kexue.fm/archives/7947

    Notation:
        some parameter are hard-coded in this code, such as in_feature and out_feature of scaling layer, and nesting list of MRL.
    """

    def __init__(
        self,
        model_name_or_path: str,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.mean,
        freeze_pos_emb: bool = True,
        add_scaling_layer: bool = False,
        use_mrl: bool = False,
        extend_pe: bool = False,
        max_length: int = 512,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](
            pretrained_model
        )
        super().__init__(embedder)
        self.retri_contrst_loss = build_loss(
            "retri_contrast", temperature=0.01, use_all_pair=True
        )
        self.cosent_loss = build_loss("cosent", temperature=0.05)
        self.cls_contrast_loss = build_loss("cls_contrast", temperature=0.05)
        self.use_mrl = use_mrl
        self.add_scaling_layer = add_scaling_layer

        if add_scaling_layer:
            """
            Here we hard code the scaling layer pretrain path, input_dim and output_dim, you can modify it by yourself
            """
            self.scaling_layer = ScalingLayer(origin_dim=1024, scaling_dim=1792)
            if os.path.exists(
                os.path.join(model_name_or_path, "2_Dense/pytorch_model.bin")
            ):
                scaling_layer_state_dict = torch.load(
                    os.path.join(model_name_or_path, "2_Dense/pytorch_model.bin")
                )
                self.scaling_layer.load_state_dict(
                    scaling_layer_state_dict, strict=True
                )
                print("load scaling layer successfully")
            else:
                print("not found pretrain, random init scaling layer")

        if use_mrl:
            self.mrl_nesting_list = [
                256,
                512,
                768,
                1024,
                1280,
                1536,
                1792,
            ]  # hard code here

        if extend_pe:
            sp = 0  # TODO, hard code here, for xlm roberta, this should be 2
            # re-init the position embeddings
            org_pe = self.embedder.encoder.embeddings.position_embeddings
            pad_idx = self.embedder.encoder.embeddings.position_embeddings.padding_idx
            extended_pe = torch.nn.Embedding(
                max_length + sp, org_pe.embedding_dim, padding_idx=pad_idx
            )
            for start_idx in range(
                0, max_length + sp, org_pe.num_embeddings
            ):  # 迭代式地去复制,从而扩增embedding
                end_idx = min(start_idx + org_pe.num_embeddings, max_length + sp)
                extended_pe.weight.data[start_idx:end_idx] = org_pe.weight.data[
                    : end_idx - start_idx
                ].clone()
            self.embedder.encoder.embeddings.position_embeddings = extended_pe
            self.embedder.encoder.embeddings.position_ids = torch.arange(
                max_length + sp
            ).expand((1, -1))
            self.embedder.encoder.embeddings.token_type_ids = torch.zeros(
                self.embedder.encoder.embeddings.position_ids.size(), dtype=torch.long
            )
            self.embedder.encoder.config.max_position_embeddings = max_length + sp

        if not extend_pe and freeze_pos_emb:  # extend pe时, 不能 freeze pos emb
            for name, param in self.embedder.encoder.embeddings.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False

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
        return {"loss": loss}

    def compute_cls_contrast_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor = None,
        type: str = "cls_contrast",
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = (
                    text_embeddings[..., :num_feat],
                    text_pos_embeddings[..., :num_feat],
                    text_neg_embeddings[..., :num_feat],
                )
                loss += self.cls_contrast_loss(emb, pos_emb, neg_emb) / len(
                    self.mrl_nesting_list
                )
        else:
            loss = self.cls_contrast_loss(
                text_embeddings, text_pos_embeddings, text_neg_embeddings
            )
        print("cls contrast loss: ", loss)
        return {"loss": loss}

    def compute_retri_contrast_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor = None,
        type: str = "retri_contrast",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                if text_neg_embeddings is not None:
                    emb, pos_emb, neg_emb = (
                        text_embeddings[..., :num_feat],
                        text_pos_embeddings[..., :num_feat],
                        text_neg_embeddings[..., :num_feat],
                    )
                else:
                    emb, pos_emb = (
                        text_embeddings[..., :num_feat],
                        text_pos_embeddings[..., :num_feat],
                    )
                    neg_emb = None
                loss += self.retri_contrst_loss(emb, pos_emb, neg_emb, **kwargs) / len(
                    self.mrl_nesting_list
                )
        else:
            loss = self.retri_contrst_loss(
                text_embeddings, text_pos_embeddings, text_neg_embeddings, **kwargs
            )
        print("triplet loss: ", loss)
        return {"loss": loss}

    def compute_cosent_loss(
        self,
        text_ids: torch.Tensor,
        text_pair_ids: torch.Tensor,
        labels: torch.Tensor,
        type: str = "cosent",
    ):
        text_embeddings = self.get_embedding(text_ids)
        text_pair_embeddings = self.get_embedding(text_pair_ids)
        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, emb_pair = (
                    text_embeddings[..., :num_feat],
                    text_pair_embeddings[..., :num_feat],
                )
                predict_labels = torch.cosine_similarity(emb, emb_pair, dim=-1)
                loss += self.cosent_loss(predict_labels, labels) / len(
                    self.mrl_nesting_list
                )
        else:
            predict_labels = torch.cosine_similarity(
                text_embeddings, text_pair_embeddings, dim=-1
            )
            loss = self.cosent_loss(predict_labels, labels)
        print("cosent loss: ", loss)
        return {"loss": loss, "predict_labels": predict_labels}

    def forward(self, **kwargs):
        if kwargs["type"] == "cls_contrast":
            return self.compute_cls_contrast_loss(**kwargs)
        elif kwargs["type"] == "retri_contrast":
            return self.compute_retri_contrast_loss(**kwargs)
        elif kwargs["type"] == "cosent":
            return self.compute_cosent_loss(**kwargs)
        else:
            raise NotImplementedError("not suuport current input kwargs")


class GPTEmbedder(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        loss_kwargs: dict,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
        freeze_pos_emb: bool = False,
        add_scaling_layer: bool = False,
        use_mrl: bool = False,
        add_cls_head: bool = False,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](
            pretrained_model
        )
        super().__init__(embedder)
        self.criterion = build_loss(**loss_kwargs)
        self.cosent_loss = build_loss("cosent", temperature=0.05)
        self.cls_contrast_loss = build_loss("cls_contrast", temperature=0.05)
        self.use_mrl = use_mrl
        self.add_scaling_layer = add_scaling_layer

        if add_scaling_layer:
            scaling_layer_state_dict = torch.load(
                os.path.join(model_name_or_path, "2_Dense/pytorch_model.bin")
            )
            self.scaling_layer = ScalingLayer(
                origin_dim=1024, scaling_dim=1792
            )  # hard code here
            self.scaling_layer.load_state_dict(scaling_layer_state_dict, strict=True)

        if use_mrl:
            self.mrl_nesting_list = [
                256,
                512,
                768,
                1024,
                1280,
                1536,
                1792,
            ]  # hard code here

        if freeze_pos_emb:
            for name, param in self.embedder.encoder.embeddings.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False

        if add_cls_head:
            self.cls_head = torch.nn.Linear(1024, 2)  # hard code here

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
        return {"loss": loss}

    def compute_cls_contrast_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor = None,
        type: str = "cls_contrast",
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = (
                    text_embeddings[..., :num_feat],
                    text_pos_embeddings[..., :num_feat],
                    text_neg_embeddings[..., :num_feat],
                )
                loss += self.cls_contrast_loss(emb, pos_emb, neg_emb) / len(
                    self.mrl_nesting_list
                )
        else:
            loss = self.cls_contrast_loss(
                text_embeddings, text_pos_embeddings, text_neg_embeddings
            )
        print("cls contrast loss: ", loss)
        return {"loss": loss}

    def compute_triplet_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor = None,
        type: str = "triplet_loss",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = (
                    text_embeddings[..., :num_feat],
                    text_pos_embeddings[..., :num_feat],
                    text_neg_embeddings[..., :num_feat],
                )
                loss += self.criterion(emb, pos_emb, neg_emb, **kwargs) / len(
                    self.mrl_nesting_list
                )
        else:
            loss = self.criterion(
                text_embeddings, text_pos_embeddings, text_neg_embeddings, **kwargs
            )
        print("triplet loss: ", loss)
        return {"loss": loss}

    def compute_scored_pair_loss(
        self,
        text_ids: torch.Tensor,
        text_pair_ids: torch.Tensor,
        labels: torch.Tensor,
        type: str = "cosent",
    ):
        text_embeddings = self.get_embedding(text_ids)
        text_pair_embeddings = self.get_embedding(text_pair_ids)
        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, emb_pair = (
                    text_embeddings[..., :num_feat],
                    text_pair_embeddings[..., :num_feat],
                )
                predict_labels = torch.cosine_similarity(emb, emb_pair, dim=-1)
                loss += self.cosent_loss(predict_labels, labels) / len(
                    self.mrl_nesting_list
                )
        else:
            predict_labels = torch.cosine_similarity(
                text_embeddings, text_pair_embeddings, dim=-1
            )
            loss = self.cosent_loss(predict_labels, labels)
        print("cosent loss: ", loss)
        return {"loss": loss, "predict_labels": predict_labels}

    def forward(self, **kwargs):
        if "type" in kwargs and "cls_contrast" == kwargs["type"]:
            return self.compute_cls_contrast_loss(**kwargs)
        elif "text_ids" in kwargs and "text_pos_ids" in kwargs:
            return self.compute_triplet_loss(**kwargs)
        elif "text_ids" in kwargs and "text_pair_ids" in kwargs and "labels" in kwargs:
            return self.compute_scored_pair_loss(**kwargs)
        elif "text_ids" in kwargs and "text_labels" in kwargs:
            return self.compute_cls_loss(**kwargs)
        else:
            raise NotImplementedError("not suuport current input kwargs")
