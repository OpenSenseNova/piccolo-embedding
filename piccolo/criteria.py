import torch


class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature


class PairInBatchNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_pos_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(
            sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long
        )
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class PairInBatchHardNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, neg_num=7, use_all_pair=False):
        super().__init__()
        self.temperature = temperature
        self.neg_num = neg_num
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_all_pair = use_all_pair

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor | None,
        text_neg_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_neg_embeddings is None:
            """For no neg"""
            sim_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = sim_matrix / self.temperature
            labels = torch.arange(
                sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long
            )
            loss = self._cross_entropy_loss(sim_matrix, labels)
            return loss

        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        if self.use_all_pair:
            sim_pos_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = torch.cat([sim_pos_matrix, sim_neg_matrix], dim=1)
            labels = torch.arange(
                sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device
            )
        else:
            sim_pos_vector = torch.cosine_similarity(
                text_embeddings, text_pos_embeddings, dim=-1
            )
            sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
            labels = torch.zeros(
                sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device
            )
        sim_matrix = sim_matrix / self.temperature
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class RetriContrastLoss(ContrastLoss):
    """
    loss for retrieval
    if use_all_pair is set to true, it will use the query-query pair as neg,
    otherwise it use query-passage as neg
    """

    def __init__(self, temperature: float = 0.05, use_all_pair=False):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_all_pair = use_all_pair

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor | None,
        text_neg_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_neg_embeddings is None:
            sim_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = sim_matrix / self.temperature
            labels = torch.arange(
                sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long
            )
            loss = self._cross_entropy_loss(sim_matrix, labels)
            return loss

        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        if self.use_all_pair:
            sim_pos_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = torch.cat([sim_pos_matrix, sim_neg_matrix], dim=1)
            labels = torch.arange(
                sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device
            )
        else:
            sim_pos_vector = torch.cosine_similarity(
                text_embeddings, text_pos_embeddings, dim=-1
            )
            sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
            labels = torch.zeros(
                sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device
            )
        sim_matrix = sim_matrix / self.temperature
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class CoSentLoss(ContrastLoss):
    """
    loss for sts and pair classification.
    here we hard code the cosent loss weight to 0.04
    """

    bias: torch.Tensor

    def __init__(self, temperature: float = 0.05, cosent_w: float = 0.04) -> None:
        super().__init__(temperature)
        self.register_buffer("bias", torch.tensor([0.0]))
        self.cosent_w = cosent_w

    def forward(
        self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor
    ) -> torch.Tensor:
        predict_similarity = predict_similarity / self.temperature
        cosine_similarity_diff = -(
            predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1)
        )
        smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff[~smaller_mask]
        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff, self.bias))
        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0) * self.cosent_w
        return loss


class ClsContrastLoss(torch.nn.Module):
    """
    loss for clustering and classification
    here we hard code the cls contrast loss weight to 0.2
    """

    def __init__(self, temperature: float = 0.05, cls_w=0.2):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cls_w = cls_w

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        bs = text_embeddings.shape[0]
        assert (
            text_neg_embeddings.shape[0] % bs == 0
        ), "neg num is not equal for each sample"
        neg_num = int(text_neg_embeddings.shape[0] // bs)

        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_pos_vector = torch.cosine_similarity(
            text_embeddings, text_pos_embeddings, dim=-1
        )

        # find the neg for eatch training sample
        neg_matrix = []
        for i in range(bs):
            neg_matrix.append(sim_neg_matrix[i, i * neg_num : (i + 1) * neg_num])
        sim_neg_matrix = torch.stack(neg_matrix)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.zeros(
            sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device
        )
        loss = self._cross_entropy_loss(sim_matrix, labels) * self.cls_w
        return loss
