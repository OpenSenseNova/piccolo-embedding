import torch
import torch.distributed as dist
from uniem.utils import dist_gather_tensor

class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature


class PairInBatchNegCoSentLoss(ContrastLoss):
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
        sim_matrix_diag = sim_matrix.diag()
        sim_matrix_diff = sim_matrix - sim_matrix_diag.unsqueeze(1)
        loss = torch.logsumexp(sim_matrix_diff, dim=1).mean()
        return loss


class TripletInBatchNegCoSentLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, add_swap_loss: bool = False):
        super().__init__(temperature)
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_softmax_loss = PairInBatchNegCoSentLoss(temperature)
        else:
            self._pair_contrast_softmax_loss = None

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        sim_matrix_diff = sim_matrix - sim_matrix[:, 0].unsqueeze(1)
        loss = torch.logsumexp(sim_matrix_diff, dim=1).mean()
        if self._pair_contrast_softmax_loss:
            loss += self._pair_contrast_softmax_loss(text_pos_embeddings, text_embeddings)
        return loss


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
        labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class PairInBatchHardNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, neg_num=7, use_all_pair=False):
        super().__init__()
        self.temperature = temperature
        self.neg_num = neg_num
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_all_pair=use_all_pair

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor | None,
        text_neg_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_neg_embeddings is None:
            '''没有negative的情况下'''
            sim_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = sim_matrix / self.temperature
            labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)
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
            labels = torch.arange(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        else:
            sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
            sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
            labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        sim_matrix = sim_matrix / self.temperature
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss
    
class PairInBatchHardNegSoftmaxContrastLossV2(ContrastLoss):
    def __init__(self, temperature: float = 0.05, hard_loss_weight: int = 1):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.hard_loss_weight = hard_loss_weight
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor | None,
        text_neg_index: torch.Tensor | None
    ) -> torch.Tensor:
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_pos_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_pos_embeddings.unsqueeze(0),
            dim=-1,
        )
        
        # compute info NCE Loss with all neg
        sim_matrix = torch.cat([sim_pos_matrix, sim_neg_matrix], dim=1) / self.temperature
        labels = torch.arange(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = self._cross_entropy_loss(sim_matrix, labels)
        
        # compute info NCE Loss with hard neg
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        # mask = torch.zeros_like(sim_neg_matrix)
        # for i, index in enumerate(text_neg_index):
        #     mask[index][i] = 1
        # sim_neg_matrix_w_mask = mask * sim_neg_matrix
        # sim_neg_matrix_w_mask[~mask.bool()] = -1 # ignore other negative
        sim_hard_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1) / self.temperature
        labels_hard = torch.zeros(sim_hard_matrix.size(0), dtype=torch.long, device=sim_hard_matrix.device)
        loss_hard = self._cross_entropy_loss(sim_hard_matrix, labels_hard)
        print('easy loss: {}, hard loss: {}'.format(loss, loss_hard * self.hard_loss_weight))
        return loss + self.hard_loss_weight * loss_hard


class TripletInBatchNegSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, add_swap_loss: bool = False):
        super().__init__(temperature)
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_softmax_loss = PairInBatchNegSoftmaxContrastLoss(temperature)
        else:
            self._pair_contrast_softmax_loss = None

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
        if self._pair_contrast_softmax_loss:
            loss += self._pair_contrast_softmax_loss(text_pos_embeddings, text_embeddings)
        return loss


class PairInBatchNegSigmoidContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = text_embeddings.size(0)
        sim_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_pos_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = sim_matrix / self.temperature
        sim_matrix_diag = sim_matrix.diag()

        sim_diff_matrix = sim_matrix_diag.unsqueeze(1) - sim_matrix
        diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_diff_matrix = sim_diff_matrix.masked_fill(diag_mask, 1e9)

        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).sum() / (batch_size**2 - batch_size)
        return loss


class TripletInBatchNegSigmoidContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, add_swap_loss: bool = False):
        super().__init__(temperature)
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_sigmoid_loss = PairInBatchNegSigmoidContrastLoss(temperature)
        else:
            self._pair_contrast_sigmoid_loss = None

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        sim_pos_vector = sim_pos_vector / self.temperature
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_neg_matrix = sim_neg_matrix / self.temperature
        sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
        if self._pair_contrast_sigmoid_loss:
            loss += self._pair_contrast_sigmoid_loss(text_pos_embeddings, text_embeddings)
        return loss


class CoSentLoss(ContrastLoss):
    bias: torch.Tensor

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__(temperature)
        self.register_buffer('bias', torch.tensor([0.0]))

    def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        predict_similarity = predict_similarity / self.temperature
        cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
        smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff[~smaller_mask]
        # cosine_similarity_diff = cosine_similarity_diff.masked_fill(smaller_mask, -1e12)
        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff, self.bias))
        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0) * 0.02
        return loss

class ClsContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
        ) -> torch.Tensor:
        bs = text_embeddings.shape[0]
        assert text_neg_embeddings.shape[0] % bs == 0, 'neg num is not equal for each sample'
        neg_num = int(text_neg_embeddings.shape[0] // bs)

        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)

        # 找到每个sample自己的neg
        neg_matrix = []
        for i in range(bs):
            neg_matrix.append(sim_neg_matrix[i, i * neg_num : (i + 1) * neg_num])
        sim_neg_matrix = torch.stack(neg_matrix)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class ALLNegContrastLoss(ContrastLoss):
    # TODO: 处理一下没有negative,只用pos做大规模pretrain的情况

    def __init__(self, temperature: float = 0.05, neg_num=7, use_all_pair=False):
        super().__init__()
        self.temperature = temperature
        self.neg_num = neg_num
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_all_pair=use_all_pair
        self.cross_device_neg = True
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor | None,
        text_neg_index: torch.Tensor | None,
        need_norm: bool = False
        ) -> torch.Tensor:
        # 如果要使用hard negative reweight, 可以在gather前算一次hard loss
        def pad_neg(emb, num):
            pad_num = num - emb.shape[0]
            pad_tensor = torch.zeros((pad_num, emb.shape[1]), dtype=emb.dtype, device=emb.device)
            new_tensor = torch.cat([emb, pad_tensor])
            return new_tensor
        
        def remove_padding(emb, num, bs):
            valid_emb = []
            for i, n in enumerate(num):
                valid_emb.append(emb[i * bs: i * bs + int(n)])
            valid_emb = torch.concat(valid_emb)
            return valid_emb
        
        if self.cross_device_neg:
            # gather text and text_pos
            all_text_e = dist_gather_tensor(text_embeddings)
            all_text_pe = dist_gather_tensor(text_pos_embeddings)

            # negative的gather稍显复杂, 由于每个gpu的neg num不一样,需要先padding,然后gather,然后再去除padding, 不过实际上不去除padding应该也没关系
            bs = text_embeddings.shape[0]
            org_neg_num = torch.tensor([text_neg_embeddings.shape[0]], device=text_embeddings.device, dtype=torch.int)
            text_neg_embeddings = pad_neg(text_neg_embeddings, bs * self.neg_num)
            all_text_ne = dist_gather_tensor(text_neg_embeddings)
            all_neg_num = dist_gather_tensor(org_neg_num)
            all_text_ne = remove_padding(all_text_ne, all_neg_num, bs)
        else:
            all_text_e, all_text_pe, all_text_ne = text_embeddings, text_pos_embeddings, text_neg_embeddings

        if need_norm:
            all_text_e = torch.nn.functional.normalize(all_text_e, dim=-1)
            all_text_pe = torch.nn.functional.normalize(all_text_pe, dim=-1)
            all_text_ne = torch.nn.functional.normalize(all_text_ne, dim=-1)

        sim_neg_matrix = torch.matmul(all_text_e, all_text_ne.transpose(0, 1))
        sim_pos_matrix = torch.matmul(all_text_e, all_text_pe.transpose(0, 1))
        sim_matrix = torch.cat([sim_pos_matrix, sim_neg_matrix], dim=1)

        if need_norm:
            # 只有是算的cos similarity的时候,才需要用温度来缩放
            sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)

        # local_sim_matrix = sim_matrix[RANK * BATCH_SIZE: (RANK+1) * BATCH_SIZE]
        # local_labels = labels[RANK * BATCH_SIZE: (RANK+1) * BATCH_SIZE]
        loss = self._cross_entropy_loss(sim_matrix, labels)
        # print('rank: {}, loss: {}'.format(RANK, loss))
        return loss

'''
这个loss一直不太work,不知道为啥,反正就是,上面那个代码,开启local的就不太好.
'''
# class ALLNegContrastLossLocal(PairInBatchNegSoftmaxContrastLoss):
#     def forward(
#         self,
#         text_embeddings: torch.Tensor,
#         text_pos_embeddings: torch.Tensor,
#     ) -> torch.Tensor:
#         all_text_pe = dist_gather_tensor(text_pos_embeddings)
#         RANK, BATCH_SIZE = dist.get_rank(), text_embeddings.shape[0]
#         labels = torch.arange(RANK * BATCH_SIZE, (RANK+1) * BATCH_SIZE, device=text_embeddings.device, dtype=torch.long)
        
#         sim_matrix = torch.cosine_similarity(
#             text_embeddings.unsqueeze(1),
#             all_text_pe.unsqueeze(0),
#             dim=-1,
#         )
#         sim_matrix = sim_matrix / self.temperature
#         loss = self._cross_entropy_loss(sim_matrix, labels)
#         return loss
