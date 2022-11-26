#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/8 22:03
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmocr.models.builder import LOSSES

# Not verified yet.
class CustomLabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing and ignore_index.
    """
    def __init__(self, smoothing=0.1, ignore_index=-1, reduction="mean"):
        super(CustomLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        assert reduction in ['mean'], f"Unsupported: {reduction}"
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2

        with torch.no_grad():
            # prepare smoothing label
            off_value = self.smoothing / x.shape[-1]
            on_value = 1. - self.smoothing + off_value
            label = torch.full(x.shape, off_value, device=x.device)
            label.scatter_(1, target.unsqueeze(-1), on_value)

        logprobs = F.log_softmax(x, dim=-1)
        # shape: N
        loss = torch.sum(-label * logprobs, dim=-1)
        if self.reduction == "mean":
            return torch.masked_select(loss, target != self.ignore_index).mean()


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


#  Implementation of Label smoothing with CrossEntropy and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)


@LOSSES.register_module()
class MASTERTFLoss(nn.Module):
    def __init__(self, ocr_ignore_index=-1, kie_ignore_index=-1,
                 reduction='none', ignore_first_char=False,
                 rec_weights=1.0, kie_weights=1.0,
                 crf_weights=1.0,
                 allow_multi_step=False,
                 label_smoothing=0.0):
        super(MASTERTFLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']

        self.label_smoothing = label_smoothing
        if self.label_smoothing > 0:
            self.loss_ce_ocr = LabelSmoothingCrossEntropy(epsilon=label_smoothing,
                                                          reduction=reduction,
                                                          ignore_index=ocr_ignore_index)
            self.loss_ce_kie = LabelSmoothingCrossEntropy(epsilon=label_smoothing,
                                                          reduction=reduction,
                                                          ignore_index=ocr_ignore_index)
            # self.loss_ce_ocr = CustomLabelSmoothingCrossEntropy(smoothing=label_smoothing,
            #                                                     ignore_index=ocr_ignore_index,
            #                                                     reduction=reduction)
            # self.loss_ce_kie = CustomLabelSmoothingCrossEntropy(smoothing=label_smoothing,
            #                                                     ignore_index=kie_ignore_index,
            #                                                     reduction=reduction)
        else:
            self.loss_ce_ocr = nn.CrossEntropyLoss(
                ignore_index=ocr_ignore_index,
                reduction=reduction
            )
            self.loss_ce_kie = nn.CrossEntropyLoss(
                ignore_index=kie_ignore_index,
                reduction=reduction
            )
        self.ignore_first_char = ignore_first_char
        self.rec_weights = rec_weights
        self.kie_weights = kie_weights
        self.crf_weights = crf_weights
        self.allow_multi_step = allow_multi_step

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs: dict, each val is list[Tensor], which corresponds to output of different
                iteration. -> Shape (B x N, L, C)
            targets_dict: dict,
            img_metas:

        Returns:
            losses: dict
        """
        losses = dict()
        if not self.allow_multi_step:
            for key in ['REC', 'KIE', 'CRF']:
                if key in outputs:
                    assert len(outputs[key]) <= 1
            # assert len(outputs['REC']) <= 1
            # assert len(outputs['KIE']) <= 1
            # if 'CRF' in outputs:
            #     assert len(outputs['CRF']) <= 1
        if 'REC' in outputs:
            for idx, logit in enumerate(outputs['REC']):
                rec_loss = self.loss_ce_ocr(logit.reshape(-1, logit.shape[-1]), targets_dict['texts'][:, :, 1:].reshape(-1))
                losses[f'loss_rec_{idx}'] = self.rec_weights * rec_loss

        for idx, logit in enumerate(outputs['KIE']):
            kie_loss = self.loss_ce_kie(logit.reshape(-1, logit.shape[-1]), targets_dict['entities'][:, :, 1:].reshape(-1))
            losses[f'loss_kie_{idx}'] = self.kie_weights * kie_loss

        if 'CRF' in outputs:
            for idx, logit in enumerate(outputs['CRF']):
                if len(logit.shape) > 0:
                    crf_loss = torch.sum(logit) / logit.shape[0]
                    losses[f'loss_crf_{idx}'] = self.crf_weights * crf_loss
                else:
                    losses[f'loss_crf_{idx}'] = self.crf_weights * logit
        if 'contrast_pos_loss' in outputs:
            losses['loss_contrast_pos'] = outputs['contrast_pos_loss']

        kv_catcher_keys = ['loss_entity', 'loss_attn_map']
        for key in kv_catcher_keys:
            if key in outputs:
                losses[key] = outputs[key]

        return losses


# if __name__ == '__main__':
#     criterion = CustomLabelSmoothingCrossEntropy()
#     # criterion = LabelSmoothingCrossEntropy()
#     x = torch.FloatTensor([
#         [0.1, 0.2, 0.3, 1.5],
#         [0.9, 0.0, 0.5, 5],
#         [0., 0.0, 1.5, 0.0]
#     ])
#     # x = torch.FloatTensor([
#     #     [1.0, 0.0, 0.0, 0.0],
#     #     [0.0, 1, 0.0, 0.0],
#     #     [0., 0.0, 0.0, 1.0]
#     # ])
#
#     x.mean()
#     target = torch.LongTensor([0, 1, 3])
#
#     loss = criterion(x, target)
#     print(loss)
