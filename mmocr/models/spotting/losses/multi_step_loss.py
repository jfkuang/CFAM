#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/15 17:49
# @Author : WeiHua
import torch.nn as nn
from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class MultiStepLoss(nn.Module):
    def __init__(self, ignore_index=-1, reduction='none',
                 ignore_first_char=False, rec_weights=1.0,
                 kie_weights=1.0):
        super(MultiStepLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction
        )
        self.ignore_first_char = ignore_first_char
        self.rec_weights = rec_weights
        self.kie_weights = kie_weights

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
        rec_cost_list = []
        kie_cost_list = []
        rec_loss, kie_loss = None, None
        cnt = 0
        for logit in outputs.get('REC', []):
            cur_loss = self.loss_ce(logit.reshape(-1, logit.shape[-1]), targets_dict['texts'].reshape(-1))
            if cnt == 0:
                rec_loss = cur_loss
            else:
                rec_loss += cur_loss
            cnt += 1
            rec_cost_list.append(cur_loss.detach().cpu())
        if not isinstance(rec_loss, type(None)):
            losses['loss_rec'] = self.rec_weights * rec_loss / cnt
            # losses['rec_cost_list'] = rec_cost_list

        cnt = 0
        for logit in outputs.get('KIE', []):
            cur_loss = self.loss_ce(logit.reshape(-1, logit.shape[-1]), targets_dict['entities'].reshape(-1))
            if cnt == 0:
                kie_loss = cur_loss
            else:
                kie_loss += cur_loss
            cnt += 1
            kie_cost_list.append(cur_loss.detach().cpu())
        if not isinstance(kie_loss, type(None)):
            losses['loss_kie'] = self.kie_weights * kie_loss / cnt
            # losses['kie_cost_list'] = kie_cost_list

        return losses
        


