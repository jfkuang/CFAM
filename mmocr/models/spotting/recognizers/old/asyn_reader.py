#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/12 17:12
# @Author : WeiHua
import ipdb
import torch
import torch.nn as nn

from mmocr.models.textrecog.recognizer.base import BaseRecognizer
from mmocr.models.builder import DETECTORS, build_loss
from mmocr.models import PositionalEncoding
from mmocr.models.spotting.modules.ViTSTR import CustomViTSTR


# todo:finish this part
@DETECTORS.register_module()
class AsynReader(BaseRecognizer):
    def __init__(self,
                 rec_cfg,
                 kie_cfg,
                 fusion_pe_args={},
                 loss=None,
                 use_ins_mask=False,
                 use_seq_mask=False,
                 share_fuser=False,
                 share_attn=False,
                 rec_only=False,
                 iter_num=3,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        """
        Asynchronous Reader for REC and KIE task.
        Args:
            rec_cfg:
            kie_cfg:
            fusion_pe_args:
            share_fuser:
            share_attn:
            rec_only:
            train_cfg:
            test_cfg:
        """
        super().__init__(init_cfg=init_cfg)
        self.rec_cfg = rec_cfg.copy()
        self.kie_cfg = kie_cfg.copy()
        self.use_ins_mask = use_ins_mask
        self.use_seq_mask = use_seq_mask
        self.share_fuser = share_fuser
        self.share_attn = share_attn
        self.rec_only = rec_only
        self.iter_num = iter_num
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # prepare fusion components
        self.fusion_pe = PositionalEncoding(**fusion_pe_args)
        # prepare rec & kie branch
        self.make_rec_module(**rec_cfg)
        self.make_kie_module(**kie_cfg)
        # build loss
        self.loss = build_loss(loss)
        self.apply(self.set_inplace)

    def make_rec_module(self, node_attn_cfg, pred_cfg, node_attn_layers, txt_classes):
        # self-attention for node-level modeling
        transformer_encoder_layer = nn.TransformerEncoderLayer(**node_attn_cfg)
        self.rec_fusion_attn = nn.TransformerEncoder(transformer_encoder_layer,
                                                     num_layers=node_attn_layers)
        # self-attention for text recognition, similar to DeiT
        self.rec_attn = CustomViTSTR(**pred_cfg)
        self.rec_attn.reset_classifier(txt_classes, cls_type='REC')

    def make_kie_module(self, node_attn_cfg, pred_cfg, entity_classes,
                        node_attn_layers, use_crf=False):
        if not self.share_fuser:
            # self-attention for node-level modeling
            transformer_encoder_layer = nn.TransformerEncoderLayer(**node_attn_cfg)
            self.kie_fusion_attn = nn.TransformerEncoder(transformer_encoder_layer,
                                                         num_layers=node_attn_layers)
        if self.share_attn:
            self.rec_attn.reset_classifier(entity_classes, cls_type='KIE')
        else:
            self.kie_attn = CustomViTSTR(**pred_cfg)
            self.kie_attn.reset_classifier(entity_classes, cls_type='KIE')
        if use_crf:
            # add CRF-layer
            raise NotImplementedError()

    def set_inplace(self, m):
        if isinstance(m, nn.ReLU):
            m.inplace = True

    def forward_train(self, vis_feature=None, spa_feature=None, labels=None,
                      num_boxes=None, **kwargs):

        """
        Args:
            vis_feature: Tensor, (B, N, C, H, W), H = W = 14 (to match DEIT)
            spa_feature: Tensor, (B, N, K, C), K is the length of each spatial feature,
                could be 8, 12 or 16
            labels: dict, includes keys "texts", "entities"
            num_boxes: list[int], actual instances num of each image

        Returns:
            losses: dict
        """
        logits = {
            'REC': [],
            'KIE': []
        }
        # -------------------- prepare masks and features -------------------- #
        # True -> Pad, for all masks in this class.
        # Tensor of bool type, (B, N), indicates which instance is padding.
        if self.use_ins_mask:
            instance_mask = self.prep_ins_mask(num_boxes, vis_feature)
        else:
            instance_mask = None
        # Tensor of bool type, (B x N, L), indicates which position of a
        #   sequence is padding.
        if self.use_seq_mask:
            sequence_mask = self.prep_seq_mask(labels['texts'], ignore_val=-1)
        else:
            sequence_mask = None
        txt_feature = None
        if not self.rec_only:
            kie_feature = None
        # -------------------- Asynchronous forward -------------------- #
        for idx in range(self.iter_num):
            if self.rec_only:
                rec_nodes = self.rec_fusion(vis_feature, txt_feature, ins_mask=instance_mask,
                                            _seq_mask=sequence_mask)
                txt_feature, txt_logits = self.rec_pred(rec_nodes, vis_feature, txt_feature=txt_feature)
                del rec_nodes
                logits['REC'].append(txt_logits)
                del txt_logits
            else:
                raise NotImplementedError()
        return self.loss(logits, labels)

    def forward_test(self, vis_feature=None, spa_feature=None, num_boxes=None):
        pass

    def rec_fusion(self, vis_feature, txt_feature=None, kie_feature=None,
                   _seq_mask=None, ins_mask=None):
        """
        Fusion multi-modal feature and model the relation across instances,
         then create node features for all instances, which are provided for
         recognition branch.
        Args:
            vis_feature: Tensor, (B, N, C, H, W), H = W = 14 (to match DeiT)
            txt_feature: Tensor, text hidden feature of last step, (B, N, L, C)
            kie_feature: Tensor, KIE hidden feature, (B, N, L, C)
            _seq_mask: Tensor of bool type, (B x N, L), True -> Pad
            ins_mask: Tensor of bool type, (B, N), True -> Pad

        Returns:
            Tensor: fused node feature for recognition, (B, N, C)
        """
        B, N, C, H, W = vis_feature.shape
        # todo: try to add position embedding for both visual and sequence feature ?
        # avg-pool feature with different length to nodes -> (B x N, C)
        node_feature = torch.sum(vis_feature.view(B * N, C, -1).permute(0, 2, 1), dim=1).div(H * W)
        # Fuse textual feature
        if not isinstance(txt_feature, type(None)):
            seq_len = txt_feature.shape[2]
            if not isinstance(_seq_mask, type(None)):
                seq_mask = ~_seq_mask
                # remove padding of texts, (B x N, L, C)
                txt_node = txt_feature.view(B * N, seq_len, C) * seq_mask.detach().unsqueeze(2).float()
                # add all chars together, (B x N, C)
                txt_node = torch.sum(txt_node, dim=1)
                # (B x N, C)
                txt_len = seq_mask.float().sum(dim=1).unsqueeze(1).expand_as(txt_node)
                # avoid divide zero denominator
                txt_len = txt_len + txt_len.eq(0).float()
                # nomalize
                node_feature += txt_node.div(txt_len)
                del txt_node
            else:
                # # (B x N, C)
                node_feature += torch.sum(txt_feature.view(-1, seq_len, C), dim=1).div(seq_len)

        # Fuse kie feature
        if not isinstance(kie_feature, type(None)):
            seq_len = kie_feature.shape[2]
            if not isinstance(_seq_mask, type(None)):
                seq_mask = ~_seq_mask
                # remove padding of texts, (B x N, L, C)
                kie_node = kie_feature.reshape(-1, seq_len, C) * seq_mask.detach().unsqueeze(2).float()
                # add all chars together, (B x N, C)
                kie_node = torch.sum(kie_node, dim=1)
                # (B x N, C)
                kie_len = seq_mask.float().sum(dim=1).unsqueeze(1).expand_as(kie_node)
                # avoid divide zero denominator
                kie_len = kie_len + kie_len.eq(0).float()
                # nomalize
                node_feature += kie_node.div(kie_len)
                del kie_node
            else:
                # (B x N, C)
                node_feature += torch.sum(kie_feature.reshape(-1, seq_len, C), dim=1).div(seq_len)
        # (B, N, C)
        # add positional embedding and send to Transformer Encoder
        node_feature = self.fusion_pe(node_feature.view(B, N, C))
        return self.rec_fusion_attn(node_feature.permute(1, 0, 2),
                                    src_key_padding_mask=ins_mask).permute(1, 0, 2)

    def rec_pred(self, node_feature, visual_feature, kie_feature=None, txt_feature=None):
        """
        Fusion node-level feature and sequence feature to predict sequence embedding.
        Currently we only use node_feature and visual_feature
        Args:
            node_feature: (B, N, C)
            visual_feature: (B, N, C, H, W)
            kie_feature: (B, N, L, C)
            txt_feature: (B, N, L, C)

        Returns:
            txt_feature: (B, N, L, C)
            pred_logits: (BN, L, C)
        """
        B, N, C = node_feature.shape
        # (B x N, H x W, C)
        seq_feature = visual_feature.reshape(B * N, C, -1).permute(0, 2, 1) + node_feature.reshape(-1, C).unsqueeze(1)

        return self.rec_attn(seq_feature, cls_type='REC', rec_memory=txt_feature,
                             kie_memory=kie_feature, num_instance=N)


    def kie_fusion(self, vis_feature, spa_feature, txt_feature=None,
                   kie_feature=None, _seq_mask=None, ins_mask=None):
        """
        Fusion multi-modal feature and model the relation across instances,
         then create node features for all instances, which are provided for
         key information extraction branch.
        Args:
            vis_feature: Tensor, (B, N, C, H, W), H = W = 14 (to match DeiT)
            spa_feature: Tensor, (B, N, K, C), K is the length of each spatial feature,
                could be 8, 12 or 16
            txt_feature: Tensor, text hidden feature of last step, (B, N, L, C)
            kie_feature: Tensor, KIE hidden feature, (B, N, L, C)
            _seq_mask: Tensor of bool type, (B x N, L), True -> Pad
            ins_mask: Tensor of bool type, (B, N), True -> Pad

        Returns:
            Tensor: fused node feature for key information extraction, (B, N, C)
        """
        B, N, C, H, W = vis_feature.shape
        # todo: try to add position embedding for both visual and sequence feature ?
        # avg-pool feature with different length to nodes -> (B, N, C)
        node_feature = torch.sum(vis_feature.reshape(B, N, C, -1).permute(0, 1, 3, 2), dim=2).div(H * W)

        raise NotImplementedError()

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def prep_ins_mask(self, num_instances, x):
        """
        Prepare instance mask
        Args:
            num_instances: List[int]
            x: visual feature, (B, N, C, H, W)

        Returns:
            instance mask: Tensor of bool type, (B, N)
                True -> Pad
        """
        device = x.device
        B, N = x.shape[:2]
        assert B == len(num_instances)
        ins_mask = torch.full((B, N), fill_value=True, device=device, dtype=torch.bool)
        for idx, num_boxes in enumerate(num_instances):
            ins_mask[idx, :num_boxes] = False
        return ins_mask

    def prep_seq_mask(self, gt_texts, ignore_val=-1):
        """
        Prepare sequence mask
        Args:
            gt_texts: Tensor, (B, N, L)
            ignore_val: val which indicates pad

        Returns:
            sequence mask: Tensor of bool type, (B x N, L)
        """
        B, N, L = gt_texts.shape
        return gt_texts.eq(ignore_val).view(-1, L).detach()

    def extract_feat(self, imgs):
        super(AsynReader, self).extract_feat(imgs)







