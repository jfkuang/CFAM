#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/12 17:12
# @Author : WeiHua
import torch
import torch.nn as nn

from mmocr.models.textrecog.recognizer.base import BaseRecognizer
from mmocr.models.builder import DETECTORS, build_loss
from mmocr.models import PositionalEncoding
from mmocr.models.spotting.modules.old.ViTSTRLocal import CustomViTSTRLocal


# todo:finish this part
@DETECTORS.register_module()
class AsynReaderLocal(BaseRecognizer):
    def __init__(self,
                 rec_cfg,
                 fusion_pe_args={},
                 loss=None,
                 use_ins_mask=False,
                 use_seq_mask=False,
                 rec_only=False,
                 ignore_val=-1,
                 end_val=1,
                 iter_num=3,
                 iter_num_local=2,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        """
        Asynchronous Reader for REC and KIE task.
        Args:
            rec_cfg:
            fusion_pe_args:
            rec_only:
            train_cfg:
            test_cfg:
        """
        super().__init__(init_cfg=init_cfg)
        self.rec_cfg = rec_cfg.copy()
        self.ignore_val = ignore_val
        self.end_val = end_val
        self.use_ins_mask = use_ins_mask
        self.use_seq_mask = use_seq_mask
        self.rec_only = rec_only
        self.iter_num = iter_num
        self.iter_num_local = iter_num_local
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # prepare fusion components
        self.fusion_pe = PositionalEncoding(**fusion_pe_args)
        # prepare rec & kie branch
        self.make_rec_module(**rec_cfg)
        # build loss
        self.loss = build_loss(loss)
        # set inplace
        self.apply(self.set_inplace)

    def make_rec_module(self, node_attn_cfg, pred_cfg, node_attn_layers,
                        txt_classes, entity_classes, use_crf):
        # self-attention for node-level modeling
        transformer_encoder_layer = nn.TransformerEncoderLayer(**node_attn_cfg)
        self.rec_fusion_attn = nn.TransformerEncoder(transformer_encoder_layer,
                                                     num_layers=node_attn_layers)
        # self-attention for text recognition, similar to DeiT
        self.rec_attn = CustomViTSTRLocal(**pred_cfg)
        self.rec_attn.reset_classifier(txt_classes, cls_type='REC')
        if not self.rec_only:
            self.rec_attn.reset_classifier(entity_classes, cls_type='KIE',
                                           use_crf=use_crf)

    def set_inplace(self, m):
        if isinstance(m, nn.ReLU):
            m.inplace = True
        # if isinstance(m, nn.Linear):
        #     trunc_normal_init(m)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.BatchNorm2d):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.ReLU):
        #     m.inplace = True

    # @force_fp32(apply_to=('logits', 'labels'))
    def _loss(self, logits, labels):
        return self.loss(logits, labels)

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
        kie_feature = None
        # -------------------- Asynchronous forward -------------------- #
        for _ in range(self.iter_num):
            nodes = self.rec_fusion(vis_feature,
                                    spa_feature,
                                    txt_feature=txt_feature,
                                    kie_feature=kie_feature,
                                    ins_mask=instance_mask,
                                    _seq_mask=sequence_mask)
            txt_feature, kie_feature, _logits = self.rec_pred(nodes, vis_feature,
                                                              txt_feature=txt_feature,
                                                              kie_feature=kie_feature,
                                                              rec_only=self.rec_only)
            del nodes
            logits['REC'] += _logits['REC']
            logits['KIE'] += _logits['KIE']
            del _logits
        return self._loss(logits, labels)
        # return self.loss(logits, labels)

    def forward_test(self, vis_feature=None, spa_feature=None, num_boxes=None):
        pass

    def rec_fusion(self, vis_feature, spa_feature, txt_feature=None,
                   kie_feature=None, _seq_mask=None, ins_mask=None):
        """
        Fusion multi-modal feature and model the relation across instances,
         then create node features for all instances, which are provided for
         recognition branch.
        Args:
            vis_feature: Tensor, (B, N, C, H, W), H = W = 14 (to match DeiT)
            spa_feature: Tensor, (B, N, K, C)
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
        node_feature += torch.sum(spa_feature.view(B * N, -1, C), dim=1).div(spa_feature.shape[2])
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

    def rec_pred(self, node_feature, visual_feature, kie_feature=None, txt_feature=None, rec_only=True):
        """
        Fusion node-level feature and sequence feature to predict sequence embedding.
        Currently we only use node_feature and visual_feature
        Args:
            node_feature: (B, N, C)
            visual_feature: (B, N, C, H, W)
            kie_feature: (B, N, L, C)
            txt_feature: (B, N, L, C)
            rec_only: if recognition only

        Returns:
            txt_feature: (B, N, L, C)
            pred_logits: (BN, L, C)
        """
        B, N, C = node_feature.shape
        # (B x N, H x W, C)
        seq_feature = visual_feature.reshape(B * N, C, -1).permute(0, 2, 1) + node_feature.reshape(-1, C).unsqueeze(1)

        return self.rec_attn(seq_feature,
                             rec_only=rec_only,
                             rec_memory=txt_feature,
                             kie_memory=kie_feature,
                             num_instance=N,
                             iter_num=self.iter_num_local)

    def simple_test(self, vis_feature=None, spa_feature=None,
                    num_boxes=None, imgs=None, img_metas=None):
        """
        test forward for one image.
        Args:
            vis_feature: visual feature, (B, N, C, H, W), H = W = 14 (to match DEIT)
            spa_feature: spatial feature, (B, N, K, C), K is the length of each spatial feature,
                could be 8, 12 or 16
            num_boxes: list[int], actual instances num of each image
            imgs: father class required
            img_metas: father class required

        Returns:
            results (dict):
        """
        results = {
            'REC': {
                'indexes': [],
                'scores': [],
            },
            'KIE': {
                'indexes': [],
                'scores': [],
            }
        }
        B, N = vis_feature.shape[:2]
        # -------------------- prepare masks and features -------------------- #
        if self.use_ins_mask:
            instance_mask = self.prep_ins_mask(num_boxes, vis_feature)
        else:
            instance_mask = None
        sequence_mask = None
        txt_feature = None
        kie_feature = None
        # -------------------- Asynchronous forward -------------------- #
        soft_max_func = nn.Softmax(dim=2)
        for _ in range(self.iter_num):
            nodes = self.rec_fusion(vis_feature,
                                    spa_feature,
                                    txt_feature=txt_feature,
                                    kie_feature=kie_feature,
                                    ins_mask=instance_mask,
                                    _seq_mask=sequence_mask)
            txt_feature, kie_feature, _logits = self.rec_pred(nodes, vis_feature,
                                                              txt_feature=txt_feature,
                                                              kie_feature=kie_feature,
                                                              rec_only=self.rec_only)
            del nodes
            for key in ['REC', 'KIE']:
                for logit in _logits[key]:
                    # logit: (B x N, L, C)
                    scores, indexes = soft_max_func(logit).topk(1, dim=2)
                    scores = scores.squeeze(-1).reshape(B, N, -1)
                    indexes = indexes.squeeze(-1).reshape(B, N, -1)
                    # (B, N, L)
                    results[key]['indexes'].append(indexes)
                    results[key]['scores'].append(scores)
            del _logits
            if self.use_seq_mask:
                # prepare sequence_mask
                sequence_mask = self.prep_seq_mask(results['REC']['indexes'][-1],
                                                   test_mode=True,
                                                   end_val=1)
            if self.rec_cfg.use_crf:
                raise NotImplementedError()
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError()

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

    def prep_seq_mask(self, texts, ignore_val=-1, test_mode=False, end_val=1):
        """
        Prepare sequence mask
        Args:
            texts: Tensor, (B, N, L)
            ignore_val: val which indicates pad
            test_mode: if it is test mode
            end_val: index of '<END>' flag

        Returns:
            sequence mask: Tensor of bool type, (B x N, L)
        """
        B, N, L = texts.shape
        if not test_mode:
            return texts.eq(ignore_val).view(-1, L).detach()
        _texts = texts.reshape(-1, L)
        mask = torch.full_like(_texts, fill_value=True, dtype=torch.bool).detach()
        for n_idx, text in enumerate(_texts):
            end_pos = torch.nonzero(text == end_val)
            if len(end_pos) == 0:
                mask[n_idx, :] = False
            else:
                mask[n_idx, :end_pos[0, 0]+1] = False
        return mask

    def load_part_weights(self, pretrain_dir):
        pretrain_dict = dict()
        for key, val in torch.load(pretrain_dir).items():
            pretrain_dict[key.replace('module.vitstr.', "")] = val
        model_dict = self.rec_attn.state_dict()

        valid_key = []
        same_key_name = []
        invalid_key = []
        for key, val in pretrain_dict.items():
            if key in model_dict:
                if val.shape == model_dict[key].shape:
                    model_dict[key] = val
                    valid_key.append(key)
                else:
                    same_key_name.append(key)
            else:
                invalid_key.append(key)
        self.rec_attn.load_state_dict(model_dict)
        print(f"Load {len(valid_key)} keys from {pretrain_dir}, {len(invalid_key)+len(same_key_name)} are not used.")

    def extract_feat(self, imgs):
        super(AsynReaderLocal, self).extract_feat(imgs)







