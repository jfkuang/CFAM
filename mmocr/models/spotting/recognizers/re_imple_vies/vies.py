#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/4 15:18
# @Author : WeiHua

"""
Codes mainly build upon https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_ie/trie
"""

import ipdb
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....textrecog.recognizer.base import BaseRecognizer
from mmocr.models.builder import DETECTORS, build_loss
from mmocr.models import PositionalEncoding
from mmcv.runner import force_fp32
from mmocr.datasets.vie_e2e_dataset import load_dict
from ...modules import build_text_encoder, build_text_decoder
from mmocr.utils import get_root_logger
from .custom_davar_builder import build_connect
from .kie_decoder import KIEDecoderTRIE


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


@DETECTORS.register_module()
class CustomTRIE(BaseRecognizer):
    def __init__(self,
                 rec_cfg,
                 loss=None,
                 use_ins_mask=False,
                 rec_only=False,
                 ocr_dict_file=None,
                 kie_dict_file=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        """
        Asynchronous Reader for REC and KIE task.
        Args:
            rec_cfg:
            rec_only:
            train_cfg:
            test_cfg:
        """
        super().__init__(init_cfg=init_cfg)
        self.rec_cfg = rec_cfg.copy()
        self.use_ins_mask = use_ins_mask
        self.rec_only = rec_only
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.logger = get_root_logger()
        # prepare alphabet
        self.ocr_dict = {
            '<GO>': 0,
            '<END>': 1,
            '<PAD>': 2,
            **{
                val: ind
                for ind, val in enumerate(load_dict(ocr_dict_file), 3)
            }
        }
        self.rev_ocr_dict = dict()
        for key, val in self.ocr_dict.items():
            self.rev_ocr_dict[val] = key
        if not rec_only:
            self.entity_dict = {
                'O': 0,
                '<PAD>': 1,
            }
            for ind, val in enumerate(load_dict(kie_dict_file), 1):
                self.entity_dict['B-' + val] = 2 * ind
                self.entity_dict['I-' + val] = 2 * ind + 1
            self.rev_entity_dict = dict()
            for key, val in self.entity_dict.items():
                self.rev_entity_dict[val] = key
        else:
            self.entity_dict = None
            self.rev_entity_dict = None

        # make module
        self.make_module(**rec_cfg)
        # build loss
        loss.update(ocr_ignore_index=self.ocr_dict['<PAD>'])
        if not rec_only:
            loss.update(kie_ignore_index=self.entity_dict['<PAD>'])
        self.loss = build_loss(loss)
        # set inplace
        self.apply(self.set_inplace)

    def make_module(self, text_encoder_args, d_model, fusion_pe_args,
                    rec_pe_args, feat_pe_args, rec_decoder_args,
                    rec_layer_num, max_seq_len, infor_context_module,
                    crf_args, rec_dual_layer_num=1):
        # build text encoder for visual memory
        text_encoder_args.update(d_model=d_model)
        self.text_encoder = build_text_encoder(**text_encoder_args)

        # build positional encoding
        fusion_pe_args.update(d_hid=d_model)
        self.fusion_pe = PositionalEncoding(**fusion_pe_args)

        # prepare rec & kie branch
        # prepare embedding & pe
        self.ocr_embedding = Embeddings(d_model=d_model, vocab=len(self.ocr_dict))
        rec_pe_args.update(d_hid=d_model)
        self.rec_pe = PositionalEncoding(**rec_pe_args)
        feat_pe_args.update(d_hid=d_model)
        self.feat_pe = PositionalEncoding(**feat_pe_args)

        # prepare transformer layer for recognition
        decoder_type = rec_decoder_args.pop('type', 'transformer')
        self.rec_decoder_type = decoder_type
        self.rec_layers = build_text_decoder(type=decoder_type, num_block=rec_layer_num - rec_dual_layer_num,
                                             **rec_decoder_args)
        self.rec_ocr_layer = build_text_decoder(type=decoder_type, num_block=rec_dual_layer_num,
                                                **rec_decoder_args)
        self.ocr_fc = nn.Linear(d_model, len(self.ocr_dict))
        self.rec_norm = nn.LayerNorm(rec_decoder_args.size)
        self.max_seq_len = max_seq_len

        # modules for information extraction
        self.infor_context_module = build_connect(infor_context_module)
        self.use_crf = crf_args.get('use_crf', False)
        self.kie_decoder = KIEDecoderTRIE(self.ocr_dict, self.entity_dict, **crf_args,
                                          d_model=d_model)


    def set_inplace(self, m):
        if isinstance(m, nn.ReLU):
            m.inplace = True
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # @force_fp32(apply_to=('logits', 'labels'))
    def _loss(self, logits, labels):
        return self.loss(logits, labels)

    def forward_train(self, vis_feature=None, labels=None,
                      num_boxes=None, sorted_idx=None, det_boxes=None,
                      img_metas=None, **kwargs):
        """
        Args:
            vis_feature: Tensor, (B, N, C, H, W)
            labels: dict, includes keys "texts", "entities"
            num_boxes: list[int], actual instances num of each image
            det_boxes: Tensor, (B, N, K), K -> [x, y, x, y, ...]

        Returns:
            losses: dict
        """
        B, N, C, H, W = vis_feature.shape

        if 'DefaultEncoder' in self.text_encoder.__repr__() and self.text_encoder.equal_dim:
            pass
        else:
            vis_feature = self.text_encoder(vis_feature.reshape(-1, C, H, W)).reshape(B, N, -1, H, W)
        # -------------------- prepare masks and features -------------------- #
        # True -> Pad, for all masks in this class.
        # Tensor of bool type, (B, N), indicates which instance is padding.
        if self.use_ins_mask:
            instance_mask = self.prep_ins_mask(num_boxes, vis_feature)
        else:
            instance_mask = None
        # -------------------- Auto-Regression forward -------------------- #
        # check if the sorted_idx and num_boxes are matched
        if num_boxes and sorted_idx:
            assert len(num_boxes) == len(sorted_idx)
            for num_, sort_ in zip(num_boxes, sorted_idx): assert num_ == len(sort_)
        # dict, includes 'REC', 'CRF' (optional)
        logits = self.rec_pred(vis_feature,
                               gt_texts=labels['texts'],
                               gt_entities=labels.get('entities', None),
                               ins_mask=instance_mask,
                               sorted_idx=sorted_idx,
                               det_boxes=det_boxes,
                               img_metas=img_metas)
        return self._loss(logits, labels)

    def forward_test(self, vis_feature=None, spa_feature=None, num_boxes=None):
        pass

    def rec_pred(self, visual_feature, gt_texts,
                 gt_entities=None, ins_mask=None,
                 sorted_idx=None, det_boxes=None,
                 img_metas=None):
        """
        Fusion node-level feature and sequence feature to predict sequence of texts and iob-tags.
        Currently we only use node_feature and visual_feature.
        Args:
            node_feature: (B, N, C)
            visual_feature: (B, N, C, H, W)
            gt_texts: (B, N, L)
            gt_entities: (B, N, L), optional
            sorted_idx: list of lists, optional
            det_boxes: Tensor, (B, N, K), K -> [x, y, x, y, ...]
m
        Returns:
            txt_feature: (B, N, L, C)
            pred_logits: (BN, L, C)
        """
        B, N, C, H, W = visual_feature.shape
        global_feature = visual_feature.reshape(B * N, C, -1).permute(0, 2, 1)
        L = gt_texts.shape[-1]
        global_feature = self.feat_pe(global_feature)
        seq_mask = self.prep_seq_mask(gt_texts[:, :, :-1], pad_val=self.ocr_dict['<PAD>'])
        # (B x N, L, C)
        query_seq = self.ocr_embedding(gt_texts.reshape(-1, L))
        return self.rec_attn(query_seq[:, :-1, :], global_feature, None, seq_mask,
                             gt_entities=gt_entities, gt_texts=gt_texts,
                             sorted_idx=sorted_idx, vis_feature=visual_feature,
                             det_boxes=det_boxes, img_metas=img_metas)

    def rec_greedy_pred(self, visual_feature, gt_texts=None, det_boxes=None,
                        ins_mask=None, sorted_idx=None, img_metas=None):
        """
        Fusion node-level feature and sequence feature to predict sequence of texts and iob-tags.
        Currently we only use node_feature and visual_feature.
        Args:
            node_feature: (B, N, C)
            visual_feature: (B, N, C, H, W)
            gt_texts: (B, N, L), optional

        Returns:
            logits: dict('REC', 'CRF'), where 'CRF' is optional
        """
        B, N, C, H, W = visual_feature.shape
        global_feature = visual_feature.reshape(B * N, C, -1).permute(0, 2, 1)
        global_feature = self.feat_pe(global_feature)
        # prepare initial query input
        GO = torch.zeros(B, N).long().to(visual_feature.device)
        GO[:, :] = self.ocr_dict['<GO>']
        # B, N, 1
        GO = GO.unsqueeze(-1)
        ocr_input = GO
        if not isinstance(gt_texts, type(None)):
            # B x N, L
            gt_texts = gt_texts.reshape(-1, gt_texts.shape[-1])
        output = None
        soft_max_func = nn.Softmax(dim=2)
        for i in range(self.max_seq_len-1):
            seq_mask = self.prep_seq_mask(ocr_input, pad_val=self.ocr_dict['<PAD>'])
            cur_l = ocr_input.shape[-1]
            # (B x N, L)
            ocr_input = ocr_input.reshape(-1, cur_l)
            query_seq = self.ocr_embedding(ocr_input)

            output = self.rec_attn(query_seq, global_feature, None, seq_mask,
                                   sorted_idx=sorted_idx,
                                   vis_feature=visual_feature,
                                   det_boxes=det_boxes,
                                   img_metas=img_metas,
                                   last_iter=i == (self.max_seq_len-2))
            if not isinstance(gt_texts, type(None)):
                ocr_input = gt_texts[:, :ocr_input.shape[-1]+1]
                ocr_input = ocr_input.reshape(B, N, -1)
            else:
                ocr_prob = soft_max_func(output['REC'][0])

                _, next_word = torch.max(ocr_prob, dim=-1)
                ocr_input = torch.cat([ocr_input, next_word[:, -1].unsqueeze(-1)], dim=1)
                ocr_input = ocr_input.reshape(B, N, -1)

        if not self.rec_only:
            self.kie_decoder.crf_decode(output, shape_=(B, N), sorted_idx=sorted_idx)

        return output

    def rec_attn(self, x, feature, src_mask, tgt_mask,
                 gt_entities=None, gt_texts=None,
                 sorted_idx=None, vis_feature=None,
                 det_boxes=None, img_metas=None,
                 last_iter=False):
        """
        Transformer for recognition
        Args:
            x: (B x N, L, C)
            feature:
            src_mask:
            tgt_mask:
            layout_info: optional, (B, N, C)
            context_info: optional, (B, N, L, C)
            gt_entities: (B, N, L), optional
            gt_texts: (B, N, L), optional
            sorted_idx: list of lists, each list represents the order of current boxes, for example,
                [1, 3, 2, 4], indicates the 2nd instance in current order corresponds to the 3rd
                in the sorted order.
            vis_feature: Tensor, (B, N, C, H, W)
            det_boxes: Tensor, (B, N, K), K -> [x, y, x, y, ...]
        Returns:

        """
        logits = {
            'REC': [],
            'KIE': []
        }
        # main process of transformer decoder.
        x = self.rec_pe(x)

        # origin transformer layer
        for idx, layer in enumerate(self.rec_layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # ocr classification head
        ocr_logit = None
        for idx, layer in enumerate(self.rec_ocr_layer):
            if idx == 0:
                ocr_logit = layer(x, feature, src_mask, tgt_mask)
            else:
                ocr_logit = layer(ocr_logit, feature, src_mask, tgt_mask)
        logits['REC'].append(self.ocr_fc(self.rec_norm(ocr_logit)))

        # multimodal context module
        # with shape (B, N, C)
        if not self.training and not last_iter:
            return logits

        multimodal_context = self.infor_context_module([vis_feature, ocr_logit],
                                                       boxes=det_boxes,
                                                       img_metas=img_metas)

        self.kie_decoder(ocr_logits=ocr_logit,
                         multi_modal_context=multimodal_context,
                         texts=gt_texts[:, :, 1:] if self.training else None,
                         tags=gt_entities[:, :, 1:] if self.training else None,
                         logits_logger=logits,
                         sorted_idx=sorted_idx)

        return logits

    def simple_test(self, vis_feature=None, num_boxes=None, gt_texts=None,
                    sorted_idx=None, imgs=None, det_boxes=None, img_metas=None):
        """
        test forward for one image.
        Args:
            vis_feature: visual feature, (B, N, C, H, W)
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
        B, N, C, H, W = vis_feature.shape
        vis_feature = self.text_encoder(vis_feature.reshape(-1, C, H, W)).reshape(B, N, -1, H, W)
        # -------------------- prepare masks and features -------------------- #
        if self.use_ins_mask:
            instance_mask = self.prep_ins_mask(num_boxes, vis_feature)
        else:
            instance_mask = None
        # -------------------- Auto-Regression forward -------------------- #
        # dict, keys = 'REC', 'CRF' (optional)
        output = self.rec_greedy_pred(vis_feature, gt_texts=gt_texts,
                                      ins_mask=instance_mask, sorted_idx=sorted_idx,
                                      det_boxes=det_boxes, img_metas=img_metas)
        soft_max_func = nn.Softmax(dim=2)
        for key in ['REC']:
            if len(output[key]) == 0 :
                continue
            for logit in output[key]:
                # logit = output[key][0]
                # logit: (B x N, L, C)
                scores, indexes = soft_max_func(logit).topk(1, dim=2)
                scores = scores.squeeze(-1).reshape(B, N, -1)
                indexes = indexes.squeeze(-1).reshape(B, N, -1)
                if key == 'REC' and not isinstance(gt_texts, type(None)):
                    indexes[:, :, :gt_texts.shape[-1]-1] = gt_texts[:, :, 1:]
                    scores[:, :, :gt_texts.shape[-1]-1] = 1.0
                # (B, N, L)
                results[key]['indexes'].append(indexes)
                results[key]['scores'].append(scores)
        if 'CRF' in output.keys():
            if len(output['CRF']) > 0:
                if self.use_crf:
                    for tags in output['CRF']:
                        # tags' shape: (B, N, L)
                        # results['REC']['indexes'].append(results['REC']['indexes'][-1])
                        # results['REC']['scores'].append(results['REC']['scores'][-1])
                        results['KIE']['indexes'].append(tags)
                        results['KIE']['scores'].append(torch.ones_like(tags, device=tags.device))
                else:
                    for logit in output['CRF']:
                        # logit: (B x N, L, C)
                        scores, indexes = soft_max_func(logit).topk(1, dim=2)
                        scores = scores.squeeze(-1).reshape(B, N, -1)
                        indexes = indexes.squeeze(-1).reshape(B, N, -1)
                        # (B, N, L)
                        # results['REC']['indexes'].append(results['REC']['indexes'][-1])
                        # results['REC']['scores'].append(results['REC']['scores'][-1])
                        results['KIE']['indexes'].append(indexes)
                        results['KIE']['scores'].append(scores)

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

    def prep_seq_mask(self, texts, pad_val=2):
        """
        Prepare sequence mask.
        The sequence format is [go, char, char, ..., end, pad, pad, ...],
        Args:
            texts: Tensor, (B, N, L)
            pad_val: val which indicates pad
        Returns:
            sequence mask: Tensor of bool type, (B x N, 1, L, L)
                True -> Non-Pad
        """
        B, N, L = texts.shape
        trg_pad_mask = (texts.reshape(B*N, L) != pad_val).unsqueeze(1).unsqueeze(3).byte()
        trg_sub_mask = torch.tril(torch.ones((L, L), dtype=torch.uint8, device=texts.device))

        return trg_pad_mask & trg_sub_mask

    def prep_crf_mask(self, texts_, end_val=1):
        """
        Prepare CRF mask.
        The sequence format is [go, char, char, ..., end, pad, pad, ...],
        Args:
            texts: Tensor, (B, N, L) or (B*N, L)
            end_val: val which indicates EOS

        Returns:
            CRF mask: Tensor of bool type, (BxN, L)
                True -> Non-Pad
        """
        if len(texts_.shape) == 3:
            B, N, L = texts_.shape
            texts = texts_.reshape(-1, L)
        else:
            texts = texts_.clone()
        crf_mask = torch.ones(texts.shape, dtype=bool, device=texts_.device)
        for i in range(texts.shape[0]):
            end_flag_pos = torch.nonzero(texts[i] == end_val)
            if end_flag_pos.shape[0] > 0:
                crf_mask[i, end_flag_pos[0, 0]:] = False
        return crf_mask


    def prep_ins_refine_mask(self, texts, pad_val=2):
        """
            Prepare mask for instance level refiner.
            The mask's format is
                [Masked, Keep, Keep, ..., Keep],
                [Keep, Masked, Keep, ..., Keep],
                [Keep, Keep, Masked, ..., Keep],
            token with PAD will also be masked.
            Args:
                texts: Tensor, (B, N, L)
                pad_val: val which indicates pad
            Returns:
                sequence mask: Tensor of bool type, (L, L)
                    False -> Masked
        """
        B, N, L = texts.shape
        # BN, 1, L, 1
        trg_pad_mask = (texts.reshape(B * N, L) == pad_val).unsqueeze(1).unsqueeze(3).bool()
        if self.ins_lvl_refine_mask_current:
            # L, L
            refine_mask = torch.diag_embed(torch.ones(L, device=texts.device)).bool()
        else:
            refine_mask = torch.diag_embed(torch.zeros(L, device=texts.device)).bool()
        return (~(trg_pad_mask | refine_mask)).byte()

    def extract_feat(self, imgs):
        super(CustomTRIE, self).extract_feat(imgs)








