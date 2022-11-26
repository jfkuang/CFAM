#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/8 13:15
# @Author : WeiHua

import ipdb
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...textrecog.recognizer.base import BaseRecognizer
from mmocr.models.builder import DETECTORS, build_loss
from mmocr.models import PositionalEncoding
from mmocr.models.ner.utils.bert import BertEncoder
from mmcv.runner import force_fp32
from mmocr.datasets.vie_e2e_dataset import load_dict
from ..modules import build_text_encoder, build_text_decoder, KIEDecoderSerial
from mmocr.utils import get_root_logger


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


def build_node_attn(attn_type, d_model, node_attn_layers, node_attn_cfg):
    assert attn_type in ['origin', 'bert']
    if attn_type == 'origin':
        # origin transformer encoder
        node_attn_cfg.update(d_model=d_model)
        transformer_encoder_layer = nn.TransformerEncoderLayer(**node_attn_cfg)
        return nn.TransformerEncoder(transformer_encoder_layer,
                                     num_layers=node_attn_layers)
    else:
        # BERT encoder
        node_attn_cfg.update(num_hidden_layers=node_attn_layers)
        node_attn_cfg.update(hidden_size=d_model)
        return BertEncoder(**node_attn_cfg)


@DETECTORS.register_module()
class AutoRegReaderSerialLocalIE(BaseRecognizer):
    def __init__(self,
                 rec_cfg,
                 loss=None,
                 use_ins_mask=False,
                 rec_only=False,
                 ocr_dict_file=None,
                 kie_dict_file=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 query_cfg=None):
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
        if query_cfg:
            query_type = query_cfg.get('query_type', 'OCR_ONLY')
            assert query_type in ['OCR_ONLY', 'BOTH']
            ocr_query_weight = query_cfg.get('ocr_weight', 0.5)
            kie_query_weight = query_cfg.get('kie_weight', 0.5)
            forward_manner = query_cfg.get('forward_manner', 'PARALLEL')
            assert forward_manner in ['PARALLEL', 'SEQUENCE']
        else:
            query_type = 'OCR_ONLY'
            ocr_query_weight = kie_query_weight = 0.5
            forward_manner = 'PARALLEL'
        self.query_type = query_type
        self.ocr_query_weight = ocr_query_weight
        self.kie_query_weight = kie_query_weight
        self.forward_manner = forward_manner
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

    def make_module(self, text_encoder_args, d_model, fusion_pe_args, node_attn_cfg,
                    node_attn_layers, rec_pe_args, feat_pe_args, rec_decoder_args,
                    rec_layer_num, max_seq_len, crf_args, rec_dual_layer_num=1):
        # build text encoder for visual memory
        text_encoder_args.update(d_model=d_model)
        self.text_encoder = build_text_encoder(**text_encoder_args)

        # build positional encoding
        fusion_pe_args.update(d_hid=d_model)
        self.fusion_pe = PositionalEncoding(**fusion_pe_args)

        # build global modeling module
        self.get_global_modeling_status(node_attn_cfg)
        self.node_attn_layers = node_attn_layers
        if self.use_layout_node_attn or self.use_context_node_attn:
            if self.use_layout_node_attn:
                self.node_feat_proj = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model)
                )
            self.rec_fusion_attn = build_node_attn(self.node_attn_type, d_model, node_attn_layers, node_attn_cfg)
            # prepare modality fuse for query
            num_modal = 1
            if self.layout_to_query and self.use_layout_node_attn:
                num_modal += 1
            if self.context_to_query and self.use_context_node_attn:
                num_modal += 1
            if num_modal > 1:
                # self.query_modal_fuser = nn.Conv1d(in_channels=int(d_model * num_modal),
                #                                    out_channels=d_model,
                #                                    kernel_size=1)
                self.query_modal_fuser = nn.Sequential(
                    nn.Linear(int(d_model * num_modal), d_model),
                    nn.LayerNorm(d_model)
                )
            if not self.rec_only:
                # prepare modality fuse for kie's input embedding
                num_modal = 1
                if self.use_layout_node_attn and self.layout_to_kie:
                    num_modal += 1
                if self.use_context_node_attn and self.context_to_kie:
                    num_modal += 1
                # self.kie_modal_fuser = nn.Conv1d(in_channels=int(d_model * num_modal),
                #                                  out_channels=d_model,
                #                                  kernel_size=1)
                self.kie_modal_fuser = nn.Sequential(
                    nn.Linear(int(d_model * num_modal), d_model),
                    nn.LayerNorm(d_model)
                )

        # prepare rec & kie branch
        # prepare embedding & pe
        self.ocr_embedding = Embeddings(d_model=d_model, vocab=len(self.ocr_dict))
        if not self.rec_only and self.query_type == 'BOTH':
            self.kie_embedding = Embeddings(d_model=d_model,
                                            vocab=len(self.entity_dict))
        rec_pe_args.update(d_hid=d_model)
        self.rec_pe = PositionalEncoding(**rec_pe_args)
        feat_pe_args.update(d_hid=d_model)
        self.feat_pe = PositionalEncoding(**feat_pe_args)

        # prepare transformer layer for recognition and extraction
        decoder_type = rec_decoder_args.pop('type', 'transformer')
        self.rec_decoder_type = decoder_type
        self.rec_layers = build_text_decoder(type=decoder_type, num_block=rec_layer_num - rec_dual_layer_num,
                                             **rec_decoder_args)
        self.rec_ocr_layer = build_text_decoder(type=decoder_type, num_block=rec_dual_layer_num,
                                                **rec_decoder_args)
        self.ocr_fc = nn.Linear(d_model, len(self.ocr_dict))
        if not self.rec_only:
            self.rec_kie_layer = build_text_decoder(type=decoder_type, num_block=rec_dual_layer_num,
                                                    **rec_decoder_args)
            num_modal = 1
            if self.use_layout_node_attn and self.use_layout_last:
                num_modal += 1
            if self.use_context_node_attn and self.use_context_last:
                num_modal += 1
            if num_modal > 1:
                self.kie_fc = nn.Linear(int(num_modal*d_model), len(self.entity_dict))
                # self.kie_modal_fuser_last = nn.Conv1d(in_channels=int(d_model*num_modal),
                #                                       out_channels=int(d_model*num_modal),
                #                                       kernel_size=1)
                self.kie_modal_fuser_last = nn.Sequential(
                    nn.Linear(int(num_modal*d_model), int(num_modal*d_model)),
                    nn.LayerNorm(int(num_modal*d_model))
                )
            else:
                self.kie_fc = nn.Linear(d_model, len(self.entity_dict))
            if self.query_type == 'BOTH':
                self.local_kie_fuser = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model)
                )

        # todo: fix a possible bug: rec_norm here is used for both rec and kie branches,
        #   maybe use separate norm for each branch.
        self.rec_norm = nn.LayerNorm(rec_decoder_args.size)
        self.max_seq_len = max_seq_len
        # CRF layer is not implemented yet.
        self.use_crf = crf_args.get('use_crf', False)
        self.keep_kie_res = crf_args.pop('keep_kie_res', True)
        self.nar_kie_mask = crf_args.pop('nar_kie_mask', 'NONE')
        assert self.nar_kie_mask in ['NONE', 'ABI', 'NO']
        self.kie_decoder = KIEDecoderSerial(self.ocr_dict, self.entity_dict, **crf_args,
                                            d_model=d_model)
        # print global modeling module's cfg
        self.print_global_modeling_cfg()

    def get_global_modeling_status(self, node_attn_cfg):
        self.node_attn_type = node_attn_cfg.pop('type', 'origin')
        self.sort_layout = node_attn_cfg.pop('sort_layout', False)
        self.sort_context = node_attn_cfg.pop('sort_context', False)
        # default: node feature will be stack with KIE's input embedding
        self.use_layout_node_attn = node_attn_cfg.pop('use_layout_node', False)
        self.use_context_node_attn = node_attn_cfg.pop('use_context_node', False)
        self.use_layout_last = node_attn_cfg.pop('use_layout_last', False)
        self.use_context_last = node_attn_cfg.pop('use_context_last', False)
        self.layout_to_kie = node_attn_cfg.pop('layout_to_kie', self.use_layout_node_attn)
        self.context_to_kie = node_attn_cfg.pop('context_to_kie', self.use_context_node_attn)
        self.context_inside = node_attn_cfg.pop('context_inside', self.use_context_node_attn)
        self.layout_to_query = node_attn_cfg.pop('layout_to_query', False)
        self.context_to_query = node_attn_cfg.pop('context_to_query', False)

    def set_inplace(self, m):
        if isinstance(m, nn.ReLU):
            m.inplace = True
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
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
                      num_boxes=None, sorted_idx=None, **kwargs):
        """
        Args:
            vis_feature: Tensor, (B, N, C, H, W)
            spa_feature: Tensor, (B, N, C)
            labels: dict, includes keys "texts", "entities"
            num_boxes: list[int], actual instances num of each image

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
        if self.use_layout_node_attn:
            # layout-based node-level attention
            nodes = self.rec_fusion(vis_feature,
                                    spa_feature,
                                    ins_mask=instance_mask,
                                    sorted_idx=sorted_idx)
        else:
            nodes = None
        # dict, includes 'REC', 'KIE' (optional)
        logits = self.rec_pred(nodes, vis_feature,
                               gt_texts=labels['texts'],
                               gt_entities=labels.get('entities', None),
                               ins_mask=instance_mask,
                               sorted_idx=sorted_idx)
        return self._loss(logits, labels)

    def forward_test(self, vis_feature=None, spa_feature=None, num_boxes=None):
        pass

    def rec_fusion(self, vis_feature, spa_feature, ins_mask=None, sorted_idx=None):
        """
        Fusion multi-modal feature and model the relation across instances,
         then create node features for all instances, which are provided for
         recognition branch.
        Args:
            vis_feature: Tensor, (B, N, C, H, W), H = W = 14 (to match DeiT)
            spa_feature: Tensor, (B, N, C)
            ins_mask: Tensor of bool type, (B, N), True -> Pad
            sorted_idx: list of list, [[sorted_idx of origin box list]]

        Returns:
            Tensor: fused node feature for recognition, (B, N, C)
        """
        B, N, C, H, W = vis_feature.shape
        # todo: try to add position embedding for both visual and sequence feature ?
        # avg-pool feature with different length to nodes -> (B x N, C)
        node_feature = torch.sum(vis_feature.view(B * N, C, -1), dim=-1).div(H * W)
        node_feature += spa_feature.view(B * N, C)
        node_feature = self.node_feat_proj(node_feature)
        if self.sort_layout:
            # (B, N, C)
            # add positional embedding and send to Transformer Encoder
            node_feature = self.fusion_pe(node_feature.view(B, N, C), sorted_idx=sorted_idx)
        else:
            # (B, N, C)
            # add positional embedding and send to Transformer Encoder
            node_feature = self.fusion_pe(node_feature.view(B, N, C))
        if self.node_attn_type == 'origin':
            return self.rec_fusion_attn(node_feature.permute(1, 0, 2),
                                        src_key_padding_mask=ins_mask).permute(1, 0, 2)
        else:
            return self.rec_fusion_attn(node_feature, attention_mask=ins_mask.unsqueeze(1).unsqueeze(2)*-1e6,
                                        head_mask=[None] * self.node_attn_layers)[0]

    def context_fusion(self, textual_feature, seq_mask, ins_mask, num_ins, sorted_idx=None):
        """
        modeling inter-instance relation via textual feature, which represents the context information.
        Args:
            textual_feature: (B*N, L, C)
            seq_mask: Tensor of bool type, (B x N, 1, L, L), True -> Non-Pad
            ins_mask: Tensor of bool type, (B, N), True -> Pad

        Returns:

        """
        BN, L, C = textual_feature.shape
        assert BN % num_ins == 0
        B = BN // num_ins
        # B x N, L, C -> B xN, 1, L, C -> B x N, L, L, C
        context_feat = textual_feature.unsqueeze(1).expand(BN, L, L, C)
        # B x N, L, C
        context_feat = context_feat.masked_fill(seq_mask.squeeze(1).unsqueeze(-1) == 0, 0).sum(2).div(
            seq_mask.squeeze(1).sum(2).unsqueeze(-1) + 1e-6
        )
        # B, N, L, C -> B, L, N, C -> B x L, N, C
        context_feat = context_feat.reshape(B, num_ins, L, C).permute(0, 2, 1, 3).reshape(B * L, num_ins, C)
        if self.sort_context:
            sort_ = []
            for tmp_ in sorted_idx:
                sort_ += [tmp_] * L
            context_feat = self.fusion_pe(context_feat, sort_)
        else:
            context_feat = self.fusion_pe(context_feat)

        # B x L, N
        ins_mask_ = ins_mask.unsqueeze(1).expand(B, L, num_ins).reshape(-1, num_ins)
        if self.node_attn_type == 'origin':
            context_feat = self.rec_fusion_attn(context_feat.permute(1, 0, 2),
                                                src_key_padding_mask=ins_mask_).permute(1, 0, 2)
        else:
            context_feat = self.rec_fusion_attn(context_feat,
                                                attention_mask=ins_mask_.unsqueeze(1).unsqueeze(2) * -1e6,
                                                head_mask=[None] * self.node_attn_layers)[0]
        context_feat = context_feat.reshape(B, L, num_ins, C).permute(0, 2, 1, 3).reshape(-1, L, C)
        return context_feat

    def rec_pred(self, node_feature, visual_feature, gt_texts,
                 gt_entities=None, ins_mask=None, sorted_idx=None):
        """
        Fusion node-level feature and sequence feature to predict sequence of texts and iob-tags.
        Currently we only use node_feature and visual_feature.
        Args:
            node_feature: (B, N, C)
            visual_feature: (B, N, C, H, W)
            gt_texts: (B, N, L)
            gt_entities: (B, N, L), optional
            sorted_idx: list of lists, optional
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
        if self.nar_kie_mask == 'ABI':
            nar_kie_mask = self.prep_nar_seq_mask(gt_texts[:, :, :-1], pad_val=self.ocr_dict['<PAD>'])
        elif self.nar_kie_mask == 'NO':
            nar_kie_mask = self.prep_no_seq_mask(gt_texts[:, :, :-1], pad_val=self.ocr_dict['<PAD>'])
        else:
            nar_kie_mask = None
        # (B x N, L, C)
        query_seq = self.ocr_embedding(gt_texts.reshape(-1, L))
        if self.query_type == 'OCR_ONLY' or self.rec_only:
            kie_query_logits = None
        else:
            kie_query_logits = self.kie_embedding(gt_entities.reshape(-1, L))
        # put context-based attention after layout-based, to further utilize the layout info
        if self.use_context_node_attn and not self.context_inside:
            # L' = L - 1
            context_attn = self.context_fusion(query_seq[:, :-1, :], seq_mask, ins_mask, N,
                                               sorted_idx=sorted_idx)
        else:
            context_attn = None
        return self.rec_attn(query_seq[:, :-1, :], global_feature, None, seq_mask,
                             layout_info=node_feature, context_info=context_attn,
                             gt_entities=gt_entities, gt_texts=gt_texts,
                             sorted_idx=sorted_idx, seq_mask=seq_mask,
                             ins_mask=ins_mask, num_ins=N,
                             kie_pre_logits=kie_query_logits[:, :-1, :] if self.query_type == 'BOTH' else None,
                             nar_kie_mask=nar_kie_mask)

    def rec_greedy_pred(self, node_feature, visual_feature, gt_texts=None,
                        ins_mask=None, sorted_idx=None):
        """
        Fusion node-level feature and sequence feature to predict sequence of texts and iob-tags.
        Currently we only use node_feature and visual_feature.
        Args:
            node_feature: (B, N, C)
            visual_feature: (B, N, C, H, W)
            gt_texts: (B, N, L), optional

        Returns:
            logits: dict('REC', 'KIE'), where 'KIE' is optional
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
        if self.query_type == 'BOTH' and not self.rec_only:
            kie_input = torch.zeros(B, N).long().to(visual_feature.device)
            kie_input[:, :] = self.entity_dict['O']
            kie_input = kie_input.unsqueeze(-1)
        else:
            kie_input = None
        output = None
        soft_max_func = nn.Softmax(dim=2)
        for i in range(self.max_seq_len-1):
            seq_mask = self.prep_seq_mask(ocr_input, pad_val=self.ocr_dict['<PAD>'])
            if self.nar_kie_mask == 'ABI':
                nar_kie_mask = self.prep_nar_seq_mask(ocr_input, pad_val=self.ocr_dict['<PAD>'])
            elif self.nar_kie_mask == 'NO':
                nar_kie_mask = self.prep_no_seq_mask(ocr_input, pad_val=self.ocr_dict['<PAD>'])

            else:
                nar_kie_mask = None
            cur_l = ocr_input.shape[-1]
            # (B x N, L)
            ocr_input = ocr_input.reshape(-1, cur_l)
            query_seq = self.ocr_embedding(ocr_input)
            if self.query_type == 'OCR_ONLY' or self.rec_only:
                kie_query_logits = None
            else:
                kie_input = kie_input.reshape(-1, cur_l)
                kie_query_logits = self.kie_embedding(kie_input)

            if self.use_context_node_attn and not self.context_inside:
                context_attn = self.context_fusion(query_seq, seq_mask, ins_mask, N,
                                                   sorted_idx=sorted_idx)
            else:
                context_attn = None

            output = self.rec_attn(query_seq, global_feature, None, seq_mask, layout_info=node_feature,
                                   context_info=context_attn, sorted_idx=sorted_idx,
                                   seq_mask=seq_mask, num_ins=N, ins_mask=ins_mask,
                                   kie_pre_logits=kie_query_logits,
                                   nar_kie_mask=nar_kie_mask)
            if not isinstance(gt_texts, type(None)):
                ocr_input = gt_texts[:, :ocr_input.shape[-1]+1]
                ocr_input = ocr_input.reshape(B, N, -1)
            else:
                ocr_prob = soft_max_func(output['REC'][0])

                _, next_word = torch.max(ocr_prob, dim=-1)
                ocr_input = torch.cat([ocr_input, next_word[:, -1].unsqueeze(-1)], dim=1)
                ocr_input = ocr_input.reshape(B, N, -1)
            if not self.rec_only and self.query_type == 'BOTH':
                kie_prob = soft_max_func(output['KIE'][0])

                _, next_word = torch.max(kie_prob, dim=-1)
                kie_input = torch.cat([kie_input, next_word[:, -1].unsqueeze(-1)], dim=1)
                kie_input = kie_input.reshape(B, N, -1)

        if not self.rec_only:
            self.kie_decoder.crf_decode(output, shape_=(B, N), sorted_idx=sorted_idx)

        return output

    def rec_attn(self, x, feature, src_mask, tgt_mask, layout_info=None,
                 context_info=None, gt_entities=None, gt_texts=None,
                 sorted_idx=None, seq_mask=None, ins_mask=None,
                 num_ins=None, kie_pre_logits=None,
                 nar_kie_mask=None):
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
        Returns:

        """
        logits = {
            'REC': [],
            'KIE': []
        }
        if (self.use_layout_node_attn and self.layout_to_query) or (self.use_context_node_attn and self.context_to_query):
            if self.use_layout_node_attn and self.layout_to_query:
                x = torch.cat(
                    [x, layout_info.reshape(-1, x.shape[-1]).unsqueeze(1).expand(x.shape)], dim=2)
            if self.use_context_node_attn and self.context_to_query:
                x = torch.cat([x, context_info.reshape(x.shape)], dim=2)
            # x = self.query_modal_fuser(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.query_modal_fuser(x)
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
        if not self.rec_only:
            # kie classification head
            if self.use_context_node_attn and self.context_inside:
                context_info = self.context_fusion(x, seq_mask, ins_mask, num_ins,
                                                   sorted_idx=sorted_idx)
            # fuse with previous iteration
            if self.query_type == 'BOTH':
                x = self.local_kie_fuser(x+kie_pre_logits)
            if self.forward_manner == 'PARALLEL':
                if (self.use_layout_node_attn and self.layout_to_kie) or (self.use_context_node_attn and self.context_to_kie):
                    src_x_shape = x.shape
                    if self.use_layout_node_attn and self.layout_to_kie:
                        x = torch.cat([x, layout_info.reshape(-1, x.shape[-1]).unsqueeze(1).expand(x.shape)], dim=2)
                    if self.use_context_node_attn and self.context_to_kie:
                        x = torch.cat([x, context_info.reshape(src_x_shape)], dim=2)
                    # x = self.kie_modal_fuser(x.permute(0, 2, 1)).permute(0, 2, 1)
                    x = self.kie_modal_fuser(x)
                kie_logit = None
                for idx, layer in enumerate(self.rec_kie_layer):
                    if idx == 0:
                        if self.nar_kie_mask != 'NONE':
                            kie_logit = layer(x, feature, src_mask, nar_kie_mask)
                        else:
                            kie_logit = layer(x, feature, src_mask, tgt_mask)
                    else:
                        if self.nar_kie_mask != 'NONE':
                            kie_logit = layer(kie_logit, feature, src_mask, nar_kie_mask)
                        else:
                            kie_logit = layer(kie_logit, feature, src_mask, tgt_mask)
            else:
                raise NotImplementedError
                if self.use_layout_node_attn or self.use_context_node_attn:
                    raise NotImplementedError
                kie_logit = None
                for idx, layer in enumerate(self.rec_kie_layer):
                    if idx == 0:
                        kie_logit = layer(ocr_logit, feature, src_mask, tgt_mask)
                    else:
                        kie_logit = layer(kie_logit, feature, src_mask, tgt_mask)
            if (self.use_layout_node_attn and self.use_layout_last) or (self.use_context_node_attn and self.use_context_last):
                kie_logit = self.rec_norm(kie_logit)
                if self.use_layout_node_attn and self.use_layout_last:
                    kie_logit = torch.cat(
                        [kie_logit, layout_info.reshape(-1, x.shape[-1]).unsqueeze(1).expand(x.shape)], dim=2)
                if self.use_context_node_attn and self.use_context_last:
                    kie_logit = torch.cat([kie_logit, context_info.reshape(kie_logit.shape)], dim=2)
                # kie_logit = self.kie_modal_fuser_last(kie_logit.permute(0, 2, 1)).permute(0, 2, 1)
                kie_logit = self.kie_modal_fuser_last(kie_logit)

                # logits['KIE'].append(self.kie_fc(kie_logit))
                self.kie_decoder(kie_logits=kie_logit,
                                 kie_cls_res=self.kie_fc(kie_logit),
                                 texts=gt_texts[:, :, 1:] if self.training else None,
                                 tags=gt_entities[:, :, 1:] if self.training else None,
                                 logits_logger=logits,
                                 sorted_idx=sorted_idx,
                                 rec_logits=ocr_logit)
            else:
                # ['KIE'].append(self.kie_fc(self.rec_norm(kie_logit)))
                self.kie_decoder(kie_logits=kie_logit,
                                 kie_cls_res=self.kie_fc(self.rec_norm(kie_logit)),
                                 texts=gt_texts[:, :, 1:] if self.training else None,
                                 tags=gt_entities[:, :, 1:] if self.training else None,
                                 logits_logger=logits,
                                 sorted_idx=sorted_idx,
                                 rec_logits=ocr_logit)
        return logits

    def simple_test(self, vis_feature=None, spa_feature=None,
                    num_boxes=None, gt_texts=None,
                    sorted_idx=None,
                    imgs=None, img_metas=None):
        """
        test forward for one image.
        Args:
            vis_feature: visual feature, (B, N, C, H, W)
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
        B, N, C, H, W = vis_feature.shape
        vis_feature = self.text_encoder(vis_feature.reshape(-1, C, H, W)).reshape(B, N, -1, H, W)
        # -------------------- prepare masks and features -------------------- #
        if self.use_ins_mask:
            instance_mask = self.prep_ins_mask(num_boxes, vis_feature)
        else:
            instance_mask = None
        # -------------------- Auto-Regression forward -------------------- #
        if self.use_layout_node_attn:
            nodes = self.rec_fusion(vis_feature,
                                    spa_feature,
                                    ins_mask=instance_mask,
                                    sorted_idx=sorted_idx)
        else:
            nodes = None
        # dict, keys = 'REC', 'KIE' (optional)
        output = self.rec_greedy_pred(nodes, vis_feature, gt_texts=gt_texts,
                                      ins_mask=instance_mask, sorted_idx=sorted_idx)
        soft_max_func = nn.Softmax(dim=2)
        if self.keep_kie_res:
            key_list = ['REC', 'KIE']
        else:
            key_list = ['REC']
        for key in key_list:
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
                assert len(output['CRF']) <= 1
                if self.use_crf:
                    for tags in output['CRF']:
                        # tags' shape: (B, N, L)
                        if self.keep_kie_res:
                            results['REC']['indexes'].append(results['REC']['indexes'][-1])
                            results['REC']['scores'].append(results['REC']['scores'][-1])
                        results['KIE']['indexes'].append(tags)
                        results['KIE']['scores'].append(torch.ones_like(tags, device=tags.device))
                else:
                    for logit in output['CRF']:
                        # logit: (B x N, L, C)
                        scores, indexes = soft_max_func(logit).topk(1, dim=2)
                        scores = scores.squeeze(-1).reshape(B, N, -1)
                        indexes = indexes.squeeze(-1).reshape(B, N, -1)
                        # (B, N, L)
                        if self.keep_kie_res:
                            results['REC']['indexes'].append(results['REC']['indexes'][-1])
                            results['REC']['scores'].append(results['REC']['scores'][-1])
                        results['KIE']['indexes'].append(indexes)
                        results['KIE']['scores'].append(scores)
        return results

    # def apply_crf(self, kie_logits=None, gt_texts=None, gt_entities=None, logits_logger=None):
    #     """
    #     Apply CRF form Key Information Extraction as post-process.
    #     Since our tagging mode is : O, tags of chars, O, PAD, PAD. The first
    #     and the last O represent START and END respectively. Only "tags of chars"
    #     will be applied CRF layer.
    #     Args:
    #         kie_logits: (B*N, L, C), where C is the num of entity classes
    #         gt_texts: (B, N, L) for train, (B x N, L, num_class) for test, optional
    #         gt_entities: (B, N, L), optional
    #
    #     Returns:
    #
    #     """
    #     if self.training:
    #         logits_logger['KIE'].append(kie_logits)
    #         if self.crf:
    #             B, N, L = gt_entities.shape
    #             tags = gt_entities.reshape(-1, L)[:, 1:]
    #             # only keep text part.
    #             crf_mask = self.prep_crf_mask(gt_texts, end_val=self.ocr_dict['<END>'])
    #             # used for calculating CRF loss
    #             log_likelihood = self.crf(kie_logits, tags, crf_mask[:, 1:],
    #                                       input_batch_first=True,
    #                                       keepdim=True)
    #             import ipdb
    #             ipdb.set_trace()
    #             logits_logger['CRF'].append(-log_likelihood)
    #     else:
    #         if self.crf:
    #             soft_max_func = nn.Softmax(dim=2)
    #             _, pred_texts = torch.max(soft_max_func(logits_logger['REC'][-1]), dim=-1)
    #             crf_mask = self.prep_crf_mask(pred_texts, end_val=self.ocr_dict['<END>'])
    #             tmp = self.crf.viterbi_tags(logits_logger['KIE'][-1], crf_mask,
    #                                         logits_batch_first=True)
    #             logits_logger['CRF'].append(tmp)
    #             import ipdb
    #             ipdb.set_trace()
    #             print("Debugging HERE")

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

    def prep_nar_seq_mask(self, texts, pad_val=2):
        B, N, L = texts.shape
        trg_pad_mask = (texts.reshape(B * N, L) != pad_val).unsqueeze(1).unsqueeze(3).byte()
        trg_sub_diag = torch.diag_embed(torch.diag(torch.ones((L, L), dtype=torch.uint8, device=texts.device)))
        trg_sub_mask = torch.ones((L, L), dtype=torch.uint8, device=texts.device) - trg_sub_diag

        return trg_pad_mask & trg_sub_mask

    def prep_no_seq_mask(self, texts, pad_val=2):
        B, N, L = texts.shape
        trg_pad_mask = (texts.reshape(B * N, L) != pad_val).unsqueeze(1).unsqueeze(3).byte()
        trg_sub_mask = torch.ones((L, L), dtype=torch.uint8, device=texts.device)

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


        # if self.training:
        #     if len(texts.shape) == 3:
        #         B, N, L = texts.shape
        #         crf_mask = (texts.reshape(B*N, L) != pad_val).byte()
        #     else:
        #         crf_mask = (texts != pad_val).byte()
        # else:
        #     if len(texts.shape) == 3:
        #         # todo: fix bug here
        #         B, N, L = texts.shape
        #         texts_ = texts.reshape(-1, L)
        #     else:
        #         texts_ = texts.clone()
        #     crf_mask = torch.ones(texts_.shape, dtype=bool, device=texts_.device)
        #     for i in range(texts_.shape[0]):
        #         end_flag_pos = torch.nonzero(texts_[i] == pad_val)
        #         if end_flag_pos.shape[0] > 0:
        #             crf_mask[i, end_flag_pos[0, 0]+1:] = False
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

    def print_global_modeling_cfg(self):
        global_modeling_cfg = ""
        global_modeling_cfg += f"use_context_node_attn: {self.use_context_node_attn}\n"
        global_modeling_cfg += f"use_layout_node_attn: {self.use_layout_node_attn}\n"
        global_modeling_cfg += f"layout_to_query: {self.layout_to_query}\n"
        global_modeling_cfg += f"layout_to_kie: {self.layout_to_kie}\n"
        global_modeling_cfg += f"use_layout_last: {self.use_layout_last}\n"
        global_modeling_cfg += f"context_to_query: {self.context_to_query}\n"
        global_modeling_cfg += f"context_to_kie: {self.context_to_kie}\n"
        global_modeling_cfg += f"use_context_last: {self.use_context_last}\n"
        global_modeling_cfg += f"sort layout nodes: {self.sort_layout}\n"
        global_modeling_cfg += f"sort context nodes: {self.sort_context}\n"
        global_modeling_cfg += f"context modeling inside: {self.context_inside}\n"
        self.logger.info(global_modeling_cfg)

    def extract_feat(self, imgs):
        super(AutoRegReaderSerialLocalIE, self).extract_feat(imgs)








