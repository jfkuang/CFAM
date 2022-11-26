#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/10 14:08
# @Author : WeiHua

import ipdb
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.textrecog.recognizer.base import BaseRecognizer
from mmocr.models.builder import DETECTORS, build_loss
from mmocr.models import PositionalEncoding
from mmocr.models.ner.utils.bert import BertEncoder
from mmcv.runner import force_fp32
from mmocr.datasets.vie_e2e_dataset import load_dict

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)

def clones(module, N):
    """ Produce N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #tmp = self.norm(x)
        #tmp = sublayer(tmp)
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def self_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scale Dot Product Attention'
    """

    d_k = value.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        #score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        score = score.masked_fill(mask == 0, -6.55e4) # for fp16
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):

    def __init__(self, headers, d_model, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.src_attn = MultiHeadAttention(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# todo:
#   query_type = 'BOTH' is not test yet
@DETECTORS.register_module()
class AutoRegReader(BaseRecognizer):
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

    def make_module(self, in_dim, d_model, fusion_pe_args, node_attn_cfg, node_attn_layers,
                    rec_pe_args, feat_pe_args, rec_decoder_args, rec_layer_num, max_seq_len,
                    use_crf=False):
        # convert dim to match DET and REC
        if in_dim == d_model:
            self.dim_converter = nn.Identity()
        else:
            self.dim_converter = nn.Conv2d(in_dim, d_model, kernel_size=3, padding=1)
        # prepare fusion components
        fusion_pe_args.update(d_hid=d_model)
        self.fusion_pe = PositionalEncoding(**fusion_pe_args)
        # todo: replace this to BERT encoder
        self.node_attn_type = node_attn_cfg.pop('type', 'origin')
        self.sort_node_input = node_attn_cfg.pop('sort_input', False)
        self.sort_context = node_attn_cfg.pop('sort_context', False)
        # default: node feature will be concat to KIE's input embedding
        self.use_layout_node_attn = node_attn_cfg.pop('use_layout_node', False)
        self.use_context_node_attn = node_attn_cfg.pop('use_context_node', False)
        self.use_layout_last = node_attn_cfg.pop('use_layout_last', False)
        self.use_context_last = node_attn_cfg.pop('use_context_last', False)
        self.layout_to_kie = node_attn_cfg.pop('layout_to_kie', self.use_layout_node_attn)
        self.context_to_kie = node_attn_cfg.pop('context_to_kie', self.use_context_node_attn)
        self.layout_to_query = node_attn_cfg.pop('layout_to_query', False)
        self.context_to_query = node_attn_cfg.pop('context_to_query', False)
        assert self.node_attn_type in ['origin', 'bert']
        self.node_attn_layers = node_attn_layers
        if self.use_layout_node_attn or self.use_context_node_attn:
            self.node_feat_proj = nn.Linear(d_model, d_model)
            if self.node_attn_type == 'origin':
                # origin transformer encoder
                node_attn_cfg.update(d_model=d_model)
                transformer_encoder_layer = nn.TransformerEncoderLayer(**node_attn_cfg)
                self.rec_fusion_attn = nn.TransformerEncoder(transformer_encoder_layer,
                                                             num_layers=node_attn_layers)
            else:
                # BERT encoder
                node_attn_cfg.update(num_hidden_layers=node_attn_layers)
                node_attn_cfg.update(hidden_size=d_model)
                self.rec_fusion_attn = BertEncoder(**node_attn_cfg)
            # prepare modality fuse for query
            num_modal = 1
            if self.layout_to_query and self.use_layout_node_attn:
                num_modal += 1
            if self.context_to_query and self.use_context_node_attn:
                num_modal += 1
            if num_modal > 1:
                self.query_modal_fuser = nn.Conv1d(in_channels=int(d_model * num_modal),
                                                   out_channels=d_model,
                                                   kernel_size=1)
            if not self.rec_only:
                # prepare modality fuse for kie's input embedding
                num_modal = 1
                if self.use_layout_node_attn and self.layout_to_kie:
                    num_modal += 1
                if self.use_context_node_attn and self.context_to_kie:
                    num_modal += 1
                self.kie_modal_fuser = nn.Conv1d(in_channels=int(d_model * num_modal),
                                                 out_channels=d_model,
                                                 kernel_size=1)

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
        self.rec_layers = clones(DecoderLayer(**rec_decoder_args), rec_layer_num-1)
        self.rec_ocr_layer = clones(DecoderLayer(**rec_decoder_args), 1)
        self.ocr_fc = nn.Linear(d_model, len(self.ocr_dict))
        if not self.rec_only:
            self.rec_kie_layer = clones(DecoderLayer(**rec_decoder_args), 1)
            # prepare output fc-layer of KIE branch
            num_modal = 1
            if self.use_layout_node_attn and self.use_layout_last:
                num_modal += 1
            if self.use_context_node_attn and self.use_context_last:
                num_modal += 1
            if num_modal > 1:
                self.kie_fc = nn.Linear(int(num_modal*d_model), len(self.entity_dict))
                self.kie_modal_fuser_last = nn.Conv1d(in_channels=int(d_model*num_modal),
                                                      out_channels=int(d_model*num_modal),
                                                      kernel_size=1)
            else:
                self.kie_fc = nn.Linear(d_model, len(self.entity_dict))

        self.rec_norm = nn.LayerNorm(rec_decoder_args.size)
        self.max_seq_len = max_seq_len
        # CRF layer is not implemented yet.
        if use_crf:
            raise NotImplementedError()
        else:
            self.crf = None
        # print global modeling module's cfg
        self.print_global_modeling_cfg()

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
            spa_feature: Tensor, (B, N, K, C), K is the length of each spatial feature,
                could be 8, 12 or 16
            labels: dict, includes keys "texts", "entities"
            num_boxes: list[int], actual instances num of each image

        Returns:
            losses: dict
        """
        B, N, C, H, W = vis_feature.shape
        vis_feature = self.dim_converter(vis_feature.reshape(-1, C, H, W)).reshape(B, N, -1, H, W)
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
            spa_feature: Tensor, (B, N, K, C)
            ins_mask: Tensor of bool type, (B, N), True -> Pad
            sorted_idx: list of list, [[sorted_idx of origin box list]]

        Returns:
            Tensor: fused node feature for recognition, (B, N, C)
        """
        B, N, C, H, W = vis_feature.shape
        # todo: try to add position embedding for both visual and sequence feature ?
        # avg-pool feature with different length to nodes -> (B x N, C)
        node_feature = torch.sum(vis_feature.view(B * N, C, -1).permute(0, 2, 1), dim=1).div(H * W)
        node_feature += torch.sum(spa_feature.view(B * N, -1, C), dim=1).div(spa_feature.shape[2])
        node_feature = self.node_feat_proj(node_feature)
        if self.sort_node_input:
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
        if self.query_type == 'OCR_ONLY' or self.rec_only:
            query_seq = self.ocr_embedding(gt_texts.reshape(-1, L))
        else:
            query_seq = self.ocr_query_weight * self.ocr_embedding(gt_texts.reshape(-1, L)) + \
                        self.kie_query_weight * self.kie_embedding(gt_entities.reshape(-1, L))
        # put context-based attention after layout-based, to further utilize the layout info
        if self.use_context_node_attn:
            # L' = L - 1
            # from (B x N, L', C) broadcast to (B x N, L', L', C)
            query_memory = query_seq[:, :-1, :].unsqueeze(1).expand(B*N, L-1, L-1, C)
            # (L', L')
            tril_mask = torch.tril(torch.ones((L-1, L-1), dtype=torch.uint8, device=query_seq.device))
            # (B x N, L', L', C)
            tril_mask = tril_mask.unsqueeze(0).unsqueeze(-1).expand(query_memory.shape)
            # (B x N, L', C)
            query_memory = query_memory.masked_fill(tril_mask == 0, 0).sum(2).div(tril_mask.sum(2))
            # (B x L', N, C)
            query_memory = query_memory.reshape(B, N, L-1, C).permute(0, 2, 1, 3).reshape(B*(L-1), N, C)
            if self.sort_context:
                # convert to B*L' lists
                sort_ = []
                for x in sorted_idx:
                    sort_ += [x] * (L-1)
                query_memory = self.fusion_pe(query_memory, sort_)
            else:
                query_memory = self.fusion_pe(query_memory)
            # from (B, N) broadcast to (B*L', N)
            ins_mask_ = ins_mask.unsqueeze(1).expand(B, L-1, N).reshape(-1, N)
            if self.node_attn_type == 'origin':
                context_attn = self.rec_fusion_attn(query_memory.permute(1, 0, 2),
                                                    src_key_padding_mask=ins_mask_).permute(1, 0, 2)
            else:
                context_attn = self.rec_fusion_attn(query_memory,
                                                    attention_mask=ins_mask_.unsqueeze(1).unsqueeze(2)*-1e6,
                                                    head_mask=[None] * self.node_attn_layers)[0]
            context_attn = context_attn.reshape(B, L-1, N, C).permute(0, 2, 1, 3).reshape(-1, L-1, C)
        else:
            context_attn = None
        return self.rec_attn(query_seq[:, :-1, :], global_feature, None, seq_mask,
                             layout_info=node_feature, context_info=context_attn)

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
            cur_l = ocr_input.shape[-1]
            # (B x N, L)
            ocr_input = ocr_input.reshape(-1, cur_l)
            if self.query_type == 'OCR_ONLY' or self.rec_only:
                query_seq = self.ocr_embedding(ocr_input)
            else:
                kie_input = kie_input.reshape(-1, cur_l)
                query_seq = self.ocr_query_weight * self.ocr_embedding(
                    ocr_input) + self.kie_query_weight * self.kie_embedding(kie_input)

            if self.use_context_node_attn:
                # L' = cur_length
                # from (B x N, L', C) broadcast to (B x N, L', L', C)
                query_memory = query_seq.unsqueeze(1).expand(B * N, cur_l, cur_l, C)
                # (L', L')
                tril_mask = torch.tril(torch.ones((cur_l, cur_l), dtype=torch.uint8, device=query_seq.device))
                # (B x N, L', L', C)
                tril_mask = tril_mask.unsqueeze(0).unsqueeze(-1).expand(query_memory.shape)
                # (B x N, L', C)
                query_memory = query_memory.masked_fill(tril_mask == 0, 0).sum(2).div(tril_mask.sum(2))
                # (B x L', N, C)
                query_memory = query_memory.reshape(B, N, cur_l, C).permute(0, 2, 1, 3).reshape(B * cur_l, N, C)
                if self.sort_context:
                    sort_ = []
                    for x in sorted_idx:
                        sort_ += [x] * cur_l
                    query_memory = self.fusion_pe(query_memory, sorted_idx=sort_)
                else:
                    query_memory = self.fusion_pe(query_memory)
                # from (B, N) broadcast to (B*L', N)
                ins_mask_ = ins_mask.unsqueeze(1).expand(B, cur_l, N).reshape(-1, N)
                if self.node_attn_type == 'origin':
                    context_attn = self.rec_fusion_attn(query_memory.permute(1, 0, 2),
                                                        src_key_padding_mask=ins_mask_).permute(1, 0, 2)
                else:
                    context_attn = self.rec_fusion_attn(query_memory,
                                                        attention_mask=ins_mask_.unsqueeze(1).unsqueeze(2) * -1e6,
                                                        head_mask=[None] * self.node_attn_layers)[0]
                context_attn = context_attn.reshape(B, cur_l, N, C).permute(0, 2, 1, 3).reshape(-1, cur_l, C)
            else:
                context_attn = None

            output = self.rec_attn(query_seq, global_feature, None, seq_mask, layout_info=node_feature,
                                   context_info=context_attn)
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

        return output

    def rec_attn(self, x, feature, src_mask, tgt_mask, layout_info=None,
                 context_info=None):
        """
        MASTER for recognition
        Args:
            x: (B x N, L, C)
            feature:
            src_mask:
            tgt_mask:
            layout_info: optional, (B, N, C)
            context_info: optional, (B, N, L, C)

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
            x = self.query_modal_fuser(x.permute(0, 2, 1)).permute(0, 2, 1)
        # main process of transformer decoder.
        x = self.rec_pe(x)

        # origin transformer layer
        for idx, layer in enumerate(self.rec_layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # ocr classification head
        for idx, layer in enumerate(self.rec_ocr_layer):
            assert idx == 0
            ocr_logit = layer(x, feature, src_mask, tgt_mask)
        logits['REC'].append(self.ocr_fc(self.rec_norm(ocr_logit)))
        if not self.rec_only:
            # kie classification head
            if self.forward_manner == 'PARALLEL':
                if (self.use_layout_node_attn and self.layout_to_kie) or (self.use_context_node_attn and self.context_to_kie):
                    src_x_shape = x.shape
                    if self.use_layout_node_attn and self.layout_to_kie:
                        x = torch.cat([x, layout_info.reshape(-1, x.shape[-1]).unsqueeze(1).expand(x.shape)], dim=2)
                    if self.use_context_node_attn and self.context_to_kie:
                        x = torch.cat([x, context_info.reshape(src_x_shape)], dim=2)
                    x = self.kie_modal_fuser(x.permute(0, 2, 1)).permute(0, 2, 1)
                for idx, layer in enumerate(self.rec_kie_layer):
                    assert idx == 0
                    kie_logit = layer(x, feature, src_mask, tgt_mask)
            else:
                if self.use_layout_node_attn or self.use_context_node_attn:
                    raise NotImplementedError
                for idx, layer in enumerate(self.rec_kie_layer):
                    assert idx == 0
                    kie_logit = layer(ocr_logit, feature, src_mask, tgt_mask)
            if (self.use_layout_node_attn and self.use_layout_last) or (self.use_context_node_attn and self.use_context_last):
                kie_logit = self.rec_norm(kie_logit)
                if self.use_layout_node_attn and self.use_layout_last:
                    kie_logit = torch.cat(
                        [kie_logit, layout_info.reshape(-1, x.shape[-1]).unsqueeze(1).expand(x.shape)], dim=2)
                if self.use_context_node_attn and self.use_context_last:
                    kie_logit = torch.cat([kie_logit, context_info.reshape(kie_logit.shape)], dim=2)
                kie_logit = self.kie_modal_fuser_last(kie_logit.permute(0, 2, 1)).permute(0, 2, 1)
                logits['KIE'].append(self.kie_fc(kie_logit))
            else:
                logits['KIE'].append(self.kie_fc(self.rec_norm(kie_logit)))

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
        vis_feature = self.dim_converter(vis_feature.reshape(-1, C, H, W)).reshape(B, N, -1, H, W)
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
        for key in ['REC', 'KIE']:
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

    # def load_part_weights(self, pretrain_dir):
    #     raise NotImplementedError()
    #     pretrain_dict = dict()
    #     for key, val in torch.load(pretrain_dir).items():
    #         pretrain_dict[key.replace('module.vitstr.', "")] = val
    #     model_dict = self.rec_attn.state_dict()
    #
    #     valid_key = []
    #     same_key_name = []
    #     invalid_key = []
    #     for key, val in pretrain_dict.items():
    #         if key in model_dict:
    #             if val.shape == model_dict[key].shape:
    #                 model_dict[key] = val
    #                 valid_key.append(key)
    #             else:
    #                 same_key_name.append(key)
    #         else:
    #             invalid_key.append(key)
    #     self.rec_attn.load_state_dict(model_dict)
    #     print(f"Load {len(valid_key)} keys from {pretrain_dir}, {len(invalid_key)+len(same_key_name)} are not used.")

    def print_global_modeling_cfg(self):
        print(f"\n")
        print(f"use_context_node_attn: {self.use_context_node_attn}")
        print(f"use_layout_node_attn: {self.use_layout_node_attn}")
        print(f"layout_to_query: {self.layout_to_query}")
        print(f"layout_to_kie: {self.layout_to_kie}")
        print(f"use_layout_last: {self.use_layout_last}")
        print(f"context_to_query: {self.context_to_query}")
        print(f"context_to_kie: {self.context_to_kie}")
        print(f"use_context_last: {self.use_context_last}")
        print(f"sort layout nodes: {self.sort_node_input}")
        print(f"sort context nodes: {self.sort_context}")
        print(f"\n")

    def extract_feat(self, imgs):
        super(AutoRegReader, self).extract_feat(imgs)








