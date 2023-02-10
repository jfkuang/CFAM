#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/14 21:16
# @Author : WeiHua

import torch
import torch.nn as nn
from .text_decoder import MultiHeadAttention, clones, SubLayerConnection, FeedForward
from mmocr.models import PositionalEncoding
import torch
import torch.nn as nn
from torchvision import utils as vutils
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from mmcv.ops import SigmoidFocalLoss
import ipdb
import math
from .master_encoder import MasterEncoder
from .FocalTransformer import FocalTransformerBlock
# import clip


#save img

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 3 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


#self attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        # ipdb.set_trace()
        attn = torch.matmul(q / self.temperature, k.transpose(-1, -2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


#cross attention
def self_attention(query, key, value, mask=None, dropout=None,
                   attn_map_type=None):
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
    if attn_map_type:
        if attn_map_type == 'origin':
            return torch.matmul(p_attn, value), score
        elif attn_map_type == 'sigmoid':
            return torch.matmul(p_attn, value), torch.sigmoid(score)
        else:
            raise RuntimeError(f"Not support attn-map-type: {attn_map_type}")
    else:
        return torch.matmul(p_attn, value), p_attn
class CrossAttention(nn.Module):

    def __init__(self, headers, d_model, dropout, attn_map_type=None):
        super(CrossAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.attn_map_type = attn_map_type

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout,
                                      attn_map_type=self.attn_map_type)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        # ipdb.set_trace()
        self.attn = torch.mean(self.attn, dim=1)
        return x, self.attn

#encoder代替embedding
class SwinTSEncoder(nn.Module):
    def __init__(self, input_dim, input_resolution, num_heads=8, window_size=7, d_model=-1, block_num=3, cls_num=22):
        super(SwinTSEncoder, self).__init__()
        self.equal_dim = input_dim == d_model
        self.layers = nn.ModuleList()
        self.cls_num = cls_num
        for _ in range(block_num):
            self.layers.append(
                FocalTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads,
                                      window_size=window_size, expand_size=0, shift_size=0, mlp_ratio=4.,
                                      qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                                      act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc",
                                      focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4)
            )
        if input_dim == d_model:
            self.dim_converter = nn.Identity()
        else:
            self.dim_converter = nn.Linear(input_dim, d_model)

    def forward(self, x):
        B, C, N = x.shape
        # (B, C, H, W) -> (B, L, C), where L = H * W

        for layer_ in self.layers:
            x = layer_(x)
        x = self.dim_converter(x)

        return x.reshape(B, C, self.cls_num)

class BalancedBCELoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6, empty_cls=0):
        super(BalancedBCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.empty_cls = empty_cls
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, pred, gt):
        """
        Balanced BCE loss for multi-class classification
        Args:
            pred: Tensor, (BN, cls)
            gt: Tensor, (BN, cls)

        Returns:

        """
        negative = (gt[:, self.empty_cls] == 1).byte()
        positive = (1 - negative).byte()
        pos_count = int(positive.float().sum())
        neg_count = min(int(negative.float().sum()),
                        int(pos_count * self.negative_ratio))
        # BN,
        loss = self.bce_loss(pred, gt).mean(dim=-1)
        pos_loss = loss * positive.float()
        neg_loss = loss * negative.float()
        neg_loss, _ = torch.topk(neg_loss.view(-1), neg_count)
        balanced_loss = (pos_loss.sum() + neg_loss.sum()) / (pos_count + neg_count + self.eps)
        return balanced_loss

class OneStageCatcher(nn.Module):
    def __init__(self, entity_list, entity_dict, d_model, attn_args, kernel_sizes=[2, 3, 4], fuse_type='conv',
                 dropout=0.2, use_entity_cls_loss=False, use_rec_embed=False, ocr_dict=None, use_self_attn=False,
                 use_layout=False):
        super(OneStageCatcher, self).__init__()
        self.entity_list = entity_list
        self.entity_dict = entity_dict
        self.fuse_type = fuse_type
        self.use_entity_cls_loss = use_entity_cls_loss
        self.use_rec_embed = use_rec_embed
        self.use_self_attn = use_self_attn
        self.use_layout = use_layout
        if use_rec_embed:
            self.ocr_dict = ocr_dict
        else:
            self.cls_embedding = nn.Embedding(1+len(entity_list), d_model)
        # todo: add pooling mode
        assert fuse_type in ['conv'], f"Unexpected fuse type: {fuse_type}"

        if fuse_type == 'conv':
            self.conv1ds = nn.ModuleList()
            for _ in kernel_sizes:
                self.conv1ds.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=_),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool1d(output_size=1)
                    )
                )
            self.fuse_fc = nn.Linear(d_model * len(kernel_sizes), d_model)
            self.fuse_activate = nn.ReLU()
        else:
            raise NotImplementedError

        attn_args.update(d_model=d_model)
        # attn_args.update(attn_map_type='origin')
        attn_args.update(attn_map_type='sigmoid')
        if self.use_self_attn:
            self.self_attn = MultiHeadAttention(**attn_args)
            self.self_attn_sublayer = SubLayerConnection(d_model, dropout)
        self.cross_attn = MultiHeadAttention(**attn_args)
        self.sublayer = SubLayerConnection(d_model, dropout)
        self.pe = PositionalEncoding(d_hid=d_model, n_position=1000)
        # self.head_fuse_fc = nn.Linear(attn_args.get('headers'), 1)
        # self.out_fc = nn.Linear(d_model, 2)
        self.entity_norm = nn.LayerNorm(d_model)
        self.entity_cls_loss = SigmoidFocalLoss(gamma=2.0, alpha=0.75, reduction='mean')
        self.attn_map_loss = nn.BCELoss()

    def forward(self, seq_feat, shapes, pad_mask, kvc_mask, rec_embedding=None,
                img_metas=None, layout_feat=None):
        """
        Build key-value relation between entity class embedding and instance-level feature
        Args:
            seq_feat: Tensor with shape (BN, L, C)
            shapes: (B, N, L)
            pad_mask: Tensor with shape (BN, L), True -> Non-pad
            kvc_mask: dict with keys 'attn_map_gt' and 'entity_cls_gt', where
                attn_map_gt is Tensor with shape (B, cls_num, N)
                entity_cls_gt is Tensor with shape (B, cls_num)
            rec_embedding: recognition embedding layer
            img_metas: dict

        Returns:
            loss: dict
        """
        B, N, L = shapes
        C = seq_feat.shape[-1]
        # BN, L, C -> BN, C, L
        # is mask here properly ?
        seq_feat_new = seq_feat.masked_fill(
            pad_mask.unsqueeze(-1) == 0, 0).permute(0, 2, 1)
        node_feat = []
        for conv_ in self.conv1ds:
            node_feat.append(conv_(seq_feat_new))
        # BN, C, num_conv1d -> BN, C * num_conv1d -> BN, C -> B, N, C
        node_feat = self.fuse_fc(torch.cat(node_feat, dim=-1).reshape(B*N, -1)).reshape(B, N, C)

        # # add ReLU here ? - modified by whua
        # node_feat = self.fuse_activate(node_feat)

        node_feat = self.pe(node_feat)

        if self.use_self_attn:
            node_feat = self.self_attn_sublayer(node_feat, lambda x: self.self_attn(x, x, x, kvc_mask['ins_mask']))

        entity_num = len(self.entity_list) + 1
        if self.use_rec_embed:
            raise NotImplementedError
            cls_embed = []
            # need to modify this part to add 'O'
            for entity_name in self.entity_list:
                # num_char,
                entity_index = torch.tensor(list(map(lambda x: self.ocr_dict[x], entity_name)), device=seq_feat.device)
                # num_char, C -> C
                cls_embed.append(rec_embedding(entity_index).mean(dim=0))
            # cls_num, C -> 1, cls_num, C -> B, cls_num, C
            cls_embed = torch.stack(cls_embed, dim=0).unsqueeze(0).expand(B, entity_num, C)
        else:
            # B, cls_num, C
            cls_embed = self.cls_embedding(
                torch.arange(0, entity_num, device=seq_feat.device).unsqueeze(0).expand(B, entity_num))
        cls_embed = self.pe(cls_embed)

        # B, cls_num, C
        entity_logits = self.sublayer(cls_embed, lambda x: self.cross_attn(x, node_feat, node_feat, None))

        if not self.training:
            return dict(), self.entity_norm(entity_logits)
        # # B * cls_num, C -> B * cls_num, 2
        # entity_logits = self.out_fc(entity_logits.reshape(-1, C))

        # merge multi-head attention map together
        # in pooling manner
        # B, headers, cls_num, N -> B, cls_num, N -> B, N, cls_num -> BN, cls_num
        attn_map = self.cross_attn.attn.mean(dim=1).permute(0, 2, 1).reshape(B * N, -1)
        # # in FC manner
        # # B, headers, cls_num, N -> B, N, cls_num, headers -> B, N, cls_num, 1 -> BN, cls_num
        # norm_func = nn.LayerNorm([])
        # attn_map = torch.sigmoid(self.head_fuse_fc(self.cross_attn.attn.permute(0, 3, 2, 1)).reshape(-1, entity_num))

        # calculate loss
        losses = dict()
        if self.use_entity_cls_loss:
            losses['loss_entity'] = self.entity_cls_loss(entity_logits, kvc_mask['entity_cls_gt'].reshape(-1))
        # B, N, cls_num -> B, N, cls_num -> BN, cls_num
        attn_map_gt = kvc_mask['attn_map_gt'].permute(0, 2, 1).reshape(B * N, -1)
        losses['loss_attn_map'] = self.attn_map_loss(attn_map, attn_map_gt)
        return losses, self.entity_norm(entity_logits)


class CustomOneStageCatcher(nn.Module):
    def __init__(self, entity_list, entity_dict, d_model, attn_args, kernel_sizes=[2, 3, 4], fuse_type='conv',
                 dropout=0.2, use_entity_cls_loss=False, use_rec_embed=False, ocr_dict=None, layer_num=1,
                 use_layout=False):
        super(CustomOneStageCatcher, self).__init__()
        self.entity_list = entity_list
        self.entity_dict = entity_dict
        self.fuse_type = fuse_type
        self.use_entity_cls_loss = use_entity_cls_loss
        self.use_rec_embed = use_rec_embed
        self.use_layout = use_layout
        if use_rec_embed:
            self.ocr_dict = ocr_dict
        else:
            self.cls_embedding = nn.Embedding(1+len(entity_list), d_model)
        if use_layout:
            self.input_norm = nn.LayerNorm(d_model)
        # todo: add pooling mode
        assert fuse_type in ['conv'], f"Unexpected fuse type: {fuse_type}"

        if fuse_type == 'conv':
            self.conv1ds = nn.ModuleList()
            for _ in kernel_sizes:
                self.conv1ds.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=_),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool1d(output_size=1)
                    )
                )
            self.fuse_fc = nn.Linear(d_model * len(kernel_sizes), d_model)
            self.fuse_activate = nn.ReLU()
        else:
            raise NotImplementedError

        attn_args.update(d_model=d_model)
        # self.self_attn = clones(MultiHeadAttention(**attn_args), layer_num)
        # self.self_attn_sublayer = clones(SubLayerConnection(d_model, dropout), layer_num)
        # self.cross_attn = MultiHeadAttention(**attn_args)
        # self.sublayer = SubLayerConnection(d_model, dropout)
        self.pe = PositionalEncoding(d_hid=d_model, n_position=1000)
        self.out_fc = nn.Linear(d_model, len(entity_list) + 1)
        self.out_norm = nn.BatchNorm1d(len(entity_list) + 1)
        self.entity_norm = nn.LayerNorm(d_model)
        # self.entity_cls_loss = nn.BCELoss()
        self.entity_cls_loss = BalancedBCELoss()
        self.pred_results = None

    def forward(self, seq_feat, shapes, pad_mask, kvc_mask, rec_embedding=None,
                img_metas=None, layout_feat=None, gt_texts=None, rev_ocr_dict=None):
        """
        Build key-value relation between entity class embedding and instance-level feature
        Args:
            seq_feat: Tensor with shape (BN, L, C)
            shapes: (B, N, L)
            pad_mask: Tensor with shape (BN, L), True -> Non-pad
            kvc_mask: dict with keys 'attn_map_gt' and 'entity_cls_gt', where
                attn_map_gt is Tensor with shape (B, cls_num, N)
                entity_cls_gt is Tensor with shape (B, cls_num)
            rec_embedding: recognition embedding layer
            img_metas: dict

        Returns:
            loss: dict
        """
        B, N, L = shapes
        C = seq_feat.shape[-1]
        # BN, L, C -> BN, C, L
        # is mask here properly ?
        seq_feat_new = seq_feat.masked_fill(
            pad_mask.unsqueeze(-1) == 0, 0).permute(0, 2, 1)
        node_feat = []
        for conv_ in self.conv1ds:
            node_feat.append(conv_(seq_feat_new))
        # BN, C, num_conv1d -> BN, C * num_conv1d -> BN, C -> B, N, C
        node_feat = self.fuse_fc(torch.cat(node_feat, dim=-1).reshape(B*N, -1)).reshape(B, N, C)
        if self.use_layout:
            node_feat = self.input_norm(node_feat + layout_feat)

        # # add ReLU here ? - modified by whua
        # node_feat = self.fuse_activate(node_feat)

        # # B, N, C
        # node_feat = self.pe(node_feat)
        # for attn_layer, sub_layer in zip(self.self_attn, self.self_attn_sublayer):
        #     node_feat = sub_layer(node_feat, lambda x: attn_layer(x, x, x, kvc_mask['ins_mask']))
        # B, N, cls_num -> BN, cls_num
        node_pred = self.out_norm(self.out_fc(node_feat).reshape(B * N, -1))
        node_pred = torch.sigmoid(node_pred)

        # calculate loss
        losses = dict()

        # transcripts = []
        # for ind in range(gt_texts.shape[1]):
        #     line_ = ""
        #     for char_idx in gt_texts[0, ind]:
        #         if char_idx == 1:
        #             break
        #         line_ += rev_ocr_dict[char_idx.item()]
        #     transcripts.append(line_)
        # binary_pred = (node_pred > 0.5).long()
        # attn_map = self.self_attn[0].attn
        # node_kv = dict()
        # for info_ in binary_pred[:, 1:].nonzero():
        #     idx, cls_idx = info_
        #     cls = self.entity_list[cls_idx]
        #     if cls not in node_kv:
        #         node_kv[cls] = [transcripts[idx]]
        #     else:
        #         node_kv[cls].append(transcripts[idx])
        # print(f"file: {img_metas[0]['filename'].split('/')[-1]}\n{node_kv}\n")
        # self.pred_results = node_kv
        # for k, v in node_kv.items():
        #     print(f"{k} : {v}")
        # print("\n\n")
        # if self.training:
        #     binary_pred = (node_pred > 0.5).long()
        #     wrong_num = (binary_pred != kvc_mask['attn_map_gt'].permute(0, 2, 1).reshape(B * N, -1)).sum()
        #     print(f"Incorrect num: {wrong_num} of {img_metas[0]['filename'].split('/')[-1]}")
        # # import ipdb
        # # ipdb.set_trace()

        if not self.training:
            return losses, node_feat

        # B, N, cls_num -> B, N, cls_num -> BN, cls_num
        attn_map_gt = kvc_mask['attn_map_gt'].permute(0, 2, 1).reshape(B * N, -1)
        losses['loss_entity'] = self.entity_cls_loss(node_pred, attn_map_gt)
        return losses, node_feat
        # return losses, self.entity_norm(node_feat)
    # (node_pred[:, 0] == 0).nonzero()
    # attn_map_gt[:, 1:].nonzero()



#contrastive learning：entity list embedding and feature做矩阵乘法
#input: tensor of entity list;  tensor of instance feature
#output: loss
#contrast learning input:
# instance_feature : context/texture feature
# embedding: context encoder/texure encoder

class OneStageContrast(nn.Module):
    def __init__(self, entity_list, d_model, kvc_type=None, kvc_weights=1.0):
        super(OneStageContrast, self).__init__()
        self.entity_list = entity_list
        self.entity_cls_loss = torchvision.ops.sigmoid_focal_loss
        self.kvc_weights = kvc_weights
        self.kvc_type = kvc_type
        #embedding
        self.cls_embedding = nn.Embedding(1 + len(entity_list), d_model)

        #self-attention
        self.self_attention = ScaledDotProductAttention(d_model)
        self.cross_attention = CrossAttention(headers=8, d_model=d_model, dropout=0.1)
        #clip
        # self.text = clip.tokenize(
        #     ["SS", "CE-PS", "CE-P1", "CE-D", "CE-PP", "TF-PS", "TF-P1", "TF-D", "TF-PP", "SO-PS", "SO-P1", "SO-D",
        #      "SO-PP", "CAR-PS", "CAR-P1", "CAR-D", "CAR-PP", "PRO-PS", "PRO-P1", "PRO-D", "PRO-PP", "Others"])
    def forward(self, context_feature, texture_feature, seq_mask, shape, kvc_mask, training, context_encoder=False, texture_encoder=False, context_embedding=False, texture_embedding=False):
        """
        Build key-value relation between entity class embedding and instance-level feature
        Args:
            context_feature: (BN, HW, C)
            texture_feature: (BN, L, C)->(B,N,L,C)->(B,N,C)
            shape: (B,N,C,H,W)
            kvc_mask: dict with keys 'attn_map_gt' and 'entity_cls_gt', where
                attn_map_gt is Tensor with shape (B, cls_num, N)
                entity_cls_gt is Tensor with shape (B, cls_num)
        Returns:
            loss: dict, kie_logits
        """
        # ipdb.set_trace()
        #instance_feature: B,N,L,C->B,N,C
        # ipdb.set_trace()
        B, N, C, H, W = shape

        #texture_feature:(B*N,L,C)->(B,N,L,C)->(B,N,C)
        _, L, _ = texture_feature.shape
        texture_feature = texture_feature.reshape(B, N, L, C)
        # BUG: this should be a masked mean
        mask = seq_mask[:, :, :, 0].permute(1, 0, 2)
        texture_feature = (texture_feature * mask.unsqueeze(-1)).sum(-2) / (mask.sum(-1).unsqueeze(-1) + 1e-5)
        # B,N,L,C->B,N,C
        # texture_feature = torch.mean(texture_feature, dim=-2)
        # norm
        texture_feature = texture_feature / texture_feature.norm(dim=1, keepdim=True)

        #context_feature:(B*N,H*W,C)->(B,N,HW,C)->(B,N,C)
        context_feature = context_feature.reshape(B, N, -1, C)
         #B,N,H*W,C->B,N,C
        context_feature = torch.mean(context_feature, dim=-2)
        context_feature = context_feature/context_feature.norm(dim=1, keepdim=True)

        #instance_feature:B,N,C
        if texture_encoder and context_encoder:
            instance_feature = texture_feature + context_feature
        elif texture_encoder and not context_encoder:
            instance_feature = texture_feature
        elif not texture_encoder and context_encoder:
            instance_feature = context_feature

        #embedding初始化entity_feature  B,cls_num->B,cls_num,C->B,C,cls_num
        cls_num = len(self.entity_list) + 1
        # entity_feature:(B,cls,C)->(B,C,cls)
        if self.kvc_type == 'clip':
            type='clip'
            # #use clip
            # device = instance_feature.device
            # model, preprocess = clip.load("ViT-B/32", device=device)
            # text = self.text.to(device)
            # with torch.no_grad():
            #     entity_feature = model.encode_text(text)
            # entity_feature = entity_feature.unsqueeze(0)
            # # ipdb.set_trace()
            # entity_feature = entity_feature.transpose(-1, -2).float()

        elif self.kvc_type =='none':
            type='none'
        else:
            # use embedding加入parameter将初始化的entity变成可更新参数
            # B,cls->B,cls,C
            # NOTE: the entity index can be not a parameter.
            entity_feature = self.cls_embedding(
                nn.Parameter(torch.arange(0, cls_num, device=instance_feature.device).unsqueeze(0).expand(B, cls_num),
                         requires_grad=False))
            # NOTE: a better normalization?
            entity_feature = entity_feature / entity_feature.norm(dim=1, keepdim=True)
            #B, cls, C->B, C, cls_num
            entity_feature = entity_feature.transpose(-1, -2)

            # try some encoder
            device = entity_feature.device
            # NOTE: the linear should be defined in the `__init__`
            encoder = nn.Linear(N, cls_num).to(device)
            if context_embedding and not texture_embedding:
                entity_feature = encoder(context_feature.reshape(B, C, -1))
            elif not context_embedding and not texture_embedding:
                entity_feature = encoder(texture_feature.reshape(B, C, -1))
            elif context_embedding and texture_embedding:
                entity_feature = encoder(instance_feature.reshape(B, C, -1))



        # ipdb.set_trace()
        if self.kvc_type == 'matrix' or self.kvc_type == 'clip':
            #contrastive learning:(B,N,C)*(B,C,cls)->(B,N,cls)
            logit_instance = torch.bmm(instance_feature, entity_feature)
        elif self.kvc_type == 'self_attention':
            #self-attention: (B,N,C)*(B,C,N)->(B,N,N)
            _, logit_instance = self.self_attention(instance_feature, instance_feature, instance_feature)
        elif self.kvc_type == 'cross_attention1':
            #entity_feature:(B,C,cls)->(B,cls,C)
            entity_feature = entity_feature.transpose(-1, -2)
            #(1)use entity_feature as query cross_attention:(B,cls,C)*(B,C,N)->(B,cls,N)->(B,N,cls)
            #logit_instance:(B,N,cls)
            _, logit_instance = self.cross_attention(entity_feature, instance_feature, instance_feature)
            logit_instance = logit_instance.transpose(-1, -2)
        elif self.kvc_type == 'cross_attention2':
            # (2)use instance_feature as query  entity_feature:(B,C,cls)->(B,cls,C)
            entity_feature = entity_feature.transpose(-1, -2)
            #logit_instance:(B,N,C),output:(B,N,cls)
            logit_instance, output = self.cross_attention(instance_feature, entity_feature, entity_feature)
        elif self.kvc_type == 'none':
            logit_instance = instance_feature

        # save_image_tensor(logit_instance, "logit_instance.jpg")
        # ipdb.set_trace()
        if training and self.kvc_type != 'none':
            if self.kvc_type == 'matrix' or self.kvc_type == 'cross_attention' or self.kvc_type == 'clip':
                #gt_entity:(B,cls,N)->(B,N,cls)
                gt_entity = kvc_mask['attn_map_gt']
                gt_entity = gt_entity.transpose(-1, -2)
                losses = dict()
                # loss赋予一定权重
                losses['loss_entity'] = self.kvc_weights * self.entity_cls_loss(logit_instance, gt_entity)
                return losses, logit_instance
            elif self.kvc_type == 'self_attention':
                # gt_entity:(B,cls,N)->(B,N,cls)->(B,N,N)
                gt_entity = kvc_mask['attn_map_gt']
                gt_entity = gt_entity.transpose(-1, -2)
                gt_entity = torch.matmul(gt_entity, gt_entity.transpose(-1, -2))
                losses = dict()
                # loss赋予一定权重
                losses['loss_entity'] = self.kvc_weights * self.entity_cls_loss(logit_instance, gt_entity)
                return losses, logit_instance
            elif self.kvc_type == 'cross_attention1':
                gt_entity = kvc_mask['attn_map_gt']
                gt_entity = gt_entity.transpose(-1, -2)
                losses = dict()
                # loss赋予一定权重
                losses['loss_entity'] = self.kvc_weights * self.entity_cls_loss(logit_instance, gt_entity)
                return losses, logit_instance
            elif self.kvc_type == 'cross_attention2':
                gt_entity = kvc_mask['attn_map_gt']
                gt_entity = gt_entity.transpose(-1, -2)
                losses = dict()
                # loss赋予一定权重
                losses['loss_entity'] = self.kvc_weights * self.entity_cls_loss(output, gt_entity)
                return losses, logit_instance
        else:
            return logit_instance

class LogitContrast(nn.Module):
    def __init__(self, entity_list, d_model, kvc_weights=1.0):
        super(LogitContrast, self).__init__()
        self.entity_list = entity_list
        self.entity_cls_loss = torchvision.ops.sigmoid_focal_loss
        self.kvc_weights = kvc_weights
        #embedding
        self.cls_embedding = nn.Embedding(1 + len(entity_list), d_model)
    def forward(self, kie_logit, ocr_logit, shape, kvc_mask):
        """
        Build key-value relation between entity class embedding and instance-level feature
        Args:
            kie_logit: (BN, L, C)
            ocr_logit:  (BN, L, C)
            shape: (B,N,C,H,W)
            kvc_mask: dict with keys 'attn_map_gt' and 'entity_cls_gt', where
                attn_map_gt is Tensor with shape (B, cls_num, N)
                entity_cls_gt is Tensor with shape (B, cls_num)
        Returns:
            loss: dict, kie_logits
        """
        B, N, C, H, W = shape
        # ocr_logit:(BN,L,C)->(B,NL,C)
        ocr_logit = ocr_logit.reshape(B, -1, C)
        # norm
        ocr_logit = ocr_logit / ocr_logit.norm(dim=1, keepdim=True)

        kie_logit = kie_logit.reshape(B, -1, C)
        # norm
        kie_logit = kie_logit / kie_logit.norm(dim=1, keepdim=True)
        kie_logit = kie_logit.transpose(-1, -2)

        # ipdb.set_trace()
        # calculate matrix:(B,NL,C)*(B,NL,C)->(B,NL,NL)
        logit_instance = torch.bmm(ocr_logit, kie_logit)

        # gt_entity:(B,NL,NL)
        gt_entity = kvc_mask['attn_map_gt']
        gt_entity = gt_entity.transpose(-1, -2)

        # 计算返回loss
        losses = dict()
        # loss赋予一定权重
        # losses['loss_logit'] = self.kvc_weights * self.entity_cls_loss(logit_instance, gt_entity)

        return losses, logit_instance

class FeatureContrast(nn.Module):
    def __init__(self, entity_list, d_model, kvc_weights=1.0):
        super(FeatureContrast, self).__init__()
        self.entity_list = entity_list
        self.entity_cls_loss = torchvision.ops.sigmoid_focal_loss
        self.kvc_weights = kvc_weights
        self.cls_embedding = nn.Embedding(1 + len(entity_list), d_model)

    def forward(self, ocr_feature, kie_feature, shape, kvc_mask):
        """
        Build key-value relation between entity class embedding and instance-level feature
        Args:
            ocr_feature: (BN, L, C)->(B,N,L,C)->(B,N,C)
            kie_feature:(context_feature): (BN, HW, C)
            shape: (B,N,C,H,W)
            gt_entities: (B,N,L)
            kvc_mask: dict with keys 'attn_map_gt' and 'entity_cls_gt', where
                attn_map_gt is Tensor with shape (B, cls_num, N)
                entity_cls_gt is Tensor with shape (B, cls_num)
        Returns:
            loss: dict, kie_logit
        """
        B, N, C, H, W = shape
        # ocr_logit:(BN,L,C)->(B,NL,C)
        ocr_feature = ocr_feature.reshape(B, -1, C)
        # norm
        ocr_feature = ocr_feature / ocr_feature.norm(dim=1, keepdim=True)

        kie_feature = kie_feature.reshape(B, -1, C)
        # norm
        kie_feature = kie_feature / kie_feature.norm(dim=1, keepdim=True)
        kie_feature = kie_feature.transpose(-1, -2)

        # ipdb.set_trace()
        # calculate matrix:(B,NL,C)*(B,NL,C)->(B,NL,NL)
        logit_feature = torch.bmm(ocr_feature, kie_feature)

        # gt_entity:(B,NL,NL)
        gt_entity = kvc_mask['attn_map_gt']
        gt_entity = gt_entity.transpose(-1, -2)

        # 计算返回loss
        losses = dict()
        # loss赋予一定权重
        losses['loss_feature'] = self.kvc_weights * self.entity_cls_loss(logit_feature, gt_entity)

        return losses, logit_feature

KV_CATCHER = {
    'OneStageCatcher': OneStageCatcher,
    'CustomOneStageCatcher': CustomOneStageCatcher,
    'OneStageContrast': OneStageContrast,
    'LogitContrast': LogitContrast,
    'FeatureContrast': FeatureContrast
}


def build_kv_catcher(cfg):
    func = KV_CATCHER[cfg.pop('type')]
    return func(**cfg)
