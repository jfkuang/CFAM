#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/25 15:00
# @Author : WeiHua
import torch
import torch.nn as nn
from .text_decoder import MultiHeadAttention, clones, SubLayerConnection, FeedForward
import torch.nn.functional as F

class CrossLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout, drop_path=0.,
                 detach_rec=False, glb_cross=False, kernel_sizes=[2, 3, 4]):
        super(CrossLayer, self).__init__()
        self.detach_rec = detach_rec
        self.glb_cross = glb_cross
        self.size = size

        self.kie_self_attn = MultiHeadAttention(**self_attn)
        self.kie_cross_attn = MultiHeadAttention(**cross_attn)
        self.sublayer_kie = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 3)
        self.kie_feed_forward = FeedForward(**feed_forward)

        if self.glb_cross:
            self.conv1ds = nn.ModuleList()
            for _ in kernel_sizes:
                self.conv1ds.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=size, out_channels=size, kernel_size=_),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool1d(output_size=1)
                    ))
            self.fuse_fc = nn.Linear(size * len(kernel_sizes), size)

        if not self.detach_rec:
            self.rec_self_attn = MultiHeadAttention(**self_attn)
            self.rec_cross_attn = MultiHeadAttention(**cross_attn)
            self.sublayer_rec = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 3)
            self.rec_feed_forward = FeedForward(**feed_forward)

    def forward(self, rec_feat, kie_feat, cross_mask, self_mask, num_ins=-1, pad_mask=None):
        """

        Args:
            rec_feat: Tensor with shape (BN, L, C)
            kie_feat: Tensor with shape (BN, L, C)
            cross_mask:
            self_mask:
            num_ins: N
            pad_mask: Tensor with shape (BN, L), True -> Non-Pad

        Returns:

        """
        if not self.detach_rec:
            rec_feat_hid = self.sublayer_rec[0](rec_feat, lambda x: self.rec_self_attn(x, x, x, self_mask))
            kie_feat_hid = self.sublayer_kie[0](kie_feat, lambda x: self.kie_self_attn(x, x, x, self_mask))

            rec_feat = self.sublayer_rec[1](rec_feat_hid,
                                            lambda x: self.rec_cross_attn(x, kie_feat_hid, kie_feat_hid, cross_mask))
            kie_feat = self.sublayer_kie[1](kie_feat_hid,
                                            lambda x: self.kie_cross_attn(x, rec_feat_hid, rec_feat_hid, cross_mask))
            rec_feat = self.sublayer_rec[2](rec_feat, self.rec_feed_forward)
            kie_feat = self.sublayer_kie[2](kie_feat, self.kie_feed_forward)
            return rec_feat, kie_feat
        else:
            if self.glb_cross:
                B = rec_feat.shape[0] // num_ins
                L, C = rec_feat.shape[-2:]
                # BN, L, C -> BN, C, L
                glb_rec_feat = rec_feat.masked_fill(
                    pad_mask.unsqueeze(-1) == 0, 0).permute(0, 2, 1)
                rec_feats = []
                for conv_ in self.conv1ds:
                    rec_feats.append(conv_(glb_rec_feat))
                # BN, C, num_conv1d -> BN, C * num_conv1d -> BN, C -> B, N, C
                rec_feats = self.fuse_fc(torch.cat(rec_feats, dim=-1).reshape(B * num_ins, -1)).reshape(B, num_ins, C)
                # B, N, C -> B, 1, N, C -> B, N, N, C -> BN, N, C
                rec_feats = rec_feats.unsqueeze(1).expand(B, num_ins, num_ins, C).reshape(B * num_ins, -1, C)
                kie_feat = self.sublayer_kie[0](kie_feat, lambda x: self.kie_self_attn(x, x, x, self_mask))
                kie_feat = self.sublayer_kie[1](kie_feat,
                                                lambda x: self.kie_cross_attn(x, rec_feats, rec_feats, None))
                return rec_feat, self.sublayer_kie[2](kie_feat, self.kie_feed_forward)

            kie_feat = self.sublayer_kie[0](kie_feat, lambda x: self.kie_self_attn(x, x, x, self_mask))
            kie_feat = self.sublayer_kie[1](kie_feat,
                                            lambda x: self.kie_cross_attn(x, rec_feat, rec_feat, cross_mask))
            return rec_feat, self.sublayer_kie[2](kie_feat, self.kie_feed_forward)


class InteractBlock(nn.Module):
    def __init__(self, num_block, d_model, keep_pre=False, kie_dict_size=-1,
                 rec_dict_size=-1, **kwargs):
        super(InteractBlock, self).__init__()
        # current only test num_block=1
        assert num_block == 1
        # 0 -> REC, 1 -> KIE
        self.rec_type_embed = nn.Parameter(torch.zeros(d_model))
        self.kie_type_embed = nn.Parameter(torch.zeros(d_model))
        self.kie_proj = nn.Linear(d_model, d_model)
        self.rec_proj = nn.Linear(d_model, d_model)
        self.kie_norm = nn.LayerNorm(d_model)
        self.rec_norm = nn.LayerNorm(d_model)

        self.keep_pre = keep_pre
        if keep_pre:
            self.rec_out_norm = nn.LayerNorm(d_model)
            self.rec_out_fc = nn.Linear(d_model, rec_dict_size)
            self.kie_out_norm = nn.LayerNorm(d_model)
            self.kie_out_fc = nn.Linear(d_model, kie_dict_size)

        kwargs.update(size=d_model)
        self.cross_layers = clones(CrossLayer(**kwargs), num_block)

    def forward(self, feat_rec, feat_kie, cross_mask, self_mask, logits_logger, num_ins=-1, pad_mask=None):
        """
        Cross-modal interaction module
        Args:
            feat_rec: Tensor with shape (BN, L, C)
            feat_kie: Tensor with shape (BN, L, C)
            cross_mask: mask for cross-attention, (BN, 1, L, L)
            self_mask: mask for self-attention, (BN, 1, L, L)

        Returns:

        """
        if self.keep_pre:
            logits_logger['REC'].append(self.rec_out_fc(self.rec_out_norm(feat_rec)))
            logits_logger['KIE'].append(self.kie_out_fc(self.kie_out_norm(feat_kie)))

        feat_rec = self.rec_norm(self.rec_proj(feat_rec) + self.rec_type_embed)
        feat_kie = self.kie_norm(self.kie_proj(feat_kie) + self.kie_type_embed)
        for idx, layer in enumerate(self.cross_layers):
            feat_rec, feat_kie = layer(feat_rec, feat_kie, cross_mask, self_mask,
                                       num_ins=num_ins, pad_mask=pad_mask)
        return feat_rec, feat_kie

# consider auto-regression ?
class CrossMimic(nn.Module):
    def __init__(self, margin=0.5, contrast_with_ins=False):
        super(CrossMimic, self).__init__()
        self.loss_func = nn.CosineEmbeddingLoss(margin=margin)
        self.contrast_with_ins = contrast_with_ins

    def run(self, feat_rec, feat_common, logits_logger, shapes, ie_mask):
        """
        Calculate global similarity loss for mimic learning
        Args:
            feat_rec: Tensor with shape (BN, L, C)
            feat_common: Tensor with shape (BN, L, C)
            logits_logger: dict
            shapes: B, N, L
            ie_mask: Tensor with shape (BN, L), True -> Key-Info

        Returns:

        """
        B, N, L = shapes
        C = feat_rec.shape[-1]
        # BN, L, C
        feat_rec_new = feat_rec.detach().masked_fill(
            ie_mask.unsqueeze(-1) == 0, 0)
        feat_common_new = feat_common.masked_fill(
            ie_mask.unsqueeze(-1) == 0, 0)
        # BN, L, C -> B, NL, C -> B, C
        t_glb = feat_rec_new.reshape(B, N * L, C).sum(dim=1).div(
            ie_mask.reshape(B, -1).sum(dim=1).unsqueeze(-1))
        c_glb = feat_common_new.reshape(B, N * L, C).sum(dim=1).div(
            ie_mask.reshape(B, -1).sum(dim=1).unsqueeze(-1))
        contrast_pos_loss = self.loss_func(c_glb, t_glb,
                                           torch.ones(B, device=feat_rec.device))
        logits_logger.update(contrast_pos_loss=contrast_pos_loss)
        return feat_common

# todo: modify mimic and contrast, maybe refer to:
#   [1] https://blog.csdn.net/yyhaohaoxuexi/article/details/113824125
#   [2] https://blog.csdn.net/taoqick/article/details/124781102

class GlobalCrossMimic(nn.Module):
    def __init__(self, loss_weight=0.1, min_margin=-0.9, max_margin=0.9):
        super(GlobalCrossMimic, self).__init__()
        self.loss_weight = loss_weight
        self.min_margin = min_margin
        self.max_margin = max_margin

    def run(self, feat_rec, feat_kie, logits_logger, shapes, pad_mask):
        """
        Calculate global similarity loss for mimic learning
        Args:
            feat_rec: Tensor with shape (BN, L, C)
            feat_kie: Tensor with shape (BN, L, C)
            logits_logger: dict
            shapes: B, N, L
            # seq_mask: Tensor with shape (BN, 1, L, L)
            # ins_mask: Tensor with shape (B, N)
            pad_mask: Tensor with shape (BN, L), True -> Non-pad

        Returns:

        """
        B, N, L = shapes
        C = feat_rec.shape[-1]
        # BN, L, C
        feat_rec_new = feat_rec.masked_fill(
            pad_mask.unsqueeze(-1) == 0, 0)
        feat_kie_new = feat_kie.masked_fill(
            pad_mask.unsqueeze(-1) == 0, 0)
        # BN, L, C -> B, NL, C -> B, C
        feat_rec_new = feat_rec_new.reshape(B, N * L, C).sum(dim=1).div(
            pad_mask.reshape(B, -1).sum(dim=1))
        feat_kie_new = feat_kie_new.reshape(B, N * L, C).sum(dim=1).div(
            pad_mask.reshape(B, -1).sum(dim=1))

        # (B, )
        simi_dist = torch.cosine_similarity(feat_rec_new, feat_kie_new)
        simi_dist = torch.clamp(simi_dist, min=self.min_margin, max=self.max_margin)
        logits_logger.update(contrast_pos_loss=self.loss_weight * torch.exp(simi_dist).mean())


class TaskAwareMimic(nn.Module):
    def __init__(self, scale_param=0.01, contrast_with_ins=False):
        super(TaskAwareMimic, self).__init__()
        self.scale_param = scale_param
        self.loss_func = nn.CrossEntropyLoss()
        self.contrast_with_ins = contrast_with_ins

    def run(self, feat_rec, feat_kie, logits_logger, shapes, pad_mask, ie_mask):
        """
        Calculate global similarity loss for mimic learning
        Args:
            feat_rec: Tensor with shape (BN, L, C)
            feat_kie: Tensor with shape (BN, L, C)
            logits_logger: dict
            shapes: B, N, L
            pad_mask: Tensor with shape (BN, L), True -> Non-pad
            ie_mask: Tensor with shape (BN, L), True -> Key-Info

        Returns:

        """
        B, N, L = shapes
        C = feat_rec.shape[-1]
        # BN, L, C
        feat_rec_new = feat_rec.masked_fill(
            ie_mask.unsqueeze(-1) == 0, 0)
        feat_kie_new = feat_kie.masked_fill(
            ie_mask.unsqueeze(-1) == 0, 0)
        # BN, L, C -> B, NL, C -> B, C
        t_glb = feat_rec_new.reshape(B, N * L, C).sum(dim=1).div(
            ie_mask.reshape(B, -1).sum(dim=1).unsqueeze(-1))
        e_glb = feat_kie_new.reshape(B, N * L, C).sum(dim=1).div(
            ie_mask.reshape(B, -1).sum(dim=1).unsqueeze(-1))

        # BN, L -> B, N, L -> B, N
        ie_select = ie_mask.reshape(B, N, L).sum(dim=-1) != 0

        # # BN, L, C -> BN, C -> B, N, C
        t_ins_all = feat_rec_new.sum(dim=1).div(
            ie_mask.sum(dim=1).unsqueeze(1)).reshape(B, N, C)
        e_ins_all = feat_kie_new.sum(dim=1).div(
            ie_mask.sum(dim=1).unsqueeze(1)).reshape(B, N, C)

        total_loss = None
        for batch_idx in range(B):
            # todo: finish this part
            # get instance-index which contains key info
            # valid_ins_num, C -> N, C for short
            t_ins_norm = F.normalize(t_ins_all[batch_idx, ie_select[batch_idx]])
            e_ins_norm = F.normalize(e_ins_all[batch_idx, ie_select[batch_idx]])
            if self.contrast_with_ins:
                # N+2, C
                batched_te_norm = F.normalize(torch.stack([t_glb[batch_idx], e_glb[batch_idx]]))
                batched_te_norm = torch.cat([batched_te_norm, e_ins_norm])
                batched_et_norm = F.normalize(torch.stack([e_glb[batch_idx], t_glb[batch_idx]]))
                batched_et_norm = torch.cat([batched_et_norm, t_ins_norm], dim=0)
                # N, N+2
                simi_score_t_te = torch.matmul(t_ins_norm, batched_te_norm.transpose(0, 1))
                simi_score_e_et = torch.matmul(e_ins_norm, batched_et_norm.transpose(0, 1))
                label = torch.zeros(simi_score_t_te.shape[0], device=simi_score_t_te.device).long()
                loss_t_proto = self.loss_func(simi_score_t_te / self.scale_param, label)
                loss_e_proto = self.loss_func(simi_score_e_et / self.scale_param, label)
            else:
                # current, we only consider one pos pair and one neg pair at a time
                # 2, C
                batched_te_norm = F.normalize(torch.stack([t_glb[batch_idx], e_glb[batch_idx]]))
                # N, 2
                simi_score_t_te = torch.matmul(t_ins_norm, batched_te_norm.transpose(0, 1))
                simi_score_e_te = torch.matmul(e_ins_norm, batched_te_norm.transpose(0, 1))
                label_t = torch.zeros(simi_score_t_te.shape[0], device=simi_score_t_te.device).long()
                label_e = torch.ones(simi_score_t_te.shape[0], device=simi_score_t_te.device).long()
                loss_t_proto = self.loss_func(simi_score_t_te / self.scale_param, label_t)
                loss_e_proto = self.loss_func(simi_score_e_te / self.scale_param, label_e)

            if batch_idx == 0:
                total_loss = loss_e_proto + loss_t_proto
            else:
                total_loss = total_loss + loss_e_proto + loss_t_proto

            # # previous version
            # # get instance-index which contains key info
            # # valid_ins_num, C -> N, C for short
            # t_ins = t_ins_all[batch_idx, ie_select[batch_idx]]
            # e_ins = e_ins_all[batch_idx, ie_select[batch_idx]]
            #
            # # current, we only consider one pos pair and one neg pair at a time
            # e_pos_simi = torch.exp(torch.cosine_similarity(e_ins, e_glb[batch_idx].unsqueeze(0)))
            # e_neg_simi = torch.exp(torch.cosine_similarity(e_ins, t_glb[batch_idx].unsqueeze(0)))
            # loss_e_proto = -torch.log(e_pos_simi / (e_pos_simi + e_neg_simi))
            #
            # t_pos_simi = torch.exp(torch.cosine_similarity(t_ins, t_glb[batch_idx].unsqueeze(0)))
            # t_neg_simi = torch.exp(torch.cosine_similarity(t_ins, e_glb[batch_idx].unsqueeze(0)))
            # loss_t_proto = -torch.log(t_pos_simi / (t_pos_simi + t_neg_simi))
            #
            # if batch_idx == 0:
            #     total_loss = (loss_e_proto+loss_t_proto).mean()
            # else:
            #     total_loss = total_loss + (loss_e_proto+loss_t_proto).mean()
        logits_logger.update(contrast_pos_loss=total_loss/B)


MIMIC_FUNC = {
    "CrossMimic": CrossMimic,
    "GlobalCrossMimic": GlobalCrossMimic,
    "TaskAwareMimic": TaskAwareMimic
}

def build_mimic(cfg):
    func = MIMIC_FUNC[cfg.pop('type')]
    return func(**cfg)


