#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/15 15:02
# @Author : WeiHua

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
from functools import partial

from mmcv.runner import auto_fp16


class CustomViTSTR(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, in_chans=256, embed_dim=768, depth=12, out_depth=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 num_patches=196, max_len=60):
        """
        Args:
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        # OpenMM AMP
        self.fp16_enabled = False
        self.max_len = max_len
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.out_block = nn.ModuleList([
            Block(
                dim=in_chans, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(out_depth)
        ])
        self.out_norm = norm_layer(in_chans)

        # Classifier head
        self.rec_head = None
        self.kie_head = None

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # Make sure the dimension can match with other model
        self.input_converter = nn.Linear(in_chans, embed_dim, bias=False)
        self.output_converter = nn.Linear(embed_dim, in_chans, bias=False)
        # convert the output sequence length to expected one
        self.seq_converter = nn.Linear(num_patches + 1, max_len)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def reset_classifier(self, num_classes, cls_type='REC'):
        assert cls_type in ['REC', 'KIE']
        if cls_type == 'REC':
            self.rec_head = nn.Linear(self.in_chans, num_classes) if num_classes > 0 else nn.Identity()
            self.rec_head.apply(self._init_weights)
        else:
            self.kie_head = nn.Linear(self.in_chans, num_classes) if num_classes > 0 else nn.Identity()
            self.kie_head.apply(self._init_weights)

    def forward_features(self, x):
        """
        Args:
            x: N, L, C
        """
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    @auto_fp16(apply_to=('x',))
    def forward(self, x, cls_type='REC', rec_memory=None,
                kie_memory=None, num_instance=None):
        """
        Args:
            x: sequence feature, B x N, L, C
            cls_type: 'REC', 'KIE'
            rec_memory: Tensor, optional
            kie_memory: Tensor, optional
            num_instance: N
        Returns:
            x: sequence feature, B, N, L, C
            logits: B x N, L, Cls
        """
        x = self.input_converter(x)
        x = self.forward_features(x)
        x = self.output_converter(x)

        # B, L, C -> B, C, L -> B, C, max_len -> B, max_len, C
        x = self.seq_converter(x.permute(0, 2, 1)).permute(0, 2, 1)
        # # directly slice to save memory
        # x = x[:, :self.max_len]

        cnt = 1
        if not isinstance(rec_memory, type(None)):
            B, N, L, C = rec_memory.shape
            x += rec_memory.reshape(-1, L, C)
            cnt += 1
        if not isinstance(kie_memory, type(None)):
            B, N, L, C = kie_memory.shape
            x += kie_memory.reshape(-1, L, C)
            cnt += 1
        if cnt > 1:
            x /= cnt
        for blk in self.out_block:
            x = blk(x)
        x = self.out_norm(x)

        # batch, seqlen, embsize
        b, s, e = x.size()
        seq_pred = x.reshape(b * s, e)
        if cls_type == 'REC':
            seq_pred = self.rec_head(seq_pred).view(b, s, -1)
        else:
            seq_pred = self.kie_head(seq_pred).view(b, s, -1)
        return x.reshape(-1, num_instance, x.shape[-2], x.shape[-1]), seq_pred

# class ViTSTR(VisionTransformer):
#     '''
#     ViTSTR is basically a ViT that uses DeiT weights.
#     Modified head to support a sequence of characters prediction for STR.
#     '''
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.rec_head = None
#         self.kie_head = None
#         if self.in_c
#
#     def reset_classifier(self, num_classes, cls_type='REC'):
#         assert cls_type in ['REC', 'KIE']
#         self.num_classes = num_classes
#         if cls_type == 'REC':
#             self.rec_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#         else:
#             self.kie_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#
#     def forward_features(self, x):
#         """
#         Args:
#             x: Tensor, (B x N, C, L), in this manner, the patch_embedding is
#             no longer needed
#         Returns:
#
#         """
#         B = x.shape[0]
#         # x = self.patch_embed(x)  # removed, since it is no longer needed
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         return x
#
#     def forward(self, x, seqlen=25, cls_type='REC'):
#         x = self.forward_features(x)
#         x = x[:, :seqlen]  # naive, use FC instead
#
#         # batch, seqlen, embsize
#         b, s, e = x.size()
#         seq_pred = x.reshape(b * s, e)
#         if cls_type == 'REC':
#             seq_pred = self.rec_head(seq_pred).view(b, s, self.num_classes)
#         else:
#             seq_pred = self.kie_head(seq_pred).view(b, s, self.num_classes)
#         return x, seq_pred