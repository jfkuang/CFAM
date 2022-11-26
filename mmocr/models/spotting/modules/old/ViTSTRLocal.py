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


class CustomViTSTRLocal(nn.Module):
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

        self.out_block_rec = nn.ModuleList([
            Block(
                dim=in_chans, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(out_depth)
        ])
        self.out_block_kie = nn.ModuleList([
            Block(
                dim=in_chans, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(out_depth)
        ])
        self.out_norm = norm_layer(in_chans)

        # Classifier head
        self.rec_head = None
        self.kie_head = None
        self.crf_head = nn.Identity()

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

    def reset_classifier(self, num_classes, cls_type='REC', use_crf=False):
        assert cls_type in ['REC', 'KIE']
        if cls_type == 'REC':
            self.rec_head = nn.Linear(self.in_chans, num_classes) if num_classes > 0 else nn.Identity()
            self.rec_head.apply(self._init_weights)
        else:
            self.kie_head = nn.Linear(self.in_chans, num_classes) if num_classes > 0 else nn.Identity()
            self.kie_head.apply(self._init_weights)
            if use_crf:
                raise NotImplementedError()

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
    def forward(self, x, rec_only=True, rec_memory=None,
                kie_memory=None, num_instance=None,
                iter_num=2):
        """
        Args:
            x: sequence feature, B x N, L, C
            cls_type: 'REC_ONLY', 'REC_KIE'
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

        logits = dict(REC=[], KIE=[])
        for i in range(iter_num):
            # forward once for recognition
            cnt = 1
            memory = None
            if not isinstance(rec_memory, type(None)):
                if len(rec_memory.shape) == 4:
                    B, N, L, C = rec_memory.shape
                    rec_memory = rec_memory.view(-1, L, C)
                memory = x + rec_memory
                cnt += 1
            if not isinstance(kie_memory, type(None)):
                if len(kie_memory.shape) == 4:
                    B, N, L, C = kie_memory.shape
                    kie_memory = kie_memory.view(-1, L, C)
                if cnt > 1:
                    memory += kie_memory
                else:
                    memory = x + kie_memory
                cnt += 1
            if cnt > 1:
                memory /= cnt
                for blk in self.out_block_rec:
                    memory = blk(memory)
            else:
                memory = self.out_block_rec[0](x)
                for blk in self.out_block_rec[1:]:
                    memory = blk(memory)
            rec_memory = self.out_norm(memory)
            del memory
            # (B x N, L, C)
            b, s, e = rec_memory.size()
            logits['REC'].append(self.rec_head(rec_memory.reshape(b * s, e)).view(b, s, -1))
            if rec_only:
                continue

            # forward once for key information extraction
            cnt = 1
            memory = None
            if not isinstance(rec_memory, type(None)):
                if len(rec_memory.shape) == 4:
                    B, N, L, C = rec_memory.shape
                    rec_memory = rec_memory.view(-1, L, C)
                memory = x + rec_memory
                cnt += 1
            if not isinstance(kie_memory, type(None)):
                if len(kie_memory.shape) == 4:
                    B, N, L, C = kie_memory.shape
                    kie_memory = kie_memory.view(-1, L, C)
                if cnt > 1:
                    memory += kie_memory
                else:
                    memory = x + kie_memory
                cnt += 1
            if cnt > 1:
                memory /= cnt
                for blk in self.out_block_kie:
                    memory = blk(memory)
            else:
                memory = self.out_block_kie[0](x)
                for blk in self.out_block_kie[1:]:
                    memory = blk(memory)
            kie_memory = self.out_norm(memory)
            del memory
            # (B x N, L, C)
            b, s, e = rec_memory.size()
            logits['KIE'].append(self.kie_head(kie_memory.reshape(b * s, e)).view(b, s, -1))
        if rec_only:
            rec_memory = rec_memory.view(-1, num_instance, x.shape[-2], x.shape[-1])
            return rec_memory, None, logits
        else:
            rec_memory = rec_memory.view(-1, num_instance, x.shape[-2], x.shape[-1])
            kie_memory = kie_memory.view(-1, num_instance, x.shape[-2], x.shape[-1])
            return rec_memory, kie_memory, logits
