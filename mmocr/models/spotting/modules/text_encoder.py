#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/24 15:23
# @Author : WeiHua
import torch

from .master_encoder import MasterEncoder
from .FocalTransformer import FocalTransformerBlock
import torch.nn as nn

# from .transformer import DeformableTransformerEncoder, DeformableTransformerEncoderLayer, PositionEmbedding

class SwinTSEncoder(nn.Module):
    def __init__(self, input_dim, input_resolution, num_heads=8, window_size=7, d_model=-1, block_num=3):
        super(SwinTSEncoder, self).__init__()
        self.equal_dim = input_dim == d_model
        self.layers = nn.ModuleList()
        for _ in range(block_num):
            self.layers.append(
                FocalTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads,
                                      window_size=window_size, expand_size=0, shift_size=0, mlp_ratio=4.,
                                      qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                                      act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc",
                                      focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4)
            )
        # self.layers = nn.Sequential(
        #     FocalTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads,
        #                           window_size=window_size, expand_size=0, shift_size=0, mlp_ratio=4.,
        #                           qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
        #                           act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc",
        #                           focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),
        #     FocalTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads,
        #                           window_size=window_size, expand_size=0, shift_size=0, mlp_ratio=4.,
        #                           qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
        #                           act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc",
        #                           focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),
        #     FocalTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads,
        #                           window_size=window_size, expand_size=0, shift_size=0, mlp_ratio=4.,
        #                           qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
        #                           act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc",
        #                           focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4)
        # )
        if input_dim == d_model:
            self.dim_converter = nn.Identity()
        else:
            self.dim_converter = nn.Linear(input_dim, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, L, C), where L = H * W
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        for layer_ in self.layers:
            x = layer_(x)
        x = self.dim_converter(x)

        return x.permute(0, 2, 1).reshape(B, C, H, W)

class DefaultEncoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super(DefaultEncoder, self).__init__()
        if input_dim == d_model:
            self.equal_dim = True
        else:
            self.equal_dim = False
            self.layer = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # input shape: B*N, C, H, W
        if self.equal_dim:
            return x
        else:
            B, C, H, W = x.shape
            x = self.layer(x.reshape(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
            return x.reshape(B, -1, H, W)


class DeformEncoder(nn.Module):
    def __init__(self, input_dim, d_model, encoder_layer, num_block):
        super(DeformEncoder, self).__init__()
        self.layer = DeformableTransformerEncoder(encoder_layer, n_layer=num_block)
        self.pe = PositionEmbedding(input_dim // 2, normalize=True)
        if input_dim == d_model:
            self.equal_dim = True
        else:
            self.equal_dim = False
            self.dim_converter = nn.Linear(input_dim, d_model)

    def valid_ratio(self, mask):
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / height
        valid_ratio_w = valid_w.float() / width
        valid_ratio = torch.stack((valid_ratio_w, valid_ratio_h), -1)

        return valid_ratio

    def forward(self, x):
        # input shape: (B*N, C, H, W)
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, HW, C)
        x = x.reshape(B, C, -1).permute(0, 2, 1)

        # expand to match later cuda operation
        if x.shape[0] > 64:
            # print(f"pre shape:{x.shape[0]}")
            valid_batch = x.shape[0]
            expand_batch = (valid_batch // 64 + 1) * 64
            B = expand_batch
            expand_batch -= valid_batch
            x = torch.cat([x, torch.zeros((expand_batch, H*W, C), device=x.device)], dim=0)
            # print(f"aft shape:{x.shape[0]}")
        else:
            valid_batch = x.shape[0]

        # 2-d position embedding
        # (B, H, W) -> (B, C, H, W) -> (B, C, L) -> (B, L, C)
        pos = self.pe(torch.zeros((B, H, W), device=x.device).bool()).flatten(2).transpose(1, 2)
        shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=x.device)
        # level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        level_start = torch.as_tensor([0], device=x.device)
        valid_ratio = self.valid_ratio(torch.zeros((B, H, W), device=x.device).bool()).unsqueeze(1)
        # (B, L, C)
        x = self.layer(x, pos, shapes, level_start, None, valid_ratio)
        if self.equal_dim:
            return x.transpose(1, 2).reshape(B, C, H, W)[:valid_batch]
        x = self.dim_converter(x).permute(0, 2, 1)
        return x.reshape(B, -1, H, W)[:valid_batch]


def build_deform_encoder(input_dim, dim_ff=1024, n_level=1, num_heads=8, n_point=4,
                         dropout=0.1, num_block=3, d_model=-1):
    encoder_layer = DeformableTransformerEncoderLayer(
        dim=input_dim, dim_ff=dim_ff, n_level=n_level,
        n_head=num_heads, n_point=n_point, dropout=dropout
    )
    return DeformEncoder(input_dim, d_model, encoder_layer, num_block)


def build_text_encoder(type, **kwargs):
    assert type in ['master', 'swints', 'none', 'deform'], f"Unsupported text encoder: {type}"
    if type == 'master':
        return MasterEncoder(**kwargs)
    elif type == 'swints':
        return SwinTSEncoder(**kwargs)
    elif type == 'none':
        return DefaultEncoder(**kwargs)
    elif type == 'deform':
        return build_deform_encoder(**kwargs)




