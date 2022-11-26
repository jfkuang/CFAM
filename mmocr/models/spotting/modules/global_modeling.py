#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/19 15:03
# @Author : WeiHua
import torch
import torch.nn as nn

# class LayoutModeling(nn.Module):
#     pass
#
#
# class ContextModeling(nn.Module):
#     pass


class GlobalModeling(nn.Module):
    """
    The Global Modeling module is able to dynamically switch parts of layout-modeling,
    context-modeling, layout & context fusion.
    """
    def __init__(self, d_model, use_layout, use_context, attn_args, concat_src=False):
        super(GlobalModeling, self).__init__()
        self.use_layout = use_layout
        self.use_context = use_context
        self.concat_src = concat_src
        if use_layout:
            layout_attn_args = attn_args.get('layout', {})
            layout_attn_args.update(d_model=d_model)
            tf_encoder_layer = nn.TransformerEncoderLayer(**layout_attn_args)
            self.layout_attn = nn.TransformerEncoder(tf_encoder_layer,
                                                     num_layers=1)
        if use_context:
            context_attn_args = attn_args.get('context', {})
            context_attn_args.update(d_model=d_model)
            tf_encoder_layer = nn.TransformerEncoderLayer(**context_attn_args)
            self.context_attn = nn.TransformerEncoder(tf_encoder_layer,
                                                      num_layers=1)
    # TODO: CHECKPOINT
    def layout_modeling(self):
        pass


