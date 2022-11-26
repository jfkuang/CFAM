#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/28 15:29
# @Author : WeiHua
import math
from mmdet.models import ResNet
from mmdet.models import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import Sequential
from torch import nn as nn


@BACKBONES.register_module()
class CustomResNet(ResNet):
    """
    Custom ResNet backbone. Compare with the version implemented by mmdet,
    it enables to choose plug plugin to which layer/block of a stage.
    :params
    :block_to_insert_plugin: list of lists or None, the first level indicates which stage,
        while the second indicates which block of corresponding stage.
        Examples: the ResNet-50's structure is (3, 4, 6, 3), so it should be like:
        [
            [True, False, False],
            [False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False]
        ]
    """
    def __init__(self, block_to_insert_plugin=None, **kwargs):
        self.block_to_insert_plugin = block_to_insert_plugin
        super(CustomResNet, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        # specify idx of stage by channels
        if kwargs['plugins']:
            if self.block_to_insert_plugin:
                stage_idx = int(math.log(kwargs['planes']/self.base_channels, 2))
                block_wise_plugins = []
                for add_plugin in self.block_to_insert_plugin[stage_idx]:
                    if add_plugin:
                        block_wise_plugins.append(kwargs['plugins'])
                    else:
                        block_wise_plugins.append([])
                kwargs['block_wise_plugins'] = block_wise_plugins

        """Pack all blocks in a stage into a ``CustomResLayer``."""
        return CustomResLayer(**kwargs)


class CustomResLayer(Sequential):
    """Custom ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 block_wise_plugins=None,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            if block_wise_plugins:
                kwargs['plugins'] = block_wise_plugins[0]
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for block_idx in range(1, num_blocks):
                if block_wise_plugins:
                    kwargs['plugins'] = block_wise_plugins[block_idx]
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for block_idx in range(num_blocks - 1):
                if block_wise_plugins:
                    kwargs['plugins'] = block_wise_plugins[block_idx]
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            if block_wise_plugins:
                kwargs['plugins'] = block_wise_plugins[-1]
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(CustomResLayer, self).__init__(*layers)
