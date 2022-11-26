#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/12 19:40
# @Author : WeiHua

from mmocr.models.builder import DETECTORS

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

@DETECTORS.register_module()
class CSRNetDecoder(nn.Module):
    def __init__(self, load_weights=False, ratio=4, in_channels=256, num_cls=4, using_bn=False,
                 loss_weight=1.0, size_average=False):
        super(CSRNetDecoder, self).__init__()
        self.seen = 0
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend_feat = [512, 256, 128, 64]
        # self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=in_channels, batch_norm=using_bn, dilation=True)
        self.output_layer = nn.Conv2d(64, num_cls, kernel_size=1)
        self.ratio = ratio
        self.interval_feats_dim = 64
        if not load_weights:
            # mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            # self.frontend.load_state_dict(mod.features[0:23].state_dict())
        self.loss_func = nn.MSELoss(size_average=size_average)
        self.loss_weight = loss_weight

    def forward(self, x, density_maps):
        """
        CSRNet Forward
        Args:
            x:
            density_maps: Tensor with shape (B, K, H, W), where K is the number of entity classes

        Returns:

        """
        H, W = density_maps.shape[2:]
        # x = self.frontend(x)
        x = self.backend(x)
        # B, K, H, W
        x = self.output_layer(x)
        # x = F.upsample(x, scale_factor=self.ratio)
        loss_dict = dict(loss_cnt=self.loss_weight * self.loss_func(
            x.reshape(-1, H, W), density_maps.reshape(-1, H, W)))
        return loss_dict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
