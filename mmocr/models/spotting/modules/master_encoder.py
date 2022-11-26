#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/9 16:41
# @Author : WeiHua
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class GCAContextBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 ratio,
                 headers,
                 pooling_type='att',
                 att_scale=False,
                 fusion_type='channel_add'):
        super(GCAContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']

        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert in_channels % headers == 0 and in_channels >= 8  # in_channels must be divided by headers evenly

        self.headers = headers
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False

        self.single_header_inplanes = int(in_channels / headers)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            # for concat
            self.cat_conv = nn.Conv2d(2 * self.in_channels, self.in_channels, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.view(batch * self.headers, self.single_header_inplanes, height, width)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.view(batch * self.headers, self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.view(batch * self.headers, 1, height * width)

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / torch.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = torch.cat([out, channel_concat_term.expand(-1, -1, H, W)], dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.in_channels, H, W])
            out = nn.functional.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, gcb_config=None):
        super(MEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.gcb_config = gcb_config

        if self.gcb_config is not None:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = GCAContextBlock(in_channels=planes,
                                                 ratio=gcb_ratio,
                                                 headers=gcb_headers,
                                                 att_scale=att_scale,
                                                 fusion_type=fusion_type)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.gcb_config is not None:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_gcb_config(gcb_config, layer):
    if gcb_config is None or not gcb_config['layers'][layer]:
        return None
    else:
        return gcb_config


class MasterEncoder(nn.Module):

    def __init__(self, layers, input_dim=3, gcb_config=None, encoder_type='gca', with_mlp=False, d_model=-1):
        super(MasterEncoder, self).__init__()
        self.encoder_type = encoder_type
        assert encoder_type in ['gca', 'conv', 'mlp']
        self.with_mlp = with_mlp
        if encoder_type == 'conv':
            module_list = []
            for idx, layer in enumerate(layers):
                if idx == 0:
                    module_list.append(nn.Conv2d(input_dim, layer, kernel_size=3, padding=1))
                else:
                    module_list.append(nn.Conv2d(layers[idx-1], layer, kernel_size=3, padding=1))
                module_list.append(nn.BatchNorm2d(layer))
                module_list.append(nn.ReLU(inplace=True))
            self.layers = nn.ModuleList(module_list)
            self.conv_mlp = nn.Sequential(
                nn.Linear(layers[-1], 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, layers[-1])
            )
            if with_mlp:
                self.mlp = nn.Sequential(
                    nn.Linear(layers[-1], 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, layers[-1])
                )
        elif encoder_type == 'mlp':
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, input_dim)
            )
        else:
            self.inplanes = 256
            if input_dim != self.inplanes:
                self.dim_converter = nn.Conv2d(input_dim, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.dim_converter = nn.Identity()
            self.layer1 = self._make_layer(MEBasicBlock, 256, layers[0], stride=1, gcb_config=get_gcb_config(gcb_config, 0))
            self.conv1 = nn.Conv2d(self.inplanes, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(256)
            self.relu1 = nn.ReLU(inplace=True)

            self.layer2 = self._make_layer(MEBasicBlock, 256, layers[1], stride=1, gcb_config=get_gcb_config(gcb_config, 1))
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(256)
            self.relu2 = nn.ReLU(inplace=True)

            self.layer3 = self._make_layer(MEBasicBlock, 256, layers[2], stride=1, gcb_config=get_gcb_config(gcb_config, 2))
            self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.relu3 = nn.ReLU(inplace=True)

            self.layer4 = self._make_layer(MEBasicBlock, 256, layers[3], stride=1, gcb_config=get_gcb_config(gcb_config, 3))
            self.conv4 = nn.Conv2d(256, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(input_dim)
            self.relu4 = nn.ReLU(inplace=True)
            if with_mlp:
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, input_dim)
                )
        if input_dim == d_model:
            self.equal_dim = True
            self.out_dim_converter = nn.Identity()
        else:
            self.equal_dim = False
            self.out_dim_converter = nn.Linear(input_dim, d_model)
        self.init_weights()

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, gcb_config=gcb_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # keep the size of x unchanged
        if self.encoder_type == 'conv':
            for layer_ in self.layers:
                x = layer_(x)
            x = x.permute(0, 2, 3, 1)
            ori_shape = x.shape
            x = self.conv_mlp(x.reshape(-1, ori_shape[-1])).reshape(ori_shape).permute(0, 3, 1, 2)
        elif self.encoder_type == 'mlp':
            x = x.permute(0, 2, 3, 1)
            ori_shape = x.shape
            x = self.layers(x.reshape(-1, ori_shape[-1])).reshape(ori_shape).permute(0, 3, 1, 2)
        else:
            x = self.dim_converter(x)
            x = self.layer1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            x = self.layer2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)

            x = self.layer3(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)

            x = self.layer4(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
        if self.with_mlp:
            x = x.permute(0, 2, 3, 1)
            ori_shape = x.shape
            x = self.mlp(x.reshape(-1, ori_shape[-1])).reshape(ori_shape).permute(0, 3, 1, 2)

        if self. equal_dim:
            return x
        else:
            B, C, H, W = x.shape
            x = self.out_dim_converter(x.reshape(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
            return x.reshape(B, -1, H, W)
