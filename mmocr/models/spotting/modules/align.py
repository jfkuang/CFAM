#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/22 16:55
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.cnn import ConvModule



class db_like_fuser(torch.nn.Module):
    def __init__(self,
                 feat_num_in=4,
                 in_channels=256,
                 out_channels=512,
                 bias_on_smooth=False,
                 bn_re_on_smooth=False,
                 conv_after_concat=False
                 ):
        super(db_like_fuser, self).__init__()
        assert feat_num_in == 4, f"feat_num_in: {feat_num_in}"
        self.feat_num_in = feat_num_in
        self.smooth_convs = torch.nn.ModuleList()
        for i in range(feat_num_in):
            norm_cfg = None
            act_cfg = None
            if bn_re_on_smooth:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')
            smooth_conv = ConvModule(
                in_channels,
                out_channels,
                3,
                bias=bias_on_smooth,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.smooth_convs.append(smooth_conv)
        self.conv_after_concat = conv_after_concat
        if self.conv_after_concat:
            norm_cfg = dict(type='BN')
            act_cfg = dict(type='ReLU')
            self.out_conv = ConvModule(
                out_channels * self.num_outs,
                out_channels * self.num_outs,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C, H_i, W_i)`. It usually expects the output
                of former FPN.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        assert len(inputs) == self.feat_num_in
        # build outputs
        # part 1: from original levels
        outs = [
            self.smooth_convs[i](inputs[i])
            for i in range(self.feat_num_in)
        ]

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(
                outs[i], size=outs[0].shape[2:], mode='nearest')
        out = torch.cat(outs, dim=1)

        if self.conv_after_concat:
            out = self.out_conv(out)

        return out


class db_fuser(torch.nn.Module):
    def __init__(self,
                 feat_num_in=4,
                 in_channels=256,
                 out_channels=512,
                 bias_on_smooth=False,
                 bias_on_out=False,
                 smooth_norm=None,
                 smooth_act=None,
                 conv_after_concat=False
                 ):
        super(db_fuser, self).__init__()
        assert feat_num_in == 4, f"feat_num_in: {feat_num_in}"
        self.feat_num_in = feat_num_in
        self.smooth_convs = torch.nn.ModuleList()
        for i in range(feat_num_in):
            norm_cfg = smooth_norm
            act_cfg = smooth_act
            # if bn_re_on_smooth:
            #     norm_cfg = dict(type='BN')
            #     act_cfg = dict(type='ReLU')
            smooth_conv = ConvModule(
                in_channels,
                out_channels,
                3,
                bias=bias_on_smooth,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.smooth_convs.append(smooth_conv)
        self.out5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels //
                      4, 3, padding=1, bias=bias_on_out),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels //
                      4, 3, padding=1, bias=bias_on_out),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels //
                      4, 3, padding=1, bias=bias_on_out),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            out_channels, out_channels // 4, 3, padding=1, bias=bias_on_out)
        self.conv_after_concat = conv_after_concat
        if self.conv_after_concat:
            norm_cfg = dict(type='BN')
            act_cfg = dict(type='ReLU')
            self.out_conv = ConvModule(
                out_channels * self.num_outs,
                out_channels * self.num_outs,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        self.out2.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out5.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C, H_i, W_i)`. It usually expects the output
                of former FPN.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        assert len(inputs) == self.feat_num_in
        # build outputs
        # part 1: from original levels
        c2, c3, c4, c5 = inputs

        # smooth input features through conv
        c2 = self.smooth_convs[0](c2)
        c3 = self.smooth_convs[1](c3)
        c4 = self.smooth_convs[2](c4)
        c5 = self.smooth_convs[3](c5)

        # Up-sample feature while combining with lower feature
        out4 = F.interpolate(c5, size=c4.shape[2:], mode='nearest') + c4
        out3 = F.interpolate(out4, size=c3.shape[2:], mode='nearest') + c3
        out2 = F.interpolate(out3, size=c2.shape[2:], mode='nearest') + c2

        p5 = self.out5(c5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        out = torch.cat((p5, p4, p3, p2), dim=1)
        if self.conv_after_concat:
            out = self.out_conv(out)

        return out


def feature_mask(image_segments, _polygons, _bboxes, batched=False):
    """
    Convert polygons to roi's mask and perform mask align.
    Args:
        image_segments: output of roi_align, (B, N, C, H, W)
        _polygons: polygons coordinates, list of lists, each contains polygons of each sample.
            [[arrays, each represents a polygon] * B]
        _bboxes: horizontal boxes coordinates, list of Tensors, each shape is (N, 4),
            where 4 indicates (tl_x, tl_y, br_x, br_y).
            [Tensor(N, 4) * B]
    Returns:

    """
    if batched:
        return feature_mask_batched(image_segments, _polygons, _bboxes)
    B, N, C, H, W = image_segments.shape
    # (height, width)
    roi_size = image_segments.shape[-2:]
    masks = []
    for i in range(B):
        bboxes = _bboxes[i].cpu()
        boxes_width = (bboxes[:, 2] - bboxes[:, 0])
        boxes_height = (bboxes[:, 3] - bboxes[:, 1])
        for j in range(len(_polygons[i])):
            # crop polygon mask and resize to the output size of roi_align
            if boxes_width[j] < 1 or boxes_height[j] < 1:
                masks.append(torch.ones(roi_size, dtype=torch.uint8))
                continue  # PAD
            cur_polygon = _polygons[i][j]
            cur_polygon[0::2] -= bboxes[j, 0].item()
            cur_polygon[1::2] -= bboxes[j, 1].item()
            cur_polygon[0::2] /= boxes_width[j]
            cur_polygon[1::2] /= boxes_height[j]
            cur_polygon[0::2] *= roi_size[1]
            cur_polygon[1::2] *= roi_size[0]

            cur_mask = maskUtils.frPyObjects(
                [cur_polygon], roi_size[0], roi_size[1]
            )
            rle = maskUtils.merge(cur_mask)
            mask = torch.from_numpy(maskUtils.decode(rle))
            masks.append(mask)
        for _ in range(N-len(_polygons[i])):
            masks.append(torch.ones(roi_size, dtype=torch.uint8))
    # B, N, H, W
    masks = torch.stack(masks, dim=0).to(image_segments.device, dtype=torch.float32).reshape(B, N, H, W)
    return image_segments * masks.unsqueeze(2)


def feature_mask_batched(image_segments, _polygons, _bboxes):
    """
    Convert polygons to roi's mask and perform mask align.
    Args:
        image_segments: output of roi_align, (B, N, C, H, W)
        _polygons: polygons coordinates, a Tensor that contains polygons of each sample.
            (B, N, L)
        _bboxes: horizontal boxes coordinates, shape is (B, N, 4),
            where 4 indicates (tl_x, tl_y, br_x, br_y)
    Returns:

    """
    B, N, C, H, W = image_segments.shape
    # (height, width)
    roi_size = image_segments.shape[-2:]
    masks = []
    bboxes = _bboxes.cpu()
    boxes_width = (bboxes[:, :, 2] - bboxes[:, :, 0]).unsqueeze(-1)
    boxes_height = (bboxes[:, :, 3] - bboxes[:, :, 1]).unsqueeze(-1)
    polygons = _polygons.cpu()
    # polygons' relative coordinates
    polygons[:, :, 0::2] -= bboxes[:, :, 0].unsqueeze(-1)
    polygons[:, :, 1::2] -= bboxes[:, :, 1].unsqueeze(-1)
    polygons[:, :, 0::2] /= boxes_width
    polygons[:, :, 1::2] /= boxes_height
    polygons[:, :, 0::2] *= roi_size[1]
    polygons[:, :, 1::2] *= roi_size[0]
    for i in range(bboxes.shape[0]):
        for j in range(bboxes.shape[1]):
            # crop polygon mask and resize to the output size of roi_align
            if boxes_width[i, j] < 1 or boxes_height[i, j] < 1:
                masks.append(torch.ones(roi_size, dtype=torch.uint8))
                continue  # PAD
            else:
                cur_mask = maskUtils.frPyObjects(
                    [polygons[i, j]], roi_size[0], roi_size[1]
                )
                rle = maskUtils.merge(cur_mask)
                mask = torch.from_numpy(maskUtils.decode(rle))
                masks.append(mask)
    # B, N, H, W
    masks = torch.stack(masks, dim=0).to(image_segments.device, dtype=torch.float32).reshape(B, N, H, W)
    return image_segments * masks.unsqueeze(2)


# def feature_mask(image_segments, _polygons, _bboxes):
#     """
#     Convert polygons to roi's mask and perform mask align.
#     Args:
#         image_segments: output of roi_align, (B, N, C, H, W)
#         _polygons: polygons coordinates, a Tensor that contains polygons of each sample.
#             (B, N, L)
#         _bboxes: horizontal boxes coordinates, shape is (B, N, 4),
#             where 4 indicates (tl_x, tl_y, br_x, br_y)
#     Returns:
#
#     """
#     B, N, C, H, W = image_segments.shape
#     # (height, width)
#     roi_size = image_segments.shape[-2:]
#     masks = []
#     bboxes = _bboxes.cpu()
#     boxes_width = (bboxes[:, :, 2] - bboxes[:, :, 0]).unsqueeze(-1)
#     boxes_height = (bboxes[:, :, 3] - bboxes[:, :, 1]).unsqueeze(-1)
#     polygons = _polygons.cpu()
#     # polygons' relative coordinates
#     polygons[:, :, 0::2] -= bboxes[:, :, 0].unsqueeze(-1)
#     polygons[:, :, 1::2] -= bboxes[:, :, 1].unsqueeze(-1)
#     polygons[:, :, 0::2] /= boxes_width
#     polygons[:, :, 1::2] /= boxes_height
#     polygons[:, :, 0::2] *= roi_size[1]
#     polygons[:, :, 1::2] *= roi_size[0]
#     for i in range(bboxes.shape[0]):
#         for j in range(bboxes.shape[1]):
#             # crop polygon mask and resize to the output size of roi_align
#             if boxes_width[i, j] < 1 or boxes_height[i, j] < 1:
#                 masks.append(torch.ones(roi_size, dtype=torch.uint8))
#                 continue  # PAD
#             else:
#                 cur_mask = maskUtils.frPyObjects(
#                     [polygons[i, j]], roi_size[0], roi_size[1]
#                 )
#                 rle = maskUtils.merge(cur_mask)
#                 mask = torch.from_numpy(maskUtils.decode(rle))
#                 masks.append(mask)
#     # B, N, H, W
#     masks = torch.stack(masks, dim=0).to(image_segments.device, dtype=torch.float32).reshape(B, N, H, W)
#     return image_segments * masks.unsqueeze(2)

