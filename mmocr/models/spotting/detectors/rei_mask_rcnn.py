#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/8 17:16
# @Author : WeiHua
# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import ipdb
import numpy as np
from functools import cmp_to_key

import torch

from mmocr.core import seg2boundary
from mmocr.models.builder import DETECTORS
from mmocr.models.textdet.detectors import TextDetectorMixin
from mmdet.models.detectors import MaskRCNN



@DETECTORS.register_module()
class ReI_OCRMaskRCNN(TextDetectorMixin, MaskRCNN):
    """Mask RCNN tailored for OCR."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 text_repr_type='quad',
                 show_score=False,
                 init_cfg=None):
        TextDetectorMixin.__init__(self, show_score)
        MaskRCNN.__init__(
            self,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        assert text_repr_type in ['quad', 'poly']
        self.text_repr_type = text_repr_type
        self.sort_box_for_test = test_cfg.get('sort_test', False)
        # check point: https://blog.csdn.net/JNingWei/article/details/120300014

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ret_det=True,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

            ret_det (bool) : if return detection results during training

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
            x (list[Tensor]): a list of Tensors of extracted feature
            det_boxes (list[Tensor]): a list of Tensors with shape (N, 4), N means
                num of boxes, 4 means (tl_x, tl_y, br_x, br_y).
            det_masks (list[Tensor]): a list of Tensors with shape (N, img_h, img_w),
                N means num of boxes.

        """
        x = self.extract_feat(img)

        losses = dict()
        if not self.with_rpn and not self.with_roi_head:
            return losses, x, None

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        if ret_det:
            roi_losses, det_boxes, det_masks = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                                           gt_bboxes, gt_labels,
                                                                           gt_bboxes_ignore, gt_masks,
                                                                           ret_det=ret_det, **kwargs)
            losses.update(roi_losses)
            det_boxes_list = []
            det_polys_list = []
            with torch.no_grad():
                for det_box_batch, det_mask_batch in zip(det_boxes, det_masks):
                    boxes_batch = []
                    polys_batch = []
                    for det_box, det_mask in zip(det_box_batch, det_mask_batch):
                        poly = seg2boundary(det_mask.cpu().numpy(), self.text_repr_type)
                        if not poly:
                            continue
                        polys_batch.append(poly)
                        tl_x, tl_y, w, h = cv2.boundingRect(np.array(polys_batch[-1], dtype=np.float32).reshape(-1, 2))
                        boxes_batch.append([tl_x, tl_y, tl_x + w, tl_y + h])
                    det_boxes_list.append(boxes_batch)
                    det_polys_list.append(polys_batch)
            return losses, x, (det_boxes_list, det_polys_list)
        else:
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     ret_det=ret_det, **kwargs)
            losses.update(roi_losses)
            return losses, x, None

    def get_boundary(self, results, return_bbox=False, sort_det=False,
                     max_h=-1, max_w=-1):
        """Convert segmentation into text boundaries.

        Args:
           results (tuple): The result tuple. The first element is
               segmentation while the second is its scores.
           return_bbox (bool): Whether to return bounding boxes
           sort_det (bool): Whether to sort detection result with respect
                to reading order.
        Returns:
           dict: A result dict containing 'boundary_result'.
        """

        assert isinstance(results, tuple)

        instance_num = len(results[1][0])
        boundaries = []
        bboxes = []
        for i in range(instance_num):
            seg = results[1][0][i]
            score = results[0][0][i][-1]
            boundary = seg2boundary(seg, self.text_repr_type, score)


            # from mmocr.models.textdet.postprocess.utils import unclip
            # expand_boundary = unclip(np.array(boundary[:-1]).reshape(-1, 2),
            #                          unclip_ratio=1.2)
            # if len(expand_boundary) == 0 or isinstance(expand_boundary[0], list):
            #     continue
            # expand_boundary = expand_boundary.reshape(-1).tolist()
            # expand_boundary.append(boundary[-1])
            # boundary = expand_boundary


            if boundary is not None:
                boundaries.append(boundary)
                if return_bbox:
                    tl_x, tl_y, w, h = cv2.boundingRect(np.array(boundary[:-1], dtype=np.float32).reshape(-1, 2))
                    bboxes.append([tl_x, tl_y, tl_x+w, tl_y+h])

        if sort_det:
            assert len(boundaries) == len(bboxes)
            to_sort_polys = [(idx, poly[:-1]) for idx, poly in enumerate(boundaries)]
            # sorted_data = sorted(to_sort_polys, key=compare_key)
            sorted_data = sorted(to_sort_polys, key=cmp_to_key(compare_key))
            sorted_index = [x[0] for x in sorted_data]
            boundaries = [boundaries[idx] for idx in sorted_index]
            if return_bbox:
                bboxes = [bboxes[idx] for idx in sorted_index]

        results = dict(boundary_result=boundaries)
        if return_bbox:
            results['box_result'] = bboxes
        return results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        # if trying to support test with batch_size > 1, may refer to OCRMaskRCNN.
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        # results = super().simple_test(img, img_metas, proposals, rescale)

        boundaries = self.get_boundary(results[0], return_bbox=True, sort_det=self.sort_box_for_test,
                                       max_h=img.shape[2]-1, max_w=img.shape[3]-1)

        return boundaries, x


# def compare_key(x):
#     #  x is (index, box), where box is list[x, y, x, y...]
#     points = x[1]
#     box = np.array(x[1], dtype=np.float32).reshape(-1, 2)
#     rect = cv2.minAreaRect(box)
#     center = rect[0]
#     return center[1], center[0]
def compare_key(a, b):
    box = np.array(a[1], dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(box)
    a_center_x = rect[0][0]
    a_center_y = rect[0][1]
    a_box_h = min(rect[1][0], rect[1][1])

    box = np.array(b[1], dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(box)
    b_center_x = rect[0][0]
    b_center_y = rect[0][1]
    b_box_h = min(rect[1][0], rect[1][1])
    if a_center_y > b_center_y:
        if (a_center_y - b_center_y) > 0.5 * min(a_box_h, b_box_h):
            return 1
        elif a_center_x > b_center_x:
            return 1
        else:
            return -1
    else:
        if (b_center_y - a_center_y) > 0.5 * min(a_box_h, b_box_h):
            return -1
        elif a_center_x > b_center_x:
            return 1
        else:
            return -1

# # Re-implement this
# @DETECTORS.register_module()
# class ReI_OCRMaskRCNN(OCRMaskRCNN):
#     """Override forward of Mask RCNN tailored for OCR, i.e., forward_train & forward_test are modified."""
#
#     def forward_train(self,
#                       img,
#                       img_metas,
#                       gt_bboxes,
#                       gt_labels,
#                       gt_bboxes_ignore=None,
#                       gt_masks=None,
#                       proposals=None,
#                       ret_det=True,
#                       **kwargs):
#         """
#         Args:
#             img (Tensor): of shape (N, C, H, W) encoding input images.
#                 Typically these should be mean centered and std scaled.
#
#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmdet/datasets/pipelines/formatting.py:Collect`.
#
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#
#             gt_labels (list[Tensor]): class indices corresponding to each box
#
#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss.
#
#             gt_masks (None | Tensor) : true segmentation masks for each box
#                 used if the architecture supports a segmentation task.
#
#             proposals : override rpn proposals with custom proposals. Use when
#                 `with_rpn` is False.
#
#             ret_det (bool) : if return detection results during training
#
#         Returns:
#             losses (dict[str, Tensor]): a dictionary of loss components
#             x (list[Tensor]): a list of Tensors of extracted feature
#             det_boxes (list[Tensor]): a list of Tensors with shape (N, 4), N means
#                 num of boxes, 4 means (tl_x, tl_y, br_x, br_y).
#             det_masks (list[Tensor]): a list of Tensors with shape (N, img_h, img_w),
#                 N means num of boxes.
#
#         """
#         x = self.extract_feat(img)
#
#         losses = dict()
#
#         # RPN forward and loss
#         if self.with_rpn:
#             proposal_cfg = self.train_cfg.get('rpn_proposal',
#                                               self.test_cfg.rpn)
#             rpn_losses, proposal_list = self.rpn_head.forward_train(
#                 x,
#                 img_metas,
#                 gt_bboxes,
#                 gt_labels=None,
#                 gt_bboxes_ignore=gt_bboxes_ignore,
#                 proposal_cfg=proposal_cfg,
#                 **kwargs)
#             losses.update(rpn_losses)
#         else:
#             proposal_list = proposals
#         if ret_det:
#             roi_losses, det_boxes, det_masks = self.roi_head.forward_train(x, img_metas, proposal_list,
#                                                                            gt_bboxes, gt_labels,
#                                                                            gt_bboxes_ignore, gt_masks,
#                                                                            ret_det=ret_det, **kwargs)
#             losses.update(roi_losses)
#             return losses, x, (det_boxes, det_masks)
#         else:
#             roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
#                                                      gt_bboxes, gt_labels,
#                                                      gt_bboxes_ignore, gt_masks,
#                                                      ret_det=ret_det, **kwargs)
#             losses.update(roi_losses)
#             return losses, x, None
