#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/10 13:11
# @Author : WeiHua
import cv2
import numpy as np
import torch
from mmdet.models import StandardRoIHead
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from mmdet.core import bbox2roi

import os


@HEADS.register_module()
class ReI_StandardRoIHead(StandardRoIHead):
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ret_det=False,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            ret_det (Bool): if returning detection results

        Returns:
            dict[str, Tensor]: a dictionary of loss components
            list[Tensor]: a list of Tensors with shape (N, 4), N means
                num of boxes, 4 means (tl_x, tl_y, br_x, br_y).
            list[Tensor]: a list of Tensors with shape (N, img_h, img_w),
                N means num of boxes.
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # type : class SamplingResult
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            if ret_det:  # to be checked
                assert len(img_metas) == len(sampling_results)
                det_boxes = []
                with torch.no_grad():
                    # bbox_results['bbox_pred']: (samples of all imgs, 4) -> delta (X, Y, W, H)
                    bbox_pred_delta = bbox_results['bbox_pred']
                    st_cnt = 0
                    for i, img_meta in enumerate(img_metas):
                        num_samples = len(sampling_results[i].bboxes)
                        # (N_sample, 4), 4 represents (tl_x, tl_y, br_x, br_y)
                        cur_bboxes = self.bbox_head.bbox_coder.decode(sampling_results[i].bboxes,
                                                                      bbox_pred_delta[st_cnt:st_cnt+num_samples],
                                                                      max_shape=img_meta['img_shape'])
                        # only keep positive boxes
                        det_boxes.append(cur_bboxes[:len(sampling_results[i].pos_inds)])
                        st_cnt += num_samples

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            if ret_det:
                total_pos = sum([len(x) for x in det_boxes])
                assert mask_results['mask_pred'].shape[0] == total_pos, "Positive samples doesn't match"
                det_masks = []
                with torch.no_grad():
                    st_cnt = 0
                    for i, img_meta in enumerate(img_metas):
                        img_shape = img_meta['img_shape']
                        # N_pos_samples, h, w
                        cur_masks, _ = _do_paste_mask(mask_results['mask_pred'][st_cnt:st_cnt+len(det_boxes[i]), ...],
                                                      det_boxes[i], img_shape[0], img_shape[1],
                                                      skip_empty=False)
                        st_cnt += len(det_boxes[i])
                        det_masks.append((cur_masks >= self.test_cfg.mask_thr_binary).to(dtype=torch.bool))
        if ret_det:
            # with torch.no_grad():
            #     self.vis_det_train(det_boxes, det_masks, img_metas)
            return losses, det_boxes, det_masks
        return losses

    def vis_det_train(self, det_boxes, det_masks, img_metas,
                      save_dir='/usr/local/dataset/whua/ie_e2e_log/det_train_vis'):
        """
        Visualize detection results during train
        Args:
            det_boxes (list[Tensor]): Tensor with shape N, 4 -> 4 indicates
                (tl_x, tl_y, br_x, br_y)
            det_masks (list[Tensor]): Tensor with shape N, h, w -> h, w indicates
                the height and width of image
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

        Returns:
        """
        assert len(det_boxes) == len(det_masks) == len(img_metas)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # 1. mask src with mask
        # 2. draw box on mask
        for det_boxes_, det_masks_, img_metas_ in zip(det_boxes, det_masks, img_metas):
            boxes = det_boxes_.detach().cpu().detach().numpy()
            masks = det_masks_.detach().cpu().detach().numpy()
            img_name = img_metas_['filename'].split('/')[-1]
            img_name_meta = img_name.split('.')[0]
            out_dir = os.path.join(save_dir, img_name_meta)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for idx, (box, mask_) in enumerate(zip(boxes, masks)):
                mask = (255*mask_).astype(np.uint8)
                full_box = np.array([
                    [box[0], box[1]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                    [box[0], box[3]]
                ])
                cv2.polylines(mask, [full_box.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
                cv2.imwrite(os.path.join(out_dir, f"{idx}.jpg"), mask)

    # def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
    #                         img_metas):
    #     """Run forward function and calculate loss for box head in training."""
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     bbox_results = self._bbox_forward(x, rois)
    #
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'], rois,
    #                                     *bbox_targets)
    #     if 'nan' in str(loss_bbox['loss_bbox'].item()):
    #         import ipdb
    #         ipdb.set_trace()
    #
    #     bbox_results.update(loss_bbox=loss_bbox)
    #     return bbox_results