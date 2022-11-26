#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/6 15:29
# @Author : WeiHua

import os
import shutil
import numpy as np
import glob
import cv2

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CustomVisualizer:
    def __init__(self,
                 vis_box=True,
                 vis_mask=False,
                 vis_text=False,
                 vis_entity=False,
                 vis_dir=None,
                 vis_num_limit=None,
                 rm_exist_dir=True):
        """
        Visualize loaded and augmented annotations, and save them to specific directory.
        Args:
            vis_box: if visualize box and poly (if exists in 'bbox_fields')
            vis_mask: if visualize binary mask
            vis_text: if visualize text of each instance
            vis_entity: if visualize entity of each instance
            vis_dir: visualization directory
            vis_num_limit: number of samples limitation for visualization
            rm_exist_dir: if removing existed vis_dir
        """
        assert vis_dir, "You have to specify a directory to save visualization"
        self.vis_box = vis_box
        self.vis_mask = vis_mask
        self.vis_text = vis_text
        self.vis_entity = vis_entity
        self.vis_dir = vis_dir
        self.vis_num_limit = vis_num_limit

        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        elif rm_exist_dir:
            shutil.rmtree(vis_dir)
            os.mkdir(vis_dir)

        if self.vis_box:
            box_vis_path = os.path.join(self.vis_dir, 'box_vis')
            if not os.path.exists(box_vis_path):
                os.mkdir(box_vis_path)
            for key in ['gt_polys', 'gt_bboxes', 'rboxes']:
                cur_vis_path = os.path.join(box_vis_path, key)
                if not os.path.exists(cur_vis_path):
                    os.mkdir(cur_vis_path)

        if self.vis_mask:
            mask_vis_path = os.path.join(self.vis_dir, 'mask_vis')
            if not os.path.exists(mask_vis_path):
                os.mkdir(mask_vis_path)
            for key in ['gt_masks']:
                cur_vis_path = os.path.join(mask_vis_path, key)
                if not os.path.exists(cur_vis_path):
                    os.mkdir(cur_vis_path)

    def visualize_box(self, results: dict):
        assert len(results['img_fields']) == 1
        box_vis_path = os.path.join(self.vis_dir, 'box_vis')
        # if not os.path.exists(box_vis_path):
        #     os.mkdir(box_vis_path)
        img_name = results['ori_filename'].replace('\\', '/').split('/')[-1]
        img_name_meta = img_name.split('.')[0]
        for key in results.get('bbox_fields', []):
            cur_vis_path = os.path.join(box_vis_path, key)
            # if not os.path.exists(cur_vis_path):
            #     os.mkdir(cur_vis_path)
            # Fixme: Not sure if this manner will cause trouble when there are
            #   multiple workers.
            if self.vis_num_limit:
                if len(glob.glob(os.path.join(cur_vis_path, '*.jpg'))) > self.vis_num_limit:
                    continue
            src_img = results[results['img_fields'][0]].copy()
            # cv2.imwrite(os.path.join(cur_vis_path, 'src_'+img_name), src_img)
            if self.vis_text:
                text_saver = open(os.path.join(cur_vis_path, img_name_meta+'_text.txt'), 'w')
            if self.vis_entity:
                entity_saver = open(os.path.join(cur_vis_path, img_name_meta+'_entity.txt'), 'w')
            for idx, box in enumerate(results[key]):
                # [x1, y1, x2, y2, ...], ndarray
                cv2.polylines(src_img, [box.reshape((-1, 2)).astype(np.int32)],
                              isClosed=True, color=(0, 0, 255), thickness=1)
                if self.vis_text or self.vis_entity:
                    cv2.putText(src_img, str(idx), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    if self.vis_text:
                        text_saver.write(
                            f"idx:{idx}, text:{results['ori_texts'][idx]}, index:{results['gt_labels']['texts'][idx]}\n"
                        )
                    if self.vis_entity:
                        entity_saver.write(
                            f"idx:{idx}, entity:{results['ori_entities'][idx]}, index:{results['gt_labels']['entities'][idx]}\n"
                        )
            if self.vis_text:
                text_saver.close()
            if self.vis_entity:
                entity_saver.close()
            cv2.imwrite(os.path.join(cur_vis_path, img_name), src_img)

    def visualize_mask(self, results):
        mask_vis_path = os.path.join(self.vis_dir, 'mask_vis')
        # if not os.path.exists(mask_vis_path):
        #     os.mkdir(mask_vis_path)
        for key in results['mask_fields']:
            cur_vis_path = os.path.join(mask_vis_path, key)
            # if not os.path.exists(cur_vis_path):
            #     os.mkdir(cur_vis_path)
            # Fixme: Not sure if this manner will cause trouble when there are
            #   multiple workers.
            if self.vis_num_limit:
                if len(glob.glob(os.path.join(cur_vis_path, '*'))) > self.vis_num_limit:
                    continue
            img_name = results['ori_filename'].replace('\\', '/').split('/')[-1]
            img_name_meta = img_name.split('.')[0]
            mask = results[key]
            masks_dir = os.path.join(cur_vis_path, img_name_meta)
            if not os.path.exists(masks_dir):
                os.mkdir(masks_dir)
            for i in range(mask.masks.shape[0]):
                cv2.imwrite(os.path.join(masks_dir, f"{i}.jpg"), (255*mask.masks[i]).astype(np.uint8))

    def __call__(self, results):
        if self.vis_box:
            self.visualize_box(results)
        if self.vis_mask:
            self.visualize_mask(results)
        return results
