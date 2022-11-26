#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/6 16:14
# @Author : WeiHua
import json
import os
import shutil

import numpy as np
import cv2
import mmcv

from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, RandomFlip
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor
from mmcv.parallel import DataContainer as DC

from .custom_e2e_visualizer import CustomVisualizer

@PIPELINES.register_module()
class E2EFormatBundle(DefaultFormatBundle):
    """
    Convert specified data / annotations to tensor and DataContainer.
    The batched operation can be seen at "def collate" in "mmcv/parallel/collate.py"
    """
    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            # if self.img_to_float is True and img.dtype == np.uint8:
            #     # Normally, image is of uint8 type without normalization.
            #     # At this time, it needs to be forced to be converted to
            #     # flot32, otherwise the model training and inference
            #     # will be wrong. Only used for YOLOX currently .
            #     img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        # for key in ['gt_bboxes', 'gt_bboxes_ignore', 'gt_polys']:
        for key in ['gt_bboxes', 'gt_bboxes_ignore']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        # polys will maintain list[array] format.
        if 'gt_polys' in results:
            results['gt_polys'] = DC(results['gt_polys'], cpu_only=True)
        # convert instances, texts and entities label to DataContainer
        for key in results['gt_labels'].keys():
            # suppose to contain 'instances', 'texts', 'entities'
            # if key in ['texts', 'entities']:
            #     results['gt_labels'][key] = DC(to_tensor(results['gt_labels'][key]).unsqueeze(0),
            #                                    stack=True, pad_dims=1, padding_value=0)
            #     continue
            results['gt_labels'][key] = DC(to_tensor(results['gt_labels'][key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'density_maps' in results:
            results['density_maps'] = DC(to_tensor(results['density_maps']), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class MakeKeyDensityMap:
    """
    Make density map for counting branch. It only focuses on instance which contains key info.
    """
    def __init__(self, gaussian_k=6, scale_ratio=4):
        self.gaussian_k = gaussian_k
        self.scale_ratio = scale_ratio

    def __call__(self, results):
        entity_dict = results['counting']['entity_dict']
        entity_list = results['counting']['entity_list']
        assert 'O' not in entity_list, f"Invalid entity list: {entity_list}"
        h, w = results['pad_shape'][:2]
        h //= self.scale_ratio
        w //= self.scale_ratio
        density_maps = []
        entities = results['gt_labels']['entities']
        polygons = results['gt_polys']
        for cls_idx, cls in enumerate(entity_list):
            density_map = np.zeros((h, w), dtype=np.float32)
            B_idx = entity_dict[f'B-{cls}']
            I_idx = entity_dict[f'I-{cls}']
            for i in range(len(entities)):
                if B_idx in entities[i] or I_idx in entities[i]:
                    gt = np.mean(polygons[i].reshape(-1, 2), axis=0) / self.scale_ratio
                    gt[0], gt[1] = np.clip(gt[0], 0, w - 1), np.clip(gt[1], 0, h - 1)
                    density_map[int(gt[1]), int(gt[0])] = 1
            density_maps.append(gaussian_filter(density_map, self.gaussian_k))
        results['density_maps'] = np.stack(density_maps, axis=0)
        return results


@PIPELINES.register_module()
class RemoveInvalidDet(CustomVisualizer):
    """
    Due to the bad annotation, some bounding box / polygon may exceed the boundary of the
    image. After clipping these box / polygon, the bounding box's top-left point and bottom-right
    point may have the same x or y coordinate, which cause nan error when encode it to delta of
    (x, y, w, h)
    """
    def __init__(self, bbox_key='gt_bboxes', vis_dir=None, rm_exist_dir=True):
        self.bbox_key = bbox_key
        self.vis_dir = vis_dir
        super(RemoveInvalidDet, self).__init__(
            vis_box=True,
            vis_mask=True,
            vis_text=True,
            vis_entity=False,
            vis_dir=vis_dir,
            rm_exist_dir=rm_exist_dir
        )

    def __call__(self, results):
        assert self.bbox_key in results['bbox_fields']
        abnormal_flag = False
        for box in results[self.bbox_key]:
            if box[0] == box[2] or box[1] == box[3]:
                abnormal_flag = True
                break
        if abnormal_flag:
            if self.vis_dir:
                self.visualize_mask(results)
                self.visualize_box(results)
        return results

@PIPELINES.register_module()
class MaskOutInstance():
    """Masking instances which are outside the image"""
    def __call__(self, results):
        if results['ann_info']['outside_poly']:
            assert len(results['img_fields']) == 1
            img_key = results['img_fields'][0]
            poly_list = []
            for poly_ in results['ann_info']['outside_poly']:
                poly = np.array(poly_).reshape(-1, 2).astype(np.int32)
                poly_list.append(poly)
            if len(poly_list) > 0:
                results[img_key] = cv2.fillPoly(results[img_key], poly_list, color=(0, 0, 0))
        return results

"""
code from: https://github.com/hikopensource/DAVAR-Lab-OCR
###################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    transforms.py
# Abstract       :    Implementations of some transformations

# Current Version:    1.0.1
# Date           :    2020-11-25
#####################################################################################################
"""
@PIPELINES.register_module()
class CustomResize(Resize):
    """ Resize images & bbox & mask. Add new specialities of
        - support poly boxes resize
        - support cbboxes resize
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        """
        Args:
            img_scale (tuple or list[tuple]): Images scales for resizing.
            multiscale_mode (str): Either "range" or "value".
            ratio_range (tuple[float]): (min_ratio, max_ratio)
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image.
            bbox_clip_border (bool, optional): Whether clip the objects outside
                the border of the image. Defaults to True.
            backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
                These two backends generates slightly different results. Defaults
                to 'cv2'.
            override (bool, optional): Whether to override `scale` and
                `scale_factor` so as to call resize twice. Default False. If True,
                after the first resizing, the existed `scale` and `scale_factor`
                will be ignored so the second resizing can be allowed.
                This option is a work-around for multiple times of resize in DETR.
                Defaults to False.
        """
        super().__init__(img_scale=img_scale, multiscale_mode=multiscale_mode, ratio_range=ratio_range,
                         keep_ratio=keep_ratio, bbox_clip_border=bbox_clip_border, backend=backend,
                         override=override)

    def _resize_bboxes(self, results):
        """ Resize bboxes (support 'gt_bboxes', 'gt_poly_bboxes').
            Refactor this function to support multiple points

        Args:
            results (dict): input data flow

        Returns:
            dict: updated data flow. All keys in `bbox_fields` will be updated according to `scale_factor`.
        """
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = []
            for box in results[key]:
                tmp_box = np.array(box, dtype=np.float32)
                tmp_box[0::2] *= results['scale_factor'][0]
                tmp_box[1::2] *= results['scale_factor'][1]
                if self.bbox_clip_border:
                    tmp_box[0::2] = np.clip(tmp_box[0::2], 0, img_shape[1])
                    tmp_box[1::2] = np.clip(tmp_box[1::2], 0, img_shape[0])
                bboxes.append(tmp_box)
            if len(results[key]) > 0:
                results[key] = bboxes

    def _resize_cbboxes(self, results):
        """ Resize cbboxes (support 'gt_cbboxes').

        Args:
            results (dict): input data flow

        Returns:
            dict: updated data flow. All keys in `cbbox_fields` will be updated according to `scale_factor`.
        """
        img_shape = results['img_shape']
        for key in results.get('cbbox_fields', []):
            cbboxes = []
            for cbox in results[key]:
                tmp_cbox = np.array(cbox, dtype=np.float32)
                new_tmp_cbox = []
                for ccbox in tmp_cbox:
                    ccbox = np.array(ccbox, dtype=np.float32)
                    ccbox[0::2] *= results['scale_factor'][0]
                    ccbox[1::2] *= results['scale_factor'][1]
                    new_tmp_cbox.append(ccbox)
                tmp_cbox = np.array(new_tmp_cbox, dtype=np.float32)
                if self.bbox_clip_border:
                    tmp_cbox[:, 0::2] = np.clip(tmp_cbox[:, 0::2], 0, img_shape[1])
                    tmp_cbox[:, 1::2] = np.clip(tmp_cbox[:, 1::2], 0, img_shape[0])
                cbboxes.append(tmp_cbox)
            results[key] = cbboxes

    def __call__(self, results):
        """ Main process of davar_resize

        Args:
            results (dict): input data flow.

        Returns:
            dict: updated data flow.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, 'scale and scale_factor cannot be both set.'
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_cbboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)

        return results


@PIPELINES.register_module()
class DavarRandomFlip(RandomFlip):
    """ Random flip the image & bbox & mask """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        """
        Args:
            flip_ratio (float | list[float]): the flipping probability.
            direction (str | list[str]): the flipping direction.
        """
        super().__init__(flip_ratio=flip_ratio, direction=direction)

    def _flip_img(self, results):
        """ Flip img horizontally.

        Args:
            results (dict): input data flow

        Returns:
            dict: updated data flow
        """
        direction = results['flip_direction']
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imflip(results[key], direction=direction)

    def _flip_cbboxes(self, results):
        """ Flip cbboxes horizontally.

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow
        """
        img_shape = results['img_shape']
        direction = results['flip_direction']
        for key in results.get('cbbox_fields', []):
            # For any n-point poly cbboxes.
            cbboxes = []
            for cbox in results[key]:
                tmp_cbox = np.array(cbox, dtype=np.float32)
                flipped_cbox = tmp_cbox.copy()
                if tmp_cbox.shape[1] == 4:
                    if direction == 'horizontal':
                        w = img_shape[1]
                        flipped_cbox[:, 0] = w - tmp_cbox[:, 2]
                        flipped_cbox[:, 2] = w - tmp_cbox[:, 0]
                    elif direction == 'vertical':
                        h = img_shape[0]
                        flipped_cbox[:, 1] = h - tmp_cbox[:, 3]
                        flipped_cbox[:, 3] = h - tmp_cbox[:, 1]
                    elif direction == 'diagonal':
                        w = img_shape[1]
                        h = img_shape[0]
                        flipped_cbox[:, 0] = w - tmp_cbox[:, 2]
                        flipped_cbox[:, 2] = w - tmp_cbox[:, 0]
                        flipped_cbox[:, 1] = h - tmp_cbox[:, 3]
                        flipped_cbox[:, 3] = h - tmp_cbox[:, 1]
                else:
                    if direction == 'horizontal':
                        w = img_shape[1]
                        flipped_cbox[:, 0::2] = w - tmp_cbox[:, 0::2]
                    elif direction == 'vertical':
                        h = img_shape[0]
                        flipped_cbox[:, 1::2] = h - tmp_cbox[:, 1::2]
                    elif direction == 'diagonal':
                        w = img_shape[1]
                        h = img_shape[0]
                        flipped_cbox[:, 0::2] = w - tmp_cbox[:, 0::2]
                        flipped_cbox[:, 1::2] = h - tmp_cbox[:, 1::2]
                cbboxes.append(flipped_cbox)
            results[key] = cbboxes

    def _flip_bboxes(self, results):
        """ Flip bboxes horizontally.

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow
        """
        img_shape = results['img_shape']
        direction = results['flip_direction']
        for key in results.get('bbox_fields', []):
            bboxes = []
            for box in results[key]:
                tmp_box = np.array(box, dtype=np.float32)
                flipped_box = tmp_box.copy()
                if tmp_box.shape[0] == 4:
                    if direction == 'horizontal':
                        w = img_shape[1]
                        flipped_box[0] = w - tmp_box[2]
                        flipped_box[2] = w - tmp_box[0]
                    elif direction == 'vertical':
                        h = img_shape[0]
                        flipped_box[1] = h - tmp_box[3]
                        flipped_box[3] = h - tmp_box[1]
                    elif direction == 'diagonal':
                        w = img_shape[1]
                        h = img_shape[0]
                        flipped_box[0] = w - tmp_box[2]
                        flipped_box[2] = w - tmp_box[0]
                        flipped_box[1] = h - tmp_box[3]
                        flipped_box[3] = h - tmp_box[1]
                else:
                    if direction == 'horizontal':
                        w = img_shape[1]
                        flipped_box[0::2] = w - tmp_box[0::2]
                    elif direction == 'vertical':
                        h = img_shape[0]
                        flipped_box[1::2] = h - tmp_box[1::2]
                    elif direction == 'diagonal':
                        w = img_shape[1]
                        h = img_shape[0]
                        flipped_box[0::2] = w - tmp_box[0::2]
                        flipped_box[1::2] = h - tmp_box[1::2]
                bboxes.append(flipped_box)
            if len(results[key]) > 0:
                results[key] = bboxes

    def _flip_masks(self, results):
        """ Flip masks horizontally.

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow
        """
        direction = results['flip_direction']
        for key in results.get('mask_fields', []):
            results[key] = results[key].flip(direction)

    def _flip_seg(self, results):
        """ Flip seg horizontally.

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow
        """
        for key in results.get('seg_fields', []):
            direction = results['flip_direction']
            results[key] = mmcv.imflip(results[key], direction=direction)

    def __call__(self, results):
        """ Call function to flip bounding boxes, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            self._flip_img(results)
            self._flip_bboxes(results)
            self._flip_cbboxes(results)
            self._flip_masks(results)
            self._flip_seg(results)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class RandomRotate():
    """ Randomly rotate images and corresponding annotations"""

    def __init__(self,
                 angles=None,
                 borderValue=(255, 255, 255)
                 ):
        """
        Args:
            angles (list | tuple):   the rotated degree, contains a single value or multiple values
                                    - If `angles` contains single value, this value represents `rotating fixed angle`;
                                    - If `angles` contains two values(tuple or list), angles represents `rotating
                                      random angle in the interval ranged by these two values`
                                    - If `angles` contains more than two values, angles represents `randomly choosing
                                      a value in this list as rotation angle`
            borderValue (tuple): filled pixels value when rotate.
        """
        self.angles = angles
        self.angle = 0
        self.borderValue = borderValue

    def _rotate_img(self, results):
        """ Randomly rotating images and corresponding annotations

        Args:
            results (dict): data flow in pipeline (contains `img` and `bboxes_fields`)

        Returns:
            dict: updated data flow in pipeline
        """
        angle = self.angle
        height, width = results['img_shape'][:2]
        center = ((width - 1) * 0.5, (height - 1) * 0.5)

        # Compute rotation matrix
        mat_rotation = cv2.getRotationMatrix2D(center, -angle, 1)

        # Compute height and width of new image after rotation
        cos = np.abs(mat_rotation[0, 0])
        sin = np.abs(mat_rotation[0, 1])
        width_new = height * sin + width * cos
        height_new = height * cos + width * sin
        mat_rotation[0, 2] += (width_new - width) * 0.5
        mat_rotation[1, 2] += (height_new - height) * 0.5
        width_new = int(np.round(width_new))
        height_new = int(np.round(height_new))

        # Rotate image
        img_rotation = cv2.warpAffine(results['img'], mat_rotation, (width_new, height_new),
                                      borderValue=self.borderValue)
        results['img'] = img_rotation
        results['img_shape'] = results['img'].shape

        # Rotate corresponding annotations: all boxes in bbox_fields
        for key in results.get('bbox_fields', []):
            gt_boxes_ret = []
            for poly in results[key]:
                rot_array = []
                poly_length = len(poly)

                if poly_length == 4:
                    poly = [poly[0], poly[1], poly[2], poly[1], poly[2], poly[3], poly[0], poly[3]]

                # Convert to np.array of shape (:, 2)
                for i in range(0, len(poly), 2):
                    rot_array.append(np.array([poly[i], poly[i + 1]]))

                # Rotate corresponding annotations
                rot_array = np.array([rot_array])
                rot_array = cv2.transform(rot_array, mat_rotation).squeeze().reshape(len(poly))

                if poly_length == 4:
                    x_coords = rot_array[0::2]
                    y_coords = rot_array[1::2]
                    rot_array = np.array([
                        np.min(x_coords),
                        np.min(y_coords),
                        np.max(x_coords),
                        np.max(y_coords)
                    ])

                gt_boxes_ret.append(rot_array)
            if len(results[key]) > 0:
                results[key] = gt_boxes_ret

        # Rotate gt_bboxes according to gt_poly_bboxes
        if 'gt_bboxes' in results and 'gt_poly_bboxes' in results:
            gt_bboxes = []
            gt_bboxes_ignore = []

            for poly in results['gt_poly_bboxes']:
                poly = np.array(poly, dtype=np.double)
                x_coords = poly[0::2]
                y_coords = poly[1::2]
                aligned_poly = [
                    np.min(x_coords),
                    np.min(y_coords),
                    np.max(x_coords),
                    np.max(y_coords)
                ]
                gt_bboxes.append(aligned_poly)

            for poly in results['gt_poly_bboxes_ignore']:
                poly = np.array(poly, dtype=np.double)
                x_coords = poly[0::2]
                y_coords = poly[1::2]
                aligned_poly = [
                    np.min(x_coords),
                    np.min(y_coords),
                    np.max(x_coords),
                    np.max(y_coords)
                ]
                gt_bboxes_ignore.append(aligned_poly)

            if len(gt_bboxes) == 0:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            else:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

            if len(gt_bboxes_ignore) == 0:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            else:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)

            results['gt_bboxes'] = gt_bboxes
            results['gt_bboxes_ignore'] = gt_bboxes_ignore

        # Rotate corresponding annotations: all boxes in cbbox_fields
        for key in results.get('cbbox_fields', []):
            gt_cboxes_ret = []
            for instance in results[key]:
                tmp_cboxes = []
                for poly in instance:
                    rot_array = []
                    # Convert to np.array of shape (:, 2)
                    for i in range(0, len(poly), 2):
                        rot_array.append(np.array([poly[i], poly[i + 1]]))
                    rot_array = np.array([rot_array])
                    # Rotate corresponding annotations
                    rot_array = cv2.transform(rot_array, mat_rotation).squeeze().reshape(len(poly))
                    tmp_cboxes.append(rot_array)
                    gt_cboxes_ret.append(tmp_cboxes)
            results[key] = gt_cboxes_ret

        # Rotate corresponding annotations: all masks in mask_fields
        for key in results.get('mask_fields', []):
            mask = results[key].masks.transpose((1, 2, 0))
            if len(results[key].masks) == 0:
                results[key] = results[key].resize((height_new, width_new))
            else:
                # Rotate mask
                mask_rotation = cv2.warpAffine(mask, mat_rotation, (width_new, height_new),
                                               borderValue=self.borderValue)
                if mask_rotation.ndim == 2:
                    # case when only one mask, (h, w)
                    mask_rotation = mask_rotation[:, :, None]  # (h, w, 1)
                mask_rotation = mask_rotation.transpose((2, 0, 1))
                results[key].masks = mask_rotation
                results[key].height = height_new
                results[key].width = width_new

    def __call__(self, results):
        """ Main process of random_rotate

        Args:
            results (dict): input data flow

        Returns:
            dict: updated data flow.
        """
        # `angles` must be a list or tuple
        assert isinstance(self.angles, (list, tuple))
        if len(self.angles) == 1:
            angle = self.angles[0]
        elif len(self.angles) == 2:
            angle_max = max(self.angles)
            angle_min = min(self.angles)
            angle = np.random.randint(angle_min, angle_max)
        else:
            angle = np.random.choice(self.angles)
        self.angle = angle
        self._rotate_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(angles={})'.format(self.angles)
        return repr_str


@PIPELINES.register_module()
class ResizeNormalize:
    """ Package resize normalize in a pipeline, Support for different interpolate modes"""

    def __init__(self, size,
                 interpolation=2,
                 mean=(127.5, 127.5, 127.5),
                 std=(127.5, 127.5, 127.5), ):
        """
        Args:
            size (tuple): image resize size
            interpolation (int): interpolation type, including [0, 1, 2, 3]
            mean (tuple): image normalization mean
            std (tuple): image normalization std
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, results):
        """
        Args:
            results (dict): dict with training image

        Returns:
            dict: dict with training image after resize and normalization
        """
        img = results['img']

        # different interpolation type corresponding the OpenCV
        if self.interpolation == 0:
            self.interpolation = cv2.INTER_NEAREST
        elif self.interpolation == 1:
            self.interpolation = cv2.INTER_LINEAR
        elif self.interpolation == 2:
            self.interpolation = cv2.INTER_CUBIC
        elif self.interpolation == 3:
            self.interpolation = cv2.INTER_AREA
        else:
            raise Exception("Unsupported interpolation type !!!")

        # Deal with the image error during image loading
        if img is None:
            return None
        try:
            img = cv2.resize(img, self.size, self.interpolation)
        except cv2.error:
            return None
        img = np.array(img, np.float32)

        # normalize the image
        img = mmcv.imnormalize(img, self.mean, self.std, to_rgb=False)
        results['img'] = img
        return results


@PIPELINES.register_module()
class DavarRandomCrop:
    """ Randomly crop images and make sure to contain text instances."""

    def __init__(self,
                 max_tries=50,
                 min_crop_side_ratio=0.1,
                 instance_key='gt_bboxes'):
        """
        Args:
            max_tries (int): maximum number of attempts.
            min_crop_side_ratio (float): minimum crop ratio.
            instance_key (str): crop according to instance_key
        """
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.instance_key = instance_key

    def _random_crop(self, img, polys):
        """ Randomly crop image

        Args:
            img (ndarray): image to be cropped
            polys (ndarray): location of polys.

        Returns:
            tuple: location of the cropping box
        """
        h, w, _ = img.shape
        pad_h = h // 10
        pad_w = w // 10
        h_array = np.zeros((h + 2 * pad_h), dtype=np.int32)
        w_array = np.zeros((w + 2 * pad_w), dtype=np.int32)

        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[0::2])
            maxx = np.max(poly[0::2])
            miny = np.min(poly[1::2])
            maxy = np.max(poly[1::2])
            w_array[minx + pad_w:maxx + pad_w] = 1
            h_array[miny + pad_h:maxy + pad_h] = 1

        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        for _ in range(self.max_tries):
            xx = np.random.choice(w_axis, size=2)
            x_min = np.min(xx) - pad_w
            x_max = np.max(xx) - pad_w
            x_min = np.clip(x_min, 0, w)
            x_max = np.clip(x_max, 0, w)
            yy = np.random.choice(h_axis, size=2)
            y_min = np.min(yy) - pad_h
            y_max = np.max(yy) - pad_h
            y_min = np.clip(y_min, 0, h)
            y_max = np.clip(y_max, 0, h)

            if x_max - x_min < self.min_crop_side_ratio * w or \
                    y_max - y_min < self.min_crop_side_ratio * h:
                continue
            for poly in polys:
                if np.all((poly[0::2] >= x_min) & (poly[1::2] >= y_min) & \
                          (poly[0::2] <= x_max) & (poly[1::2] <= y_max)):
                    return x_min, y_min, x_max, y_max
        return 0, 0, w, h

    def __call__(self, results):
        """ Main process of davar_random_crop

        Args:
            results (dict): input data flow.

        Returns:
            dict: updated data flow.
        """
        img = results['img']
        polys = results[self.instance_key]
        x_min, y_min, x_max, y_max = self._random_crop(img, polys)
        kept_idx = []
        for idx, poly in enumerate(polys):
            if np.all((poly[0::2] >= x_min) & (poly[1::2] >= y_min) & \
                      (poly[0::2] <= x_max) & (poly[1::2] <= y_max)):
                kept_idx.append(idx)
        kept_idx = np.array(kept_idx)

        # modified by whua
        if len(kept_idx) / len(polys) < 0.5:
            # at least keep 50% text instances
            return results

        # crop img
        results['img'] = img[y_min: y_max, x_min: x_max, :]
        results['img_shape'] = results['img'].shape
        # crop mask
        for key in results.get('mask_fields', []):
            results[key] = results[key].crop(np.array([x_min, y_min, x_max, y_max]))
        # crop box
        if 'gt_bboxes' in results.get('bbox_fields', []):
            kept_bbox_idx = []
            for box_idx, box in enumerate(results['gt_bboxes']):
                box = np.array(box)
                if np.all((np.min(box[0::2]) >= x_min) & (np.min(box[1::2]) >= y_min) & \
                          (np.max(box[0::2]) <= x_max) & (np.max(box[1::2]) <= y_max)):
                    kept_bbox_idx.append(box_idx)
            for key in results.get('bbox_fields', []):
                bboxes = []
                for box_idx in kept_bbox_idx:
                    box = np.array(results[key][box_idx])
                    box[0::2] = (box[0::2] - x_min)
                    box[1::2] = (box[1::2] - y_min)
                    bboxes.append(box)
                # no valid box in img
                if len(bboxes) == 0:
                    if key == 'gt_bboxes':
                        bboxes = np.zeros((0, 4), dtype=np.float32)
                    else:
                        bboxes = np.zeros((0, 8), dtype=np.float32)
                results[key] = bboxes
        else:
            for key in results.get('bbox_fields', []):
                bboxes = []
                for box in results[key]:
                    box = np.array(box)
                    if np.all((np.min(box[0::2]) >= x_min) & (np.min(box[1::2]) >= y_min) & \
                              (np.max(box[0::2]) <= x_max) & (np.max(box[1::2]) <= y_max)):
                        box[0::2] = (box[0::2] - x_min)
                        box[1::2] = (box[1::2] - y_min)
                        bboxes.append(box)
                # no valid box in img
                if len(bboxes) == 0:
                    if key == 'gt_bboxes':
                        bboxes = np.zeros((0, 4), dtype=np.float32)
                    else:
                        bboxes = np.zeros((0, 8), dtype=np.float32)
                results[key] = bboxes

            # nums = None
            # for key in results.get('bbox_fields', []):
            #     if not nums:
            #         nums = len(results[key])
            #         continue
            #     if nums != len(results[key]):
            #         for key in results.get('bbox_fields', []):
            #             print(f"{key}: {len(results[key])}")

        # # calculate the kept text and label
        # for key in ['gt_labels', 'gt_texts']:
        #     if key in results:
        #         results[key] = [results[key][idx] for idx in kept_idx]
        # modified by whua
        for key in ['gt_labels']:
            if key in results:
                for label_key in results[key].keys():
                    results[key][label_key] = results[key][label_key][kept_idx]


        # calculate the kept mask
        for key in ['gt_masks']:
            if key in results:
                ori_mask = results[key].masks
                kept_mask = [ori_mask[idx] for idx in kept_idx]
                if len(kept_mask) > 0:
                    kept_mask = np.stack(kept_mask)
                else:
                    kept_mask = np.empty((0, results[key].height, results[key].width), dtype=np.float32)
                results[key] = BitmapMasks(kept_mask, results[key].height, results[key].width)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
