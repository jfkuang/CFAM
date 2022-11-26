#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/6 14:00
# @Author : WeiHua

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale, min_scale = 1500, 1000

# pipeline for end-to-end text spotting but follow the manner of vie
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # may be replace to 'ScaleAspectJitter'
    dict(type='CustomResize',
         img_scale=(max_scale, min_scale), keep_ratio=True, multiscale_mode='range'),
    dict(type='RandomFlip', flip_ratio=0.),
    # dict(type='RemoveInvalidDet', vis_dir='/usr/local/dataset/whua/ie_e2e_log/e2e_openmm_abnormal'),
    # dict(type='CustomVisualizer', vis_mask=True, vis_text=True, vis_entity=False,
    #      vis_dir='/apdcephfs/share_887471/common/whua/logs/ie_e2e_log/vis_res', vis_num_limit=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='E2EFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_polys'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # may be replace to 'ScaleAspectJitter'
    dict(type='CustomResize',
         img_scale=(max_scale, min_scale), keep_ratio=True, multiscale_mode='range'),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='E2EFormatBundle'),
    dict(
        type='Collect',
        keys=['img'])
        # keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
