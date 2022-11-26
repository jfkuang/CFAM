#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/5 16:38
# @Author : WeiHua

dataset_type = 'VIEE2EDataset'
data_root = '/home/whua/code/MaskTextSpotterV3-master/datasets/synthtext/SynthText/'
ann_root = '/home/whua/code/MaskTextSpotterV3-master/datasets/synthtext/e2e_format/'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))

train = dict(
    type=dataset_type,
    ann_file=f'{ann_root}/train.txt',
    loader=loader,
    dict_file=f'{ann_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=False,
    class_file=None,
    data_type='ocr',
    max_seq_len=60)

test = dict(
    type=dataset_type,
    ann_file=f'{ann_root}/train.txt',
    loader=loader,
    dict_file=f'{ann_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=False,
    class_file=None,
    data_type='ocr',
    max_seq_len=60)

train_list = [train]

test_list = [test]
