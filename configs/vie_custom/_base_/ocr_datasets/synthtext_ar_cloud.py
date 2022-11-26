#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/5 16:38
# @Author : WeiHua

dataset_type = 'VIEE2EDataset'
data_root = '/apdcephfs/share_887471/common/ocr_benchmark/benchmark/SynthText'
ann_root = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/SynthText/'

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
    max_seq_len=60,
    check_outside=True,
    order_type='shuffle',
    auto_reg=True)

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
    max_seq_len=60,
    check_outside=True,
    auto_reg=True)

train_list = [train]

test_list = [test]
