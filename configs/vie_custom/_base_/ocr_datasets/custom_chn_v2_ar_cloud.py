#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/2 22:53
# @Author : WeiHua

dataset_type = 'VIEE2EDataset'
data_root = '/data/whua/dataset/custom_chn_synth/merged'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='CustomLineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations'],
        optional_keys=['entity_dict']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/annotation.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=False,
    class_file=None,
    data_type='ocr',
    max_seq_len=80,
    order_type='shuffle',
    auto_reg=True,
    pre_parse_anno=True)
"""
avg:11.705568749328094, max:25, min:1
avg_height:441.12469581024425, avg_width:1158.8753041897558
ext key:[]
max_len:74
avg_ins_height:69.38160162741796, avg_ins_width:264.441446308023
Total instance num: 2395451, total image num: 204642
"""


train_list = [train]

test_list = [train]
