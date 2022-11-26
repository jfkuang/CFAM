#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：synth_chn_ar_cloud_kjf.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/6/4 17:12 
'''
dataset_type = 'VIEE2EDataset'
#1032
# data_root = '/data/jfkuang/syntext'
#1803
data_root = '/home/jfkuang/data/syntext'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='CustomLineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations'],
        optional_keys=['entity_dict']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/custom_json_format.txt',
    loader=loader,
    dict_file=f'{data_root}/custom_dict.json',
    img_prefix=f'{data_root}/syn_130k_images',
    pipeline=None,
    test_mode=False,
    class_file=None,
    data_type='ocr',
    max_seq_len=75,
    order_type='shuffle',
    auto_reg=True,
    pre_parse_anno=True)
"""
    Total sample num: 134514
    max_pt_num:4
    avg_ins:11.237142602257014, max:321, min:1
    avg_height:420.61183222564193, avg_width:487.8559555139242
    max_len:71
    avg_ins_height:35.7767097812647, avg_ins_width:81.25013148728493 
"""

train_list = [train]

test_list = [train]
