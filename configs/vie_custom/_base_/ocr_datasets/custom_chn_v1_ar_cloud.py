#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/2 22:53
# @Author : WeiHua

dataset_type = 'VIEE2EDataset'
data_root_chn_v1 = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/custom_chn_v1/merged'
# data_root = '/data/whua/dataset/ie_e2e/nfv1/ie_e2e_data/mm_format/table'
# data_root = '/mnt/whua/ie_e2e_data/mm_format/table'
# data_root_cord = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/cord/e2e_format'
# data_root_nfv3 = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/nfv3/e2e_format'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='CustomLineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations'],
        optional_keys=['entity_dict']))

train_chn_v1 = dict(
    type=dataset_type,
    ann_file=f'{data_root_chn_v1}/annotation.txt',
    loader=loader,
    dict_file=f'{data_root_chn_v1}/dict.json',
    img_prefix=data_root_chn_v1,
    pipeline=None,
    test_mode=False,
    class_file=None,
    data_type='ocr',
    max_seq_len=55,
    order_type='shuffle',
    auto_reg=True)
"""
avg:42.735463852540974, max:70, min:21
avg_height:1280.0, avg_width:720.0
ext key:[]
max_len:49
avg_ins_height:32.0, avg_ins_width:183.05079478268453
total 184024 images
"""

train_list = [train_chn_v1]

test_list = [train_chn_v1]
