#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/2 22:53
# @Author : WeiHua

dataset_type = 'VIEE2EDataset'
data_root = '/data/whua/ie_e2e/sroie/e2e_format'
# data_root = '/data/whua/dataset/ie_e2e/nfv1/ie_e2e_data/mm_format/table'
# data_root = '/mnt/whua/ie_e2e_data/mm_format/table'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='CustomLineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations'],
        optional_keys=['entity_dict']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/train_update_screen.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=False,
    class_file=f'{data_root}/class_list.json',
    data_type='vie',
    max_seq_len=72,
    order_type='shuffle',
    auto_reg=True,
    pre_parse_anno=True)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/test_screen.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=True,
    class_file=f'{data_root}/class_list.json',
    data_type='vie',
    max_seq_len=72,
    order_type='origin',
    auto_reg=True)

train_list = [train]

test_list = [test]
