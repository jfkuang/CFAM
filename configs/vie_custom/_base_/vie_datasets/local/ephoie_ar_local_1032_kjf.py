#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e
@File    ：ephoie_ar_local_1032_kjf.py
@IDE     ：PyCharm
@Author  ：jfkuang
@Date    ：2022/6/4 13:19
'''


dataset_type = 'VIEE2EDataset'
data_root = '/home/jfkuang/data/ephoie_e2e_format/e2e_format'
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
    ann_file=f'{data_root}/train.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=False,
    class_file=f'{data_root}/class_list.json',
    data_type='vie',
    max_seq_len=80,
    order_type='shuffle',
    auto_reg=True)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/test.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=True,
    class_file=f'{data_root}/class_list.json',
    data_type='vie',
    max_seq_len=80,
    order_type='origin',
    auto_reg=True)

train_list = [train]

test_list = [test]
