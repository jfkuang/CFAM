#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：sroie_3090.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/10/13 17:46 
'''
dataset_type = 'VIEE2EDataset'
data_root = '/data3/jfkuang/data/sroie/e2e_format'

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
    auto_reg=True)

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
