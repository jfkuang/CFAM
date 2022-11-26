#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：nfv5_3125_ar_local.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/10/4 19:28 
'''

dataset_type = 'VIEE2EDataset'
data_root = '/root/paddlejob/workspace/env_run/dkliang/projects/synchronous/V100_4/ie_e2e-newtest/ie_e2e_dataset/nfv5_3125'

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
    max_seq_len=125,
    order_type='shuffle',
    auto_reg=True,
    pre_parse_anno=True)

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
    max_seq_len=125,
    order_type='origin',
    auto_reg=True)

train_list = [train]

test_list = [test]
