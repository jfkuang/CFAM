#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/2 22:53
# @Author : WeiHua

dataset_type = 'VIEE2EDataset'
data_root = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/custom_english/merged'
# data_root = '/data/whua/dataset/ie_e2e/nfv1/ie_e2e_data/mm_format/table'
# data_root = '/mnt/whua/ie_e2e_data/mm_format/table'
data_root_cord = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/cord/e2e_format'
data_root_nfv3 = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/nfv3/e2e_format'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='CustomLineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations'],
        optional_keys=['entity_dict']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/annotation_max_145.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root,
    pipeline=None,
    test_mode=False,
    class_file=None,
    data_type='ocr',
    max_seq_len=150,
    order_type='shuffle',
    auto_reg=True)

# test = dict(
#     type=dataset_type,
#     ann_file=f'{data_root}/annotation.txt',
#     loader=loader,
#     dict_file=f'{data_root}/dict.json',
#     img_prefix=data_root,
#     pipeline=None,
#     test_mode=True,
#     class_file=None,
#     data_type='ocr',
#     max_seq_len=170,
#     order_type='origin',
#     auto_reg=True)

# test_sroie = dict(
#     type=dataset_type,
#     ann_file=f'{data_root_sroie}/annotation.txt',
#     loader=loader,
#     dict_file=f'{data_root}/dict.json',
#     img_prefix=data_root_sroie,
#     pipeline=None,
#     test_mode=True,
#     class_file=None,
#     data_type='ocr',
#     max_seq_len=170,
#     order_type='origin',
#     auto_reg=True)

test_cord = dict(
    type=dataset_type,
    ann_file=f'{data_root_cord}/test.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root_cord,
    pipeline=None,
    test_mode=True,
    class_file=None,
    data_type='ocr',
    max_seq_len=150,
    order_type='origin',
    auto_reg=True)

test_nfv3 = dict(
    type=dataset_type,
    ann_file=f'{data_root_nfv3}/test.txt',
    loader=loader,
    dict_file=f'{data_root}/dict.json',
    img_prefix=data_root_nfv3,
    pipeline=None,
    test_mode=True,
    class_file=None,
    data_type='ocr',
    max_seq_len=150,
    order_type='origin',
    auto_reg=True)

train_list = [train]

test_list = [test_cord, test_nfv3]
