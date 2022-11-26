#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/9 22:14
# @Author : WeiHua

"""
default ->
    1280
    1e4
    feature-fuse-v1
    rec & kie weights = 10
    Adam optimizer
    shuffle instances' order
    4 decoder layer
    no node-level modeling
    no text encoder
    with data augmentation

v0 -> fuse without sum during up-sampling -> default
v1 -> fuse with sum during up-sampling

"""