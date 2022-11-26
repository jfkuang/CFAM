#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：vie_modules.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/10/21 15:52 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class VCM(nn.Module):
    def __int__(self, shape, in_dim, out_dim):
        super(VCM, self).__int__()
        self.dim = in_dim
        self.shape = shape
        self.conv1 = nn.Conv2d(in_dim, 10, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(10, out_dim, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, context_feature):
        #context_feature:(BN,HW,C)
        #return:(B,N,C)
        B, N, C, H, W = self.shape
        context_feature = context_feature.unsqueeze(3).reshape(B, N, C, -1)
        feature = self.conv1(context_feature)
        feature = self.pool1(feature)
        feature = self.fc(feature)
        return feature

class SCM(nn.Module):
    def __int__(self, shape, in_dim, out_dim):
        super(SCM, self).__int__()
        self.dim = in_dim
        self.shape = shape
        self.conv1 = nn.Conv2d(in_dim, out_dim, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 5)
    def forward(self, texture_feature):
        #texture_feature:(BN,L,C)
        #return:(B,N,C)
        feature1 = self.conv1(texture_feature)
        feature2 = self.conv2(texture_feature)
        feature3 = self.conv3(texture_feature)
