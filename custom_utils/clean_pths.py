#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/11 11:20
# @Author : WeiHua
import glob
import shutil

from tqdm import tqdm
import os


if __name__ == '__main__':
    dirs = glob.glob('/apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/*')
    for dir_ in tqdm(dirs):
        pths = glob.glob(os.path.join(dir_, '*.pth'))
        for pth_ in pths:
            if 'epoch_' in pth_:
                num_epoch = pth_.split('/')[-1].split('.')[0]
                num_epoch = int(num_epoch[6:])
                if num_epoch < 270:
                    os.remove(pth_)
