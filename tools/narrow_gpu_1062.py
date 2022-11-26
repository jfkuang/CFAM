#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：narrow_gpu_1062.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/10/15 22:30 
'''

import os
import sys
import time

cmd = 'sh train_1062.sh'


def get_info():
    gpu_memory_list = []
    gpu_status = os.popen('nvidia-smi | grep %').readlines()
    for list_ in gpu_status:
        list_ = list_.split('|')
        gpu_memory = int(list_[2].split('/')[0].split('M')[0].strip())
        gpu_memory_list.append(gpu_memory)
        # 主要是针对memory操作，如果对power需要操作的话同理
        # gpu_power = int(list_[1].split('  ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_memory_list


def setup_python(interval=1):
    gpu_memory_list = get_info()
    i = 0
    while (True):
        if gpu_memory_list[i] <= 7000:  # 设置条件，满足则运行python程序
            if i == 0 or i == 1 or i == 2 or i == 3:  # 指定特定的GPU
                print('\n' + cmd)
                os.system(cmd)  # 运行python脚本
                break
                #gpu_memory_list = get_info()  # 遍历一次GPU资源后重新查询
        else:
            gpu_memory_str = 'gpu memory:%d MiB' % gpu_memory_list[i]
            sys.stdout.write('\r' + str(i) + ' ' + gpu_memory_str)  # 写入缓冲
            sys.stdout.flush()  # 输出并清空缓冲
            time.sleep(interval)
            i += 1
            if i % 4 == 0 and i != 0:  # 遍历一次GPU资源后重新查询
                i = 0
                gpu_memory_list = get_info()


if __name__ == '__main__':
    setup_python()
