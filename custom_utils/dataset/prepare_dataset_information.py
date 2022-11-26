#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：prepare_dataset_information.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/10/19 20:16 
'''
import json
import ipdb
train_path = "/home/whua/dataset/ie_e2e_dataset/nfv5_3125/train.txt"
test_path = "/home/whua/dataset/ie_e2e_dataset/nfv5_3125/test.txt"
f = open(train_path, 'r')
f1 = open(test_path, 'r')
save_path = "/home/jfkuang/nf_anantation.txt"
train_datas = f.readlines()
# train_nums = 0
train_dict = {'SS':0, 'CE-PS':0, 'CE-P1':0, 'CE-D':0, 'CE-PP':0, 'TF-PS':0, 'TF-P1':0, 'TF-D':0, 'TF-PP':0, 'SO-PS':0, 'SO-P1':0, 'SO-D':0, 'SO-PP':0, 'CAR-PS':0, 'CAR-P1':0, 'CAR-D':0, 'CAR-PP':0, 'PRO-PS':0, 'PRO-P1':0, 'PRO-D':0, 'PRO-PP':0}
for data in train_datas:
    # ipdb.set_trace()
    # train_nums += len(json.loads(data)['annotations'])
    data = json.loads(data)
    for key in data['entity_dict']:
        ipdb.set_trace()
        train_dict[key] = train_dict[key]+1


test_datas = f1.readlines()
# test_nums = 0
test_dict = {'SS':0, 'CE-PS':0, 'CE-P1':0, 'CE-D':0, 'CE-PP':0, 'TF-PS':0, 'TF-P1':0, 'TF-D':0, 'TF-PP':0, 'SO-PS':0, 'SO-P1':0, 'SO-D':0, 'SO-PP':0, 'CAR-PS':0, 'CAR-P1':0, 'CAR-D':0, 'CAR-PP':0, 'PRO-PS':0, 'PRO-P1':0, 'PRO-D':0, 'PRO-PP':0}
for data in test_datas:
    # ipdb.set_trace()
    # test_nums += len(json.loads(data)['annotations'])
    data = json.loads(data)
    for key in data['entity_dict']:
        test_dict[key] = test_dict[key] + 1

# print(train_nums)
# print(test_nums)
# print(train_nums+test_nums)
#
# save_file = open(save_path, 'w')
# save_file.write("train_dict:\n")
# for k, v in train_dict.items():
#     save_file.write(k + ':' + str(v))
#     save_file.write(',')
# print("\n")
# save_file.write("test_dict:\n")
# for k, v in test_dict.items():
#     save_file.write(k + ':' + str(v))
#     save_file.write(',')
# print("train_dict:")
# for key in train_dict:
#     print(key+':'+str(train_dict[key]))
# print('\n')
# print("test_dict:")
# for key in test_dict:
#     print(key+':'+str(test_dict[key]))
# print('\n')

#NFV5:  train:85157 test:25998 total:111155
        #entity分布：
        #train:SS:2164,CE-PS:2171,CE-P1:235,CE-D:40,CE-PP:5,TF-PS:2171,TF-P1:224,TF-D:2112,TF-PP:94,SO-PS:2121,SO-P1:124,SO-D:2083,SO-PP:87,CAR-PS:2164,CAR-P1:226,CAR-D:2105,CAR-PP:91,PRO-PS:2160,PRO-P1:226,PRO-D:418,PRO-PP:18
        #test:SS:494,CE-PS:548,CE-P1:439,CE-D:96,CE-PP:9,TF-PS:545,TF-P1:428,TF-D:401,TF-PP:23,SO-PS:405,SO-P1:113,SO-D:334,SO-PP:13,CAR-PS:537,CAR-P1:426,CAR-D:380,CAR-PP:18,PRO-PS:533,PRO-P1:429,PRO-D:142,PRO-PP:8

#SROIE: train:33614 test:18702 total:52316
#CORD:  train:21556  test:2356  total:23912
#EPHOIE: train:12411 test:3343  total:15754