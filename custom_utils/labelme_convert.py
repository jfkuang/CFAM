#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/5/26 16:16
# @Author : WeiHua

import os
import shutil
import json
from tqdm import tqdm
import cv2
import numpy as np

from functools import cmp_to_key


"""
labelme format:
text===Name_of_Key_1===Val_of_Key_1===index of this instance in sentence===index of this instance in entity value===...
"""
def merge_to_labelme(anno_files, out_dir, img_path, store_interval=100):
    cls_list = ["CE-P1", "CE-PS", "CE-D", "TF-P1", "TF-PS", "TF-D", "CAR-P1", "CAR-PS", "CAR-D", "PRO-P1", "PRO-PS", "PRO-D", "SS", "SO-P1", "SO-PS", "SO-D"]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    annos = []
    for anno_file in anno_files:
        with open(anno_file, 'r', encoding='utf-8') as f:
            for line_ in f.readlines():
                if line_.strip():
                    annos.append(json.loads(line_.strip()))
    for idx, item_ in tqdm(enumerate(annos)):
        save_dir = os.path.join(out_dir, str(idx//store_interval))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_img_dir = os.path.join(save_dir, "image_files")
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
        file_name = item_['file_name'].split("/")[-1][:-4]
        img = cv2.imread(os.path.join(img_path, item_['file_name']))
        out_dict = {
            "version": "4.5.13",
            "flags": {},
            "shapes": [],
            "imagePath": f"image_files/{item_['file_name'].split('/')[-1]}",
            "imageData": None,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1]
        }
        sorted_idx = get_poly_sort_idx([x['polygon'] for x in item_['annotations']])
        appear_entity = dict()
        for sort_idx in sorted_idx:
            instance = item_['annotations'][sort_idx]
            assert len(instance['entity']) == len(instance['text'])
            label = instance['text']

            cur_entity = None
            cur_entity_val = ""
            for local_idx, tag in enumerate(instance['entity']):
                if tag == 'O':
                    continue
                assert tag[:2] in ["B-", "I-"]
                if not cur_entity:
                    cur_entity = tag[2:]
                    assert cur_entity in cls_list, f"Out of entity list: {cur_entity}"
                    cur_entity_val = instance['text'][local_idx]
                else:
                    if cur_entity != tag[2:]:
                        assert len(cur_entity_val) > 0, f"Invalid annotation:{instance}"
                        index_in_sentence = instance['text'][:local_idx].count(cur_entity_val)-1
                        if cur_entity in appear_entity:
                            appear_entity[cur_entity] += 1
                        else:
                            appear_entity[cur_entity] = 1
                        index_in_entity = appear_entity[cur_entity] - 1
                        add_info = f"{cur_entity}==={cur_entity_val}==={index_in_sentence}==={index_in_entity}"
                        label += f"==={add_info}"
                        cur_entity = tag[2:]
                        cur_entity_val = instance['text'][local_idx]
                    else:
                        cur_entity_val += instance['text'][local_idx]

            if cur_entity:
                assert len(cur_entity_val) > 0, f"Invalid annotation:{instance}"
                index_in_sentence = instance['text'].count(cur_entity_val) - 1
                if cur_entity in appear_entity:
                    appear_entity[cur_entity] += 1
                else:
                    appear_entity[cur_entity] = 1
                index_in_entity = appear_entity[cur_entity] - 1
                add_info = f"{cur_entity}==={cur_entity_val}==={index_in_sentence}==={index_in_entity}"
                label += f"==={add_info}"

            cur_shape = {
                "label": label,
                "points": np.array(instance['polygon']).reshape(-1, 2).tolist(),
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            out_dict["shapes"].append(cur_shape)
        with open(os.path.join(save_dir, f"{file_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(out_dict, f, ensure_ascii=False)
        shutil.copyfile(os.path.join(img_path, item_['file_name']), os.path.join(save_img_dir, f"{file_name}.jpg"))


def get_poly_sort_idx(polys):
    def compare_key(a, b):
        box = np.array(a[1], dtype=np.float32).reshape(-1, 2)
        rect = cv2.minAreaRect(box)
        a_center_x = rect[0][0]
        a_center_y = rect[0][1]
        a_box_h = min(rect[1][0], rect[1][1])

        box = np.array(b[1], dtype=np.float32).reshape(-1, 2)
        rect = cv2.minAreaRect(box)
        b_center_x = rect[0][0]
        b_center_y = rect[0][1]
        b_box_h = min(rect[1][0], rect[1][1])
        if a_center_y > b_center_y:
            if (a_center_y - b_center_y) > 0.5 * min(a_box_h, b_box_h):
                return 1
            elif a_center_x > b_center_x:
                return 1
            else:
                return -1
        else:
            if (b_center_y - a_center_y) > 0.5 * min(a_box_h, b_box_h):
                return -1
            elif a_center_x > b_center_x:
                return 1
            else:
                return -1

    to_sort_polys = [(idx, poly_) for idx, poly_ in enumerate(polys)]
    sorted_data = sorted(to_sort_polys, key=cmp_to_key(compare_key))
    return [x[0] for x in sorted_data]


if __name__ == '__main__':
    anno_files = ["E:/Dataset/Nutrition_Facts_Formal/V4/e2e_format/train_regular.txt",
                  "E:/Dataset/Nutrition_Facts_Formal/V4/e2e_format/test_regular.txt"]
    out_dir = "E:/Dataset/Nutrition_Facts_Formal/V4/temp"
    img_path = "E:/Dataset/Nutrition_Facts_Formal/V4/e2e_format"
    merge_to_labelme(anno_files, out_dir, img_path, store_interval=100)
