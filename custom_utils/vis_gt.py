#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/11 11:49
# @Author : WeiHua
import json
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm


def get_poly_sort_idx(polys):
    def compare_key(x):
        #  x is (index, box), where box is list[x, y, x, y...]
        points = x[1]
        box = np.array(x[1], dtype=np.float32).reshape(-1, 2)
        rect = cv2.minAreaRect(box)
        center = rect[0]
        return center[1], center[0]
    to_sort_polys = [(idx, poly_) for idx, poly_ in enumerate(polys)]
    sorted_data = sorted(to_sort_polys, key=compare_key)
    sorted_idx = [x[0] for x in sorted_data]
    return sorted_idx


def vis_gt(src_dir, out_dir, anno_list):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for anno_file in anno_list:
        annos = []
        with open(os.path.join(src_dir, anno_file), 'r', encoding='utf-8') as f:
            for line_ in f.readlines():
                if line_.strip() == "":
                    continue
                annos.append(json.loads(line_.strip()))
        for anno_ in tqdm(annos):
            img = cv2.imread(os.path.join(src_dir, anno_['file_name']))
            out_file = anno_['file_name'].split('/')[-1].replace('jpg', 'txt')
            saver = open(os.path.join(out_dir, out_file), 'w', encoding='utf-8')
            to_sort_polys = [np.array(x['polygon']) for x in anno_['annotations']]
            sorted_idx = get_poly_sort_idx(to_sort_polys)
            for i, idx_ in enumerate(sorted_idx):
                poly_ = np.array(anno_['annotations'][idx_]['polygon'], np.int)
                cv2.polylines(img, [poly_.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                cv2.putText(img, str(i), (poly_[0], poly_[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                saver.write(f"\nidx:{idx_}\n")
                saver.write(f"TEXT:{anno_['annotations'][idx_]['text']}\n")
                if 'entity' not in anno_['annotations'][idx_]:
                    continue
                saver.write(f"_KIE:{anno_['annotations'][idx_]['entity']}\n")
            cv2.imwrite(os.path.join(out_dir, out_file.replace('txt', 'jpg')), img)
            saver.close()

if __name__ == '__main__':
    src_dir = r'E:\Dataset\KIE\SROIE\e2e_format'
    out_dir = r'E:\Dataset\KIE\SROIE\e2e_format\vis'
    anno_list = ['train_update.txt']
    vis_gt(src_dir, out_dir, anno_list)




