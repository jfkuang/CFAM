#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/2 15:18
# @Author : WeiHua
import os
import shutil
import json
import glob
import cv2
import numpy as np
from tqdm import tqdm

class_list = []
# todo: crop the dataset
def convert_e2e_format(src_dir, out_img_dir, out_file, name_prefix='train', cnt_prefix=0):
    imgs = glob.glob(os.path.join(src_dir, 'image/*.png'))
    for idx, img_file_ in tqdm(enumerate(imgs, cnt_prefix)):
        img_file = img_file_.replace('\\', '/')
        img_name = img_file.split('/')[-1]
        img = cv2.imread(img_file)
        height, width = img.shape[:2]
        json_file = img_name.replace('png', 'json')
        with open(os.path.join(src_dir, f'json/{json_file}'), 'r', encoding='utf-8') as f:
            anno = json.load(f)
        assert anno['meta']['image_size']['width'] == width
        assert anno['meta']['image_size']['height'] == height
        if len(anno['roi']) != 0:
            # using roi to crop image
            roi = []
            for i in range(1, 5):
                roi.append(anno['roi'][f'x{i}'])
                roi.append(anno['roi'][f'y{i}'])
            tl_x = min(roi[0::2])
            tl_y = min(roi[1::2])
            br_x = max(roi[0::2]) + 2
            br_y = max(roi[1::2]) + 2

            tl_x = max(tl_x, 0)
            tl_y = max(tl_y, 0)
            br_x = min(br_x, width)
            br_y = min(br_y, height)
            img = img[tl_y:br_y, tl_x:br_x, :]
            height, width = img.shape[:2]
        else:
            tl_x = 0
            tl_y = 0
        out_img_name = f"{name_prefix}_{idx}.png"
        out_info = dict(
            file_name=f"image_files/{out_img_name}",
            height=height,
            width=width,
            annotations=[]
        )
        for item_ in anno['valid_line']:
            category = item_['category']
            if category not in class_list:
                class_list.append(category)
            for word in item_['words']:
                if len(word['text']) == 0:
                    continue
                quad = []
                for i in range(1, 5):
                    quad.append(word['quad'][f'x{i}'])
                    quad.append(word['quad'][f'y{i}'])
                quad = np.array(quad, np.int)
                quad[0::2] -= tl_x
                quad[1::2] -= tl_y
                quad = quad.tolist()
                entity = []
                for char_idx in range(len(word['text'])):
                    if char_idx == 0:
                        entity.append(f"B-{category}")
                    else:
                        entity.append(f"I-{category}")
                txt = dict(
                    polygon=quad,
                    text=word['text'],
                    entity=entity)
                out_info['annotations'].append(txt)
        out_str = json.dumps(out_info)
        out_file.write(out_str+"\n")
        cv2.imwrite(os.path.join(out_img_dir, out_img_name), img)
        # shutil.copyfile(img_file, os.path.join(out_img_dir, out_img_name))

    return len(imgs)+cnt_prefix


def prepare_cord(src_dir, out_dir, data_type='train'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_img_dir = os.path.join(out_dir, 'image_files')
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    out_file = open(os.path.join(out_dir, f"{data_type}.txt"), 'w', encoding='utf-8')
    if isinstance(src_dir, list):
        cnt_prefix = 0
        for idx, dir_ in enumerate(src_dir):
            cnt_prefix = convert_e2e_format(dir_, out_img_dir, out_file,
                                            name_prefix=data_type,
                                            cnt_prefix=cnt_prefix)
    elif isinstance(src_dir, str):
        _ = convert_e2e_format(src_dir, out_img_dir, out_file,
                               name_prefix=data_type)
    else:
        raise ValueError(f"Unsupported input of src_dir:{src_dir}")
    out_file.close()
    with open(os.path.join(out_dir, f"{data_type}_class.json"), 'w', encoding='utf-8') as f:
        json.dump(class_list, f, ensure_ascii=False)

def vis_cord():
    # get ocr dict
    with open('../dict_default.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    # files = ['E:/Dataset/KIE/CORD/e2e_format/test.txt']
    files = ['E:/Dataset/KIE/CORD/e2e_format/train.txt']
    img_dir = 'E:/Dataset/KIE/CORD/e2e_format'
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                img = cv2.imread(os.path.join(img_dir, info_['file_name']))
                for idx, anno in enumerate(info_['annotations']):
                    poly = np.array(anno['polygon'], dtype=np.int)
                    cv2.polylines(img, [poly.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                    cv2.putText(img, str(idx), (poly[0], poly[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    print(f"idx:{idx}, text:{anno['text']}, entity:{anno['entity']}")
                    assert len(anno['text']) == len(anno['entity'])
                cv2.imshow('img', img)
                cv2.waitKey(0)

if __name__ == '__main__':
    # src_dir = 'E:/Dataset/KIE/CORD/test'
    # # src_dir = ['E:/Dataset/KIE/CORD/train', 'E:/Dataset/KIE/CORD/dev']
    # out_dir = 'E:/Dataset/KIE/CORD/e2e_format'
    # data_type = 'test'
    # # data_type = 'train'
    # prepare_cord(src_dir, out_dir, data_type=data_type)
    # print(f"class_list:{class_list}")

    # vis_cord()

    # # validate entity class
    # with open('E:/Dataset/KIE/CORD/e2e_format/test_class.json', 'r', encoding='utf-8') as f:
    #     test_classes = json.load(f)
    # with open('E:/Dataset/KIE/CORD/e2e_format/train_class.json', 'r', encoding='utf-8') as f:
    #     train_classes = json.load(f)
    # print(len(train_classes), len(test_classes))
    # assert len(test_classes) <= len(train_classes)
    # for cls in test_classes:
    #     assert cls in train_classes, f"cls:{cls}"

    # get ocr dict
    avg_width = 0
    avg_height = 0
    img_cnt = 0
    max_text_len = 0
    with open('../dict_default.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    ext_dict = []
    files = ['E:/Dataset/KIE/CORD/e2e_format/train.txt', 'E:/Dataset/KIE/CORD/e2e_format/test.txt']
    max_num = 0
    min_num = 999
    avg_num = 0
    avg_ins_width = 0
    avg_ins_height = 0
    total_instance_num = 0
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                avg_height += info_['height']
                avg_width += info_['width']
                img_cnt += 1
                num_ins = len(info_['annotations'])
                max_num = max(max_num, num_ins)
                min_num = min(min_num, num_ins)
                avg_num += num_ins
                for anno in info_['annotations']:
                    avg_ins_width += (max(anno['polygon'][0::2]) - min(anno['polygon'][0::2]))
                    avg_ins_height += (max(anno['polygon'][1::2]) - min(anno['polygon'][1::2]))
                    max_text_len = max(max_text_len, len(anno['text']))
                    for char_ in anno['text']:
                        if char_ not in default_dict:
                            if char_ not in ext_dict:
                                ext_dict.append(char_)
    full_key = default_dict + ext_dict
    print(f"avg_height:{avg_height/img_cnt}, avg_width:{avg_width/img_cnt}")
    print(f"max_num:{max_num}, min_num:{min_num}, avg:{avg_num/img_cnt}")
    print(f"ext key:{ext_dict}")
    print(f"max_len:{max_text_len}")
    print(f"avg_ins_height:{avg_ins_height / avg_num}, avg_ins_width:{avg_ins_width / avg_num}")
    # # with open('E:/Dataset/KIE/CORD/e2e_format/dict.json', 'w', encoding='utf-8') as f:
    # #     json.dump(full_key, f, ensure_ascii=False)

    """
    avg_height:1488.863, avg_width:930.88
    max_num:135, min_num:5, avg:23.912
    ext key:[]
    max_len:32
    avg_ins_height:38.95997825359652, avg_ins_width:96.0955587152894
    """