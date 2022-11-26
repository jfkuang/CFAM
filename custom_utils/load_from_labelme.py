#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/4 15:10
# @Author : WeiHua
import glob
import json
import os.path
import shutil

import cv2
import numpy as np
from tqdm import tqdm
import random


PRE_DEFINE_KEY = ['SS',
                  'CE-PS', 'CE-P1', 'CE-D', 'CE-PP',
                  'TF-PS', 'TF-P1', 'TF-D', 'TF-PP',
                  'SO-PS', 'SO-P1', 'SO-D', 'SO-PP',
                  'CAR-PS', 'CAR-P1', 'CAR-D', 'CAR-PP',
                  'PRO-PS', 'PRO-P1', 'PRO-D', 'PRO-PP']

KEP_MAPPING = {
        '：': ':',
        '（': '(',
        '）': ')'
    }


def clean_ocr(in_str: str):
    for key, val in KEP_MAPPING.items():
        in_str = in_str.replace(key, val)
    return in_str


def parse_key_info(label, anno_file):
    if '===' not in label:
        text = clean_ocr(label)
        entity = ['O'] * len(text)
        return text, entity
    # text===CLS===VAL===IDX===IDX===CLS===VAL===IDX===IDX
    info_ = label.split('===')
    assert len(info_) >= 5, f"Invalid anno: {label} \n file: {anno_file}"
    assert (len(info_)-1) % 4 == 0, f"Invalid anno: {label} \n file: {anno_file}"
    text = clean_ocr(info_[0])
    entity = ['O'] * len(text)
    kv_num = (len(info_) - 1) // 4
    for idx in range(kv_num):
        try:
            entity_cls = info_[4*idx+1].upper()
            entity_val = clean_ocr(info_[4*idx+2])
            pos_idx = int(info_[4*idx+3])
        except Exception as e:
            print(f"Invalid anno: {label} \n file: {anno_file}")
            raise RuntimeError(e)
        assert entity_cls in PRE_DEFINE_KEY and entity_val in text, \
            f"Invalid anno: {label} \n file: {anno_file}"
        tmp = text.split(entity_val)
        try:
            st_idx = len(entity_val.join(tmp[:(pos_idx + 1)]))
        except Exception as e:
            import ipdb
            ipdb.set_trace()
            print(tmp)
        end_idx = st_idx + len(entity_val)
        if end_idx > len(entity):
            raise RuntimeError(f"Invalid anno: {label} \n file: {anno_file}")
        for _ in range(st_idx, end_idx):
            assert entity[_] == 'O', f"Invalid anno: {label} \n file: {anno_file}"
        entity[st_idx] = f'B-{entity_cls}'
        for _ in range(st_idx+1, end_idx):
            entity[_] = f'I-{entity_cls}'
    return text, entity


def parse_key_dict(lm_annos, anno_file):
    kv_pair = dict()
    for anno in lm_annos:
        label = anno['label']
        if '===' not in label:
            continue
        info_ = label.split('===')
        assert len(info_) >= 5, f"Invalid anno: {label} \n file: {anno_file}"
        assert (len(info_) - 1) % 4 == 0, f"Invalid anno: {label} \n file: {anno_file}"
        text = clean_ocr(info_[0])
        kv_num = (len(info_) - 1) // 4
        for idx in range(kv_num):
            try:
                entity_cls = info_[4 * idx + 1].upper()
                entity_val = clean_ocr(info_[4 * idx + 2])
                pos_idx = int(info_[4 * idx + 3])
                tag_idx = int(info_[4 * idx + 4])
            except Exception as e:
                print(f"Invalid anno: {label} \n file: {anno_file}")
                raise RuntimeError(e)
            assert entity_cls in PRE_DEFINE_KEY and entity_val in text, \
                f"Invalid anno: {label} \n file: {anno_file}"
            if entity_cls not in kv_pair:
                kv_pair[entity_cls] = [[entity_val, tag_idx]]
            else:
                kv_pair[entity_cls].append([entity_val, tag_idx])
    kv_dict = {}
    assert len(kv_pair) > 0, f"Invalid file: {anno_file}"
    for key, val in kv_pair.items():
        sorted_val = sorted(val, key=lambda x: x[-1], reverse=False)
        pre_idx = -1
        val_full = ""
        for val_part in sorted_val:
            val_full += val_part[0]
            assert val_part[1] == pre_idx+1, f"Invalid file: {anno_file}, key:{key}, val:{val}"
            pre_idx = val_part[1]
        kv_dict[key] = val_full
    return kv_dict


def split_train_test(anno_list, train_ratio_hard=0.5, train_num=1500):
    # all_cls = []
    # for info_ in tqdm(anno_list):
    #     for ins_ in info_['annotations']:
    #         assert len(ins_['text']) == len(ins_['entity'])
    #         for entity_tag in ins_['entity']:
    #             if entity_tag[0] in ['I', 'B']:
    #                 cur_type = entity_tag[2:]
    #             else:
    #                 cur_type = 'O'
    #             if cur_type not in all_cls:
    #                 all_cls.append(cur_type)
    # print(f"true cls num: {len(all_cls)}, default cls num: {len(PRE_DEFINE_KEY)}")
    # import ipdb
    # ipdb.set_trace()

    # split annos to nf_v1, google and openfood
    nf_data = []
    google_data = []
    openfood_data = []
    for anno_ in anno_list:
        file_name = anno_['file_name'].split('/')[-1]
        if file_name[:2] == 'nf':
            nf_data.append(anno_)
        elif file_name[:3] == 'GOG':
            google_data.append(anno_)
        elif file_name.split('_')[0] in ['NZL', 'SGP', 'UK']:
            openfood_data.append(anno_)
        else:
            raise ValueError(f"Not match: {anno_}")
    random.shuffle(nf_data)
    random.shuffle(google_data)
    random.shuffle(openfood_data)
    print(f"After split:")
    print(f"NF_V1: {len(nf_data)}")
    print(f"Google: {len(google_data)}")
    print(f"OpenFood: {len(openfood_data)}")

    # split train & test
    train_list = []
    test_list = []
    # OpenFood
    num_train = int(len(openfood_data) * train_ratio_hard)
    train_list += openfood_data[:num_train]
    test_list += openfood_data[num_train:]
    print(f"OpenFood:")
    print(f"\ttrain:{num_train}, test:{len(openfood_data)-num_train}")
    # Google
    if len(google_data) > 0:
        num_train = int(len(google_data) * train_ratio_hard)
        train_list += google_data[:num_train]
        test_list += google_data[num_train:]
        print(f"Google:")
        print(f"\ttrain:{num_train}, test:{len(google_data) - num_train}")
    # NF-V1
    num_train = train_num - len(train_list)
    train_list += nf_data[:num_train]
    test_list += nf_data[num_train:]
    print(f"NF-V1:")
    print(f"\ttrain:{num_train}, test:{len(nf_data) - num_train}")

    # verify if train data covers all entity classes
    train_cls = []
    for info_ in tqdm(train_list):
        for ins_ in info_['annotations']:
            assert len(ins_['text']) == len(ins_['entity'])
            for entity_tag in ins_['entity']:
                if entity_tag[0] in ['I', 'B']:
                    cur_type = entity_tag[2:]
                else:
                    cur_type = 'O'
                if cur_type == 'O':
                    continue
                if cur_type not in train_cls:
                    train_cls.append(cur_type)
    assert len(train_cls) == len(PRE_DEFINE_KEY), f"{train_cls}"
    for cls in train_cls:
        if cls not in PRE_DEFINE_KEY:
            raise RuntimeError(f"Unexpected entity: {cls}")
    # verify if test data covers all entity classes
    test_cls = []
    for info_ in tqdm(test_list):
        for ins_ in info_['annotations']:
            assert len(ins_['text']) == len(ins_['entity'])
            for entity_tag in ins_['entity']:
                if entity_tag[0] in ['I', 'B']:
                    cur_type = entity_tag[2:]
                else:
                    cur_type = 'O'
                if cur_type == 'O':
                    continue
                if cur_type not in test_cls:
                    test_cls.append(cur_type)
    assert len(test_cls) == len(PRE_DEFINE_KEY), f"{test_cls}"
    for cls in test_cls:
        if cls not in PRE_DEFINE_KEY:
            raise RuntimeError(f"Unexpected entity: {cls}")
    # check if exist duplicate
    test_file_names = [x['file_name'] for x in test_list]
    for x in train_list:
        assert x['file_name'] not in test_file_names, f"Exists duplicate: {x['file_name']}"
    print(f"In total, Num_train:{len(train_list)}, Num_test:{len(test_list)}")
    return train_list, test_list


def load_labelme(dir_list, out_dir, default_dict=None,
                 train_ratio_hard=0.4, train_num=1650):
    """
    Load label-me format annotation and converted to required format.
    Args:
        dir_list: list of directories where store annotation files
        out_dir: output directory path

    Returns:

    """
    anno_files = []
    img_files = []
    for dir_ in dir_list:
        annos = glob.glob(os.path.join(dir_, '*.json'))
        annos = [x.replace('\\', '/') for x in annos]
        images = [os.path.join(dir_, f"image_files/{x.split('/')[-1].replace('json', 'jpg')}") for x in annos]
        images = [x.replace('\\', '/') for x in images]
        anno_files.extend(annos)
        img_files.extend(images)
    print(f"In total, load {len(img_files)} samples from {len(dir_list)} directories.")
    out_anno_list = []
    for i in tqdm(range(len(anno_files))):
        # pre-check file path
        anno_file = anno_files[i]
        img_file = img_files[i]
        img_name = img_file.split('/')[-1]
        assert os.path.exists(img_file), f"Missing: {img_file}"

        # load annotation
        with open(anno_file, 'r', encoding='utf-8') as f:
            anno_lm = json.load(f)
        img = cv2.imread(img_file)
        height, width, _ = img.shape
        assert height == anno_lm['imageHeight']
        assert width == anno_lm['imageWidth']

        out_anno = {
            'file_name': f"image_files/{img_name}",
            'height': height,
            'width': width,
            'annotations': [],
            'entity_dict': None
        }
        # convert annotation
        for anno in anno_lm['shapes']:
            text, entity = parse_key_info(anno['label'], anno_file)

            # for _ in entity:
            #     if _ == 'O':
            #         continue
            #     else:
            #         if 'O' in entity:
            #             import ipdb
            #             ipdb.set_trace()
            #             break

            polygon = anno['points']
            if len(polygon) == 2:
                assert anno['shape_type'] == 'rectangle'
                min_x, min_y = polygon[0]
                max_x, max_y = polygon[1]
                polygon = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            elif len(polygon) >= 2:
                # assert len(polygon) % 2 == 0, f"Invalid sample: {anno} \n of file: {anno_file}"
                polygon = np.array(polygon)
            else:
                raise RuntimeError(f"Invalid sample: {anno} \n of file: {anno_file}")

            ins_anno = {
                'polygon': polygon.reshape(-1).tolist(),
                'text': text,
                'entity': entity
            }
            out_anno['annotations'].append(ins_anno)

        out_anno['entity_dict'] = parse_key_dict(anno_lm['shapes'], anno_file)
        out_anno_list.append(out_anno)

    # check ocr dict
    if default_dict:
        ext_dict = []
        with open(default_dict, 'r', encoding='utf-8') as f:
            default_dict = json.load(f)
        for sample_ in out_anno_list:
            for anno in sample_['annotations']:
                for char_ in anno['text']:
                    if char_ not in default_dict:
                        if char_ not in ext_dict:
                            ext_dict.append(char_)
        print(f"External chars: {ext_dict}, total num: {len(ext_dict)}")

    # copy & paste image data
    print(f"Copy images ...")
    out_img_dir = os.path.join(out_dir, 'image_files')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.mkdir(out_img_dir)
    for img_file in tqdm(img_files):
        img_name = img_file.replace('\\', '/').split('/')[-1]
        shutil.copyfile(img_file, os.path.join(out_img_dir, img_name))

    # split train & test set with respect to data type
    max_tries = 10
    num_tries = 0
    while True:
        try:
            train_list, test_list = split_train_test(out_anno_list, train_ratio_hard=train_ratio_hard, train_num=train_num)
            print("Split train ~ test success.")
            break
        except Exception as e:
            num_tries += 1
            if num_tries > max_tries:
                raise RuntimeError(f"After trying {num_tries} times, still failed.")
            print(f"Error: {e} occurs, retry...")
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for info_ in train_list:
            out_str = json.dumps(info_, ensure_ascii=False)
            f.write(out_str + '\n')
    with open(os.path.join(out_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        for info_ in test_list:
            out_str = json.dumps(info_, ensure_ascii=False)
            f.write(out_str + '\n')
    with open(os.path.join(out_dir, 'class_list.json'), 'w', encoding='utf-8') as f:
        json.dump(PRE_DEFINE_KEY, f, ensure_ascii=False)
    print(f"Finish loading and saving dataset.")


def get_ocr_dict():
    # get ocr dict
    avg_width = 0
    avg_height = 0
    img_cnt = 0
    max_text_len = 0
    with open(r'E:\Dataset\Nutrition_Facts_Formal\V5_31_37/dict.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    ext_dict = []
    files = [r"E:\Dataset\Nutrition_Facts_Formal\V5_31_37/train.txt",
             r"E:\Dataset\Nutrition_Facts_Formal\V5_31_37/test.txt"]
    total_instance_num = 0
    max_instance_num = 0
    min_instance_num = 999
    avg_ins_width = 0
    avg_ins_height = 0
    max_pt_num = 0
    entity_cls_list = ['O'] + PRE_DEFINE_KEY
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                avg_height += info_['height']
                avg_width += info_['width']
                img_cnt += 1
                num_instance = len(info_['annotations'])
                total_instance_num += num_instance
                max_instance_num = max(max_instance_num, num_instance)
                min_instance_num = min(min_instance_num, num_instance)
                for anno in info_['annotations']:
                    max_pt_num = max(max_pt_num, len(anno['polygon'])//2)
                    assert len(anno['text']) == len(anno['entity'])
                    rect = cv2.minAreaRect(np.array(anno['polygon'], dtype=np.float32).reshape(-1, 2))
                    avg_ins_height += min(rect[1][0], rect[1][1])
                    avg_ins_width += max(rect[1][0], rect[1][1])
                    max_text_len = max(max_text_len, len(anno['text']))
                    if len(anno['text']) > 110:
                        print("\n")
                        print(info_['file_name'])
                        print(anno['text'])
                        print("\n")
                    for char_ in anno['text']:
                        if char_ not in default_dict:
                            if char_ not in ext_dict:
                                ext_dict.append(char_)
                    for entity_ in anno['entity']:
                        entity = entity_.replace('B-', '')
                        entity = entity.replace('I-', '')
                        assert entity in entity_cls_list, f"Entity:{entity_}"
    print(f"Total sample num: {img_cnt}")
    print(f"max_pt_num:{max_pt_num}")
    print(f"avg_ins_num:{total_instance_num / img_cnt}, max:{max_instance_num}, min:{min_instance_num}")
    full_key = default_dict + ext_dict
    print(f"avg_height:{avg_height / img_cnt}, avg_width:{avg_width / img_cnt}")
    print(f"ext key:{ext_dict}")
    print(f"max_text_len:{max_text_len}")
    print(f"avg_ins_height: {avg_ins_height/total_instance_num}")
    print(f"avg_ins_width: {avg_ins_width/total_instance_num}")
    """
    Total sample num: 3000
    max_pt_num:42
    avg_ins_num:37.05166666666667, max:206, min:1
    avg_height:566.795, avg_width:489.36466666666666
    ext key:[]
    max_text_len:122
    avg_ins_height: 26.73226101775652
    avg_ins_width: 92.61864199692633
    """


def vis_data():
    files = [r"E:\Dataset\Nutrition_Facts_Formal\V5_31_37/train.txt",
             r"E:\Dataset\Nutrition_Facts_Formal\V5_31_37/test.txt"]
    img_dir = r"E:\Dataset\Nutrition_Facts_Formal\V5_31_37"
    # specify_file = 'SGP_82.jpg'
    specify_file = None
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                if specify_file:
                    if specify_file not in info_['file_name']:
                        continue
                img = cv2.imread(os.path.join(img_dir, info_['file_name']))
                for idx, anno in enumerate(info_['annotations']):
                    poly = np.array(anno['polygon'], dtype=np.int)
                    cv2.polylines(img, [poly.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                    cv2.putText(img, str(idx), (poly[0], poly[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                    print(f"idx:{idx}, text:{anno['text']}, entity:{anno['entity']}")
                cv2.imshow('img', img)
                cv2.waitKey(0)


if __name__ == '__main__':
    dir_list = [
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\0\0',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\1\1_splited\image',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\2\2',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\3\3',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\4\4',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\5\5',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\6\6',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\7\7 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\8\8',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\9\9',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\10\10',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\11\11',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\12\12',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\13\image_13\image_13',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\14\14',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\15\15',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\16\16 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\17\17ie_ljt\17ie\17',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\18\18 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\19\19 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\20\20 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\21\image_21\image_21',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\22\22 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\23\23ie_ljt\23ie\23',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\24_full\data',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\25\25 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\26\26 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\27\27 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\28\28 (2)',
        r'E:\Dataset\Nutrition_Facts_Formal\V4\CHECKED\29\29 (2)'
    ]
    # out_dir = r"E:\Dataset\Nutrition_Facts_Formal/V5_2200"
    out_dir = r"E:\Dataset\Nutrition_Facts_Formal/V5_31_28"
    default_dict = r'E:\Dataset\Nutrition_Facts_Formal\e2e_format/dict.json'
    load_labelme(dir_list, out_dir, default_dict, train_ratio_hard=0.2, train_num=2250)

    # get_ocr_dict()

    # vis_data()
