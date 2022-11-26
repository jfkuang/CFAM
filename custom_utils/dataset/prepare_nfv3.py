#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/6 10:57
# @Author : WeiHua
import os
import shutil

import cv2
import json
import glob
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon
import random


def prepare_nfv3(nfv1_dir, nfv3_dir, out_dir, train_split, save=True):
    out_img_dir = os.path.join(out_dir, 'image_files')
    if save:
        if os.path.exists(out_img_dir):
            shutil.rmtree(out_img_dir)
        os.mkdir(out_img_dir)
    entity_list = ['O', 'SS',
                   'CE-PS', 'CE-P1', 'CE-D',
                   'TF-PS', 'TF-P1', 'TF-D',
                   'SF-PS', 'SF-P1', 'SF-D',
                   'SO-PS', 'SO-P1', 'SO-D',
                   'CAR-PS', 'CAR-P1', 'CAR-D',
                   'PRO-PS', 'PRO-P1', 'PRO-D']
    entity_to_rm = ['SF-PS', 'SF-P1', 'SF-D']
    entity_map = {
        "Serving size": "SS",
        "Calories": "CE-PS",
        "Fat_aps": "TF-PS",
        "Fat_dv": "TF-D",
        "Sodium_aps": "SO-PS",
        "Sodium_dv": "SO-D",
        "Carb._aps": "CAR-PS",
        "Carb._dv": "CAR-D",
        "Protein": "PRO-PS"
    }
    # load v3 first
    tmp_v3 = glob.glob(os.path.join(nfv3_dir, '*.json'))
    data_v3 = {}
    v3_not_exists = []
    for file_ in tqdm(tmp_v3):
        cur_file_name = file_.replace('\\', '/').split('/')[-1].replace('.json', '')
        if not os.path.exists(os.path.join(nfv3_dir, f'imgs/{cur_file_name}.jpg')):
            v3_not_exists.append(cur_file_name)
            continue
        # assert os.path.exists(os.path.join(nfv3_dir, f'imgs/{cur_file_name}.jpg')), f"{cur_file_name} not exists"
        with open(file_, 'r', encoding='utf-8') as f:
            data_v3[cur_file_name] = json.load(f)
    print(f"Load {len(data_v3)} from v3, {len(v3_not_exists)} not exists, pass them.")
    # load files those not exist in v3 from v1
    anns = ['train.txt', 'test.txt']
    data_v1 = dict()
    v1_duplicate = []
    for ann in anns:
        with open(os.path.join(nfv1_dir, ann), 'r', encoding='utf-8') as f:
            for line_ in f.readlines():
                tmp = line_.strip()
                if not tmp:
                    continue
                info_ = json.loads(tmp)
                file_name = info_['file_name'].split('/')[-1].replace('.jpg', '')
                assert file_name not in data_v1.keys(), f"file {file_name} appears multiple times."
                if file_name in data_v3.keys():
                    v1_duplicate.append(file_name)
                    continue
                data_v1[file_name] = info_
    print(f"Load {len(data_v1)} from v1, {len(v1_duplicate)} already exist in v3, ignore them.")
    # Add '%' to certain entity value for v1, and copy image
    added_cnt = 0
    for key in data_v1.keys():
        if save:
            shutil.copyfile(os.path.join(nfv1_dir, f'image_files/{key}.jpg'), os.path.join(out_img_dir, f'{key}.jpg'))
        for i in range(len(data_v1[key]['annotations'])):
            cur_entity = data_v1[key]['annotations'][i]['entity']
            cur_text = data_v1[key]['annotations'][i]['text']
            # add '%' to certain entity -> number%
            if cur_text[-1] == '%' and cur_entity[-1] == 'O' and cur_entity[0] != 'O':
                is_same_entity = True
                tmp = cur_entity[0][2:]
                for ent_idx in range(len(cur_text)-1):
                    cur_cls = cur_entity[ent_idx]
                    if cur_cls == 'O':
                        is_same_entity = False
                        break
                    if cur_cls[2:] != tmp:
                        is_same_entity = False
                        break
                    if cur_text[ent_idx] < '0' or cur_text[ent_idx] > '9':
                        is_same_entity = False
                        break
                if is_same_entity:
                    added_cnt += 1
                    cur_entity[-1] = 'I-' + tmp
            # replace to v3-format entity class
            for ent_idx in range(len(cur_entity)):
                if cur_entity[ent_idx][0] in ['I', 'B']:
                    entity_type = entity_map[cur_entity[ent_idx][2:]]
                    assert entity_type in entity_list
                    if entity_type in entity_to_rm:
                        new_entity_type = 'O'
                    else:
                        new_entity_type = cur_entity[ent_idx][:2] + entity_type
                    data_v1[key]['annotations'][i]['entity'][ent_idx] = new_entity_type
    print(f"Match {added_cnt} '%' flag to certain entity.")
    # Convert v3 to e2e-format with select entity classes
    max_num_kv = 0
    out_info_v3 = list()
    unmatch_list = []
    fail_messages = []
    for file_name, anno in tqdm(data_v3.items()):
        fail_flag = False
        fail_msg = ""
        img_ = cv2.imread(os.path.join(nfv3_dir, f'imgs/{file_name}.jpg'))
        height, width, _ = img_.shape
        tl_x = tl_y = 0
        br_x = br_y = None
        table_poly = None
        # crop table
        for instance_ in anno['shapes']:
            if instance_['label'] == 'FULL-TABLE':
                table_poly = np.array(instance_['points'], dtype=np.int)
                tl_x = max(int(min(table_poly[:, 0])) - 1, 0)
                br_x = min(int(max(table_poly[:, 0])) + 1, width)
                tl_y = max(int(min(table_poly[:, 1])) - 1, 0)
                br_y = min(int(max(table_poly[:, 1])) + 1, height)
                table_poly[:, 0] -= tl_x
                table_poly[:, 1] -= tl_y
                break
        if not isinstance(table_poly, type(None)):
            img = img_[tl_y: br_y, tl_x: br_x, :]
            mask = np.zeros(img.shape, img.dtype)
            cv2.fillPoly(mask, [table_poly], color=(255, 255, 255))
            img = np.bitwise_and(img, mask)
        else:
            img = img_
        height, width, _ = img.shape
        blur_area_whole = np.zeros(img.shape, img.dtype)
        poly_to_blur = []
        output_dict = {
            'file_name': f"image_files/{file_name}.jpg",
            'height': height,
            'width': width,
            'annotations': []
        }
        # mask ### and convert entity label
        for instance_ in anno['shapes']:
            poly = np.array(instance_['points'], dtype=np.int)
            poly[:, 0] -= tl_x
            poly[:, 1] -= tl_y
            if not isinstance(table_poly, type(None)):
                if not is_a_inside_b(poly, table_poly):
                    continue
            # blur specific area and convert label
            if instance_['label'] == '###':
                ins_tl_x = max(int(min(poly[:, 0])), 0)
                ins_br_x = min(int(max(poly[:, 0])), width)
                ins_tl_y = max(int(min(poly[:, 1])), 0)
                ins_br_y = min(int(max(poly[:, 1])), height)
                if ins_tl_x >= width or ins_tl_y >= height:
                    continue
                poly_to_blur.append(poly)

                blur_mask = np.zeros(img[ins_tl_y: ins_br_y, ins_tl_x: ins_br_x, :].shape, img.dtype)
                new_poly = poly.copy()
                new_poly[:, 0] -= ins_tl_x
                new_poly[:, 1] -= ins_tl_y
                cv2.fillPoly(blur_mask, [new_poly], color=(255, 255, 255))
                blur_area_whole[ins_tl_y: ins_br_y, ins_tl_x: ins_br_x, :] = blur_mask
                continue
            content = instance_['label']
            output_ins = {
                'polygon': poly.reshape(-1).tolist()
            }
            if '===' in content:
                info_ = content.split('===')
                assert len(info_) % 2 == 1 and len(info_) > 1, f"content:{content}, id:{file_name}"
                texts = info_[0]
                entity_tags = ['O' for _ in range(len(texts))]
                kv_pair_num = len(info_) // 2
                max_num_kv = max(kv_pair_num, max_num_kv)
                offset_idx = 0
                for i in range(kv_pair_num):
                    if info_[2*(i+1)] not in texts[offset_idx:]:
                        if len(fail_msg) > 0:
                            fail_msg += f" ||| to_match:{info_[2 * (i + 1)]}, search_space:{texts[offset_idx:]}, "
                            fail_msg += f"All:{texts}, all_line:{content} ||"
                        else:
                            fail_msg += f"||file name: {file_name} ||"
                            fail_msg += f" to_match:{info_[2*(i+1)]}, search_space:{texts[offset_idx:]}, "
                            fail_msg += f"All:{texts}, all_line:{content} ||"
                        # if 'GOG_116' in file_name:
                        #     print(f"to_match:{info_[2*(i+1)]}, search_space:{texts[offset_idx:]}")
                        #     print(f"All:{texts}, all_line:{content}, offset_idx:{offset_idx}")
                        #     return
                        fail_flag = True
                        break
                    # assert info_[2*(i+1)] in texts[offset_idx:], f"origin:{texts}, cur:{texts[offset_idx:]}, target:{info_[2*(i+1)]}, file:{file_name}"
                    st_idx = texts[offset_idx:].find(info_[2*(i+1)])
                    for j in range(offset_idx+st_idx, offset_idx+st_idx+len(info_[2*(i+1)])):
                        if j == (offset_idx+st_idx):
                            entity_tags[j] = f'B-{info_[2*i+1]}'
                        else:
                            entity_tags[j] = f'I-{info_[2*i+1]}'
                    offset_idx = offset_idx+st_idx+len(info_[2*(i+1)])
                output_ins['text'] = texts
                output_ins['entity'] = entity_tags
                if fail_flag:
                    break
            else:
                output_ins['text'] = content
                output_ins['entity'] = ['O' for _ in range(len(content))]
            output_dict['annotations'].append(output_ins)
        if fail_flag:
            unmatch_list.append(file_name)
            fail_messages.append(fail_msg)
            continue
        if len(poly_to_blur) > 0:
            img = np.bitwise_or(img, blur_area_whole)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        out_info_v3.append(output_dict)
        if save:
            cv2.imwrite(os.path.join(out_img_dir, f"{file_name}.jpg"), img)
    # merge v1 & v3, then split to train & test, for OpenFood, NF, Google respectively
    print(f"Un-match exists in {len(unmatch_list)} / {len(data_v3)}, ignore them")
    # with open(os.path.join(out_dir, 'unmatch_msg.json'), 'w', encoding='utf-8') as f:
    #     json.dump(fail_messages, f, ensure_ascii=False, indent=4)
    if save:
        with open(os.path.join(out_dir, 'unmatch_list.json'), 'w', encoding='utf-8') as f:
            json.dump(unmatch_list, f)
    print(f"Current, valid v3:{len(out_info_v3)}, v1:{len(data_v1)}")

    # split to train & validate
    nf_data = []
    openfood_data = []
    google_data = []
    for key, val in data_v1.items():
        nf_data.append(val)
    for val in out_info_v3:
        file_name = val['file_name'].split('/')[-1]
        if file_name[:2] == 'nf':
            nf_data.append(val)
        elif file_name[:3] == 'GOG':
            google_data.append(val)
        elif file_name.split('_')[0] in ['NZL', 'SGP', 'UK']:
            openfood_data.append(val)
    if save:
        with open(os.path.join(out_dir, 'nf.txt'), 'w', encoding='utf-8') as saver:
            for val in nf_data:
                out_str = json.dumps(val)
                saver.write(out_str+'\n')
        with open(os.path.join(out_dir, 'openfood.txt'), 'w', encoding='utf-8') as saver:
            for val in openfood_data:
                out_str = json.dumps(val)
                saver.write(out_str+'\n')
        with open(os.path.join(out_dir, 'google.txt'), 'w', encoding='utf-8') as saver:
            for val in google_data:
                out_str = json.dumps(val)
                saver.write(out_str+'\n')
    print(f"Save: nf:{len(nf_data)}, open_food:{len(openfood_data)}, Google:{len(google_data)}")


def prepare_nfv3_only(src_dir, out_dir, save=True):
    out_img_dir = os.path.join(out_dir, 'image_files')
    if save:
        if os.path.exists(out_img_dir):
            shutil.rmtree(out_img_dir)
        os.mkdir(out_img_dir)
    entity_list = ['O', 'SS',
                   'CE-PS', 'CE-P1', 'CE-D',
                   'TF-PS', 'TF-P1', 'TF-D',
                   'SF-PS', 'SF-P1', 'SF-D',
                   'SO-PS', 'SO-P1', 'SO-D',
                   'CAR-PS', 'CAR-P1', 'CAR-D',
                   'PRO-PS', 'PRO-P1', 'PRO-D']
    entity_to_rm = ['SF-PS', 'SF-P1', 'SF-D']
    # load v3 first
    tmp_v3 = glob.glob(os.path.join(src_dir, '*.json'))
    data_v3 = {}
    v3_not_exists = []
    for file_ in tqdm(tmp_v3):
        cur_file_name = file_.replace('\\', '/').split('/')[-1].replace('.json', '')
        if not os.path.exists(os.path.join(src_dir, f'imgs/{cur_file_name}.jpg')):
            v3_not_exists.append(cur_file_name)
            continue
        # assert os.path.exists(os.path.join(nfv3_dirsrc_dir, f'imgs/{cur_file_name}.jpg')), f"{cur_file_name} not exists"
        with open(file_, 'r', encoding='utf-8') as f:
            data_v3[cur_file_name] = json.load(f)
    print(f"Load {len(data_v3)} from v3, {len(v3_not_exists)} not exists, pass them.")
    # Convert v3 to e2e-format with select entity classes
    max_num_kv = 0
    out_info_v3 = list()
    unmatch_list = []
    fail_messages = []
    for file_name, anno in tqdm(data_v3.items()):
        fail_flag = False
        fail_msg = ""
        img_ = cv2.imread(os.path.join(src_dir, f'imgs/{file_name}.jpg'))
        height, width, _ = img_.shape
        tl_x = tl_y = 0
        br_x = br_y = None
        table_poly = None
        # crop table
        for instance_ in anno['shapes']:
            if instance_['label'] == 'FULL-TABLE':
                table_poly = np.array(instance_['points'], dtype=np.int)
                tl_x = max(int(min(table_poly[:, 0])) - 1, 0)
                br_x = min(int(max(table_poly[:, 0])) + 1, width)
                tl_y = max(int(min(table_poly[:, 1])) - 1, 0)
                br_y = min(int(max(table_poly[:, 1])) + 1, height)
                table_poly[:, 0] -= tl_x
                table_poly[:, 1] -= tl_y
                break
        if not isinstance(table_poly, type(None)):
            img = img_[tl_y: br_y, tl_x: br_x, :]
            mask = np.zeros(img.shape, img.dtype)
            cv2.fillPoly(mask, [table_poly], color=(255, 255, 255))
            img = np.bitwise_and(img, mask)
        else:
            img = img_
        height, width, _ = img.shape
        blur_area_whole = np.zeros(img.shape, img.dtype)
        poly_to_blur = []
        output_dict = {
            'file_name': f"image_files/{file_name}.jpg",
            'height': height,
            'width': width,
            'annotations': []
        }
        # mask ### and convert entity label
        for instance_ in anno['shapes']:
            poly = np.array(instance_['points'], dtype=np.int)
            poly[:, 0] -= tl_x
            poly[:, 1] -= tl_y
            if not isinstance(table_poly, type(None)):
                if not is_a_inside_b(poly, table_poly):
                    continue
            # blur specific area and convert label
            if instance_['label'] == '###':
                ins_tl_x = max(int(min(poly[:, 0])), 0)
                ins_br_x = min(int(max(poly[:, 0])), width)
                ins_tl_y = max(int(min(poly[:, 1])), 0)
                ins_br_y = min(int(max(poly[:, 1])), height)
                if ins_tl_x >= width or ins_tl_y >= height:
                    continue
                poly_to_blur.append(poly)

                blur_mask = np.zeros(img[ins_tl_y: ins_br_y, ins_tl_x: ins_br_x, :].shape, img.dtype)
                new_poly = poly.copy()
                new_poly[:, 0] -= ins_tl_x
                new_poly[:, 1] -= ins_tl_y
                cv2.fillPoly(blur_mask, [new_poly], color=(255, 255, 255))
                blur_area_whole[ins_tl_y: ins_br_y, ins_tl_x: ins_br_x, :] = blur_mask
                continue
            content = instance_['label']
            output_ins = {
                'polygon': poly.reshape(-1).tolist()
            }
            if '===' in content:
                info_ = content.split('===')
                assert len(info_) % 2 == 1 and len(info_) > 1, f"content:{content}, id:{file_name}"
                texts = info_[0]
                entity_tags = ['O' for _ in range(len(texts))]
                kv_pair_num = len(info_) // 2
                max_num_kv = max(kv_pair_num, max_num_kv)
                offset_idx = 0
                for i in range(kv_pair_num):
                    if info_[2 * (i + 1)] not in texts[offset_idx:]:
                        if len(fail_msg) > 0:
                            fail_msg += f" ||| to_match:{info_[2 * (i + 1)]}, search_space:{texts[offset_idx:]}, "
                            fail_msg += f"All:{texts}, all_line:{content} ||"
                        else:
                            fail_msg += f"||file name: {file_name} ||"
                            fail_msg += f" to_match:{info_[2 * (i + 1)]}, search_space:{texts[offset_idx:]}, "
                            fail_msg += f"All:{texts}, all_line:{content} ||"
                        fail_flag = True
                        break
                    # assert info_[2*(i+1)] in texts[offset_idx:], f"origin:{texts}, cur:{texts[offset_idx:]}, target:{info_[2*(i+1)]}, file:{file_name}"
                    st_idx = texts[offset_idx:].find(info_[2 * (i + 1)])
                    for j in range(offset_idx + st_idx, offset_idx + st_idx + len(info_[2 * (i + 1)])):
                        if j == (offset_idx + st_idx):
                            entity_tags[j] = f'B-{info_[2 * i + 1]}'
                        else:
                            entity_tags[j] = f'I-{info_[2 * i + 1]}'
                    offset_idx = offset_idx + st_idx + len(info_[2 * (i + 1)])
                output_ins['text'] = texts
                output_ins['entity'] = entity_tags
                if fail_flag:
                    break
            else:
                output_ins['text'] = content
                output_ins['entity'] = ['O' for _ in range(len(content))]
            output_dict['annotations'].append(output_ins)
        if fail_flag:
            unmatch_list.append(file_name)
            fail_messages.append(fail_msg)
            continue
        if len(poly_to_blur) > 0:
            img = np.bitwise_or(img, blur_area_whole)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        out_info_v3.append(output_dict)
        if save:
            cv2.imwrite(os.path.join(out_img_dir, f"{file_name}.jpg"), img)

    print(f"Un-match exists in {len(unmatch_list)} / {len(data_v3)}, ignore them")
    with open(os.path.join(out_dir, 'unmatch_msg.json'), 'w', encoding='utf-8') as f:
        json.dump(fail_messages, f, ensure_ascii=False, indent=4)
    if save:
        with open(os.path.join(out_dir, 'unmatch_list.json'), 'w', encoding='utf-8') as f:
            json.dump(unmatch_list, f)
    print(f"Current, valid: {len(out_info_v3)}")
    # save to out_dir
    if save:
        with open(os.path.join(out_dir, 'added.txt'), 'w', encoding='utf-8') as saver:
            for val in out_info_v3:
                out_str = json.dumps(val)
                saver.write(out_str+'\n')
    print(f"Save: {len(out_info_v3)} to {out_dir}")



def is_a_inside_b(a, b, thresh=0.2):
    tmp_a = Polygon(a)
    tmp_b = Polygon(b)
    # if not tmp_b.is_valid or tmp_a.is_valid:
    #     return False
    try:
        return tmp_a.intersection(tmp_b).area / tmp_a.area > thresh
    except Exception as e:
        print(f"Error occurs when judge if poly is inside table: {e}")
        return False


def cal_iou(a, b):
    tmp_a = Polygon(a)
    tmp_b = Polygon(b)
    try:
        return tmp_a.intersection(tmp_b).area / tmp_a.union(tmp_b).area
    except Exception as e:
        print(f"Error occurs when calculating iou: {e}")
        return 0


def cal_dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def insert_pt(poly, add_num):
    if add_num == 0:
        return poly.reshape(-1, 2)
    # strategy: find the longest
    poly = poly.reshape(-1, 2)
    max_dist = 0
    max_idx = 0
    num_pt = poly.shape[0]
    for idx in range(poly.shape[0]):
        cur_dist = cal_dist(poly[idx], poly[(idx+1)%num_pt])
        if cur_dist > max_dist:
            max_dist = cur_dist
            max_idx = idx
    added_pt = (poly[max_idx] + poly[(max_idx+1)%num_pt]) / 2
    poly_list = poly.tolist()
    poly_list = poly_list[:max_idx+1] + [added_pt.tolist()] + poly_list[max_idx+1:]
    return insert_pt(np.array(poly_list), add_num-1)




def vis_nfv3():
    # get ocr dict
    avg_width = 0
    avg_height = 0
    with open('../dict_default.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    ext_dict = []
    # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/others/nf.txt']
    # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/others/openfood.txt']
    # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/others/google.txt']
    files = [r'E:\Dataset\Nutrition_Facts_Formal\V4\e2e_format/test.txt']
    # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/test.txt']
    img_dir = r'E:\Dataset\Nutrition_Facts_Formal\V4\e2e_format'
    # specify_file = 'SGP_82.jpg'
    specify_file = None
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                if specify_file:
                    if specify_file not in info_['file_name']:
                        continue
                avg_height += info_['height']
                avg_width += info_['width']
                img = cv2.imread(os.path.join(img_dir, info_['file_name']))
                for idx, anno in enumerate(info_['annotations']):
                    poly = np.array(anno['polygon'], dtype=np.int)
                    cv2.polylines(img, [poly.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                    # cv2.putText(img, str(idx), (poly[0], poly[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                    print(f"idx:{idx}, text:{anno['text']}, entity:{anno['entity']}")
                cv2.imshow('img', img)
                cv2.waitKey(0)


def analysis_data_distribution(anno_file, data_type='train'):
    annos = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            annos.append(json.loads(line_.strip()))
    nf_data_num = 0
    google_data_num = 0
    openfood_data_num = 0
    nf_hard_num = 0
    for anno_ in annos:
        file_name = anno_['file_name'].split('/')[-1]
        if file_name[:2] == 'nf':
            nf_data_num += 1
        elif file_name[:3] == 'GOG':
            google_data_num += 1
        elif file_name.split('_')[0] in ['NZL', 'SGP', 'UK']:
            openfood_data_num += 1
        else:
            nf_hard_num += 1
    print(f"In {data_type}:")
    print(f"Nutrition_Facts_V1: {nf_data_num}")
    print(f"Google: {google_data_num}")
    print(f"OpenFood: {openfood_data_num}")
    print(f"NF_Hard: {nf_hard_num}")


def split_annotation(anno_file, out_dir):
    annos = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            annos.append(json.loads(line_.strip()))
    nf_data = []
    google_data = []
    openfood_data = []
    for anno_ in annos:
        file_name = anno_['file_name'].split('/')[-1]
        if file_name[:2] == 'nf':
            nf_data.append(anno_)
        elif file_name[:3] == 'GOG':
            google_data.append(anno_)
        elif file_name.split('_')[0] in ['NZL', 'SGP', 'UK']:
            openfood_data.append(anno_)
        else:
            raise ValueError(f"Not match: {anno_}")
    keys = ['nf', 'google', 'openfood']
    data = [nf_data, google_data, openfood_data]
    for key_, data_ in zip(keys, data):
        with open(os.path.join(out_dir, f'test_split_{key_}.txt'), 'w', encoding='utf-8') as saver:
            for sample_ in data_:
                out_str = json.dumps(sample_)
                saver.write(out_str+'\n')




if __name__ == '__main__':
    data_type = 'test'
    anno_file = rf'E:\Dataset\Nutrition_Facts_Formal\e2e_format/{data_type}_even.txt'
    out_dir = rf'E:\Dataset\Nutrition_Facts_Formal\e2e_format'
    # analysis_data_distribution(anno_file, data_type=data_type)
    # split_annotation(anno_file, out_dir)


    nfv1_dir = r'E:\Dataset\NutritionFacts\nfv1_e2e_format\ie_e2e_data\mm_format\table'
    nfv3_dir = r'E:\Dataset\Nutrition_Facts_Formal\Final_ver\1'
    out_dir = r'E:\Dataset\Nutrition_Facts_Formal\e2e_format'
    train_split = 0.65
    # prepare_nfv3(nfv1_dir, nfv3_dir, out_dir, train_split, save=False)
    # src_dir = r'E:\Dataset\Nutrition_Facts_Formal\Final_ver\modified'
    # out_dir = r'E:\Dataset\Nutrition_Facts_Formal\Final_ver\modified_e2e_format'
    # prepare_nfv3_only(src_dir, out_dir, save=True)

    # # clean
    # # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/nf.txt',
    # #          'E:/Dataset/Nutrition_Facts_Formal/e2e_format/openfood.txt',
    # #          'E:/Dataset/Nutrition_Facts_Formal/e2e_format/google.txt']
    # files = ['E:/Dataset/Nutrition_Facts_Formal/Final_ver/modified_e2e_format/added.txt']
    # entity_to_rm = ['SF-PS', 'SF-P1', 'SF-D']
    # entity_map = {'CE=PS': 'CE-PS'}
    # for file_ in files:
    #     datas = []
    #     with open(file_, 'r', encoding='utf-8') as f:
    #         for line_ in f.readlines():
    #             if len(line_.strip()) > 0:
    #                 datas.append(json.loads(line_.strip()))
    #         for i in range(len(datas)):
    #             annos = datas[i]['annotations']
    #             new_annos = []
    #             for ann in annos:
    #                 if ann['text'] == 'FULL-TABLE':
    #                     continue
    #                 new_entity_tags = []
    #                 for entity_idx in range(len(ann['entity'])):
    #                     if ann['entity'][entity_idx][0] in ['I', 'B']:
    #                         cur_type = ann['entity'][entity_idx][2:]
    #                         if cur_type in entity_map:
    #                             cur_type = entity_map[cur_type]
    #                         if cur_type in entity_to_rm:
    #                             new_entity = 'O'
    #                         else:
    #                             new_entity = ann['entity'][entity_idx][:2] + cur_type
    #                         new_entity_tags.append(new_entity)
    #                     else:
    #                         new_entity_tags.append('O')
    #                 ann['entity'] = new_entity_tags
    #                 new_annos.append(ann)
    #             datas[i]['annotations'] = new_annos
    #     with open(file_, 'w', encoding='utf-8') as saver:
    #         for data in datas:
    #             out_str = json.dumps(data)
    #             saver.write(out_str+'\n')

    # # split train & test
    # entity_list = ['O', 'SS',
    #                'CE-PS', 'CE-P1', 'CE-D',
    #                'TF-PS', 'TF-P1', 'TF-D',
    #                'SO-PS', 'SO-P1', 'SO-D',
    #                'CAR-PS', 'CAR-P1', 'CAR-D',
    #                'PRO-PS', 'PRO-P1', 'PRO-D']
    # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/others/openfood.txt',
    #          'E:/Dataset/Nutrition_Facts_Formal/e2e_format/others/google.txt',
    #          'E:/Dataset/Nutrition_Facts_Formal/e2e_format/others/nf.txt']
    # train_list = []
    # test_list = []
    # for file_ in tqdm(files):
    #     datas = []
    #     with open(file_, 'r', encoding='utf-8') as f:
    #         for line_ in f.readlines():
    #             if len(line_.strip()) > 0:
    #                 datas.append(json.loads(line_.strip()))
    #     if 'nf.txt' in file_:
    #         num_train = 2000 - len(train_list)
    #         print(f"{file_}: train:{num_train}, test:{len(datas) - num_train}")
    #         random.shuffle(datas)
    #         train_list += datas[:num_train]
    #         test_list += datas[num_train:]
    #     else:
    #         train_split = 0.5
    #         num_train = int(train_split * len(datas))
    #         print(f"{file_}: train:{num_train}, test:{len(datas)-num_train}")
    #         random.shuffle(datas)
    #         train_list += datas[:num_train]
    #         test_list += datas[num_train:]
    # train_cls = []
    # for info_ in tqdm(train_list):
    #     for ins_ in info_['annotations']:
    #         assert len(ins_['text']) == len(ins_['entity'])
    #         for entity_tag in ins_['entity']:
    #             if entity_tag[0] in ['I', 'B']:
    #                 cur_type = entity_tag[2:]
    #             else:
    #                 cur_type = 'O'
    #             if cur_type not in train_cls:
    #                 train_cls.append(cur_type)
    # assert len(train_cls) == len(entity_list), f"{train_cls}"
    # print(f"Num_train:{len(train_list)}, Num_test:{len(test_list)}")
    # with open('E:/Dataset/Nutrition_Facts_Formal/e2e_format/train_even.txt', 'w', encoding='utf-8') as saver:
    #     for info_ in train_list:
    #         out_str = json.dumps(info_)
    #         saver.write(out_str+'\n')
    # with open('E:/Dataset/Nutrition_Facts_Formal/e2e_format/test_even.txt', 'w', encoding='utf-8') as saver:
    #     for info_ in test_list:
    #         out_str = json.dumps(info_)
    #         saver.write(out_str+'\n')

    # # get dict for kie
    # files = {'train': 'E:/Dataset/Nutrition_Facts_Formal/e2e_format/train.txt',
    #          'test': 'E:/Dataset/Nutrition_Facts_Formal/e2e_format/test.txt'}
    # for key in ['train', 'test']:
    #     datas = []
    #     with open(files[key], 'r', encoding='utf-8') as f:
    #         for line_ in f.readlines():
    #             if len(line_.strip()) > 0:
    #                 datas.append(json.loads(line_.strip()))
    #     cls_list = []
    #     for info_ in tqdm(datas):
    #         for ins_ in info_['annotations']:
    #             assert len(ins_['text']) == len(ins_['entity'])
    #             for entity_tag in ins_['entity']:
    #                 if entity_tag[0] in ['I', 'B']:
    #                     cur_type = entity_tag[2:]
    #                 else:
    #                     cur_type = 'O'
    #                 if cur_type not in cls_list:
    #                     cls_list.append(cur_type)
    #     with open(f'E:/Dataset/Nutrition_Facts_Formal/e2e_format/{key}_dict.json', 'w', encoding='utf-8') as saver:
    #         json.dump(cls_list, saver)

    # # clean unexpected ocr key
    # unexpect = ['É', 'é', '·', '℮', 'è', '₽', 'À', '¡', 'ó', 'ú', '®', '＜', '₹', 'ª', '¢', '₭', '¿']
    # # files = {'train': 'E:/Dataset/Nutrition_Facts_Formal/e2e_format/train_even.txt',
    # #          'test': 'E:/Dataset/Nutrition_Facts_Formal/e2e_format/test_even.txt'}
    # files = {'added': 'E:/Dataset/Nutrition_Facts_Formal/Final_ver/modified_e2e_format/added.txt'}
    # # data_dir = 'E:/Dataset/Nutrition_Facts_Formal/e2e_format'
    # data_dir = 'E:/Dataset/Nutrition_Facts_Formal/Final_ver/modified_e2e_format'
    # dataset = {}
    # # for key in ['train', 'test']:
    # for key in ['added']:
    #     file_ = files[key]
    #     invalid_file = []
    #     datas = []
    #     with open(file_, 'r', encoding='utf-8') as f:
    #         for line_ in f.readlines():
    #             if len(line_.strip()) > 0:
    #                 datas.append(json.loads(line_.strip()))
    #     new_datas = []
    #     for i in range(len(datas)):
    #         annos = datas[i]['annotations']
    #         img_name = datas[i]['file_name']
    #         poly_to_mask = []
    #         new_annos = []
    #         for ann in annos:
    #             is_invalid = False
    #             for char_ in ann['text']:
    #                 if char_ in unexpect:
    #                     poly_to_mask.append(ann['polygon'])
    #                     is_invalid = True
    #                     break
    #             if not is_invalid:
    #                 new_annos.append(ann)
    #         if len(poly_to_mask) > 0:
    #             invalid_file.append(img_name)
    #             continue
    #             # img = cv2.imread(os.path.join(data_dir, img_name))
    #             # height, width, _ = img.shape
    #             # blur_area_whole = np.zeros(img.shape, img.dtype)
    #             # for poly in poly_to_mask:
    #             #     poly = np.array(poly, dtype=np.int).reshape(-1, 2)
    #             #     ins_tl_x = max(int(min(poly[:, 0])), 0)
    #             #     ins_br_x = min(int(max(poly[:, 0])), width)
    #             #     ins_tl_y = max(int(min(poly[:, 1])), 0)
    #             #     ins_br_y = min(int(max(poly[:, 1])), height)
    #             #     blur_mask = np.zeros(img[ins_tl_y: ins_br_y, ins_tl_x: ins_br_x, :].shape, img.dtype)
    #             #     new_poly = np.array(poly, dtype=np.int)
    #             #     new_poly[:, 0] -= ins_tl_x
    #             #     new_poly[:, 1] -= ins_tl_y
    #             #     cv2.fillPoly(blur_mask, [new_poly], color=(255, 255, 255))
    #             #     blur_area_whole[ins_tl_y: ins_br_y, ins_tl_x: ins_br_x, :] = blur_mask
    #             # new_img = np.bitwise_or(img, blur_area_whole)
    #             # cv2.imshow('pre', img)
    #             # cv2.imshow('aft', new_img)
    #             # cv2.waitKey(0)
    #         new_datas.append(datas[i])
    #     dataset[key] = new_datas
    # # print(f"After clean, {len(dataset['added'])} images left.")
    # # with open(files['added'], 'w', encoding='utf-8') as f:
    # #     for sample_ in dataset['added']:
    # #         out_str = json.dumps(sample_, ensure_ascii=False)
    # #         f.write(out_str+'\n')
    # supply_num = 2000 - len(dataset['train'])
    # num_test = len(dataset['test'])
    # dataset['train'] = dataset['train'] + dataset['test'][num_test-supply_num:]
    # dataset['test'] = dataset['test'][:num_test-supply_num]
    # for key in ['train', 'test']:
    #     with open(f'E:/Dataset/Nutrition_Facts_Formal/e2e_format/{key}_even.txt', 'w', encoding='utf-8') as saver:
    #         for info_ in dataset[key]:
    #             out_str = json.dumps(info_)
    #             saver.write(out_str+'\n')
    # print(f"After clean, train:{len(dataset['train'])}, test:{len(dataset['test'])}")


    # # convert polygon by cv2
    # # files = {'train': 'E:/Dataset/Nutrition_Facts_Formal/e2e_format/train_even.txt',
    # #          'test': 'E:/Dataset/Nutrition_Facts_Formal/e2e_format/test_even.txt'}
    # files = {'added': r'E:/Dataset/Nutrition_Facts_Formal/Final_ver/modified_e2e_format/added.txt'}
    # data_dir = 'E:/Dataset/Nutrition_Facts_Formal/Final_ver/modified_e2e_format'
    # dataset = {}
    # iou_not_match = 0
    # pt_num = 20
    # # for key in ['train', 'test']:
    # for key in ['added']:
    #     file_ = files[key]
    #     invalid_file = []
    #     datas = []
    #     with open(file_, 'r', encoding='utf-8') as f:
    #         for line_ in f.readlines():
    #             if len(line_.strip()) > 0:
    #                 datas.append(json.loads(line_.strip()))
    #     new_datas = []
    #     for data in tqdm(datas):
    #         new_data = {
    #             'file_name': data['file_name'],
    #             'width': data['width'],
    #             'height': data['height'],
    #             'annotations': []
    #         }
    #         for ins_ in data['annotations']:
    #             poly = np.array(ins_['polygon'], dtype=np.int).reshape((-1, 2))
    #             # strategy 2: use approximate
    #             epsilon = 0.001 * cv2.arcLength(poly, True)
    #             approx = cv2.approxPolyDP(poly, epsilon, True)
    #             polygon = approx.reshape((-1, 2))
    #             # if cal_iou(poly, polygon) < 0.95:
    #             #     iou_not_match += 1
    #             polygon = insert_pt(polygon, pt_num-polygon.shape[0])
    #             if cal_iou(poly, polygon) < 0.95:
    #                 iou_not_match += 1
    #             assert polygon.shape[0] == pt_num
    #             ins_['polygon'] = polygon.reshape(-1).tolist()
    #             new_data['annotations'].append(ins_)
    #         new_datas.append(new_data)
    #     with open(file_, 'w', encoding='utf-8') as f:
    #         for info_ in new_datas:
    #             out_str = json.dumps(info_)
    #             f.write(out_str + '\n')
    # # print(f"After approx, max points num:{max_pt_num}, min:{min_pt_num}")
    # print(f"IoU not match num: {iou_not_match}")

    # # get ocr dict
    # avg_width = 0
    # avg_height = 0
    # img_cnt = 0
    # max_text_len = 0
    # with open('../dict_default.json', 'r', encoding='utf-8') as f:
    #     default_dict = json.load(f)
    # ext_dict = []
    # entity_cls_list = ["O", "CE-P1", "CE-PS", "CE-D", "TF-P1", "TF-PS", "TF-D", "CAR-P1", "CAR-PS", "CAR-D", "PRO-P1", "PRO-PS", "PRO-D", "SS", "SO-P1", "SO-PS", "SO-D"]
    # # files = ['E:/Dataset/Nutrition_Facts_Formal/e2e_format/train_even.txt',
    # #          'E:/Dataset/Nutrition_Facts_Formal/e2e_format/test_even.txt']
    # files = ['E:/Dataset/Nutrition_Facts_Formal/Final_ver/modified_e2e_format/added.txt']
    # avg_instance_num = 0
    # total_instance_num = 0
    # max_instance_num = 0
    # min_instance_num = 999
    # avg_ins_width = 0
    # avg_ins_height = 0
    # max_pt_num = 0
    # for file_ in files:
    #     with open(file_, 'r', encoding='utf-8') as f:
    #         for line in tqdm(f.readlines()):
    #             info_ = json.loads(line.strip())
    #             avg_height += info_['height']
    #             avg_width += info_['width']
    #             img_cnt += 1
    #             num_instance = len(info_['annotations'])
    #             total_instance_num += num_instance
    #             max_instance_num = max(max_instance_num, num_instance)
    #             min_instance_num = min(min_instance_num, num_instance)
    #             for anno in info_['annotations']:
    #                 max_pt_num = max(max_pt_num, len(anno['polygon'])//2)
    #                 assert len(anno['text']) == len(anno['entity'])
    #                 avg_ins_width += (max(anno['polygon'][0::2]) - min(anno['polygon'][0::2]))
    #                 avg_ins_height += (max(anno['polygon'][1::2]) - min(anno['polygon'][1::2]))
    #                 max_text_len = max(max_text_len, len(anno['text']))
    #                 # if len(anno['text']) > 60:
    #                 #     print(anno['text'])
    #                 for char_ in anno['text']:
    #                     if char_ not in default_dict:
    #                         if char_ not in ext_dict:
    #                             ext_dict.append(char_)
    #                 for entity_ in anno['entity']:
    #                     entity = entity_.replace('B-', '')
    #                     entity = entity.replace('I-', '')
    #                     assert entity in entity_cls_list, f"Entity:{entity_}"
    # print(f"Total sample num: {img_cnt}")
    # print(f"max_pt_num:{max_pt_num}")
    # print(f"avg:{total_instance_num / img_cnt}, max:{max_instance_num}, min:{min_instance_num}")
    # full_key = default_dict + ext_dict
    # print(f"avg_height:{avg_height / img_cnt}, avg_width:{avg_width / img_cnt}")
    # print(f"ext key:{ext_dict}")
    # print(f"max_len:{max_text_len}")
    # print(
    #     f"avg_ins_height:{avg_ins_height / total_instance_num}, avg_ins_width:{avg_ins_width / total_instance_num}")

    """
    avg:35.52663837812353, max:206, min:1
    avg_height:581.5530410183876, avg_width:480.16548797736914
    ext key:[]
    max_len:122
    avg_ins_height:29.866103753052396, avg_ins_width:94.57165237817021
    """

    vis_nfv3()

