#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/2 10:21
# @Author : WeiHua

import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import ipdb
"""
"""


def compare_kie(ie_format_anno, formal_anno_dir):
    ie_format_annos = []
    with open(ie_format_anno, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                ie_format_annos.append(json.loads(line_.strip()))
    for sample_ in ie_format_annos:
        file_name = sample_['file_name'].split('/')[-1].split('.')[0]
        with open(os.path.join(formal_anno_dir, f"{file_name}.txt"), 'r', encoding='utf-8') as f:
            formal_anno = json.load(f)
        e2e_anno = sample_['entity_dict']
        # compare formal annotation with ours
        match_flag = True
        if len(formal_anno) != len(e2e_anno):
            match_flag = False
        for key_ in formal_anno.keys():
            if key_ not in e2e_anno.keys():
                match_flag = False
                break
            elif formal_anno[key_] != e2e_anno[key_]:
                match_flag = False
                break
        if not match_flag:
            print(f"Not the same: {file_name}")
            print(f"formal:{formal_anno}\nie-e2e:{e2e_anno}")
    print("End of check.")


def cal_iou(a, b):
    a = np.array(a, dtype=np.float32).reshape(-1, 2)
    b = np.array(b, dtype=np.float32).reshape(-1, 2)
    a = Polygon(a)
    b = Polygon(b)
    if not a.is_valid or not b.is_valid:
        return 0
    return a.intersection(b).area / a.union(b).area


def compare_ocr(ie_format_anno, formal_anno_dir, formal_img_dir, iou_thresh=0.99):
    ie_format_annos = []
    with open(ie_format_anno, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                ie_format_annos.append(json.loads(line_.strip()))
    total_pred = 0
    total_gt = 0
    total_match = 0
    st_idx = 326
    for idx, sample_ in tqdm(enumerate(ie_format_annos)):
        if idx < st_idx:
            continue
        file_name = sample_['file_name'].split('/')[-1].split('.')[0]
        gt_texts = []
        gt_polys = []
        tl_x, tl_y = 9999, 9999
        br_x, br_y = 0, 0
        print(idx, file_name)
        with open(os.path.join(formal_anno_dir, f"{file_name}.txt"), 'r', encoding='utf-8') as f:
            for line_ in f.readlines():
                if not line_.strip():
                    continue
                line_ = line_.strip()
                tmp = line_.split(',')
                assert len(tmp) >= 9, f"Invalid line: {line_}"
                txt_st_idx = len(",".join(tmp[:8]))
                transcript = line_[txt_st_idx+1:]
                coords = [float(x) for x in tmp[:8]]

                tmp_box = np.array(coords, dtype=np.float32)
                min_x = int(min(tmp_box[0::2]))
                min_y = int(min(tmp_box[1::2]))
                max_x = int(max(tmp_box[0::2]))
                max_y = int(max(tmp_box[1::2]))
                tl_x = min(min_x, tl_x)
                tl_y = min(min_y, tl_y)
                br_x = max(max_x, br_x)
                br_y = max(max_y, br_y)

                gt_texts.append(transcript)
                gt_polys.append(coords)

        img = cv2.imread(str(os.path.join(formal_img_dir, f"{file_name}.jpg")))
        tl_x = max(0, tl_x)
        tl_y = max(0, tl_y)
        br_x = min(img.shape[1], br_x + 5)
        br_y = min(img.shape[0], br_y + 5)
        img = img[tl_y: br_y, tl_x: br_x, :]
        gt_polys_new = []
        for poly_ in gt_polys:
            tmp = np.array(poly_, dtype=np.float32)
            tmp[::2] -= tl_x
            tmp[1::2] -= tl_y
            gt_polys_new.append(tmp)
        if sample_['height'] != img.shape[0] or sample_['width'] != img.shape[1]:
            print(f"Unmatch:\ne2e:H, W = {sample_['height']}, {sample_['width']}")
            print(f"formal:H, W = {img.shape[0]}, {img.shape[1]}")

        # img_pred = img.copy()
        # img_gt = img.copy()
        # pred_polys = [x['polygon'] for x in sample_['annotations']]
        # for idx, poly_ in enumerate(pred_polys):
        #     poly_ = np.array(poly_, dtype=np.int)
        #     cv2.polylines(img_pred, [poly_.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
        #     cv2.putText(img_pred, str(idx), (poly_[0], poly_[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
        #     print(f"{idx}: {sample_['annotations'][idx]['text']}")
        # for idx, poly_ in enumerate(gt_polys_new):
        #     poly_ = np.array(poly_, dtype=np.int)
        #     cv2.polylines(img_gt, [poly_.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
        #     cv2.putText(img_gt, str(idx), (poly_[0], poly_[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
        #     print(f"{idx}: {gt_texts[idx]}")
        # cv2.imwrite("gt.jpg", img_gt)
        # cv2.imwrite("pred.jpg", img_pred)
        # cv2.imshow('pred', img_pred)
        # cv2.imshow('gt', img_gt)
        # cv2.waitKey(0)


        match_matrix = np.zeros((len(sample_['annotations']), len(gt_texts)), dtype=np.float32)
        for i in range(len(sample_['annotations'])):
            for j in range(len(gt_texts)):
                if cal_iou(sample_['annotations'][i]['polygon'], gt_polys_new[j]) > iou_thresh:
                    if sample_['annotations'][i]['text'] == gt_texts[j]:
                        match_matrix[i, j] = 1
        total_gt += len(gt_texts)
        total_pred += len(sample_['annotations'])
        sample_matched = 0
        not_matched_words = []
        for i in range(match_matrix.shape[0]):
            if np.sum(match_matrix[i]) > 0:
                sample_matched += 1
            else:
                not_matched_words.append(sample_['annotations'][i]['text'])
        total_match += sample_matched
        if sample_matched != len(gt_texts):
            print(f"Unmatch: {file_name}")
            print(f"not matched words:{not_matched_words}")
            pred_texts = [x['text'] for x in sample_['annotations']]
            print(f"pred:{pred_texts}")
            print(f"gt:{gt_texts}")
            # visualize
            pred_polys = [x['polygon'] for x in sample_['annotations']]
            img_pred = img.copy()
            img_gt = img.copy()
            for idx, poly_ in enumerate(pred_polys):
                poly_ = np.array(poly_, dtype=np.int)
                cv2.polylines(img_pred, [poly_.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                cv2.putText(img_pred, str(idx), (poly_[0], poly_[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                print(f"{idx}: {sample_['annotations'][idx]['text']}")
            for idx, poly_ in enumerate(gt_polys_new):
                poly_ = np.array(poly_, dtype=np.int)
                cv2.polylines(img_gt, [poly_.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                cv2.putText(img_gt, str(idx), (poly_[0], poly_[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                print(f"{idx}: {gt_texts[idx]}")

            cv2.imwrite("gt.jpg", img_gt)
            cv2.imwrite("pred.jpg", img_pred)
            cv2.imshow('pred', img_pred)
            cv2.imshow('gt', img_gt)
            cv2.waitKey(0)
            ipdb.set_trace()


    precision = total_match / total_pred
    recall = total_match / total_gt
    hmean = 2 * precision * recall / (precision + recall)
    print(f"precision:{precision}, recall:{recall}, hmean:{hmean}")
    ipdb.set_trace()
    print(f"End check.")



def check_kie(anno_file, data_dir):
    entity_classes = ["company", "address", "date", "total"]
    st_idx = 0
    with open(anno_file, 'r', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            if idx < st_idx:
                continue
            print(idx)
            info_ = json.loads(line.strip())
            img = cv2.imread(os.path.join(data_dir, info_['file_name']))
            default_entity_dict = info_['entity_dict']
            cur_entity_dict = dict()
            for idx, anno in enumerate(info_['annotations']):
                poly = np.array(anno['polygon'], dtype=np.int)
                cv2.polylines(img, [poly.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                cv2.putText(img, str(idx), (poly[0], poly[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                # print(f"idx:{idx}, text:{anno['text']}")
                assert len(anno['entity']) == len(anno['text']), f"not same length:{info_['file_name']}, txt:{anno['text']}"
                for idx, ec in enumerate(anno['entity']):
                    if ec[0] in ['B', 'I']:
                        cur_cls = ec[2:]
                        assert cur_cls in entity_classes
                        if cur_cls in cur_entity_dict and idx == 0:
                            cur_entity_dict[cur_cls] += ' '
                        if cur_cls not in cur_entity_dict:
                            cur_entity_dict[cur_cls] = anno['text'][idx]
                        else:
                            cur_entity_dict[cur_cls] += anno['text'][idx]
            not_same = []
            for key, val in default_entity_dict.items():
                if key not in cur_entity_dict:
                    not_same.append(f'miss entity:{key}')
                elif cur_entity_dict[key] != val:
                    if abs(len(cur_entity_dict[key]) - len(val)) > 3:
                        not_same.append(f"value of {key} not the same:\ndefault:{val}\ncurrent:{cur_entity_dict[key]}")
            if len(not_same) > 0:
                print(f"Compare results of {info_['file_name'].split('/')[-1]}:")
            for msg in not_same:
                print(msg)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)



if __name__ == '__main__':
    ie_format_anno = r"E:\Dataset\KIE\SROIE\e2e_format/train_update.txt"
    formal_anno_dir = r"E:\Dataset\KIE\SROIE\SROIE_VERIFIED_GITHUB\SROIE2019\0325updated.task2train(626p)" # github version
    # formal_anno_dir = r"E:\Dataset\KIE\SROIE\0325updated.task2train(626p)\0325updated.task2train(626p)" # official
    # formal_anno_dir = r"E:\Dataset\KIE\SROIE\SROIE_test_gt_task_3"
    # compare_kie(ie_format_anno, formal_anno_dir)

    ie_format_anno = r"E:\Dataset\KIE\SROIE\e2e_format/test.txt"
    # formal_anno_dir = r"E:\Dataset\KIE\SROIE\SROIE_VERIFIED_GITHUB\SROIE2019\0325updated.task1train(626p)" # github version
    formal_anno_dir = r"E:\Dataset\KIE\SROIE\SROIE_VERIFIED_GITHUB\SROIE2019\text.task1_2-test（361p)"
    # formal_img_dir = r"E:\Dataset\KIE\SROIE\SROIE_VERIFIED_GITHUB\SROIE2019\0325updated.task1train(626p)" # origin image
    formal_img_dir = r"E:\Dataset\KIE\SROIE\SROIE_test_images_task_3"
    compare_ocr(ie_format_anno, formal_anno_dir, formal_img_dir)

    anno_file = r"E:\Dataset\KIE\SROIE\e2e_format/train_update.txt"
    data_dir = r"E:\Dataset\KIE\SROIE\e2e_format"
    # check_kie(anno_file, data_dir)
