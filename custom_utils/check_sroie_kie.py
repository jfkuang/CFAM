#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/16 12:54
# @Author : WeiHua
import glob
import json
import os
from functools import cmp_to_key
import cv2
import numpy as np
from tqdm import tqdm

def sort_by_read_order(texts, entities, seq_scores, boxes):
    """
    sort text instances with respect to reading order.
    Args:
        texts: list[str]
        entities: list[list[entity of each char]]
        boxes: list[list[x1, y1, x2, y2, ...]]

    Returns:

    """
    num_sample = len(texts)
    assert num_sample == len(entities), f"sample:{num_sample}, entities:{len(entities)}"
    assert num_sample == len(boxes), f"sample:{num_sample}, boxes:{len(boxes)}"
    sorted_index = [i for i in range(num_sample)]
    to_sort_boxes = [(idx, box) for idx, box in enumerate(boxes)]
    # def compare_key(x):
    #     #  x is (index, box), where box is list[x, y, x, y...]
    #     points = x[1]
    #     box = np.array(x[1], dtype=np.float32).reshape(-1, 2)
    #     rect = cv2.minAreaRect(box)
    #     center = rect[0]
    #     return center[1], center[0]
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
    # sorted_data = sorted(to_sort_boxes, key=compare_key)
    sorted_data = sorted(to_sort_boxes, key=cmp_to_key(compare_key))
    sorted_index = [x[0] for x in sorted_data]
    return [texts[i] for i in sorted_index], [entities[i] for i in sorted_index], [seq_scores[i] for i in sorted_index]


def eval_hmean_kie(gt_kie, pred_kie):
    """
    F-1 score for kie evaluation.
    Args:
        gt_kie: list[dict], each dict represent an image
        pred_kie: list[dict], each dict represent an image

    Returns:

    """
    img_results = []
    assert len(gt_kie) == len(pred_kie)
    total_num_gt = 0
    total_num_pred = 0
    total_match_num = 0
    # for gt_dict, pred_dict in tqdm(zip(gt_kie, pred_kie)):
    for gt_dict, pred_dict in zip(gt_kie, pred_kie):
        num_gt = len(gt_dict)
        num_pred = len(pred_dict)
        match_num = 0
        for pred_key, pred_val in pred_dict.items():
            if pred_key in gt_dict.keys():
                if gt_dict[pred_key] == pred_val:
                    match_num += 1
        if num_pred == 0:
            num_pred = 1
        precision = match_num / num_pred
        recall = match_num / num_gt
        if match_num == 0:
            hmean = 0
        else:
            hmean = 2 * precision * recall / (precision + recall)
        img_results.append({'recall': recall, 'precision': precision, 'hmean': hmean})

        total_num_gt += num_gt
        total_num_pred += num_pred
        total_match_num += match_num

    total_precision = total_match_num / total_num_pred
    total_recall = total_match_num / total_num_gt
    if total_match_num == 0:
        total_hmean = 0
    else:
        total_hmean = 2 * total_precision * total_recall / (total_precision + total_recall)
    dataset_results = {
        'num_gts': total_num_gt,
        'num_preds': total_num_pred,
        'num_match': total_match_num,
        'recall': total_recall,
        'precision': total_precision,
        'hmean': total_hmean
    }
    return dataset_results, img_results


def analysis_kie(file_names, pred_kie, gt_kie, out_file):
    assert len(file_names) == len(pred_kie) == len(gt_kie)
    with open(out_file, 'w', encoding='utf-8') as saver:
        for i in tqdm(range(len(gt_kie))):
            file_name = file_names[i]
            pred = pred_kie[i]
            gt = gt_kie[i]
            saver.write(f"\n{file_name}\n")
            for key_, val_ in pred.items():
                if key_ not in gt.keys():
                    saver.write(f"PD:\t{key_}:{val_}\n")
                    saver.write(f"GT:\t{key_}:None\n")
                else:
                    if val_ != gt[key_]:
                        saver.write(f"PD:\t{key_}:{val_}\n")
                        saver.write(f"GT:\t{key_}:{gt[key_]}\n")
            for key_, val_ in gt.items():
                if key_ not in pred.keys():
                    saver.write(f"PD:\t{key_}:None\n")
                    saver.write(f"GT:\t{key_}:{val_}\n")


def eval_under_gt(anno_file, sort_ins=True, save_res=False, out_file=None):
    # load annotations
    entity_list = ["company", "address", "date", "total"]
    anno_list = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if not line_.strip():
                continue
            anno_list.append(json.loads(line_.strip()))
    pred_kie = []
    gt_kie = []
    for anno_info in anno_list:
        gt_kie.append(anno_info['entity_dict'])
        kie_dict = dict()

        # modified by whua, add box threshold to screen low confidence result
        # Not Test Yet
        pred_texts_ = [x['text'] for x in anno_info['annotations']]
        pred_entities_ = [x['entity'] for x in anno_info['annotations']]
        seq_score_ = [1.0] * len(pred_texts_)

        if sort_ins:
            cur_pred_boxes = [x['polygon'] for x in anno_info['annotations']]
            pred_texts, pred_entities, seq_score_ = sort_by_read_order(pred_texts_, pred_entities_, seq_score_,
                                                                       cur_pred_boxes)
        else:
            pred_texts = pred_texts_
            pred_entities = pred_entities_
        for pred_text, pred_entity in zip(pred_texts, pred_entities):
            entity_within_instance = dict()
            for char_idx, (txt_token, iob_tag) in enumerate(zip(pred_text, pred_entity)):
                if iob_tag[0] in ['I', 'B']:
                    cur_tag = iob_tag[2:]
                    assert cur_tag in entity_list, f"cur_tag:{cur_tag}, iob_tag:{iob_tag}"
                    if cur_tag not in entity_within_instance:
                        entity_within_instance[cur_tag] = txt_token
                    else:
                        entity_within_instance[cur_tag] += txt_token
            for key, val in entity_within_instance.items():
                if key in kie_dict:
                    if val in kie_dict[key]:
                        continue
                    kie_dict[key].append(val)
                    # if val == kie_dict[key]:
                    #     continue
                    # kie_dict[key] += ' '
                    # kie_dict[key] += val
                else:
                    kie_dict[key] = [val]
        kie_dict_merged = dict()
        for key, val in kie_dict.items():
            kie_dict_merged[key] = ' '.join(val)
            kie_dict = kie_dict_merged
        pred_kie.append(kie_dict)
    kie_result, kie_img_result = eval_hmean_kie(gt_kie, pred_kie)
    assert len(kie_img_result) == len(anno_list) == len(gt_kie) == len(pred_kie)
    invalid_anno = []
    invalid_gt = []
    invalid_pred = []
    for res, anno_info, gt, pred in zip(kie_img_result, anno_list, gt_kie, pred_kie):
        if res['hmean'] != 1.0:
            invalid_anno.append(anno_info)
            invalid_gt.append(gt)
            invalid_pred.append(pred)
    import ipdb
    ipdb.set_trace()
    print(kie_result)
    if save_res:
        analysis_kie([x['file_name'] for x in invalid_anno],
                     invalid_pred, invalid_gt, out_file)


if __name__ == '__main__':
    anno_file = r"E:\Dataset\KIE\SROIE\e2e_format/train_update_screen_v21.txt"
    out_file = "./sroie_train_gt_check_res.txt"
    eval_under_gt(anno_file, sort_ins=True, save_res=True, out_file=out_file)


