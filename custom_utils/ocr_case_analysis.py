#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/5/5 16:38
# @Author : WeiHua

import json
from tqdm import tqdm
import os
import numpy as np
from shapely.geometry import Polygon

def cal_iou(a, b):
    a = Polygon(np.array(a).reshape(-1, 2))
    b = Polygon(np.array(b).reshape(-1, 2))
    if not a.is_valid or not b.is_valid:
        return 0
    return a.intersection(b).area / a.union(b).area

def load_pred(file_name):
    pred_boxes = []
    pred_texts = []
    pred_scores = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if "Step 0:" not in line_:
                continue
            tmp = line_.strip().split("Step 0: ")[-1]
            tmp = tmp[9:].split('], text:')
            assert len(tmp) == 2, f"Invalid line: {line_}"

            box = tmp[0].split(',')
            box = [int(x) for x in box]
            assert len(box) == 8, f"Invalid line: {line_}"
            pred_boxes.append(box)

            tmp = tmp[1].split(', score:')
            assert len(tmp) == 2, f"Invalid line: {line_}"

            pred_texts.append(tmp[0])
            pred_scores.append(float(tmp[1]))
    return pred_boxes, pred_texts, pred_scores

# fix hidden bug here
def analysis_ocr(anno_file, pred_dir, seq_thresh=0.9, iou_thresh=0.5, ignore_capital=False,
                 saving_file=None, print_log=True):
    annos = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                annos.append(json.loads(line_.strip()))

    total_pred = 0
    total_pred_det = 0
    total_gt = 0
    total_match = 0
    total_match_det = 0

    if saving_file:
        saver = open(saving_file, 'w', encoding='utf-8')
    else:
        saver = None

    for anno in tqdm(annos):
        file_name = os.path.basename(anno['file_name'])
        if saver:
            saver.write(f"\n{file_name}\n")
        else:
            if print_log:
                print(file_name)

        pred_file = os.path.join(pred_dir, file_name.replace('jpg', 'txt'))

        load_polys, load_texts, load_scores = load_pred(pred_file)

        pred_polys = load_polys

        pred_poly_text = []
        pred_texts = []
        for pred_poly, pred_text, pred_score in zip(pred_polys, load_texts, load_scores):
            if pred_score > seq_thresh:
                pred_poly_text.append(pred_poly)
                if ignore_capital:
                    pred_texts.append(pred_text.upper())
                else:
                    pred_texts.append(pred_text)

        match_matrix = np.zeros((len(pred_texts), len(anno['annotations'])), dtype=bool)
        match_matrix_det = np.zeros((len(pred_polys), len(anno['annotations'])), dtype=bool)

        for i in range(len(pred_polys)):
            for j in range(len(anno['annotations'])):
                if match_matrix_det[:, j].sum() > 0:
                    continue
                if cal_iou(pred_polys[i], anno['annotations'][j]['polygon']) > iou_thresh:
                    match_matrix_det[i, j] = True

        for i in range(len(pred_poly_text)):
            for j in range(len(anno['annotations'])):
                if match_matrix[:, j].sum() > 0:
                    continue
                if cal_iou(pred_poly_text[i], anno['annotations'][j]['polygon']) > iou_thresh:
                    if ignore_capital:
                        if pred_texts[i] == anno['annotations'][j]['text'].upper():
                            match_matrix[i, j] = True
                        else:
                            msg = f"gt:|{anno['annotations'][j]['text'].upper()}|, pred:|{pred_texts[i]}|"
                            if saver:
                                saver.write(f"{msg}\n")
                            else:
                                if print_log:
                                    print(msg)
                    else:
                        if pred_texts[i] == anno['annotations'][j]['text']:
                            match_matrix[i, j] = True
                        else:
                            print(f"gt:|{anno['annotations'][j]['text']}|, pred:|{pred_texts[i]}|")
        if print_log:
            gt_texts = [x['text'] for x in anno['annotations']]
            print(f"file_name:{file_name}")
            p = match_matrix_det.sum() / match_matrix_det.shape[0]
            r = match_matrix_det.sum() / match_matrix_det.shape[1]
            hm = 2 * p * r / (p + r)
            print("det_precision:", p)
            print("det_recall:", r)
            print(f"det_hmean:{hm}")
            p = match_matrix.sum() / match_matrix.shape[0]
            r = match_matrix.sum() / match_matrix.shape[1]
            hm = 2 * p * r / (p + r)
            print("rec_precision:", p)
            print("rec_recall:", r)
            print(f"rec_hmean:{hm}")
            import ipdb
            ipdb.set_trace()

        # det
        total_pred_det += match_matrix_det.shape[0]
        total_pred += match_matrix.shape[0]
        total_gt += match_matrix.shape[1]
        total_match += match_matrix.sum()
        total_match_det += match_matrix_det.sum()

    precision = total_match / total_pred
    recall = total_match / total_gt
    hmean = 2 * precision * recall / (precision + recall)

    precision_det = total_match_det / total_pred_det
    recall_det = total_match_det / total_gt
    hmean_det = 2 * precision_det * recall_det / (precision_det + recall_det)

    print(f"prediction of : {pred_dir}")
    print(f"In total {len(annos)} samples:")
    print(f"End-to-end detection:")
    print(f"precision:{precision_det}, recall:{recall_det}, hmean:{hmean_det}")
    print(f"End-to-end recognition:")
    print(f"precision:{precision}, recall:{recall}, hmean:{hmean}")
    if saver:
        saver.close()
    import ipdb
    ipdb.set_trace()
    print(f"End of evaluation")


from mmocr.core.evaluation.vie_metric import eval_hmean_iou_e2e
from mmocr.core.evaluation import hmean_iou

def formal_eval(anno_file, pred_dir, seq_thresh=0.5, iou_thresh=0.5, ignore_capital=True):
    pred_boxes_det = []
    pred_boxes = []
    gt_boxes = []
    pred_texts = []
    gt_texts = []

    annos = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                annos.append(json.loads(line_.strip()))

    for anno in annos:
        cur_polys = []
        cur_texts = []
        for sample_ in anno['annotations']:
            cur_polys.append(sample_['polygon'])
            cur_texts.append(sample_['text'])
        gt_boxes.append(cur_polys)
        gt_texts.append(cur_texts)

        pred_file = os.path.join(pred_dir, os.path.basename(anno['file_name']).replace('jpg', 'txt'))

        load_polys, load_texts, load_scores = load_pred(pred_file)

        pred_polys = load_polys

        p_polys = []
        p_texts = []
        for pred_poly, pred_text, pred_score in zip(pred_polys, load_texts, load_scores):
            if pred_score > seq_thresh:
                p_polys.append(pred_poly)
                if ignore_capital:
                    p_texts.append(pred_text.upper())
                else:
                    p_texts.append(pred_text)
        pred_boxes.append(p_polys)
        pred_texts.append(p_texts)
        pred_boxes_det.append(pred_polys)

    gt_ignored_boxes = [[] for _ in range(len(gt_boxes))]

    det_res, _ = hmean_iou.eval_hmean_iou(
        pred_boxes_det,
        gt_boxes,
        gt_ignored_boxes)
    print(f"det_rec:\n{det_res}")

    rec_res, _ = eval_hmean_iou_e2e(
        pred_boxes,
        gt_boxes,
        pred_texts,
        gt_texts,
        gt_ignored_boxes,
        iou_thr=iou_thresh,
        precision_thr=seq_thresh,
        ignore_capital=ignore_capital)
    print(f"rec_res:\n{rec_res}")
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    anno_file = '/home/whua/dataset/ie_e2e/sroie/e2e_format/test_screen.txt'
    pred_dir = '/home/whua/logs/ie_ocr_log/vis_sroie_screen_ft_default_dp02_lr2e4_rr_roi1260_bs4_epoch_390/image_files'
    analysis_ocr(anno_file, pred_dir, seq_thresh=0.5, iou_thresh=0.5, ignore_capital=True,
                 saving_file='/home/whua/vis_sroie_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.txt',
                 print_log=False)

    # formal_eval(anno_file, pred_dir, seq_thresh=0.5, iou_thresh=0.5, ignore_capital=True)