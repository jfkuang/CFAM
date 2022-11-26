#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/21 16:11
# @Author : WeiHua
import ipdb

from mmocr.core.evaluation import hmean_iou
import numpy as np

import mmocr.utils as utils
from . import utils as eval_utils
from tqdm import tqdm
import cv2
from functools import cmp_to_key


def eval_vie_e2e(results,
                 img_infos,
                 ann_infos,
                 metrics='e2e-hmean',
                 score_thr=0.3,
                 iters_rec_preds=None,
                 iters_kie_preds=None,
                 entity_list=None,
                 sort_for_kie=True,
                 seq_scores=None,
                 box_thr=0.5,
                 kie_instance_thr=0.85):
    """
    evaluate end-to-end key information extraction model.
    Args:
        results: {'boundary_result', 'box_result', 'REC', 'KIE'}
        img_infos:
        ann_infos:
        metrics:
        score_thr:
        iters_rec_preds: list[list[list[transcript]]], 1st list -> iter,
            2nd list -> image, 3rd list -> instance
        iters_kie_preds: list[list[list[transcript]]], 1st list -> iter,
            2nd list -> image, 3rd list -> instance
        seq_scores: list[list[list[transcript]]], 1st list -> iter,
            2nd list -> image, 3rd list -> instance
        box_thr: threshold for box

    Returns:

    """
    eval_results = dict()
    print(f"During evaluation, using metric: {metrics}")
    if 'boundary_result' in results[0]:
        print("Evaluating DET metrics...")
        # prepare gt for detection evaluation
        pred_boxes = []
        for result in results:
            # pred_boxes.append([x[:-1] for x in result['boundary_result']])

            # modified by whua, add box threshold to screen low confidence result
            scores = np.array([x[-1] for x in result['boundary_result']])
            keep_ids = scores > score_thr
            pred_boxes.append([result['boundary_result'][x][:-1] for x in np.where(keep_ids)[0]])

        gt_boxes = []
        for ann_info in ann_infos:
            gt_boxes.append(ann_info['masks'])
        gt_ignored_boxes = [[] for i in range(len(gt_boxes))]
        # gt_boxes represents polygons, pred_boxes represents polygons
        det_result, det_img_result = hmean_iou.eval_hmean_iou(pred_boxes, gt_boxes,
                                                              gt_ignored_boxes)
        eval_results['DET'] = det_result

    if iters_rec_preds:
        print("Evaluating REC metrics...")
        rec_iter_results = {}
        # prepare gt for end-to-end OCR evaluation
        pred_boxes = []
        keep_ids_list = []
        for result in results:
            # pred_boxes.append([x[:-1] for x in result['boundary_result']])

            # modified by whua, add box threshold to screen low confidence result
            scores = np.array([x[-1] for x in result['boundary_result']])
            keep_ids = scores > score_thr
            keep_ids_list.append(np.where(keep_ids)[0])
            pred_boxes.append([result['boundary_result'][x][:-1] for x in np.where(keep_ids)[0]])
        gt_boxes = []
        gt_texts = []
        for ann_info in ann_infos:
            gt_boxes.append(ann_info['masks'])
            gt_texts.append(ann_info['ori_texts'])
        gt_ignored_boxes = [[] for i in range(len(gt_boxes))]
        for iter_, iter_result in enumerate(iters_rec_preds):
            print(f"Evaluating REC of iter_{iter_} ...")
            # rec_result, rec_img_result = eval_hmean_iou_e2e(pred_boxes, gt_boxes,
            #                                                 iter_result, gt_texts,
            #                                                 gt_ignored_boxes,
            #                                                 ignore_capital='sroie' in metrics)

            # modified by whua, add box threshold to screen low confidence result
            new_iter_result = []
            for img_result, keep_ids in zip(iter_result, keep_ids_list):
                new_iter_result.append([img_result[x] for x in keep_ids])
            rec_result, rec_img_result = eval_hmean_iou_e2e(pred_boxes, gt_boxes,
                                                            new_iter_result, gt_texts,
                                                            gt_ignored_boxes,
                                                            ignore_capital='sroie' in metrics)

            rec_iter_results[iter_] = rec_result
        eval_results['REC'] = rec_iter_results

    # todo: apply box_threshold for kie_evaluation
    if iters_kie_preds:
        print("Evaluating KIE metrics...")
        kie_iter_results = {}
        # prepare gt for KIE evaluation
        # list[dict], dict = {'entity_key': 'entity_val'}
        gt_kie = []
        for ann_info in ann_infos:
            if 'entity_dict' in ann_info.keys():
                gt_kie.append(ann_info['entity_dict'])
                continue
            kie_dict = dict()
            if sort_for_kie:
                gt_texts, gt_entities, _ = sort_by_read_order(ann_info['ori_texts'], ann_info['ori_entities'],
                                                              [1 for _ in range(len(ann_info['ori_texts']))], ann_info['masks'])
            else:
                gt_texts = ann_info['ori_texts']
                gt_entities = ann_info['ori_entities']
            # for gt_text, gt_entity in zip(ann_info['ori_texts'], ann_info['ori_entities']):
            for gt_text, gt_entity in zip(gt_texts, gt_entities):
                assert len(gt_text) == len(gt_entity)
                for txt_token, iob_tag in zip(gt_text, gt_entity):
                    if iob_tag[0] in ['I', 'B']:
                        cur_tag = iob_tag[2:]
                        assert cur_tag in entity_list
                        if cur_tag not in kie_dict:
                            kie_dict[cur_tag] = txt_token
                        else:
                            kie_dict[cur_tag] += txt_token
            gt_kie.append(kie_dict)
        for iter_, (iter_rec, iter_kie, iter_seq_score) in enumerate(zip(iters_rec_preds, iters_kie_preds, seq_scores)):
            print(f"Evaluating KIE of iter_{iter_} ...")
            pred_kie = []
            assert len(iter_rec) == len(iter_kie) == len(seq_scores)
            if sort_for_kie:
                # prepare box prediction for sorting
                pred_boxes = []

                # for result in results:
                #     pred_boxes.append([x[:-1] for x in result['boundary_result']])

                # modified by whua, add box threshold to screen low confidence result
                for result, keep_ids in zip(results, keep_ids_list):
                    pred_boxes.append([result['boundary_result'][x][:-1] for x in keep_ids])
            else:
                pred_boxes = None
            for idx, (pred_texts_, pred_entities_, seq_score_) in enumerate(zip(iter_rec, iter_kie, seq_scores)):
                kie_dict = dict()
                assert len(pred_texts_) == len(pred_entities_) == len(seq_score_)

                # modified by whua, add box threshold to screen low confidence result
                # Not Test Yet
                pred_texts_ = [pred_texts_[x] for x in keep_ids_list[idx]]
                pred_entities_ = [pred_entities_[x] for x in keep_ids_list[idx]]
                seq_score_ = [seq_score_[x] for x in keep_ids_list[idx]]

                if sort_for_kie:
                    cur_pred_boxes = pred_boxes[idx]
                    pred_texts, pred_entities, seq_score_ = sort_by_read_order(pred_texts_, pred_entities_, seq_score_, cur_pred_boxes)
                else:
                    pred_texts = pred_texts_
                    pred_entities = pred_entities_
                for pred_text, pred_entity, score_ in zip(pred_texts, pred_entities, seq_score_):
                    if score_ < kie_instance_thr:
                        continue
                    if 'sroie' in metrics:
                        entity_within_instance = dict()
                    for char_idx, (txt_token, iob_tag) in enumerate(zip(pred_text, pred_entity)):
                        if iob_tag[0] in ['I', 'B']:
                            cur_tag = iob_tag[2:]
                            assert cur_tag in entity_list, f"cur_tag:{cur_tag}, iob_tag:{iob_tag}"
                            if 'sroie' in metrics:
                                if cur_tag not in entity_within_instance:
                                    entity_within_instance[cur_tag] = txt_token
                                else:
                                    entity_within_instance[cur_tag] += txt_token
                            else:
                                if cur_tag not in kie_dict:
                                    kie_dict[cur_tag] = txt_token
                                else:
                                    kie_dict[cur_tag] += txt_token
                    if 'sroie' in metrics:
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
                if 'sroie' in metrics:
                    kie_dict_merged = dict()
                    for key, val in kie_dict.items():
                        kie_dict_merged[key] = ' '.join(val)
                    kie_dict = kie_dict_merged
                pred_kie.append(kie_dict)

            # if iter_ == 1:
            #     with open('/home/whua/logs/ie_e2e_log/nfv5_disen_kie_results.txt', 'w', encoding='utf-8') as f:
            #         for img_info, pred, gt in zip(img_infos, pred_kie, gt_kie):
            #             f.write(f"img:{img_info['filename']}\n")
            #             f.write(f"pred:{pred}\n  gt:{gt}\n")

            kie_result, kie_img_result = eval_hmean_kie(gt_kie, pred_kie)
            kie_iter_results[iter_] = kie_result
        eval_results['KIE'] = kie_iter_results
    return eval_results

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


def eval_hmean_iou_e2e(pred_boxes,
                       gt_boxes,
                       pred_texts,
                       gt_texts,
                       gt_ignored_boxes,
                       iou_thr=0.5,
                       precision_thr=0.5,
                       ignore_capital=False):
    """Evaluate hmean of text detection using IOU standard.

    Args:
        pred_boxes (list[list[list[float]]]): Text boxes for an img list. Each
            box has 2k (>=8) values.
        gt_boxes (list[list[list[float]]]): Ground truth text boxes for an img
            list. Each box has 2k (>=8) values.
        pred_texts (list[list[str]]): Text transcripts for an img list.
        gt_texts (list[list[str]]): Ground truth text transcripts for an img.
        gt_ignored_boxes (list[list[list[float]]]): Ignored ground truth text
            boxes for an img list. Each box has 2k (>=8) values.
        iou_thr (float): Iou threshold when one (gt_box, det_box) pair is
            matched.
        precision_thr (float): Precision threshold when one (gt_box, det_box)
            pair is matched.
        ignore_capital: Weather to ignore the capital

    Returns:
        hmean (tuple[dict]): Tuple of dicts indicates the hmean for the dataset
            and all images.
    """
    assert utils.is_3dlist(pred_boxes)
    assert utils.is_3dlist(gt_boxes)
    assert utils.is_3dlist(gt_ignored_boxes)
    assert 0 <= iou_thr <= 1
    assert 0 <= precision_thr <= 1

    img_num = len(pred_boxes)
    assert img_num == len(gt_boxes)
    assert img_num == len(gt_ignored_boxes)
    assert img_num == len(gt_texts)
    assert img_num == len(pred_texts)

    for item_box, item_text in zip(pred_boxes, pred_texts):
        assert len(item_box) == len(item_text)
    for item_box, item_text in zip(gt_boxes, gt_texts):
        assert len(item_box) == len(item_text)

    dataset_gt_num = 0
    dataset_pred_num = 0
    dataset_hit_num = 0

    img_results = []

    # for i in tqdm(range(img_num)):
    for i in range(img_num):
        gt = gt_boxes[i]
        gt_ignored = gt_ignored_boxes[i]
        pred = pred_boxes[i]

        # transcripts of instances
        gt_txt = gt_texts[i]
        pred_txt = pred_texts[i]
        if ignore_capital:
            # print(f"before:pred:{pred_txt}, gt:{gt_txt}")
            gt_txt = [x.upper() for x in gt_txt]
            pred_txt = [x.upper() for x in pred_txt]
            # print(f"after:gt:{gt_txt}, pred:{pred_txt}")

        gt_num = len(gt)
        gt_ignored_num = len(gt_ignored)
        pred_num = len(pred)

        hit_num = 0

        # get gt polygons.
        gt_all = gt + gt_ignored
        gt_polys = [eval_utils.points2polygon(p) for p in gt_all]
        gt_ignored_index = [gt_num + i for i in range(len(gt_ignored))]
        gt_num = len(gt_polys)
        pred_polys, _, pred_ignored_index = eval_utils.ignore_pred(
            pred, gt_ignored_index, gt_polys, precision_thr)

        # match.
        if gt_num > 0 and pred_num > 0:
            sz = [gt_num, pred_num]
            iou_mat = np.zeros(sz)

            gt_hit = np.zeros(gt_num, np.int8)
            pred_hit = np.zeros(pred_num, np.int8)

            for gt_id in range(gt_num):
                for pred_id in range(pred_num):
                    gt_pol = gt_polys[gt_id]
                    det_pol = pred_polys[pred_id]

                    iou_mat[gt_id,
                            pred_id] = eval_utils.poly_iou(det_pol, gt_pol)

            for gt_id in range(gt_num):
                for pred_id in range(pred_num):
                    if (gt_hit[gt_id] != 0 or pred_hit[pred_id] != 0
                            or gt_id in gt_ignored_index
                            or pred_id in pred_ignored_index):
                        continue
                    if iou_mat[gt_id, pred_id] > iou_thr and gt_txt[gt_id] == pred_txt[pred_id]:
                        # both iou and sequence prediction are taken into account
                        gt_hit[gt_id] = 1
                        pred_hit[pred_id] = 1
                        hit_num += 1

        gt_care_number = gt_num - gt_ignored_num
        pred_care_number = pred_num - len(pred_ignored_index)

        r, p, h = eval_utils.compute_hmean(hit_num, hit_num, gt_care_number,
                                           pred_care_number)

        img_results.append({'recall': r, 'precision': p, 'hmean': h})

        dataset_hit_num += hit_num
        dataset_gt_num += gt_care_number
        dataset_pred_num += pred_care_number

    dataset_r, dataset_p, dataset_h = eval_utils.compute_hmean(
        dataset_hit_num, dataset_hit_num, dataset_gt_num, dataset_pred_num)

    dataset_results = {
        'num_gts': dataset_gt_num,
        'num_dets': dataset_pred_num,
        'num_match': dataset_hit_num,
        'recall': dataset_r,
        'precision': dataset_p,
        'hmean': dataset_h
    }

    return dataset_results, img_results

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
    # match_nums = {'SS': 0, 'CE-PS': 0, 'CE-P1': 0, 'CE-D': 0, 'CE-PP': 0, 'TF-PS': 0, 'TF-P1': 0, 'TF-D': 0, 'TF-PP': 0,
    #               'SO-PS': 0, 'SO-P1': 0, 'SO-D': 0, 'SO-PP': 0, 'CAR-PS': 0, 'CAR-P1': 0, 'CAR-D': 0, 'CAR-PP': 0,
    #               'PRO-PS': 0, 'PRO-P1': 0, 'PRO-D': 0, 'PRO-PP': 0}
    # gt_nums = {'SS': 0, 'CE-PS': 0, 'CE-P1': 0, 'CE-D': 0, 'CE-PP': 0, 'TF-PS': 0, 'TF-P1': 0, 'TF-D': 0, 'TF-PP': 0,
    #            'SO-PS': 0, 'SO-P1': 0, 'SO-D': 0, 'SO-PP': 0, 'CAR-PS': 0, 'CAR-P1': 0, 'CAR-D': 0, 'CAR-PP': 0,
    #            'PRO-PS': 0, 'PRO-P1': 0, 'PRO-D': 0, 'PRO-PP': 0}
    # pred_nums = {'SS': 0, 'CE-PS': 0, 'CE-P1': 0, 'CE-D': 0, 'CE-PP': 0, 'TF-PS': 0, 'TF-P1': 0, 'TF-D': 0, 'TF-PP': 0,
    #              'SO-PS': 0, 'SO-P1': 0, 'SO-D': 0, 'SO-PP': 0, 'CAR-PS': 0, 'CAR-P1': 0, 'CAR-D': 0, 'CAR-PP': 0,
    #              'PRO-PS': 0, 'PRO-P1': 0, 'PRO-D': 0, 'PRO-PP': 0}
    # total_res = {'SS': 0.0, 'CE-PS': 0.0, 'CE-P1': 0.0, 'CE-D': 0.0, 'CE-PP': 0.0, 'TF-PS': 0.0, 'TF-P1': 0.0,
    #              'TF-D': 0.0, 'TF-PP': 0.0, 'SO-PS': 0.0, 'SO-P1': 0.0, 'SO-D': 0.0, 'SO-PP': 0.0, 'CAR-PS': 0.0,
    #              'CAR-P1': 0.0, 'CAR-D': 0.0, 'CAR-PP': 0.0, 'PRO-PS': 0.0, 'PRO-P1': 0.0, 'PRO-D': 0.0, 'PRO-PP': 0.0}
    # for gt_dict, pred_dict in tqdm(zip(gt_kie, pred_kie)):
    for gt_dict, pred_dict in zip(gt_kie, pred_kie):
        num_gt = len(gt_dict)
        num_pred = len(pred_dict)
        match_num = 0
        for pred_key, pred_val in pred_dict.items():
            if pred_key in gt_dict.keys():
                if gt_dict[pred_key] == pred_val:
                    match_num += 1

        # for gt_key, gt_val in gt_dict.items():
        #     gt_nums[gt_key] += 1
        #
        # for pred_key, pred_val in pred_dict.items():
        #     pred_nums[pred_key] += 1
        #     if pred_key in gt_dict.keys():
        #         # ipdb.set_trace()
        #         if gt_dict[pred_key] == pred_val:
        #             match_num += 1
        #             match_nums[pred_key] += 1
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
    #     # ipdb.set_trace()
    #     file_name = open("/home/jfkuang/entity_result.txt", 'w')
    #     for key, val in match_nums.items():
    #         if pred_nums[key] == 0:
    #             pred_nums[key] = 1
    #         if gt_nums[key] == 0:
    #             gt_nums[key] = 1
    #         precision = match_nums[key]/pred_nums[key]
    #         recall = match_nums[key]/gt_nums[key]
    #         if match_nums[key] == 0:
    #             total_res[key] = 0
    #         else:
    #             total_res[key] = 2*precision*recall/(precision+recall)
    #     # ipdb.set_trace()
    #         file_name.write(key + ':' + str(total_res[key]))
    #         file_name.write('\n')
    #     file_name.close()
    # for key, val in total_res.items():
    #     print(key+":"+str(val))
    #     print('\n')

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





