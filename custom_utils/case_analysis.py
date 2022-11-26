#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/3 14:37
# @Author : WeiHua
import json
import ast
from tqdm import tqdm
import editdistance


def analysis_kie(result_file, out_file):
    lines = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            lines.append(line_.strip())
    assert len(lines) % 3 == 0
    with open(out_file, 'w', encoding='utf-8') as saver:
        for i in tqdm(range(len(lines)//3)):
            file_ = lines[3*i]
            pred_ = ast.literal_eval(lines[3*i+1][5:])
            gt_ = ast.literal_eval(lines[3*i+2][3:])
            saver.write(f"\n{file_}\n")
            for key_, val_ in pred_.items():
                if key_ not in gt_.keys():
                    saver.write(f"PD:\t{key_}:{val_}\n")
                    saver.write(f"GT:\t{key_}:None\n")
                else:
                    if val_ != gt_[key_]:
                        saver.write(f"PD:\t{key_}:{val_}\n")
                        saver.write(f"GT:\t{key_}:{gt_[key_]}\n")
            for key_, val_ in gt_.items():
                if key_ not in pred_.keys():
                    saver.write(f"PD:\t{key_}:None\n")
                    saver.write(f"GT:\t{key_}:{val_}\n")

def analysis_metric(result_file):
    lines = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            lines.append(line_.strip())
    assert len(lines) % 3 == 0
    total_num_pred = 0
    total_num_gt = 0
    total_num_match = 0
    total_num_match_ignore_lost = 0
    for i in tqdm(range(len(lines) // 3)):
        file_ = lines[3 * i]
        pred_ = ast.literal_eval(lines[3 * i + 1][5:])
        gt_ = ast.literal_eval(lines[3 * i + 2][3:])
        total_num_pred += len(pred_)
        total_num_gt += len(gt_)
        num_match = 0
        num_match_ignore_lost = 0
        for key_, val_ in gt_.items():
            if key_ in pred_.keys():
                if pred_[key_] == val_:
                    num_match += 1
                    num_match_ignore_lost += 1
                elif editdistance.eval(pred_[key_], val_) <= 3:
                    num_match_ignore_lost += 1
            elif key_ in ['total', 'date']:
                total_num_pred += 1
                num_match_ignore_lost += 1
        total_num_match += num_match
        total_num_match_ignore_lost += num_match_ignore_lost
    precision = total_num_match / total_num_pred
    recall = total_num_match / total_num_gt
    fscore = 2 * precision * recall / (precision + recall)
    print(f"precision:{precision}, recall:{recall}, fs:{fscore}")
    precision_ignore_lost = total_num_match_ignore_lost / total_num_pred
    recall_ignore_lost = total_num_match_ignore_lost / total_num_gt
    fscore_ignore_lost = 2 * precision_ignore_lost * recall_ignore_lost / (precision_ignore_lost + recall_ignore_lost)
    print(f"ignore lost:")
    print(f"precision:{precision_ignore_lost}, recall:{recall_ignore_lost}, fscore:{fscore_ignore_lost}")




if __name__ == '__main__':

    result_file = r"/home/whua/logs/ie_e2e_log/nfv5_disen_kie_results.txt"
    out_file = r"/home/whua/logs/ie_e2e_log/nfv5_disen_msg.txt"
    analysis_kie(result_file, out_file)
    # analysis_metric(result_file)