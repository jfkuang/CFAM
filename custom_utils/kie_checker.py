#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/5/20 20:15
# @Author : WeiHua

import json
import os.path as osp
import cv2
import numpy as np
import copy

def check_kie(anno_json, img_dir, out_dir=None, if_show=False, auto_ajust=False, adjusted_file=None):
    entity_classes = ["company", "address", "date", "total"]
    if out_dir:
        saver = open(out_dir, 'w', encoding='utf-8')
    else:
        saver = None
    annotations = []
    with open(anno_json, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                try:
                    annotations.append(json.loads(line_.strip()))
                except Exception as e:
                    print(line_)
                    raise RuntimeError(e)
    cleaned_annotations = []
    for info_ in annotations:
        img = cv2.imread(osp.join(img_dir, info_['file_name']))
        default_entity_dict = info_['entity_dict']
        cur_entity_dict = dict()
        cleaned_info = copy.copy(info_)
        cleaned_info['annotations'] = []
        for idx, anno in enumerate(info_['annotations']):
            poly = np.array(anno['polygon'], dtype=np.int)
            cv2.polylines(img, [poly.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
            cv2.putText(img, str(idx), (poly[0], poly[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
            if len(anno['entity']) != len(anno['text']):
                if auto_ajust:
                    cls_set = []
                    for ent in anno['entity']:
                        if ent == 'O':
                            if ent not in cls_set:
                                cls_set.append(ent)
                        else:
                            if ent[2:] not in cls_set:
                                cls_set.append(ent[2:])
                    if len(cls_set) == 1:
                        if len(anno['entity']) < len(anno['text']):
                            dif_num = len(anno['text']) - len(anno['entity'])
                            anno['entity'] = anno['entity'] + dif_num * anno['entity'][-1:]
                        else:
                            anno['entity'] = anno['entity'][:len(anno['text'])]
            # print(f"idx:{idx}, text:{anno['text']}")
            assert len(anno['entity']) == len(anno['text']), f"not same length: {len(anno['text'])} vs {len(anno['entity'])}\n" \
                                                             f"{info_['file_name']}, txt:{anno['text']}, entity:{anno['entity']}"
            cleaned_info['annotations'].append(anno)
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
        cleaned_annotations.append(cleaned_info)
        not_same = []
        for key, val in default_entity_dict.items():
            if key not in cur_entity_dict:
                not_same.append(f'\nmiss entity:|{key}|')
            elif cur_entity_dict[key] != val:
                not_same.append(f"\nvalue of |{key}| not the same:\ndefault: |{val}|\ncurrent: |{cur_entity_dict[key]}|")
        for key, val in cur_entity_dict.items():
            if key not in default_entity_dict.keys():
                not_same.append(f"\nUn-expected entity in prediction: |{key}|, value: |{val}|")
        if len(not_same) > 0:
            default_msg = f"\nCompare results of {info_['file_name'].split('/')[-1]}:"
        else:
            default_msg = f"All correct in {info_['file_name'].split('/')[-1]}"
        if saver:
            saver.write(default_msg)
        else:
            print(default_msg)
        for msg in not_same:
            if saver:
                saver.write(msg)
            else:
                print(msg)
        if if_show:
            cv2.imshow('img', img)
            cv2.waitKey(0)
    if saver:
        saver.close()
    if auto_ajust:
        with open(adjusted_file, 'w', encoding='utf-8') as adj_saver:
            for info_ in cleaned_annotations:
                out_str = json.dumps(info_, ensure_ascii=False)
                adj_saver.write(out_str+'\n')
    print("Finish!")


if __name__ == '__main__':
    anno_json = r'E:\Dataset\KIE\SROIE\e2e_format/train_update_screen_v2.txt'
    img_dir = r'E:\Dataset\KIE\SROIE\e2e_format'
    check_kie(anno_json, img_dir, out_dir=None, if_show=False, auto_ajust=True,
              adjusted_file=r'E:\Dataset\KIE\SROIE\e2e_format/train_update_screen_v21.txt')

