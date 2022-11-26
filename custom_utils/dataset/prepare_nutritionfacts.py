#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/5 16:34
# @Author : WeiHua
import glob
import os
import os.path as osp
import shutil
import json
import cv2
import re

import numpy as np
from tqdm import tqdm
import string


def convert_dataset(train_dir, val_dir, out_dir,
                    entity_classes):
    """Convert MTP format data to required format for OpenMM-based code"""
    if osp.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    out_img_dir = osp.join(out_dir, 'image_files')
    os.mkdir(out_img_dir)
    iter_ = [
        {
            'file': 'train.txt',
            'dir': train_dir
        },
        {
            'file': 'test.txt',
            'dir': val_dir
        }
    ]
    char_replace = {
        '”': '"',
        '’': "'",
        '，': ',',
    }
    abandon_img_num = 0
    full_keys = "! \"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    for item_ in iter_:
        with open(osp.join(out_dir, item_['file']), 'w', encoding='utf-8') as saver:
            imgs = glob.glob(osp.join(item_['dir'], 'images/*.jpg'))
            ocr_dir = osp.join(item_['dir'], 'boxes_and_transcripts')
            kie_dir = osp.join(item_['dir'], 'entities')
            for file in tqdm(imgs):
                file_name = file.replace('\\', '/').split('/')[-1][:-4]
                shutil.copyfile(file, osp.join(out_img_dir, f'{file_name}.jpg'))
                img_shape = cv2.imread(file).shape
                info_ = dict(
                    file_name=f"image_files/{file_name}.jpg",
                    height=img_shape[0],
                    width=img_shape[1],
                    annotations=[]
                )
                boxes_and_transcripts_data = read_gt_file_with_box_entity_type(osp.join(ocr_dir, f'{file_name}.tsv'),
                                                                               read_method='match')
                boxes, transcripts, box_entity_types = [], [], []
                abandon_flag = False
                for index, points, transcript, entity_type in boxes_and_transcripts_data:
                    tl_x, tl_y, w, h = cv2.boundingRect(np.array(points, dtype=np.float32).reshape(-1, 2))
                    bbox = np.array([tl_x, tl_y, tl_x+w, tl_y+h])
                    bbox[0::2] = np.clip(bbox[0::2], 0, img_shape[1])
                    bbox[1::2] = np.clip(bbox[1::2], 0, img_shape[0])
                    if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                        # for annotation exceeds the boundary of image, remove it.
                        print(f"Invalid annotation instance detected, remove it.")
                        continue
                    if len(transcript) == 0:
                        transcript = ' '
                    new_transcript = ""
                    for char_ in transcript:
                        if char_ not in full_keys:
                            if char_ in char_replace.keys():
                                new_transcript += char_replace[char_]
                            else:
                                abandon_flag = True
                                break
                        else:
                            new_transcript += char_
                    boxes.append(points)
                    transcripts.append(new_transcript)
                    box_entity_types.append(entity_type)
                if abandon_flag:
                    abandon_img_num += 1
                    continue
                with open(osp.join(kie_dir, f'{file_name}.txt'), 'r', encoding='utf-8') as f:
                    entities = json.load(f)
                iob_tags = text2iob_label_with_box_and_within_box_exactly_level(box_entity_types,
                                                                                transcripts,
                                                                                entities,
                                                                                [],
                                                                                Entities_list=entity_classes)
                assert len(iob_tags) == len(transcripts) == len(boxes)
                for i in range(len(boxes)):
                    ann = {
                        "polygon": boxes[i],
                        "text": transcripts[i],
                        "entity": iob_tags[i]
                    }
                    info_['annotations'].append(ann)
                out_str = json.dumps(info_)
                saver.write(out_str+'\n')
    # 12 images are removed in nutrition_facts_v1 dataset.
    print(f"abandon_img_num:{abandon_img_num}")


def read_gt_file_with_box_entity_type(filepath: str, read_method='match'):
    if read_method == 'match':
        with open(filepath, 'r', encoding='utf-8') as f:
            document_text = f.read()

        # match pattern in document: index,x1,y1,x2,y2,x3,y3,x4,y4,transcript,box_entity_type
        regex = r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*," \
                r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,(.*),(.*)\n?$"

        matches = re.finditer(regex, document_text, re.MULTILINE)

        res = []
        for matchNum, match in enumerate(matches, start=1):
            index = int(match.group(1))
            points = [float(match.group(i)) for i in range(2, 10)]
            transcription = str(match.group(10))
            entity_type = str(match.group(11))
            res.append((index, points, transcription, entity_type))
        return res
    elif read_method == 'json':
        # 'ocr_idx' 'coord' 'text' 'entity'
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        res = []
        for per_data in data:
            try:
                cur_data = (per_data['ocr_idx'],
                            per_data['coord'],
                            per_data['text'],
                            per_data['entity'])
            except Exception as e:
                print(f"per_data:{per_data}")
                print(f"data:{data}")
                print(f"file:{filepath}")
                raise ValueError(e)
            res.append(cur_data)
        return res
    else:
        raise ValueError(f"Invalid read method:{read_method}")


def text2iob_label_with_box_and_within_box_exactly_level(annotation_box_types,
                                                         transcripts,
                                                         exactly_entities_label,
                                                         box_level_entities,
                                                         Entities_list=None):
    '''
     box_level_entities will perform box level tagging, others will perform exactly matching within specific box.
    :param annotation_box_types: each transcripts box belongs to the corresponding entity types
    :param transcripts: transcripts of documents
    :param exactly_entities_label: exactly entity type and entity value of documents
    :param box_level_entities: using box level label tagging, this result is same as
                    function of text2iob_label_with_box_level_match
    :return:
    '''

    def exactly_match_within_box(transcript: str, entity_type: str, entity_exactly_value: str):
        '''
        perform exactly match in the scope of current box
        :param transcript: the transcript of current box
        :param entity_type: the entity type of current box
        :param entity_exactly_value: exactly label value of corresponding entity type
        :return:
        '''
        matched = False

        # Preprocess remove the punctuations and whitespaces.
        # -- Remove this step, since some punctuations are actually exists in ground-truth (such as '(' and ')')
        # (src_seq, src_idx), (tgt_seq, _) = preprocess_transcripts(transcript), preprocess_transcripts(
        #     entity_exactly_value)
        if len(entity_exactly_value) < 5:
            src_seq, src_idx = rm_blank_in_transcripts(transcript)
            tgt_seq, _ = rm_blank_in_transcripts(entity_exactly_value)
        else:
            # -- Use raw input
            src_seq = transcript
            src_idx = [i for i in range(len(transcript))]
            tgt_seq = entity_exactly_value

        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
            return matched, None

        result_tags = ['O'] * len(transcript)
        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i + tgt_len] == tgt_seq:
                matched = True
                tag = ['I-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                tag[0] = 'B-{}'.format(entity_type)
                result_tags[src_idx[i]:src_idx[i + tgt_len - 1] + 1] = tag
                break

        return matched, result_tags

    assert Entities_list, "Entity list is required."
    tags = []
    for entity_type, transcript in zip(annotation_box_types, transcripts):
        entity_type = entity_type.strip()
        if entity_type in Entities_list:
            matched, resulted_tag = False, None
            if entity_type not in box_level_entities:
                matched, resulted_tag = exactly_match_within_box(transcript, entity_type,
                                                                 exactly_entities_label[entity_type])

            if matched:
                tags.append(resulted_tag)
            else:
                tag = ['I-{}'.format(entity_type)] * len(transcript)
                tag[0] = 'B-{}'.format(entity_type)
                tags.append(tag)
        else:
            tags.append(['O'] * len(transcript))

    return tags

def rm_blank_in_transcripts(transcripts):
    '''
    preprocess texts into separated word-level list, this is helpful to matching tagging label between source and target label,
    e.g. source: xxxx hello ! world xxxx  target: xxxx hello world xxxx,
    we want to match 'hello ! world' with 'hello world' to decrease the impact of ocr bad result.
    :param transcripts:
    :return: seq: the cleaned sequence, idx: the corresponding indices.
    '''
    seq, idx = [], []
    for index, x in enumerate(transcripts):
        if x != ' ':
            seq.append(x)
            idx.append(index)
    return seq, idx

if __name__ == '__main__':
    # train_dir = '/home/whua/datasets/table_image/train'
    # val_dir = '/home/whua/datasets/table_image/validate'
    # out_dir = '/home/whua/datasets/mm_format/table'
    train_dir = '/home/whua/code/PICK-goods/data/nf_dataset/mtp_format/table_image/train'
    val_dir = '/home/whua/code/PICK-goods/data/nf_dataset/mtp_format/table_image/validate'
    out_dir = '/mnt/whua/ie_e2e_data/mm_format/table'
    entity_classes = [
        "Serving size",
        "Calories",
        "Fat_aps",
        "Fat_dv",
        "Sodium_aps",
        "Sodium_dv",
        "Carb._aps",
        "Carb._dv",
        "Protein"
    ]
    convert_dataset(train_dir, val_dir, out_dir,
                    entity_classes)

    # prepare entity and alphabet list
    out_ocr_key = f"{out_dir}/dict.json"
    out_kie_key = f"{out_dir}/class_list.json"
    tmp_keys = string.printable[:-6]
    full_keys = "! \"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    not_exist = []
    for key in full_keys:
        if key not in tmp_keys:
            not_exist.append(key)
    print(f"Not exist keys:{not_exist}")
    full_keys = [x for x in tmp_keys] + not_exist
    with open(out_ocr_key, 'w', encoding='utf-8') as f:
        json.dump(full_keys, f, ensure_ascii=False)
    with open(out_kie_key, 'w', encoding='utf-8') as f:
        json.dump(entity_classes, f, ensure_ascii=False)








