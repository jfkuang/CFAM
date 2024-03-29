#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/27 16:26
# @Author : WeiHua
import os
import shutil

import cv2
import glob
import os.path as osp
from tqdm import tqdm
import numpy as np
import json

from prepare_nutritionfacts import text2iob_label_with_box_and_within_box_exactly_level

#todo:
#   1. generate e2e-format dataset, for test, directly use the official entity result (Done)
#   2. modify evaluate function
#       2.1 use official entity result if provided, otherwise generated by char-level entity tag (Done)
#       2.2 specifically for SROIE, when entity across lines, use ' ' to concat them. (Done)
#       2.3 remember to modify the config, especially the max_len of text and entity_class_num, text_class_num (Done)

# todo: 1. 重新整理数据集：裁剪 & 校正KIE是否有误 & 去掉***内容 (Done) 2. eval时全部转大写 (Done) 3. eval时去掉重复box (Done)
"""
Ref:
[1] https://github.com/BlackStar1313/ICDAR-2019-RRC-SROIE/tree/master/keyword_information_extraction/data
[2]https://github.com/zzzDavid/ICDAR-2019-SROIE/tree/master
"""
# def convert_sroie(img_dir, ocr_dir, kie_dir, ann_dir):
#     char_kie_ann = torch.load(ann_dir)
#     ocr_anns = dict()
#     for ocr_file in glob.glob(osp.join(ocr_dir, '/*.txt')):
#
#     for key, val in char_kie_ann.items():


def prepare_sroie(src_img, src_ocr, src_kie, out_dir, data_type, entity_classes):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_img_dir = osp.join(out_dir, 'image_files')
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    if data_type == 'train':
        saver = open(osp.join(out_dir, 'train.txt'), 'w', encoding='utf-8')
        files = glob.glob(osp.join(src_img, '*.jpg'))
        files = [x.replace('\\', '/').split('/')[-1].split('.')[0] for x in files]
        invalid_img = 0
        invalid_sample = 0
        total_img = 0
        total_sample = 0
        for file_ in tqdm(files):
            total_img += 1
            is_invalid_img = False
            ocr_file = osp.join(src_ocr, f'{file_}.txt')
            img_file = osp.join(src_img, f'{file_}.jpg')
            kie_file = osp.join(src_kie, f'{file_}.txt')
            boxes = []
            transcripts = []
            box_entity_types = []
            img = cv2.imread(img_file)
            # crop image
            tl_x, tl_y = 9999, 9999
            br_x, br_y = 0, 0
            with open(ocr_file, 'r', encoding='utf-8') as f:
                for line_ in f.readlines():
                    info = line_.strip().split('\t')
                    assert len(info) == 10, f"line:{line_}"
                    tmp_box = np.array(info[:8], dtype=np.float32)
                    # x, y, w, h = cv2.boundingRect(np.array(info[:8], dtype=np.float32).reshape(-1, 2))
                    min_x = int(min(tmp_box[0::2]))
                    min_y = int(min(tmp_box[1::2]))
                    max_x = int(max(tmp_box[0::2]))
                    max_y = int(max(tmp_box[1::2]))
                    # min_x = int(x)
                    # min_y = int(y)
                    # max_x = int(x+w)
                    # max_y = int(y+h)
                    tl_x = min(min_x, tl_x)
                    tl_y = min(min_y, tl_y)
                    br_x = max(max_x, br_x)
                    br_y = max(max_y, br_y)
            tl_x = max(0, tl_x)
            tl_y = max(0, tl_y)
            br_x = min(img.shape[1], br_x+5)
            br_y = min(img.shape[0], br_y+5)
            img = img[tl_y: br_y, tl_x: br_x, :]

            height, width = img.shape[:2]
            out_info = dict(
                file_name=f"image_files/{file_}.jpg",
                height=height,
                width=width,
                annotations=[]
            )
            with open(ocr_file, 'r', encoding='utf-8') as f:
                for line_ in f.readlines():
                    total_sample += 1
                    info = line_.strip().split('\t')
                    assert len(info) == 10, f"line:{line_}"
                    if info[8] == '***':
                        continue
                    # check if exists invalid text instance
                    cur_box = np.array(info[:8], dtype=np.float32)
                    cur_box[0::2] -= tl_x
                    cur_box[1::2] -= tl_y
                    x, y, w, h = cv2.boundingRect(cur_box.reshape(-1, 2))
                    bbox = np.array([x, y, x + w, y + h])
                    bbox[0::2] = np.clip(bbox[0::2], 0, width)
                    bbox[1::2] = np.clip(bbox[1::2], 0, height)
                    if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                        print(f"line_:{line_}, box:{cur_box}, w:{width}, h:{width}, tl_x:{tl_x}, tl_y:{tl_y}")
                        return
                        is_invalid_img = True
                        invalid_sample += 1
                    tmp_box = np.array(info[:8], dtype=np.int)
                    tmp_box[0::2] -= tl_x
                    tmp_box[1::2] -= tl_y

                    boxes.append(tmp_box.tolist())
                    transcripts.append(info[8])
                    box_entity_types.append(info[9])
            with open(kie_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
            iob_tags = text2iob_label_with_box_and_within_box_exactly_level(box_entity_types,
                                                                            transcripts,
                                                                            entities,
                                                                            ['address'],
                                                                            Entities_list=entity_classes)
            assert len(iob_tags) == len(transcripts) == len(boxes)
            for i in range(len(boxes)):
                ann = {
                    "polygon": boxes[i],
                    "text": transcripts[i],
                    "entity": iob_tags[i]
                }
                out_info['annotations'].append(ann)
            out_str = json.dumps(out_info)
            saver.write(out_str + '\n')
            if is_invalid_img:
                invalid_img += 1
            cv2.imwrite(osp.join(out_img_dir, f"{file_}.jpg"), img)
            # shutil.copyfile(img_file, osp.join(out_img_dir, f"{file_}.jpg"))
        saver.close()
        print(f"img: invalid/all: {invalid_img}/{total_img}")
        print(f"sample: invalid/all: {invalid_sample}/{total_sample}")
    else: # bug here
        saver = open(osp.join(out_dir, 'test.txt'), 'w', encoding='utf-8')
        files = glob.glob(osp.join(src_img, '*.jpg'))
        files = [x.replace('\\', '/').split('/')[-1].split('.')[0] for x in files]
        invalid_img = 0
        invalid_sample = 0
        total_img = 0
        total_sample = 0
        for file_ in tqdm(files):
            total_img += 1
            is_invalid_img = False
            ocr_file = osp.join(src_ocr, f'{file_}.txt')
            img_file = osp.join(src_img, f'{file_}.jpg')
            kie_file = osp.join(src_kie, f'{file_}.txt')
            boxes = []
            transcripts = []
            box_entity_types = []
            img = cv2.imread(img_file)
            # crop image
            tl_x, tl_y = 9999, 9999
            br_x, br_y = 0, 0
            with open(ocr_file, 'r', encoding='utf-8') as f:
                for line_ in f.readlines():
                    if line_.strip() == '':
                        continue
                    info = line_.strip().split(',')
                    tmp_box = np.array(info[:8], dtype=np.float32)
                    min_x = int(min(tmp_box[0::2]))
                    min_y = int(min(tmp_box[1::2]))
                    max_x = int(max(tmp_box[0::2]))
                    max_y = int(max(tmp_box[1::2]))
                    tl_x = min(min_x, tl_x)
                    tl_y = min(min_y, tl_y)
                    br_x = max(max_x, br_x)
                    br_y = max(max_y, br_y)
            tl_x = max(0, tl_x)
            tl_y = max(0, tl_y)
            br_x = min(img.shape[1], br_x + 5)
            br_y = min(img.shape[0], br_y + 5)
            img = img[tl_y: br_y, tl_x: br_x, :]

            height, width = img.shape[:2]
            out_info = dict(
                file_name=f"image_files/{file_}.jpg",
                height=height,
                width=width,
                annotations=[]
            )
            with open(ocr_file, 'r', encoding='utf-8') as f:
                for line_ in f.readlines():
                    if line_.strip() == '':
                        continue
                    total_sample += 1
                    info = line_.strip().split(',')
                    pts = np.array(info[:8], np.int)
                    pts[0::2] -= tl_x
                    pts[1::2] -= tl_y
                    pts = pts.tolist()
                    content = ",".join(info[8:])
                    if content == '***':
                        continue
                    # check if exists invalid text instance
                    try:
                        # check if exists invalid text instance
                        cur_box = np.array(info[:8], dtype=np.float32)
                        cur_box[0::2] -= tl_x
                        cur_box[1::2] -= tl_y
                        x, y, w, h = cv2.boundingRect(cur_box.reshape(-1, 2))
                    except Exception as e:
                        raise ValueError(f"error:{e}, pts:{pts}, line:{line_}, file:{ocr_file}")
                    bbox = np.array([x, y, x + w, y + h])
                    bbox[0::2] = np.clip(bbox[0::2], 0, width)
                    bbox[1::2] = np.clip(bbox[1::2], 0, height)
                    if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                        is_invalid_img = True
                        invalid_sample += 1
                    boxes.append(pts)
                    transcripts.append(content)
            with open(kie_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
            out_info['entity_dict'] = entities
            assert len(transcripts) == len(boxes)
            for i in range(len(boxes)):
                ann = {
                    "polygon": boxes[i],
                    "text": transcripts[i]
                }
                out_info['annotations'].append(ann)
            out_str = json.dumps(out_info)
            saver.write(out_str + '\n')
            if is_invalid_img:
                invalid_img += 1
            cv2.imwrite(osp.join(out_img_dir, f"{file_}.jpg"), img)
            # shutil.copyfile(img_file, osp.join(out_img_dir, f"{file_}.jpg"))
        saver.close()
        print(f"img: invalid/all: {invalid_img}/{total_img}")
        print(f"sample: invalid/all: {invalid_sample}/{total_sample}")


def vis_sroie():
    # get ocr dict
    avg_width = 0
    avg_height = 0
    img_cnt = 0
    max_text_len = 0
    with open('../dict_default.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    ext_dict = []
    # files = ['E:/Dataset/KIE/SROIE/e2e_format/train.txt', 'E:/Dataset/KIE/SROIE/e2e_format/train.txt']
    files = ['E:/Dataset/KIE/SROIE/e2e_format/test.txt', 'E:/Dataset/KIE/SROIE/e2e_format/train_update.txt']
    avg_instance_num = 0
    total_instance_num = 0
    max_instance_num = 0
    min_instance_num = 999
    img_dir = 'E:/Dataset/KIE/SROIE/e2e_format'
    specify_file = 'X51005255805.'
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                if specify_file:
                    if specify_file not in info_['file_name']:
                        continue
                avg_height += info_['height']
                avg_width += info_['width']
                img = cv2.imread(osp.join(img_dir, info_['file_name']))
                for idx, anno in enumerate(info_['annotations']):
                    poly = np.array(anno['polygon'], dtype=np.int)
                    cv2.polylines(img, [poly.reshape(-1, 2)], isClosed=True, thickness=1, color=(0, 0, 255))
                    cv2.putText(img, str(idx), (poly[0], poly[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    print(f"idx:{idx}, text:{anno['text']}")
                cv2.imshow('img', img)
                cv2.waitKey(0)

def check_kie():
    with open('../dict_default.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    # files = ['E:/Dataset/KIE/SROIE/e2e_format/train.txt', 'E:/Dataset/KIE/SROIE/e2e_format/train.txt']
    entity_classes = ["company", "address", "date", "total"]
    files = ['E:/Dataset/KIE/SROIE/e2e_format/train.txt']
    img_dir = 'E:/Dataset/KIE/SROIE/e2e_format'
    entity_dir = 'E:/Dataset/KIE/SROIE/0325updated.task2train(626p)/0325updated.task2train(626p)'
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                img = cv2.imread(osp.join(img_dir, info_['file_name']))
                entity_file = info_['file_name'].split('/')[-1].replace('jpg', 'txt')
                with open(osp.join(entity_dir, entity_file), 'r', encoding='utf-8') as ef:
                    default_entity_dict = json.load(ef)
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

def find_invalid_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                try:
                    tmp = json.loads(line_.strip())
                    for ins_ in tmp['annotations']:
                        if len(ins_['text']) != len(ins_['entity']):
                            print(tmp['file_name'])
                            print(ins_)
                            exit()
                    print(tmp['file_name'])
                except Exception as e:
                    print(f"Invalid line:{line_.strip()}")
                    raise RuntimeError(f"Error: {e}")

def find_invalid_json_and_refine(json_file):
    samples = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip():
                try:
                    tmp = json.loads(line_.strip())
                    fine_annos = []
                    for ins_ in tmp['annotations']:
                        if len(ins_['text']) != len(ins_['entity']):
                            # clean it
                            if len(ins_['text']) > len(ins_['entity']):
                                dis_len = len(ins_['text']) - len(ins_['entity'])
                                ins_['entity'].extend(ins_['entity'][-1:]*dis_len)
                            else:
                                ins_['entity'] = ins_['entity'][:len(ins_['text'])]
                        fine_annos.append(ins_)
                    tmp['annotations'] = fine_annos
                    samples.append(tmp)
                except Exception as e:
                    print(f"Invalid line:{line_.strip()}")
                    raise RuntimeError(f"Error: {e}")

    refined_out = json_file.replace('.txt', '_refined.txt')
    with open(refined_out, 'w', encoding='utf-8') as f:
        for ins_ in samples:
            out_str = json.dumps(ins_, ensure_ascii=False)
            f.write(out_str+'\n')


if __name__ == '__main__':
    json_file = r"E:\Dataset\KIE\SROIE\e2e_format\train_update_screen_v1_refined.txt"
    find_invalid_json(json_file)
    # find_invalid_json_and_refine(json_file)

    # vis_sroie()
    # src_dir = 'E:/Dataset/KIE/SROIE/sroie_human_anno'
    # out_dir = 'E:/Dataset/KIE/SROIE/e2e_format'
    #
    # default_train = r'E:\Dataset\KIE\SROIE\SROIE_test_images_task_3'
    # imgs = glob.glob(osp.join(default_train, '*.jpg'))
    # default_imgs = [img.replace('\\', '/').split('/')[-1].split('.')[0] for img in imgs]
    #
    # anno_train = r'E:\Dataset\KIE\SROIE\sroie_human_anno\test_ocr'
    # imgs = glob.glob(osp.join(anno_train, '*.jpg'))
    # anno_imgs = [img.replace('\\', '/').split('/')[-1].split('.')[0] for img in imgs]
    #
    # print(f"default:{len(default_imgs)}, anno:{len(anno_imgs)}")
    # not_exist = []
    # for img in default_imgs:
    #     if img not in anno_imgs:
    #         not_exist.append(img)
    # print(f"not exists: {not_exist}")

    # src_img = r'E:\Dataset\KIE\SROIE\0325updated.task2train(626p)\0325updated.task2train(626p)'
    # src_ocr = r'E:\Dataset\KIE\SROIE\sroie_human_anno\train'
    # src_kie = r'E:\Dataset\KIE\SROIE\0325updated.task2train(626p)\0325updated.task2train(626p)'
    # out_dir = 'E:/Dataset/KIE/SROIE/e2e_format'
    # data_type = 'train'

    src_img = r'E:\Dataset\KIE\SROIE\task3-test(347p)\task3-test(347p)'
    src_img = r'E:\Dataset\KIE\SROIE\sroie_human_anno\test_ocr'
    src_ocr = r'E:\Dataset\KIE\SROIE\sroie_human_anno\test_ocr'
    src_kie = r'E:\Dataset\KIE\SROIE\SROIE_test_gt_task_3'
    out_dir = 'E:/Dataset/KIE/SROIE/e2e_format'
    data_type = 'test'
    #
    entity_classes = ["company", "address", "date", "total"]
    # prepare_sroie(src_img, src_ocr, src_kie, out_dir, data_type, entity_classes)

    # with open(osp.join(out_dir, 'class_list.json'), 'w', encoding='utf-8') as f:
    #     json.dump(entity_classes, f, ensure_ascii=False)

    # # get ocr dict
    # avg_width = 0
    # avg_height = 0
    # img_cnt = 0
    # max_text_len = 0
    # with open('../dict_default.json', 'r', encoding='utf-8') as f:
    #     default_dict = json.load(f)
    # ext_dict = []
    # files = ['E:/Dataset/KIE/SROIE/e2e_format/train.txt', 'E:/Dataset/KIE/SROIE/e2e_format/test.txt']
    # # files = ['E:/Dataset/KIE/Rec_Pretrain/custom_eng_v1/annotation.txt']
    # avg_instance_num = 0
    # total_instance_num = 0
    # max_instance_num = 0
    # min_instance_num = 999
    # avg_ins_width = 0
    # avg_ins_height = 0
    # num_pts = []
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
    #                 if len(anno['polygon']) not in num_pts:
    #                     num_pts.append(len(anno['polygon']))
    #                 avg_ins_width += (max(anno['polygon'][0::2]) - min(anno['polygon'][0::2]))
    #                 avg_ins_height += (max(anno['polygon'][1::2]) - min(anno['polygon'][1::2]))
    #                 max_text_len = max(max_text_len, len(anno['text']))
    #                 # if len(anno['text']) > 60:
    #                 #     print(anno['text'])
    #                 for char_ in anno['text']:
    #                     if char_ not in default_dict:
    #                         if char_ not in ext_dict:
    #                             ext_dict.append(char_)
    # print(f"point num of instance:{num_pts}")
    # print(f"avg:{total_instance_num/img_cnt}, max:{max_instance_num}, min:{min_instance_num}")
    # full_key = default_dict + ext_dict
    # print(f"avg_height:{avg_height / img_cnt}, avg_width:{avg_width / img_cnt}")
    # print(f"ext key:{ext_dict}")
    # print(f"max_len:{max_text_len}")
    # print(f"avg_ins_height:{avg_ins_height/total_instance_num}, avg_ins_width:{avg_ins_width/total_instance_num}")


    # with open('E:/Dataset/KIE/SROIE/e2e_format/dict.json', 'w', encoding='utf-8') as f:
    #     json.dump(full_key, f, ensure_ascii=False)
    # check_kie()

    # # add gt_kie for train set
    # data_infos = []
    # with open('E:/Dataset/KIE/SROIE/e2e_format/train.txt', 'r', encoding='utf-8') as f:
    #     for line_ in f.readlines():
    #         if len(line_.strip()) == 0:
    #             continue
    #         data_infos.append(json.loads(line_.strip()))
    # kie_anno_dir = r'E:\Dataset\KIE\SROIE\0325updated.task2train(626p)\0325updated.task2train(626p)'
    # with open('E:/Dataset/KIE/SROIE/e2e_format/train_update.txt', 'w', encoding='utf-8') as saver:
    #     for data_info in data_infos:
    #         file_name = data_info['file_name'].split('/')[-1].replace('.jpg', '.txt')
    #         with open(osp.join(kie_anno_dir, file_name), 'r', encoding='utf-8') as f:
    #             kie_dict = json.load(f)
    #         data_info['entity_dict'] = kie_dict
    #         out_str = json.dumps(data_info)
    #         saver.write(out_str+'\n')



    # vis_sroie()
    """    
    avg:53.76978417266187, max:153, min:18
    avg_height:1630.6022610483042, avg_width:764.3535457348407
    ext key:['·']
    max_len:69
    avg_ins_height:36.03906877174204, avg_ins_width:207.93608318360793
    """

"""
Output dataset format:
Dataset Format: (similar to WileReceipt)
└── Nutrition Facts
  ├── class_list.json
  ├── dict.json
  ├── image_files
  ├── test.txt ~ 347
  └── train.txt ~ 626

For test.txt & train.txt:
    Each line is a dict, includes [file_name, height, width, annotations, entity_dict (optional -> required in sroie)].
    "annotations" corresponding to a list, each of which is a dict and contains keys:
    {
        "polygon": polygon, [x, y, x, y ...], expected to have the same number of vertex
        "text": transcript,
        "entity": entity tag in token-level, which is IOB tag format, e.g.,
            [entity_of_token_1, entity_of_token_2, ...]
    },
    "entity_dict" indicates a map from entity key to entity value. If it is provided, then it will
     directly be used to evaluate the KIE results. It is formatted as below:
    {
        "entity_key": "entity_val"
    }

"""




