#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/27 21:36
# @Author : WeiHua
import json
import os
import random
import shutil
from tqdm import tqdm

def gather_together(partitions, dirs, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, "image_files"))
    assert isinstance(partitions, list), f"|{partitions}| is supposed to be a list"
    assert len(partitions) == len(dirs), f"Not match: partitions:|{partitions}|, dirs:|{dirs}|"
    samples = []
    sample_names = []
    for part_, dir_ in zip(partitions, dirs):
        print(f"Loading samples from {part_} ...")
        try:
            with open(part_, 'r', encoding='utf-8') as f:
                cur_data = json.load(f)
            samples += cur_data
            for data_ in cur_data:
                if data_['file_name'] in sample_names:
                    raise RuntimeError(f"{data_['file_name']} is duplicated!")
                sample_names.append(data_['file_name'])
                shutil.copyfile(os.path.join(dir_, data_['file_name']), os.path.join(out_dir, data_['file_name']))
        except Exception as e:
            print(f"Error |{e}| occurs, retrying ...")
            cur_data = []
            with open(part_, 'r', encoding='utf-8') as f:
                for line_ in f.readlines():
                    if line_.strip() == "":
                        continue
                    cur_data.append(json.loads(line_.strip()))
            samples += cur_data
            for data_ in cur_data:
                if data_['file_name'] in sample_names:
                    raise RuntimeError(f"{data_['file_name']} is duplicated!")
                sample_names.append(data_['file_name'])
                shutil.copyfile(os.path.join(dir_, data_['file_name']), os.path.join(out_dir, data_['file_name']))
    with open(os.path.join(out_dir, "total_version.txt"), 'w', encoding='utf-8') as f:
        for data_ in samples:
            out_str = json.dumps(data_, ensure_ascii=False)
            f.write(out_str+'\n')
    print(f"In total, {len(samples)} are loaded.")


def split_train_test(data_dir, anno_file, train_ratio_hard=0.3, train_num=2000):
    samples = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            samples.append(json.loads(line_.strip()))
    # split annos to nf_v1, google and openfood
    nf_data = []
    google_data = []
    openfood_data = []
    for anno_ in samples:
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
    entity_list = ['O', 'SS',
                   'CE-PS', 'CE-P1', 'CE-D',
                   'TF-PS', 'TF-P1', 'TF-D',
                   'SO-PS', 'SO-P1', 'SO-D',
                   'CAR-PS', 'CAR-P1', 'CAR-D',
                   'PRO-PS', 'PRO-P1', 'PRO-D']
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
                if cur_type not in train_cls:
                    train_cls.append(cur_type)
    assert len(train_cls) == len(entity_list), f"{train_cls}"
    print(f"In total, Num_train:{len(train_list)}, Num_test:{len(test_list)}")
    with open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8') as saver:
        for info_ in train_list:
            out_str = json.dumps(info_)
            saver.write(out_str+'\n')
    with open(os.path.join(data_dir, 'test.txt'), 'w', encoding='utf-8') as saver:
        for info_ in test_list:
            out_str = json.dumps(info_)
            saver.write(out_str+'\n')

def get_ocr_dict():
    # get ocr dict
    avg_width = 0
    avg_height = 0
    img_cnt = 0
    max_text_len = 0
    with open('../dict_default.json', 'r', encoding='utf-8') as f:
        default_dict = json.load(f)
    ext_dict = []
    entity_cls_list = ["O", "CE-P1", "CE-PS", "CE-D", "TF-P1", "TF-PS", "TF-D", "CAR-P1", "CAR-PS", "CAR-D", "PRO-P1", "PRO-PS", "PRO-D", "SS", "SO-P1", "SO-PS", "SO-D"]
    files = [r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format/total_version_regular.txt"]
    avg_instance_num = 0
    total_instance_num = 0
    max_instance_num = 0
    min_instance_num = 999
    avg_ins_width = 0
    avg_ins_height = 0
    max_pt_num = 0
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
                    avg_ins_width += (max(anno['polygon'][0::2]) - min(anno['polygon'][0::2]))
                    avg_ins_height += (max(anno['polygon'][1::2]) - min(anno['polygon'][1::2]))
                    max_text_len = max(max_text_len, len(anno['text']))
                    # if len(anno['text']) > 60:
                    #     print(anno['text'])
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
    print(f"avg:{total_instance_num / img_cnt}, max:{max_instance_num}, min:{min_instance_num}")
    full_key = default_dict + ext_dict
    print(f"avg_height:{avg_height / img_cnt}, avg_width:{avg_width / img_cnt}")
    print(f"ext key:{ext_dict}")
    print(f"max_len:{max_text_len}")
    print(
        f"avg_ins_height:{avg_ins_height/total_instance_num}")
    """
    Total sample num: 3000
    max_pt_num:20
    avg:35.424, max:206, min:1
    avg_height:566.8573333333334, avg_width:489.234
    ext key:[]
    max_len:122
    avg_ins_height:29.967404396266186
    """

def get_origin_v4(v4_with_full_pts, origin_annos, out_file):
    regular_annos = {}
    for anno_ in origin_annos:
        with open(anno_, 'r', encoding='utf-8') as f:
            for line_ in f.readlines():
                if line_.strip() == "":
                    continue
                tmp = json.loads(line_.strip())
                regular_annos[tmp['file_name']] = tmp
    full_pts_annos = []
    with open(v4_with_full_pts, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            full_pts_annos.append(json.loads(line_))
    with open(out_file, 'w', encoding='utf-8') as f:
        for sample_ in full_pts_annos:
            try:
                out_str = json.dumps(regular_annos[sample_['file_name']])
                f.write(out_str+'\n')
            except Exception as e:
                print(f"Error occurs: {e}")
                print(f"error file name: {sample_['file_name']}")
                exit()

def clean_entity_class(files):
    # clean
    entity_to_rm = ['SF-PS', 'SF-P1', 'SF-D']
    entity_map = {'CE=PS': 'CE-PS'}
    for file_ in files:
        datas = []
        with open(file_, 'r', encoding='utf-8') as f:
            for line_ in f.readlines():
                if len(line_.strip()) > 0:
                    datas.append(json.loads(line_.strip()))
            for i in range(len(datas)):
                annos = datas[i]['annotations']
                new_annos = []
                for ann in annos:
                    if ann['text'] == 'FULL-TABLE':
                        continue
                    new_entity_tags = []
                    for entity_idx in range(len(ann['entity'])):
                        if ann['entity'][entity_idx][0] in ['I', 'B']:
                            cur_type = ann['entity'][entity_idx][2:]
                            if cur_type in entity_map:
                                cur_type = entity_map[cur_type]
                            if cur_type in entity_to_rm:
                                new_entity = 'O'
                            else:
                                new_entity = ann['entity'][entity_idx][:2] + cur_type
                            new_entity_tags.append(new_entity)
                        else:
                            new_entity_tags.append('O')
                    ann['entity'] = new_entity_tags
                    new_annos.append(ann)
                datas[i]['annotations'] = new_annos
        with open(file_, 'w', encoding='utf-8') as saver:
            for data in datas:
                out_str = json.dumps(data)
                saver.write(out_str+'\n')

if __name__ == '__main__':
    partitions = [r"E:\Dataset\Nutrition_Facts_Formal\e2e_format/final_jfkuang.txt",
                  r"E:\Dataset\Nutrition_Facts_Formal\Final_ver\modified_e2e_format/added.txt"]
    dirs = [r"E:\Dataset\Nutrition_Facts_Formal\e2e_format",
            r"E:\Dataset\Nutrition_Facts_Formal\Final_ver\modified_e2e_format"]
    out_dir = r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format"
    # gather_together(partitions, dirs, out_dir)

    data_dir = r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format"
    anno_file = r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format/total_version.txt"
    # split_train_test(data_dir, anno_file)

    v4_with_full_pts = r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format/total_version.txt"
    origin_annos = [r"E:\Dataset\Nutrition_Facts_Formal\Final_ver\modified_e2e_format/added - 副本.txt",
                    r"E:\Dataset\Nutrition_Facts_Formal\e2e_format\others/google.txt",
                    r"E:\Dataset\Nutrition_Facts_Formal\e2e_format\others/nf.txt",
                    r"E:\Dataset\Nutrition_Facts_Formal\e2e_format\others/openfood.txt"]
    out_file = r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format/total_version_regular.txt"
    # get_origin_v4(v4_with_full_pts, origin_annos, out_file)
    # clean_entity_class([out_file])
    get_ocr_dict()
    v4_with_full_pts = r"E:\Dataset\Nutrition_Facts_Formal\V4\e2e_format/test.txt"
    origin_annos = [r"E:\Dataset\Nutrition_Facts_Formal\V4/e2e_format/total_version_regular.txt"]
    out_file = r"E:\Dataset\Nutrition_Facts_Formal\V4\e2e_format/test_regular.txt"
    # get_origin_v4(v4_with_full_pts, origin_annos, out_file)