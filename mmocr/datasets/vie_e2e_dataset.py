#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/5 15:34
# @Author : WeiHua
import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import random
from functools import cmp_to_key

from mmdet.datasets.builder import DATASETS
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.utils import is_type_list
from mmocr.datasets.pipelines.box_utils import _sort_vertex

from mmcv.utils import print_log
from mmocr.core import eval_vie_e2e, convert_vie_res


@DATASETS.register_module()
class VIEE2EDataset(BaseDataset):
    def __init__(self,
                 ann_file=None,
                 loader=None,
                 dict_file=None,
                 img_prefix='',
                 pipeline=None,
                 test_mode=True,
                 class_file=None,
                 data_type='vie',
                 max_seq_len=60,
                 check_outside=False,
                 order_type='origin',
                 auto_reg=False,
                 pre_load_img=False,
                 pre_parse_anno=False,
                 specify_num=None):
        """
        Dataset for end-to-end visual information extraction. Code is similar to
        "KIEDataset".
        Args:
            ann_file (str): Annotation file path.
            loader (dict): Dictionary to construct loader to load annotation infos.
            dict_file: file contains characters, json format
            img_prefix: Image prefix to generate full image path.
            pipeline: Processing pipeline.
            test_mode: If True, try...except will be turned off in __getitem__.
            class_file: file contains entity classes, json format
            data_type: 'ocr' or 'vie' dataset
            max_seq_len: maximum text sequence length
            check_outside: check if there exists instance outside the image.
            order_type: type of instance order, ['origin', 'sort', 'shuffle']
            auto_reg: if the model is auto-regression
            specify_num: restrict the sample number of dataset to a specific number, then it
                is able to mix datasets with different ratio.
        """
        # choose to load ocr and vie dataset
        assert data_type in ['ocr', 'vie']
        assert order_type in ['origin', 'sort', 'shuffle']
        self.data_type = data_type
        self.max_seq_len = max_seq_len
        self.check_outside = check_outside
        self.order_type = order_type
        self.auto_reg = auto_reg
        self.specify_num = specify_num
        super().__init__(
            ann_file,
            loader,
            pipeline,
            img_prefix=img_prefix,
            test_mode=test_mode)
        assert os.path.exists(dict_file)
        if data_type == 'vie':
            assert os.path.exists(class_file)

        # Todo: convert to '<SOS>', '<EOS>', '<PAD>'
        if not auto_reg:
            # Same as ViTSTR, <GO> acts as the first token and padding token,
            # <END> acts as End token.
            self.ocr_dict = {
                '<GO>': 0,
                '<END>': 1,
                **{
                    val: ind
                    for ind, val in enumerate(load_dict(dict_file), 2)
                }
            }
        else:
            # Same as MASTER, <GO> acts as the first token, <END> acts as end token,
            #   <PAD> acts as padding token.
            self.ocr_dict = {
                '<GO>': 0,
                '<END>': 1,
                '<PAD>': 2,
                **{
                    val: ind
                    for ind, val in enumerate(load_dict(dict_file), 3)
                }
            }
        self.rev_ocr_dict = dict()
        for key, val in self.ocr_dict.items():
            self.rev_ocr_dict[val] = key
        if data_type == 'vie':
            # IOB tagging
            self.entity_list = load_dict(class_file)
            if not self.auto_reg:
                self.entity_dict = {
                    'O': 0,
                }
                for ind, val in enumerate(self.entity_list, 1):
                    self.entity_dict['B-'+val] = 2 * ind - 1
                    self.entity_dict['I-'+val] = 2 * ind
            else:
                self.entity_dict = {
                    'O': 0,
                    '<PAD>': 1,
                }
                self.entity_cls_dict = {
                    'O': 0,
                }
                for ind, val in enumerate(self.entity_list, 1):
                    self.entity_dict['B-' + val] = 2 * ind
                    self.entity_dict['I-' + val] = 2 * ind + 1
                    self.entity_cls_dict[val] = ind

            self.rev_entity_dict = dict()
            for key, val in self.entity_dict.items():
                self.rev_entity_dict[val] = key
            if self.auto_reg:
                # when decoding results, convert '<PAD>' to 'O'
                self.rev_entity_dict[self.entity_dict['<PAD>']] = 'O'
        else:
            self.entity_list = None
            self.entity_dict = None
            self.rev_entity_dict = None

        self.pre_load_img = pre_load_img
        if pre_load_img:
            self.pre_loading_images()

        self.pre_parse_anno = pre_parse_anno
        if pre_parse_anno:
            self.pre_parse_anno_info()

        # # run self check, not necessary
        # if not self.test_mode:
        #     self.run_check()

    def __len__(self):
        if self.specify_num:
            return self.specify_num
        return len(self.data_infos)

    def pre_loading_images(self):
        self.loaded_imgs = dict()
        for index in range(len(self)):
            img_ann_info = self.data_infos[index]
            if self.img_prefix is not None:
                filename = os.path.join(self.img_prefix, img_ann_info['file_name'])
            else:
                filename = img_ann_info['file_name']
            self.loaded_imgs[index] = cv2.imread(filename)
            assert not isinstance(self.loaded_imgs[index], type(None)), f"Invalid image {filename}"

    def prepare_train_img(self, index):
        """
        Prepare image data and annotation for training.
        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
            introduced by pipeline.

        """
        if self.specify_num:
            index = random.randint(0, len(self.data_infos)-1)
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        if self.pre_parse_anno:
            ann_info = self.pared_anno_info[index]
        else:
            ann_info = self._parse_anno_info(img_ann_info['annotations'],
                                             img_width=img_ann_info['width'],
                                             img_height=img_ann_info['height'])
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, index=index)
        return self.pipeline(results)

    def pre_parse_anno_info(self):
        self.pared_anno_info = list()
        for index in range(len(self)):
            img_ann_info = self.data_infos[index]
            ann_info = self._parse_anno_info(img_ann_info['annotations'],
                                             img_width=img_ann_info['width'],
                                             img_height=img_ann_info['height'])
            self.pared_anno_info.append(ann_info)

    def _parse_anno_info(self, annotations, img_width=None, img_height=None):
        """
        Parse annotations of polygons, texts and entity labels for one image.
        Args:
            annotations (list[dict]): Annotations of one image, where each
                dict is for one text instance.

        Returns:
            modify this part later
            dict: A dict containing the following keys:
                - polys
                - bboxes
                - rboxes
                - texts
                - text_inds
                - labels
                - label_inds
        """
        assert is_type_list(annotations, dict)
        assert len(annotations) > 0, 'Please remove data with empty annotation'
        assert 'polygon' in annotations[0]
        assert 'text' in annotations[0]
        if self.data_type == 'vie' and not self.test_mode:
            assert 'entity' in annotations[0]

        # polygon is expected to have the same number of points
        polys, bboxes, rboxes, texts, text_inds = [], [], [], [], []
        rbox_center = []
        rbox_height = []
        if self.check_outside:
            outside_poly = []
        else:
            outside_poly = None
        if self.data_type == 'vie':
            entity_labels, entity_label_inds = [], []
            entity_cls_inds = []
        for ann in annotations:
            if ann['text'] == '###':
                continue
            # polygon, rotate box and bounding box
            poly = ann['polygon']
            if self.check_outside:
                tl_x, tl_y, w, h = cv2.boundingRect(np.array(poly, dtype=np.float32).reshape(-1, 2))
                tmp_bbox = np.array([tl_x, tl_y, tl_x+w, tl_y+h])
                tmp_bbox[0::2] = np.clip(tmp_bbox[0::2], 0, img_width-1)
                tmp_bbox[1::2] = np.clip(tmp_bbox[1::2], 0, img_height-1)
                if tmp_bbox[0] == tmp_bbox[2] or tmp_bbox[1] == tmp_bbox[3]:
                    outside_poly.append(poly)
                    continue
            # Todo: check if this function works fine.
            poly = _sort_vertex(np.array(poly, dtype=np.float32).reshape(-1, 2))
            polys.append([poly.reshape(-1).tolist()])
            rbox = cv2.minAreaRect(poly)
            rbox_center.append(rbox[0])
            rbox_height.append(min(rbox[1][0], rbox[1][1]))
            rboxes.append(cv2.boxPoints(rbox).reshape(-1))
            tl_x, tl_y, w, h = cv2.boundingRect(poly)
            bboxes.append(np.array([tl_x, tl_y, tl_x+w, tl_y+h], dtype=np.float32))
            # text and entity label
            text = ann['text']
            texts.append(text)
            text_ind = [self.ocr_dict[c] for c in text]
            text_inds.append(text_ind)
            if self.data_type == 'vie':
                if ann.get('entity', None):
                    entity = ann['entity']
                    entity_labels.append(entity)
                    entity_label_ind = [self.entity_dict[c] for c in entity]
                    entity_label_inds.append(entity_label_ind)
                    entity_cls_ind = []
                    for c in entity:
                        if 'B-' in c or 'I-' in c:
                            entity_cls_ind.append(self.entity_cls_dict[c[2:]])
                        else:
                            entity_cls_ind.append(self.entity_cls_dict[c])
                    entity_cls_inds.append(entity_cls_ind)

        if self.order_type in ['shuffle', 'sort']:
            n_sample = len(polys)
            # re-order instances
            if self.order_type == 'shuffle':
                sorted_index = [i for i in range(n_sample)]
                random.shuffle(sorted_index)
            else:
                to_sort = [(idx, center, box_h) for idx, (center, box_h) in enumerate(zip(rbox_center, rbox_height))]
                # sorted_data = sorted(to_sort, key=compare_key)
                sorted_data = sorted(to_sort, key=cmp_to_key(compare_key))
                sorted_index = [x[0] for x in sorted_data]
            polys = [polys[idx] for idx in sorted_index]
            bboxes = [bboxes[idx] for idx in sorted_index]
            rboxes = [rboxes[idx] for idx in sorted_index]
            texts = [texts[idx] for idx in sorted_index]
            text_inds = [text_inds[idx] for idx in sorted_index]
            if self.data_type == 'vie':
                if len(entity_labels) > 0:
                    entity_labels = [entity_labels[idx] for idx in sorted_index]
                    entity_label_inds = [entity_label_inds[idx] for idx in sorted_index]
                    entity_cls_inds = [entity_cls_inds[idx] for idx in sorted_index]
        # else:
        #     sorted_index = [idx for idx in range(len(polys))]


        labels = dict(
            instances=np.zeros(len(polys), dtype=np.long),
            texts=self.pad_text_index(text_inds, start_val=self.ocr_dict['<GO>'], end_val=self.ocr_dict['<END>'],
                                      pad_val=self.ocr_dict.get('<PAD>', -1))
        )
        if self.data_type == 'vie':
            if len(entity_labels) > 0:
                labels['entities'] = self.pad_text_index(entity_label_inds, start_val=self.entity_dict['O'],
                                                         end_val=self.entity_dict['O'],
                                                         pad_val=self.entity_dict.get('<PAD>', -1))
                labels['entity_cls'] = self.pad_text_index(entity_cls_inds, start_val=self.entity_cls_dict['O'],
                                                           end_val=self.entity_cls_dict['O'],
                                                           pad_val=-1)
            else:
                labels['entities'] = entity_label_inds
                labels['entity_cls'] = entity_cls_inds

        ann_infos = dict(
            masks=polys,
            rboxes=np.stack(rboxes, axis=0).astype(np.float32),
            bboxes=np.stack(bboxes, axis=0).astype(np.float32),
            ori_texts=texts,
            labels=labels,
            ann_type=self.data_type,
            outside_poly=outside_poly,
            # gt_orders=sorted_index
        )
        if self.data_type == 'vie':
            ann_infos['ori_entities'] = entity_labels
        return ann_infos

    def pad_text_index(self, text_inds, start_val=0, end_val=1, pad_val=-1):
        """Pad texts to the same length"""
        cur_max_len = max([len(x) for x in text_inds]) + 2
        assert cur_max_len < self.max_seq_len, f"Max sequence length exceeds defined. cur: {cur_max_len} vs defined: {self.max_seq_len}"
        if self.auto_reg:
            max_len = min(cur_max_len, self.max_seq_len)
        else:
            max_len = self.max_seq_len
        padded_text_inds = pad_val * np.ones((len(text_inds), max_len), np.int32)
        padded_text_inds[:, 0] = start_val
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, 1:len(text_ind)+1] = np.array(text_ind)
            padded_text_inds[idx, len(text_ind)+1] = end_val
        return padded_text_inds

    def pre_pipeline(self, results, index=None):
        """Prepare results dict for pipeline."""
        # todo: modify this part later.
        # image path prefix
        results['img_prefix'] = self.img_prefix

        # removed
        # # origin orders, for merging instances together later
        # results['gt_orders'] = np.array(results['ann_info']['gt_orders'])

        # XXX_fields is used to specify the annotation name of specific domain
        # polygon annotations are used
        # convert to list[Numpy.ndarray]

        # polygons will maintain list[array], to satisfy different number of points
        # results['gt_polys'] = np.stack([np.array(x[0], dtype=np.float32) for x in results['ann_info']['masks']], axis=0)
        results['gt_polys'] = [np.array(x[0], dtype=np.float32) for x in results['ann_info']['masks']]

        results['bbox_fields'] = ['gt_polys']
        results['rboxes'] = results['ann_info']['rboxes']
        results['bbox_fields'].append('rboxes')

        results['mask_fields'] = []
        # Actually not use.
        results['seg_fields'] = []

        results['ori_texts'] = results['ann_info']['ori_texts']
        if self.data_type == 'vie':
            results['ori_entities'] = results['ann_info']['ori_entities']
        results['filename'] = os.path.join(self.img_prefix,
                                           results['img_info']['filename'])
        results['ori_filename'] = results['img_info']['filename']
        if self.pre_load_img:
            results['ori_filename'] = results['img_info']['filename']
            results['img'] = self.loaded_imgs[index]
            results['img_shape'] = results['img'].shape
            results['ori_shape'] = results['img'].shape
            results['img_fields'] = ['img']
        results['counting'] = {
            'entity_dict': self.entity_dict,
            'entity_list': self.entity_list
        }
        # # a dummy img data, seems no use
        # results['img'] = np.zeros((0, 0, 0), dtype=np.uint8)

    def run_check(self):
        print('-' * 50)
        print('-' * 20 + 'Run Self-Checking for Dataset' + '-' * 20)
        print('-' * 50)
        for i in tqdm(range(len(self))):
            self.prepare_train_img(i)

    def evaluate(self, results, metric='e2e-hmean', logger=None, **kwargs):
        # prepare annotation
        img_infos = []
        ann_infos = []
        print("Preparing ground-truth annotation for evaluation.")
        for i in range(len(self)):
        # for i in tqdm(range(len(self))):
            img_ann_info = self.data_infos[i]
            img_info = {
                'filename': img_ann_info['file_name'],
            }
            ann_info = self._parse_anno_info(img_ann_info['annotations'])
            if 'entity_dict' in img_ann_info.keys():
                ann_info['entity_dict'] = img_ann_info['entity_dict']
            img_infos.append(img_info)
            ann_infos.append(ann_info)
        rec_preds = []
        kie_preds = []
        print("Converting recognition results for evaluation.")
        # for result in tqdm(results):
        for result in results:
            rec_results, kie_results = convert_vie_res(result, len(result['boundary_result']),
                                                       self.ocr_dict, self.rev_ocr_dict,
                                                       self.rev_entity_dict,
                                                       auto_reg=self.auto_reg)
            rec_preds.append(rec_results)
            kie_preds.append(kie_results)
        if rec_preds[0]:
            max_iter = len(rec_preds[0][0])
            iters_rec_preds = []
            iters_seq_scores = []
            if kie_preds[0]:
                iters_kie_preds = []
            else:
                iters_kie_preds = None
            for iter_ in range(max_iter):
                # list[list[transcript]], 1st list -> image, 2rd list -> instances
                rec_transcripts = []
                kie_transcripts = []
                seq_scores = []
                for rec_pred, kie_pred in zip(rec_preds, kie_preds):
                    # list[transcript]
                    rec_transcripts.append([x[iter_][0] for x in rec_pred])
                    seq_scores.append([x[iter_][1] for x in rec_pred])
                    if kie_pred:
                        kie_transcripts.append([x[iter_][0] for x in kie_pred])
                iters_rec_preds.append(rec_transcripts)
                iters_seq_scores.append(seq_scores)
                if kie_preds[0]:
                    iters_kie_preds.append(kie_transcripts)
        else:
            iters_rec_preds, iters_kie_preds = None, None
        return eval_vie_e2e(results, img_infos, ann_infos, metric,
                            iters_rec_preds=iters_rec_preds,
                            iters_kie_preds=iters_kie_preds,
                            entity_list=self.entity_list,
                            seq_scores=seq_scores)

    def compute_metric(self, metric_type):
        raise NotImplementedError

def load_dict(dict_file, encoding='utf-8'):
    """
    Load tokens from file, which are stored in a list.
    Args:
        dict_file: json format
        encoding: default 'utf-8'

    Returns:
        list: a list of tokens.
    """
    assert dict_file[-4:] == 'json'
    with open(dict_file, 'r', encoding=encoding) as f:
        item_list = json.load(f)
    return item_list

# # sort y-dir first, then x-dir
# def compare_key(x):
#     #  x is (index, center), where center is list[x, y]
#     return x[1][1], x[1][0]

def compare_key(a, b):
    # a / b contains: (idx, center(x, y), box_height)
    a_center_x = a[1][0]
    a_center_y = a[1][1]
    a_box_h = a[2]

    b_center_x = b[1][0]
    b_center_y = b[1][1]
    b_box_h = b[2]
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


# ### Run Test Func ###
# if __name__ == '__main__':
#     pass



