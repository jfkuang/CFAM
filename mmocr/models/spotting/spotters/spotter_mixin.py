#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/8 15:31
# @Author : WeiHua

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from mmocr.core import imshow_e2e_result
from mmocr.datasets.vie_e2e_dataset import load_dict

# Todo: modify this module.
class SpotterMixin:
    """Base class for text detector, only to show results.

    Args:
        show_score (bool): Whether to show text instance score.
    """

    def __init__(self, show_score, show_bbox, show_text,
                 show_entity, dict_file=None, class_file=None,
                 auto_reg=False):
        self.show_score = show_score
        self.show_bbox = show_bbox
        self.show_text = show_text
        self.show_entity = show_entity
        self.auto_reg = auto_reg

        if dict_file:
            if not auto_reg:
                self.ocr_dict = {
                    '<GO>': 0,
                    '<END>': 1,
                    **{
                        val: ind
                        for ind, val in enumerate(load_dict(dict_file), 2)
                    }
                }
            else:
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
        else:
            self.ocr_dict = None
            self.rev_ocr_dict = None

        if class_file:
            # IOB tagging
            if not self.auto_reg:
                self.entity_dict = {
                    'O': 0,
                }
                for ind, val in enumerate(load_dict(class_file), 1):
                    self.entity_dict['B-' + val] = 2 * ind - 1
                    self.entity_dict['I-' + val] = 2 * ind
            else:
                self.entity_dict = {
                    'O': 0,
                    '<PAD>': 1,
                }
                self.entity_cls_dict = {
                    'O': 0,
                }
                for ind, val in enumerate(load_dict(class_file), 1):
                    self.entity_dict['B-' + val] = 2 * ind
                    self.entity_dict['I-' + val] = 2 * ind + 1
                    self.entity_cls_dict[val] = ind
            self.rev_entity_dict = dict()
            for key, val in self.entity_dict.items():
                self.rev_entity_dict[val] = key
        else:
            self.entity_dict = dict()
            self.rev_entity_dict = dict()

    # todo: modify this part to satisfy both auto_reg and non_auto_reg
    def show_result(self,
                    img,
                    result,
                    score_thr=0.5,
                    bbox_color='red',
                    poly_color='red',
                    text_color='red',
                    thickness=1,
                    font_scale=0.05,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img`.
            score_thr (float, optional): Minimum score of bboxes to be shown. (The same as box_thr in vie_metric.py)
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            poly_color (str or tuple or :obj:`Color`): Color of poly lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.imshow_pred_boundary`
        """
        img = mmcv.imread(img)
        img = img.copy()
        boundaries = None
        bboxes = None
        labels = None
        if 'boundary_result' in result.keys():
            boundaries = result['boundary_result']
            labels = [0] * len(boundaries)
        if 'box_result' in result.keys():
            bboxes = result['box_result']
            if not labels:
                labels = [0] * len(bboxes)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        if boundaries is not None:
            # convert REC & KIE results to strings
            rec_results = result.get('REC', None)
            kie_results = result.get('KIE', None)
            texts, entities = None, None
            if rec_results:
                if len(rec_results['indexes']) != 0:
                    texts = [[] for _ in range(len(bboxes))]
                    for indexes, scores in zip(rec_results['indexes'], rec_results['scores']):
                        # (B, N, L) -> (N, L)
                        index = indexes[0]
                        score = scores[0]
                        for i in range(index.shape[0]):  # N
                            seq = ""
                            seq_score = 0
                            score_cnt = 0
                            if not self.auto_reg:
                                for j in range(index.shape[1]):  # L
                                    if index[i, j] == self.ocr_dict['<END>']:
                                        seq_score += score[i, j]
                                        score_cnt += 1
                                        break
                                    elif index[i, j] == self.ocr_dict['<GO>']:
                                        seq_score += score[i, j]
                                        score_cnt += 1
                                        continue
                                    else:
                                        seq_score += score[i, j]
                                        score_cnt += 1
                                        seq += self.rev_ocr_dict[index[i, j]]
                                texts[i].append([seq, seq_score/score_cnt])
                            else:
                                invalid_flag = False
                                for j in range(index.shape[1]):  # L
                                    if index[i, j] == self.ocr_dict['<END>']:
                                        seq_score += score[i, j]
                                        score_cnt += 1
                                        break
                                    elif index[i, j] == self.ocr_dict['<GO>']:
                                        seq_score += score[i, j]
                                        score_cnt += 1
                                        continue
                                    elif index[i, j] == self.ocr_dict['<PAD>']:
                                        # <PAD> is not supposed to be exists between <GO> and <END>
                                        invalid_flag = True
                                        break
                                    else:
                                        seq_score += score[i, j]
                                        score_cnt += 1
                                        seq += self.rev_ocr_dict[index[i, j]]
                                if not invalid_flag:
                                    texts[i].append([seq, seq_score / score_cnt])
                                else:
                                    texts[i].append([' ', 0.0])
            if kie_results:
                if len(kie_results['indexes']) != 0:
                    entities = [[] for _ in range(len(bboxes))]
                    for idx, (indexes, scores) in enumerate(zip(kie_results['indexes'], kie_results['scores'])):
                        # (B, N, L) -> (N, L)
                        index = indexes[0]
                        score = scores[0]
                        for i in range(index.shape[0]):
                            if not self.auto_reg:
                                tags = []
                                tag_score = 0
                                txt_len = len(texts[i][idx][0])
                                score_cnt = 0
                                # Even '<GO>' is not predicted, so we regard this as an invalid instance
                                if txt_len == len(index[i]):
                                    tags = ['O' for _ in range(len(index[i]))]
                                else:
                                    for j in range(1, txt_len + 1):
                                        tags.append(self.rev_entity_dict[index[i, j]])
                                        tag_score += score[i, j]
                                        score_cnt += 1
                                if score_cnt == 0:
                                    score_cnt = 1
                                entities[i].append([",".join(tags), tag_score/score_cnt])
                            else:
                                tags = []
                                tag_score = 0
                                txt_len = len(texts[i][idx][0])
                                score_cnt = 0
                                # Since '<END>' is not predicted in text, we regard this as an invalid instance
                                if txt_len == len(index[i]):
                                    tags = ['O' for _ in range(len(index[i]))]
                                else:
                                    for j in range(txt_len):
                                        tags.append(self.rev_entity_dict[index[i, j]])
                                        tag_score += score[i, j]
                                        score_cnt += 1
                                if score_cnt == 0:
                                    score_cnt = 1
                                entities[i].append([tags, tag_score / score_cnt])
            imshow_e2e_result(
                img,
                boundaries,
                labels,
                score_thr=score_thr,
                boundary_color=poly_color,
                text_color=text_color,
                thickness=thickness,
                font_scale=font_scale,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                show_score=self.show_score,
                bboxes=bboxes,
                show_bbox=self.show_bbox,
                bbox_color=bbox_color,
                show_text=self.show_text,
                texts=texts,
                show_entity=self.show_entity,
                entities=entities)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, '
                          'result image will be returned')
        return img

    # def show_result(self,
    #                 img,
    #                 result,
    #                 score_thr=0.5,
    #                 bbox_color='green',
    #                 text_color='green',
    #                 thickness=1,
    #                 font_scale=0.5,
    #                 win_name='',
    #                 show=False,
    #                 wait_time=0,
    #                 out_file=None):
    #     """Draw `result` over `img`.
    #
    #     Args:
    #         img (str or Tensor): The image to be displayed.
    #         result (dict): The results to draw over `img`.
    #         score_thr (float, optional): Minimum score of bboxes to be shown.
    #             Default: 0.3.
    #         bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
    #         text_color (str or tuple or :obj:`Color`): Color of texts.
    #         thickness (int): Thickness of lines.
    #         font_scale (float): Font scales of texts.
    #         win_name (str): The window name.
    #         wait_time (int): Value of waitKey param.
    #             Default: 0.
    #         show (bool): Whether to show the image.
    #             Default: False.
    #         out_file (str or None): The filename to write the image.
    #             Default: None.imshow_pred_boundary`
    #     """
    #     img = mmcv.imread(img)
    #     img = img.copy()
    #     boundaries = None
    #     labels = None
    #     if 'boundary_result' in result.keys():
    #         boundaries = result['boundary_result']
    #         labels = [0] * len(boundaries)
    #
    #     # if out_file specified, do not show image in window
    #     if out_file is not None:
    #         show = False
    #     # draw bounding boxes
    #     if boundaries is not None:
    #         imshow_pred_boundary(
    #             img,
    #             boundaries,
    #             labels,
    #             score_thr=score_thr,
    #             boundary_color=bbox_color,
    #             text_color=text_color,
    #             thickness=thickness,
    #             font_scale=font_scale,
    #             win_name=win_name,
    #             show=show,
    #             wait_time=wait_time,
    #             out_file=out_file,
    #             show_score=self.show_score)
    #
    #     if not (show or out_file):
    #         warnings.warn('show==False and out_file is not specified, '
    #                       'result image will be returned')
    #     return img
