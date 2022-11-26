#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/20 22:31
# @Author : WeiHua
import warnings
import cv2
import mmcv
import numpy as np
import mmocr.utils as utils


def imshow_e2e_result(img,
                      boundaries_with_scores,
                      labels,
                      score_thr=0,
                      boundary_color='blue',
                      text_color='blue',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      show_score=False,
                      bboxes=None,
                      show_bbox=False,
                      bbox_color='red',
                      show_text=False,
                      texts=None,
                      show_entity=False,
                      entities=None):
    """
    (Modification of "imshow_pred_boundary" in .visualize)
    Draw boundaries, bounding boxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        boundaries_with_scores (list[list[float]]): Boundaries with scores.
        labels (list[int]): Labels of boundaries.
        score_thr (float): Minimum score of boundaries to be shown.
        boundary_color (str or tuple or :obj:`Color`): Color of boundaries.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename of the output.
        show_score (bool): Whether to show text instance score.
        bboxes (list[list[float]]): Bounding boxes
        show_bbox (bool): Whether to show bounding box.
        bbox_color (str or tuple or :obj:`Color`): Color of bounding boxes.
        show_text: Whether to show predict texts.
        texts (list[list[list[item_1, item_2]]]): texts, 1st list is instance-level, 2nd list
            is output of different step, 3rd list is [transcript, score].
        show_entity: Whether to show predict entities.
        entities (list[list[list[item_1, item_2]]]): entities, 1st list is instance-level, 2nd
            list is output of different step, 3rd list is [transcript, score].
    """
    assert isinstance(img, (str, np.ndarray))
    assert utils.is_2dlist(boundaries_with_scores)
    assert utils.is_type_list(labels, int)
    assert utils.equal_len(boundaries_with_scores, labels)
    if show_bbox:
        assert utils.equal_len(bboxes, labels)
    if show_text:
        assert utils.equal_len(bboxes, texts)
    if show_entity:
        assert utils.equal_len(bboxes, entities)
    if len(boundaries_with_scores) == 0:
        warnings.warn('0 text found in ' + out_file)
        return None

    utils.valid_boundary(boundaries_with_scores[0])
    img = mmcv.imread(img)

    scores = np.array([b[-1] for b in boundaries_with_scores])
    inds = scores > score_thr
    boundaries = [boundaries_with_scores[i][:-1] for i in np.where(inds)[0]]
    scores = [scores[i] for i in np.where(inds)[0]]
    labels = [labels[i] for i in np.where(inds)[0]]
    if show_bbox:
        bboxes = [bboxes[i] for i in np.where(inds)[0]]
        assert len(bboxes) == len(boundaries)

    if show_text:
        texts = [texts[i] for i in np.where(inds)[0]]
        assert len(texts) == len(boundaries)
    if show_entity:
        entities = [entities[i] for i in np.where(inds)[0]]
        assert len(entities) == len(boundaries)

    boundary_color = mmcv.color_val(boundary_color)
    if show_bbox:
        bbox_color = mmcv.color_val(bbox_color)
    text_color = mmcv.color_val(text_color)
    font_scale = font_scale

    if show_text or show_entity:
        if 'jpg' in out_file:
            txt_saver = open(out_file.replace('.jpg', '.txt'), 'w', encoding='utf-8')
        else:
            txt_saver = open(out_file.replace('.png', '.txt'), 'w', encoding='utf-8')
    else:
        txt_saver = None

    for idx, (boundary, score) in enumerate(zip(boundaries, scores)):
        boundary_int = np.array(boundary).astype(np.int32)

        cv2.polylines(
            img, [boundary_int.reshape(-1, 1, 2)],
            True,
            color=boundary_color,
            thickness=thickness)
        if show_text or show_entity:
            cv2.putText(img, str(idx),
                        (boundary_int[0], boundary_int[1] + 2),
                        cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=text_color)
            if show_text:
                seq_info = ""
                for iter_, item_ in enumerate(texts[idx]):
                    seq_info += f"\nStep {iter_}: polygon:{boundary_int.reshape(-1).tolist()}, text:{item_[0]}, score:{item_[1]}"
                line = f"idx:{idx}, text_output_info:{seq_info}\n"
                txt_saver.write(line)
            if show_entity:
                seq_info = ""
                for iter_, item_ in enumerate(entities[idx]):
                    seq_info += f"\nStep {iter_}: entities:{item_[0]}, score:{item_[1]}"
                line = f"idx:{idx}, entity_output_info:{seq_info}\n"
                txt_saver.write(line)

        if show_score:
            label_text = f'{score:.02f}'
            cv2.putText(img, label_text,
                        (boundary_int[0], boundary_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if show_bbox:
            cur_bbox = np.array(bboxes[idx], dtype=np.int).reshape(-1, 1, 2)
            cv2.polylines(
                img, [cur_bbox],
                isClosed=True,
                color=bbox_color,
                thickness=thickness)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    if txt_saver:
        txt_saver.close()

    return img