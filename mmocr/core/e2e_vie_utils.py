#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/21 17:56
# @Author : WeiHua


def convert_vie_res(result, num_boxes, ocr_dict, rev_ocr_dict, rev_entity_dict, auto_reg=False):
    """
    Convert REC & KIE results to strings
    Args:
        result:
        num_boxes:
        ocr_dict:
        rev_ocr_dict:
        rev_entity_dict:

    Returns:
        texts: None or list[list[list[transcript, score]]], 1st list is instance-level, 2nd list
            is output of different step
        entities: None or list[list[list[transcript, score]]], 1st list is instance-level, 2nd list
            is output of different step
    """
    rec_results = result.get('REC', None)
    kie_results = result.get('KIE', None)
    texts, entities = None, None
    if rec_results:
        if len(rec_results['indexes']) != 0:
            texts = [[] for _ in range(num_boxes)]
            for indexes, scores in zip(rec_results['indexes'], rec_results['scores']):
                # (B, N, L) -> (N, L)
                index = indexes[0]
                score = scores[0]
                for i in range(index.shape[0]):  # N
                    seq = ""
                    seq_score = 0
                    score_cnt = 0
                    if not auto_reg:
                        for j in range(index.shape[1]):  # L
                            if index[i, j] == ocr_dict['<END>']:
                                seq_score += score[i, j]
                                score_cnt += 1
                                break
                            elif index[i, j] == ocr_dict['<GO>']:
                                seq_score += score[i, j]
                                score_cnt += 1
                                continue
                            else:
                                seq_score += score[i, j]
                                score_cnt += 1
                                seq += rev_ocr_dict[index[i, j]]
                        texts[i].append([seq, seq_score / score_cnt])
                    else:
                        invalid_flag = False
                        for j in range(index.shape[1]):  # L
                            if index[i, j] == ocr_dict['<END>']:
                                seq_score += score[i, j]
                                score_cnt += 1
                                break
                            elif index[i, j] == ocr_dict['<GO>']:
                                seq_score += score[i, j]
                                score_cnt += 1
                                continue
                            elif index[i, j] == ocr_dict['<PAD>']:
                                # <PAD> is not supposed to be exists between <GO> and <END>
                                invalid_flag = True
                                break
                            else:
                                seq_score += score[i, j]
                                score_cnt += 1
                                seq += rev_ocr_dict[index[i, j]]
                        if not invalid_flag:
                            texts[i].append([seq, seq_score / score_cnt])
                        else:
                            texts[i].append([' ', 0.0])
    if kie_results:
        if len(kie_results['indexes']) != 0:
            entities = [[] for _ in range(num_boxes)]
            for idx, (indexes, scores) in enumerate(zip(kie_results['indexes'], kie_results['scores'])):
                # (B, N, L) -> (N, L)
                index = indexes[0]
                score = scores[0]
                for i in range(index.shape[0]):
                    if not auto_reg:
                        tags = []
                        tag_score = 0
                        txt_len = len(texts[i][idx][0])
                        score_cnt = 0
                        # Even '<GO>' is not predicted, so we regard this as an invalid instance
                        if txt_len == len(index[i]):
                            tags = ['O' for _ in range(len(index[i]))]
                        else:
                            for j in range(1, txt_len + 1):
                                tags.append(rev_entity_dict[index[i, j]])
                                tag_score += score[i, j]
                                score_cnt += 1
                        if score_cnt == 0:
                            score_cnt = 1
                        entities[i].append([tags, tag_score / score_cnt])
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
                                tags.append(rev_entity_dict[index[i, j]])
                                tag_score += score[i, j]
                                score_cnt += 1
                        if score_cnt == 0:
                            score_cnt = 1
                        entities[i].append([tags, tag_score / score_cnt])
    return texts, entities