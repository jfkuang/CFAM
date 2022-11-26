#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/7 16:57
# @Author : WeiHua
import cv2
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from shapely.geometry import Polygon

from mmocr.models.builder import DETECTORS
from .spotter_mixin import SpotterMixin
from mmocr.models.builder import build_head
from mmdet.models.builder import build_roi_extractor
from mmdet.models.detectors.base import BaseDetector
from mmdet.core import bbox2roi
from ..modules.spatial_embedding import PositionEmbedding2D, CustomPositionEmbedding
from ..modules import feature_mask, db_like_fuser, db_fuser
from mmocr.utils import get_root_logger
from functools import cmp_to_key

@DETECTORS.register_module()
class TwoStageSpotter(SpotterMixin, BaseDetector):
    """Two-Stage End-to-end spotter for both VIE and OCR
    If a module is implemented in OpenMM, then reimplement its forward_train
    and forward_test will be enough.
    """

    def __init__(self,
                 det_head,
                 neck=None,
                 ext_head=None,
                 cnt_head=None,
                 vis_cfg=None,
                 pretrain=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 part_pretrain=None,
                 model_type='DET'):
        assert model_type in ['DET', 'OCR', 'VIE']
        # override show_result in BaseDetector
        SpotterMixin.__init__(self, **vis_cfg)
        BaseDetector.__init__(self, init_cfg)

        # cfg for train & test respectively
        self.train_cfg = train_cfg
        if not train_cfg:
            self.use_det_res = False
        else:
            self.use_det_res = train_cfg.pop('use_det_res', False)
            if self.use_det_res:
                self.max_ins_num_per_img = train_cfg.pop('max_ins_num_per_img', 100)
        self.test_cfg = test_cfg

        self.model_type = model_type
        self.logger = get_root_logger()

        # build detection head
        self.det_head = build_head(det_head)

        if neck:
            self.feature_fuser_cfg = neck.get('feature_fuser', None)
            if self.feature_fuser_cfg:
                if self.feature_fuser_cfg.pop('sum_up', True):
                    # during up-sample, the feature will combine with lower feature
                    self.feature_fuser = db_fuser(**self.feature_fuser_cfg)
                else:
                    self.feature_fuser = db_like_fuser(**self.feature_fuser_cfg)
            else:
                self.feature_fuser = None
            self.roi_extractor = build_roi_extractor(neck['roi_extractor'])
            if model_type == 'VIE' and neck.get('spatial_embedding', None):
                # self.spatial_embedding = PositionEmbedding2D(**neck['spatial_embedding'])
                self.spatial_embedding = CustomPositionEmbedding(**neck['spatial_embedding'])
            else:
                self.spatial_embedding = None
            self.use_mask_align = neck.get('use_mask_align', False)
        elif model_type != 'DET':
            raise RuntimeError('Neck is required for end-to-end model.')

        # build extra head to perform recognition or kie
        if ext_head:
            self.ext_head = build_head(ext_head)
            # if self.training:
            #     self.logger.info(f"External head:\n{self.ext_head}")
        else:
            self.ext_head = None
        if cnt_head:
            self.cnt_head = build_head(cnt_head)
        else:
            self.cnt_head = None
        if pretrain:
            # pretrain weights for whole model
            raise NotImplementedError
        elif part_pretrain:
            # pretrain weights for part model, rec-branch for example
            assert isinstance(part_pretrain, dict)
            for key in part_pretrain.keys():
                assert key in ['EXT']
                if key == 'EXT':
                    self.ext_head.load_part_weights(part_pretrain[key])

    def forward_train(self, imgs=None, img_metas=None, gt_bboxes=None, gt_labels=None,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, gt_polys=None,
                      density_maps=None, **kwargs):
        losses = dict()
        # ------------- Detection ------------- #
        det_losses, features, det_results = self.det_head(img=imgs, img_metas=img_metas, gt_bboxes=gt_bboxes,
                                                          gt_labels=gt_labels['instances'], gt_masks=gt_masks,
                                                          ret_det=self.use_det_res)
        losses.update(det_losses)
        if self.model_type == 'DET':
            return losses
        # ------------- Feature preparation ------------- #
        # pad boxes, texts/entities -> B, N_max, L
        if self.model_type == 'OCR':
            if self.use_det_res:
                with torch.no_grad():
                    label_bboxes, label_polys, label_texts = \
                        self.add_pred(det_results[0], det_results[1], gt_bboxes, gt_polys, gt_labels['texts'])
            else:
                label_bboxes = gt_bboxes
                label_polys = gt_polys
                label_texts = gt_labels['texts']
            bboxes, texts, num_boxes = \
                self.pad_boxes_and_labels(label_bboxes,
                                          label_texts,
                                          text_pad_val=self.ocr_dict.get('<PAD>', -1))
        else:
            label_bboxes = gt_bboxes
            label_polys = gt_polys
            label_texts = gt_labels['texts']
            bboxes, texts, entities, num_boxes = \
                self.pad_boxes_and_labels(label_bboxes,
                                          label_texts,
                                          entities=gt_labels['entities'],
                                          text_pad_val=self.ocr_dict.get('<PAD>', -1),
                                          entity_pad_val=self.entity_dict.get('<PAD>', -1))
        # list[Tensor] -> [batch_ind, x1, y1, x2, y2]
        rec_rois = bbox2roi([x for x in bboxes])

        # merge features from different stages together
        if self.feature_fuser:
            features = [self.feature_fuser(features[:self.feature_fuser_cfg.get('feat_num_in', 4)])]

        # visual feature -> (N_box_of_all_images, C, H, W)
        vis_feature = self.roi_extractor(features, rec_rois)
        _, C, H, W = vis_feature.shape
        # visual feature -> B, N, C, H, W
        vis_feature = vis_feature.reshape(-1, bboxes.shape[1], C, H, W)
        # mask align
        if self.use_mask_align:
            vis_feature = feature_mask(vis_feature, label_polys, label_bboxes)
            # vis_feature = feature_mask(vis_feature, polys, bboxes)
        # spatial feature -> B, N, C
        if self.spatial_embedding:
            # todo: fix spatial_embedding, to able to accept polygons with different points
            #   currently, direct use bounding box instead (not test yet)
            spa_feature = self.spatial_embedding(bboxes, img_metas)
            # spa_feature = self.spatial_embedding(polys, img_metas)
        else:
            spa_feature = None
        if self.trie_model:
            sorted_idx = get_poly_sort_idx(label_polys)
        elif self.ext_head.sort_layout or self.ext_head.sort_context or \
                self.ext_head.use_crf or 'Serial' in str(type(self.ext_head)):
            # provide sorted index for node index sorting
            sorted_idx = get_poly_sort_idx(label_polys)
            # sorted_idx = get_poly_sort_idx(gt_polys)
        else:
            sorted_idx = None

        # ------------------ Counting-based Aux Branch ---------------- #
        if self.cnt_head:
            losses.update(self.cnt_head(features[0], density_maps))

        # ------------- Recognition & Key Info Extraction ------------- #
        labels = dict(texts=texts.long())
        if self.model_type == 'VIE':
            labels['entities'] = entities.long()
        if self.trie_model:
            ext_losses = self.ext_head.forward_train(vis_feature=vis_feature,
                                                     labels=labels,
                                                     num_boxes=num_boxes,
                                                     sorted_idx=sorted_idx,
                                                     det_boxes=bboxes,
                                                     img_metas=img_metas)
        else:
            ext_losses = self.ext_head.forward_train(vis_feature=vis_feature,
                                                     spa_feature=spa_feature,
                                                     labels=labels,
                                                     num_boxes=num_boxes,
                                                     sorted_idx=sorted_idx,
                                                     img_metas=img_metas)
        losses.update(ext_losses)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        forward_test is only implemented for non-augmentation test, i.e. simple_test.
        Args:
            imgs (Tensor): (B, C, H, W),
            img_metas (List[dict]): list of metas info of each image in a batch.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_size = len(img_metas)
        for img_id in range(batch_size):
            img_metas[img_id]['batch_input_shape'] = tuple(imgs.size()[-2:])
        return self.simple_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_metas, **kwargs):
        assert img.shape[0] == 1, "Batch bigger than 1 is not supported in test"
        # ------------- Detection ------------- #
        kwargs_det = dict(
            proposals=kwargs.get('proposals', None),
            rescale=kwargs.get('rescale', False)
        )
        # dict with keys: "boundary_result", "box_result"
        #   boundary_result: list of polygons / quads, [x, y, x, y, ..., score]
        #   box_result: list of bounding boxes, [tl_x, tl_y, br_x, br_y]
        det_results, features = self.det_head.simple_test(img, img_metas, **kwargs_det)
        if len(det_results['box_result']) == 0:
            # Zero object is detected
            det_results['box_result'] = np.array([[0, 0, 1, 1]])
            det_results['boundary_result'] = np.array([[0, 0, 1, 1, 0]])
        if self.model_type == 'DET':
            return det_results

        # scale_factor = resized / origin
        scale_x, scale_y = img_metas[0]['scale_factor'][:2]
        # (N x 4)
        resized_bboxes = torch.tensor(det_results['box_result'], dtype=torch.float32, device=img.device)
        resized_bboxes[:, 0::2] *= scale_x
        resized_bboxes[:, 1::2] *= scale_y

        # (n lists, each is a polygon)
        resized_polys = []
        for x in det_results['boundary_result']:
            cur_poly = np.array(x[:-1], dtype=np.float32)
            cur_poly[0::2] *= scale_x
            cur_poly[1::2] *= scale_y
            resized_polys.append(cur_poly)
        # # (N x 8), currently only support quad polygon
        # resized_polys = torch.tensor([x[:-1] for x in det_results['boundary_result']],
        #                              dtype=torch.float32, device=img.device)
        # resized_polys[:, 0::2] *= scale_x
        # resized_polys[:, 1::2] *= scale_y

        scores = [x[-1] for x in det_results['boundary_result']]

        # ------------- Feature preparation ------------- #
        gt_bboxes = kwargs.get('gt_bboxes', None)
        gt_polys = kwargs.get('gt_polys', None)
        gt_labels = kwargs.get('gt_labels', None)
        if not isinstance(gt_bboxes, type(None)) and self.model_type == 'VIE':
            # replace detection results to gt for eval
            assert len(gt_polys) == 1, "Current only support batch_size=1 during test or eval."
            # scale_factor = resized / origin
            scale_x, scale_y = img_metas[0]['scale_factor'][:2]
            gt_boundary_result = []
            gt_box_result = []
            for x in gt_polys[0]:
                tmp = x.copy()
                tmp[0::2] /= scale_x
                tmp[1::2] /= scale_y
                gt_boundary_result.append(tmp.tolist()+[1.0])
            for x in gt_bboxes[0]:
                tmp = x.clone()
                tmp[0::2] /= scale_x
                tmp[1::2] /= scale_y
                gt_box_result.append(tmp.tolist())
            det_results['boundary_result'] = gt_boundary_result
            det_results['box_result'] = gt_box_result

            label_bboxes = gt_bboxes
            label_polys = gt_polys
            label_texts = gt_labels['texts']

            # Use gt ocr if it is provided.
            bboxes, texts, num_boxes = \
                self.pad_boxes_and_labels(label_bboxes,
                                          label_texts,
                                          text_pad_val=self.ocr_dict.get('<PAD>', -1))

        else:
            if not isinstance(gt_bboxes, type(None)):
                # if ground-true boxes are provided
                # replace detection results to gt for eval
                assert len(gt_polys) == 1, "Current only support batch_size=1 during test or eval."
                # scale_factor = resized / origin
                scale_x, scale_y = img_metas[0]['scale_factor'][:2]
                gt_boundary_result = []
                gt_box_result = []
                for x in gt_polys[0]:
                    tmp = x.copy()
                    tmp[0::2] /= scale_x
                    tmp[1::2] /= scale_y
                    gt_boundary_result.append(tmp.tolist() + [1.0])
                for x in gt_bboxes[0]:
                    tmp = x.clone()
                    tmp[0::2] /= scale_x
                    tmp[1::2] /= scale_y
                    gt_box_result.append(tmp.tolist())
                det_results['boundary_result'] = gt_boundary_result
                det_results['box_result'] = gt_box_result
                label_bboxes = gt_bboxes
                label_polys = gt_polys
            else:
                label_bboxes = [resized_bboxes]
                label_polys = [resized_polys]
            bboxes, num_boxes = \
                self.pad_boxes_and_labels(label_bboxes)
        # list[Tensor] -> [batch_ind, x1, y1, x2, y2]
        rec_rois = bbox2roi([x for x in bboxes])

        # merge features from different stages together
        if self.feature_fuser:
            features = [self.feature_fuser(features[:self.feature_fuser_cfg.get('feat_num_in', 4)])]

        # visual feature -> (N_box_of_all_images, C, H, W)
        vis_feature = self.roi_extractor(features, rec_rois)
        _, C, H, W = vis_feature.shape
        # visual feature -> B, N, C, H, W
        vis_feature = vis_feature.reshape(-1, bboxes.shape[1], C, H, W)
        # mask align
        if self.use_mask_align:
            vis_feature = feature_mask(vis_feature, label_polys, label_bboxes)
            # vis_feature = feature_mask(vis_feature, polys, bboxes)
        # spatial feature -> B, N, L_pt, C
        if self.spatial_embedding:
            spa_feature = self.spatial_embedding(bboxes, img_metas)
        else:
            spa_feature = None
        if self.trie_model:
            sorted_idx = get_poly_sort_idx(label_polys)
        elif self.ext_head.sort_layout or self.ext_head.sort_context or \
                self.ext_head.use_crf or 'Serial' in str(type(self.ext_head)):
            # provide sorted index for node index sorting
            sorted_idx = get_poly_sort_idx(label_polys)
            # if not isinstance(gt_polys, type(None)):
            #     sorted_idx = get_poly_sort_idx(gt_polys)
            # else:
            #     sorted_idx = get_poly_sort_idx([resized_polys])
        else:
            sorted_idx = None
        # ------------- Recognition & Key Info Extraction ------------- #
        # ext_results:
        # {
        #     'REC': {
        #         'indexes': [Tensor(B, N, L)],
        #         'scores': [Tensor(B, N, L)],
        #     },
        #     'KIE': {'indexes', 'scores'}
        # }
        if self.trie_model:
            if not isinstance(gt_bboxes, type(None)):
                ext_results = self.ext_head.simple_test(vis_feature=vis_feature, num_boxes=num_boxes,
                                                        sorted_idx=sorted_idx, det_boxes=bboxes,
                                                        img_metas=img_metas, gt_texts=texts)
            else:
                ext_results = self.ext_head.simple_test(vis_feature=vis_feature, num_boxes=num_boxes,
                                                        sorted_idx=sorted_idx, det_boxes=bboxes,
                                                        img_metas=img_metas)
        else:
            if not isinstance(gt_bboxes, type(None)) and self.model_type == "VIE":
                ext_results = self.ext_head.simple_test(vis_feature=vis_feature, spa_feature=spa_feature,
                                                        num_boxes=num_boxes, gt_texts=texts, sorted_idx=sorted_idx,
                                                        img_metas=img_metas)
            else:
                ext_results = self.ext_head.simple_test(vis_feature=vis_feature, spa_feature=spa_feature,
                                                        num_boxes=num_boxes, sorted_idx=sorted_idx,
                                                        img_metas=img_metas)
        det_results.update(ext_results)
        for key in ['KIE', 'REC']:
            if key in det_results:
                det_results[key]['indexes'] = [x.cpu().numpy() for x in det_results[key]['indexes']]
                det_results[key]['scores'] = [x.cpu().numpy() for x in det_results[key]['scores']]
        # todo:
        #  add CRF layer
        return [det_results]
    
    def aug_test(self, imgs, img_metas, **kwargs):
        super(TwoStageSpotter, self).aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        super(TwoStageSpotter, self).extract_feat(imgs)

    def pad_boxes_and_labels(self, boxes, texts=None, polys=None, entities=None,
                             entity_cls=None, text_pad_val=-1, entity_pad_val=-1,
                             entity_cls_pad_val=-1):
        """
        Padding boxes and labels and stack them respectively.
        Each of them is list[Tensor].
        # 'polys' is required to have the same number of points. -> 'polys' is no longer need to be
        #   converted to Tensor, but gt polys are still required to contain same number of points.
        """
        sample_num = len(boxes)
        if texts:
            assert sample_num == len(texts)
        if polys:
            assert sample_num == len(polys)
        if entities:
            assert sample_num == len(entities)
        if entity_cls:
            assert sample_num == len(entity_cls)

        num_boxes = [len(x) for x in boxes]
        results = []
        max_num_boxes = max(num_boxes)
        pad_boxes = [F.pad(x, (0, 0, 0, max_num_boxes - len(x)))
                     for x in boxes]
        results.append(torch.stack(pad_boxes, dim=0))

        if texts:
            # adaptive choose max seq length
            # pad text to max_seq_len and max_num_box
            max_seq_len = max([x.shape[-1] for x in texts])
            pad_texts = [F.pad(x, (0, max_seq_len - x.shape[-1], 0, max_num_boxes - len(x)),
                               value=text_pad_val)
                         for x in texts]
            # pad_texts = [F.pad(x, (0, 0, 0, max_num_boxes-len(x)),
            #                    value=text_pad_val)
            #              for x in texts]
            results.append(torch.stack(pad_texts, dim=0))

        if polys:
            pad_polys = [F.pad(x, (0, 0, 0, max_num_boxes - len(x)))
                         for x in polys]
            results.append(torch.stack(pad_polys, dim=0))

        if entities:
            # adaptive choose max seq length
            # pad text to max_seq_len and max_num_box
            max_seq_len = max([x.shape[-1] for x in entities])
            pad_entities = [F.pad(x, (0, max_seq_len - x.shape[-1], 0, max_num_boxes - len(x)),
                                  value=entity_pad_val)
                            for x in entities]
            # pad_entities = [F.pad(x, (0, 0, 0, max_num_boxes - len(x)),
            #                       value=entity_pad_val)
            #                 for x in entities]
            results.append(torch.stack(pad_entities, dim=0))
        if entity_cls:
            # adaptive choose max seq length
            # pad text to max_seq_len and max_num_box
            max_seq_len = max([x.shape[-1] for x in entity_cls])
            pad_entity_cls = [F.pad(x, (0, max_seq_len - x.shape[-1], 0, max_num_boxes - len(x)),
                                  value=entity_pad_val)
                            for x in entity_cls]
            results.append(torch.stack(pad_entity_cls, dim=0))
        results.append(num_boxes)
        return results

    def add_pred(self, pred_box_list, pred_poly_list, gt_box, gt_poly, gt_texts, thresh=0.8):
        """
        add prediction with high iou to gt, to increase the diversity.
        Args:
            pred_box_list: list of lists, [ [[tl_x, tl_y, br_x, br_y] * N] * B ]
            pred_poly_list: list of lists, [ [[x1, y1, x2, y2, ...] * N] * B ]
            gt_box: list of Tensors, [ Tensor with shape N*4 ]
            gt_poly: list of lists, [ [[array of x1, y1, x2, y2, ...] * N] * B ]
            gt_texts: list of Tensors, [ Tensor with shape N*L ]

        Returns:

        """

        def cal_iou(a, b):
            tmp_a = Polygon(np.array(a).reshape(-1, 2))
            tmp_b = Polygon(np.array(b).reshape(-1, 2))
            if not tmp_a.is_valid or not tmp_b.is_valid:
                return 0
            return tmp_a.intersection(tmp_b).area / tmp_a.union(tmp_b).area

        label_bboxes = []
        label_polys = []
        label_texts = []
        for batch_idx in range(len(gt_box)):
            device = gt_box[batch_idx].device
            bboxes = gt_box[batch_idx].cpu()
            polys = gt_poly[batch_idx]
            texts = gt_texts[batch_idx].cpu()
            bboxes_extend = []
            polys_extend = []
            texts_extend = []
            iou_scores = []
            for box_, poly_ in zip(pred_box_list[batch_idx], pred_poly_list[batch_idx]):
                max_iou = -1
                match_idx = -1
                for gt_idx, gt_polygon in enumerate(gt_poly[batch_idx]):
                    iou_val = cal_iou(poly_, gt_polygon)
                    if iou_val > max_iou:
                        max_iou = iou_val
                        match_idx = gt_idx
                if max_iou > thresh:
                    # bboxes_extend.append(bboxes[match_idx])
                    # polys_extend.append(polys[match_idx])

                    bboxes_extend.append(box_)
                    polys_extend.append(np.array(poly_))

                    texts_extend.append(texts[match_idx])
                    iou_scores.append(max_iou)
            if len(iou_scores) > 0 and len(bboxes) < self.max_ins_num_per_img:
                to_sort = [(idx, val) for idx, val in enumerate(iou_scores)]
                to_sort = sorted(to_sort, key=lambda x: x[-1], reverse=True)
                sorted_idx = [x[0] for x in to_sort]

                num_keep = self.max_ins_num_per_img - len(bboxes)
                sorted_idx = sorted_idx[:num_keep]

                bboxes_extend = [bboxes_extend[x] for x in sorted_idx]
                polys_extend = [polys_extend[x] for x in sorted_idx]
                texts_extend = [texts_extend[x] for x in sorted_idx]
                label_bboxes.append(torch.cat((bboxes, torch.Tensor(bboxes_extend)), dim=0).to(device))
                polys.extend(polys_extend)
                label_polys.append(polys)
                label_texts.append(torch.cat((texts, torch.stack(texts_extend, dim=0)), dim=0).to(device))
            else:
                label_bboxes.append(bboxes.to(device))
                label_polys.append(polys)
                label_texts.append(texts.to(device))

        return label_bboxes, label_polys, label_texts

    @property
    def trie_model(self):
        return 'TRIE' in str(type(self.ext_head))

def get_poly_sort_idx(polys):
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

    sorted_idx = []
    for i in range(len(polys)):
        cur_poly = polys[i]
        # if not isinstance(polys[i], np.ndarray):
        #     cur_poly = polys[i].detach().cpu()
        # else:
        #     cur_poly = polys[i]
        to_sort_polys = [(idx, poly_) for idx, poly_ in enumerate(cur_poly)]
        # sorted_data = sorted(to_sort_polys, key=compare_key)
        sorted_data = sorted(to_sort_polys, key=cmp_to_key(compare_key))
        sorted_idx.append([x[0] for x in sorted_data])
    return sorted_idx

