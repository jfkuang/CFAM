#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/8 9:55
# @Author : WeiHua
import torch
import torch.nn as nn
from torchcrf import CRF
from mmocr.models.spotting.modules.kie_modules import BiLSTMLayer, UnionLayer


class KIEDecoderTRIE(nn.Module):
    """ KIE decoder for serial structure"""
    def __init__(self, ocr_dict, entity_dict, use_crf=False, ins_lvl_mean=False,
                 d_model=-1, lstm_args=None):
        super(KIEDecoderTRIE, self).__init__()
        self.debug_mode = False
        self.ins_lvl_mean = ins_lvl_mean
        self.ocr_dict = ocr_dict
        self.rev_ocr_dict = dict()
        for key, val in ocr_dict.items():
            self.rev_ocr_dict[val] = key
        self.entity_dict = entity_dict

        bilstm_kwargs = lstm_args.get('bilstm_kwargs', dict())
        mlp_kwargs = lstm_args.get('mlp_kwargs', dict())
        # since the input of lstm is the combination of textual and context
        bilstm_kwargs.update(input_size=2*d_model)
        if bilstm_kwargs['bidirectional']:
            mlp_kwargs.update(in_dim=2*bilstm_kwargs['hidden_size'])
        else:
            mlp_kwargs.update(in_dim=bilstm_kwargs['hidden_size'])
        mlp_kwargs.update(out_dim=len(self.entity_dict))
        self.bilstm_layer = BiLSTMLayer(bilstm_kwargs, mlp_kwargs,
                                        pad_val=self.ocr_dict['<PAD>'],
                                        apply_norm=lstm_args.get('apply_norm', False))

        self.union_layer = UnionLayer(debug_mode=self.debug_mode)
        self.soft_max_func = nn.Softmax(dim=-1)

        self.use_crf = use_crf
        if use_crf:
            # self.crf = ConditionalRandomField(num_tags=len(entity_dict))
            self.crf = CRF(len(entity_dict), batch_first=True)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.entity_dict['<PAD>'],
                reduction='mean'
            )

    def forward(self, ocr_logits, multi_modal_context, texts=None, tags=None, logits_logger=None,
                sorted_idx=None):
        """
        Decoder of KIE branch
        Args:
            ocr_logits: B*N, L, C
            multi_modal_context: B, N, C
            texts: B, N, L
            tags: B, N, L
            logits_logger: dict

        Returns:

        """
        if self.training:
            B, N, L = texts.shape
            C = ocr_logits.shape[-1]
            # -> BN, C -> BN, 1, C -> BN, L, C
            multi_modal_context = multi_modal_context.reshape(B*N, C).unsqueeze(1).expand(B*N, L, C)
            if not self.debug_mode:
                new_kie_logits, new_mask, new_tags, mask = self.union_layer(
                    torch.cat((ocr_logits, multi_modal_context), dim=-1).reshape(B, N, L, -1),
                    texts, self.ocr_dict['<END>'],
                    tags_=tags,
                    tag_pad=self.entity_dict['<PAD>'],
                    sorted_idx=sorted_idx)
            else:
                raise NotImplementedError
                new_kie_logits, new_mask, new_tags, mask, doc_texts = self.union_layer(
                    kie_logits.reshape(B, N, L, -1),
                    texts, self.ocr_dict['<END>'],
                    tags_=tags,
                    tag_pad=self.entity_dict['<PAD>'],
                    sorted_idx=sorted_idx)
                docs = []
                for i in range(B):
                    cur_doc = ""
                    for j in doc_texts[i]:
                        cur_idx = j.item()
                        if cur_idx == self.ocr_dict['<END>']:
                            break
                        cur_doc += self.rev_ocr_dict[cur_idx]
                    docs.append(cur_doc)
            new_kie_logits = self.bilstm_layer(new_kie_logits, new_mask.sum(dim=-1), (None, None))
            if self.use_crf:
                if self.ins_lvl_mean:
                    log_likelihood = self.crf(new_kie_logits, new_tags, mask=new_mask)
                    if sorted_idx:
                        total_num_box = 0
                        for idx_set in sorted_idx:
                            total_num_box += len(idx_set)
                        log_likelihood /= total_num_box
                    else:
                        raise ValueError(f"sorted_idx is required for calculating the num of boxes")
                else:
                    log_likelihood = self.crf(new_kie_logits, new_tags, mask=new_mask)
                    log_likelihood /= new_mask.sum()
                if 'CRF' not in logits_logger:
                    logits_logger['CRF'] = [-log_likelihood.reshape(1)]
                else:
                    logits_logger['CRF'].append(-log_likelihood.reshape(1))
            else:
                if 'CRF' not in logits_logger:
                    logits_logger['CRF'] = [self.criterion(
                        new_kie_logits.reshape(-1, new_kie_logits.shape[-1]),
                        new_tags.reshape(-1))]
                else:
                    logits_logger['CRF'].append(self.criterion(
                        new_kie_logits.reshape(-1, new_kie_logits.shape[-1]),
                        new_tags.reshape(-1)))
        else:
            logits_logger['ocr_logits'] = ocr_logits
            logits_logger['multi_modal_context'] = multi_modal_context


    def crf_decode(self, logits_logger, shape_, sorted_idx, gt_texts=None):
        if self.training:
            raise RuntimeError("crf_decode should be inference only")

        B, N = shape_
        L, C = logits_logger['ocr_logits'].shape[1:]

        multi_modal_context = logits_logger['multi_modal_context'].reshape(B*N, C).unsqueeze(1).expand(B*N, L, C)
        ocr_logits = logits_logger['ocr_logits']
        if not isinstance(gt_texts, type(None)):
            pred_texts = gt_texts
        else:
            _, pred_texts = torch.max(self.soft_max_func(logits_logger['REC'][-1]), dim=-1)
        new_kie_logits, new_mask, mask = self.union_layer(
            torch.cat((ocr_logits, multi_modal_context), dim=-1).reshape(B, N, L, -1),
            pred_texts.reshape(B, N, -1),
            self.ocr_dict['<END>'],
            sorted_idx=sorted_idx)

        new_kie_logits = self.bilstm_layer(new_kie_logits, new_mask.sum(dim=-1), (None, None))
        if self.use_crf:
            # tmp = self.crf.viterbi_tags(new_kie_logits, new_mask, logits_batch_first=True)
            best_paths = self.crf.decode(new_kie_logits, new_mask)
            ins_lvl_tags = self.union_layer.split_out(best_paths, mask, sorted_idx=sorted_idx)
            if 'CRF' not in logits_logger:
                logits_logger['CRF'] = [ins_lvl_tags]
            else:
                logits_logger['CRF'].append(ins_lvl_tags)
        else:
            ins_lvl_logits = self.union_layer.split_logits_out(new_kie_logits, mask, sorted_idx=sorted_idx)
            if 'CRF' not in logits_logger:
                logits_logger['CRF'] = [ins_lvl_logits]
            else:
                logits_logger['CRF'].append(ins_lvl_logits)
