#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/4 15:51
# @Author : WeiHua

from mmocr.models.spotting.recognizers.re_imple_trie.connects.multimodal_context_module import MultiModalContextModule
from mmocr.models.spotting.recognizers.re_imple_trie.connects.multimodal_feature_merge import MultiModalFusion
from mmocr.models.spotting.recognizers.re_imple_trie.connects.bert_encoder import BertEncoder
from mmocr.models.spotting.recognizers.re_imple_trie.embedding.node_embedding import NodeEmbedding
from mmocr.models.spotting.recognizers.re_imple_trie.embedding.position_embedding import PositionEmbedding2D
from mmocr.models.spotting.recognizers.re_imple_trie.embedding.sentence_embedding import SentenceEmbeddingCNN

CONNECT_MODULE = {
    "MultiModalContextModule": MultiModalContextModule,
    "MultiModalFusion": MultiModalFusion,
    "BertEncoder": BertEncoder
}

EMBEDDING_MODULE = {
    "NodeEmbedding": NodeEmbedding,
    "PositionEmbedding2D": PositionEmbedding2D,
    "SentenceEmbeddingCNN": SentenceEmbeddingCNN
}


def build_connect(cfg):
    func = CONNECT_MODULE[cfg.pop('type')]
    return func(**cfg)


def build_embedding(cfg):
    func = EMBEDDING_MODULE[cfg.pop('type')]
    return func(**cfg)
