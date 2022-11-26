#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/12 20:36
# @Author : WeiHua
import torch
import torch.nn as nn
from mmocr.models import PositionalEncoding
import cv2


class PositionEmbedding2D(nn.Module):
    """
    2D Postion Embedding layer for boxes and polygons.
    Code is modified based on https://github.com/hikopensource/DAVAR-Lab-OCR
    """
    def __init__(self,
                 max_position_embeddings=128,
                 embedding_dim=128,
                 width_embedding=False,
                 height_embedding=False,
                 fuse_method='sum',
                 pos_input_attn_cfg=None):
        """
        Args:
            max_position_embeddings (int): max normalized input dimension (similar to vocab_size).
            embedding_dim (int): size of embedding vector.
            width_embedding (bool): whether to include width-direction embedding.
            height_embedding (bool): whether to include height-direction embedding.
        """
        super().__init__()
        assert fuse_method in ['sum', 'cat'], f"Unsupported fuse method for PositionEmbedding2D: {fuse_method}"
        self.fuse_method = fuse_method
        self.max_position_embeddings = max_position_embeddings
        self.pos_embedding_dim = embedding_dim

        self.x_embedding = nn.Embedding(self.max_position_embeddings, self.pos_embedding_dim)
        self.y_embedding = nn.Embedding(self.max_position_embeddings, self.pos_embedding_dim)
        self.width_embedding = None
        if width_embedding:
            self.width_embedding = nn.Embedding(2 * self.max_position_embeddings + 1, self.pos_embedding_dim)
        self.height_embedding = None
        if height_embedding:
            self.height_embedding = nn.Embedding(2 * self.max_position_embeddings + 1, self.pos_embedding_dim)
        if self.fuse_method == 'sum':
            # sum -> fc # seems need to modify
            self.pos_input_proj = nn.Linear(self.pos_embedding_dim, self.pos_embedding_dim)
            # self.pos_input_proj_relu = nn.ReLU()
        else:
            self.pos_input_proj = nn.Linear(self.pos_embedding_dim, self.pos_embedding_dim)
            pos_input_attn_cfg.update(d_model=self.pos_embedding_dim)
            self.pos_input_attn = nn.TransformerEncoderLayer(**pos_input_attn_cfg)
            self.pos_input_attn_pe = PositionalEncoding(n_position=64, d_hid=self.pos_embedding_dim)


    @property
    def with_width_embedding(self):
        """

        Returns:
            Determine the model with the width_embedding or not
        """
        return hasattr(self, 'width_embedding') and self.width_embedding is not None

    @property
    def with_height_embedding(self):
        """

        Returns:
            Determine the model with the height_embedding or not
        """
        return hasattr(self, 'height_embedding') and self.height_embedding is not None

    def forward(self, boxes, img_metas):
        """ Forward computation

        Args:
            boxes (Tensor): bounding boxes or rotate boxes or polygons,
                Tensor,in shape of [B x N x K]. -> K: [x, y, x, y, ...]
            img_metas (list[dict]): each dict contains image meta infos,
                e.g., 'img_shape'
        Returns:
            Tensor: embeddings, in shape of [B x N x C] or [B, N, L, C]
        """
        # normalize bboxes
        assert boxes.shape[0] == len(img_metas)
        assert boxes.shape[-1] % 2 == 0
        if self.fuse_method == 'sum':
            pts_num = boxes.shape[-1]//2
            for idx, img_meta in enumerate(img_metas):
                boxes[idx, :, 0::2] /= img_meta['img_shape'][1]
                boxes[idx, :, 1::2] /= img_meta['img_shape'][0]
            # normalize to max_position_embeddings.
            boxes = torch.clamp((boxes * self.max_position_embeddings), 0, self.max_position_embeddings - 1).long()
            sum_position_embedding = None
            for i in range(pts_num):
                if isinstance(sum_position_embedding, type(None)):
                    sum_position_embedding = self.x_embedding(boxes[:, :, 2*i])
                else:
                    sum_position_embedding += self.x_embedding(boxes[:, :, 2*i])
                sum_position_embedding += self.y_embedding(boxes[:, :, 2*i+1])

            # include width-direction embedding
            if self.with_width_embedding:
                for i in range(pts_num):
                    sum_position_embedding += self.width_embedding(
                        boxes[:, :, 2 * ((i + 1) % pts_num)] - boxes[:, :, 2 * i] + self.max_position_embeddings)

            # include height-direction embedding
            if self.with_height_embedding:
                for i in range(pts_num):
                    sum_position_embedding += self.height_embedding(
                        boxes[:, :, 2 * ((i + 1) % pts_num) + 1] - boxes[:, :, 2 * i + 1] + self.max_position_embeddings)

            # sum & projection.
            # sum_position_embedding = self.pos_input_proj_relu(self.pos_input_proj(sum_position_embedding))
            sum_position_embedding = self.pos_input_proj(sum_position_embedding)
            return sum_position_embedding
        else:
            # B, N, L, C
            position_embedding = torch.empty(boxes.shape+(self.pos_embedding_dim,), device=boxes.device)
            B, N, K, C = position_embedding.shape
            for idx, img_meta in enumerate(img_metas):
                # boxes[idx][0::2] = boxes[idx][0::2] / img_meta['img_shape'][1]
                # boxes[idx][1::2] = boxes[idx][1::2] / img_meta['img_shape'][0]
                boxes[idx, :, 0::2] /= img_meta['img_shape'][1]
                boxes[idx, :, 1::2] /= img_meta['img_shape'][0]
            # normalize to max_position_embeddings.
            boxes = torch.clamp((boxes * self.max_position_embeddings), 0, self.max_position_embeddings - 1).long()
            position_embedding[:, :, 0::2, :] = self.x_embedding(boxes[:, :, 0::2])
            position_embedding[:, :, 1::2, :] = self.y_embedding(boxes[:, :, 1::2])
            position_embedding = self.pos_input_attn_pe(position_embedding.reshape(-1, K, C))
            return self.pos_input_attn(position_embedding.permute(1, 0, 2)).permute(1, 0, 2).reshape(
                B, N, K, C)

class CustomPositionEmbedding(nn.Module):
    """
    2D Postion Embedding layer for boxes and polygons.
    (min_x, min_y, max_x, max_y, width, height)
    """
    def __init__(self, embedding_dim=128, max_position_embedding=128):
        """
        Args:
            embedding_dim (int): output dimension of position feature
        """
        super().__init__()
        self.max_position_embedding = max_position_embedding
        self.embedding_dim = embedding_dim

        self.x_embedding = nn.Embedding(self.max_position_embedding, self.embedding_dim)
        self.y_embedding = nn.Embedding(self.max_position_embedding, self.embedding_dim)
        self.width_embedding = nn.Embedding(self.max_position_embedding, self.embedding_dim)
        self.height_embedding = nn.Embedding(self.max_position_embedding, self.embedding_dim)
        self.pos_input_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.pos_input_acti = nn.ReLU()

        # self.pos_input_proj = nn.Linear(6, self.embedding_dim)

    def forward(self, boxes, img_metas):
        """ Forward computation

        Args:
            boxes (Tensor): bounding boxes or rotate boxes or polygons,
                Tensor,in shape of [B x N x K]. -> K: [x, y, x, y, ...]
            img_metas (list[dict]): each dict contains image meta infos,
                e.g., 'img_shape'
        Returns:
            Tensor: embeddings, in shape of [B x N x C] or [B, N, L, C]
        """
        # normalize bboxes
        assert boxes.shape[0] == len(img_metas)
        assert boxes.shape[-1] % 2 == 0
        for idx, img_meta in enumerate(img_metas):
            boxes[idx, :, 0::2] /= img_meta['img_shape'][1]
            boxes[idx, :, 1::2] /= img_meta['img_shape'][0]
        boxes = torch.clamp((boxes * self.max_position_embedding), 0, self.max_position_embedding - 1).long()
        pos_embed = self.x_embedding(boxes[:, :, 0::2].min(dim=-1)[0])  # min_x
        pos_embed += self.y_embedding(boxes[:, :, 1::2].min(dim=-1)[0])  # min_y
        pos_embed += self.x_embedding(boxes[:, :, 0::2].max(dim=-1)[0])  # max_x
        pos_embed += self.y_embedding(boxes[:, :, 1::2].max(dim=-1)[0])  # max_y

        # width
        pos_embed += self.width_embedding(boxes[:, :, 0::2].max(dim=-1)[0] - boxes[:, :, 0::2].min(dim=-1)[0])
        # height
        pos_embed += self.height_embedding(boxes[:, :, 1::2].max(dim=-1)[0] - boxes[:, :, 1::2].min(dim=-1)[0])

        return self.pos_input_acti(self.pos_input_proj(pos_embed))

        # ext_spa_feat = []
        # ext_spa_feat.append(boxes[:, :, 0::2].min(dim=-1)[0].unsqueeze(-1))  # min_x
        # ext_spa_feat.append(boxes[:, :, 1::2].min(dim=-1)[0].unsqueeze(-1))  # min_y
        # ext_spa_feat.append(boxes[:, :, 0::2].max(dim=-1)[0].unsqueeze(-1))  # max_x
        # ext_spa_feat.append(boxes[:, :, 1::2].max(dim=-1)[0].unsqueeze(-1))  # max_y
        # # width
        # ext_spa_feat.append((boxes[:, :, 0::2].max(dim=-1)[0] - boxes[:, :, 0::2].min(dim=-1)[0]).unsqueeze(-1))
        # # height
        # ext_spa_feat.append((boxes[:, :, 1::2].max(dim=-1)[0] - boxes[:, :, 1::2].min(dim=-1)[0]).unsqueeze(-1))
        #
        # return self.pos_input_proj(torch.cat(ext_spa_feat, dim=-1))

class PositionEmbedding(nn.Module):
    """
        Embedding layer for boxes and polygons.
        Code is modified based on https://github.com/hikopensource/DAVAR-Lab-OCR
        """
    """ Embedding layer. Raw implementation: nn.Embedding(vocab_size, embedding_dim)"""
    def __init__(self,
                 max_position_embeddings=128,
                 embedding_dim=128,
                 drop_out=0.):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embedding_dim (int): dim of input features
            drop_out (float): drop_out ratio if required.
        """
        super().__init__()
        self.drop_out = drop_out
        self.max_position_embeddings = max_position_embeddings
        self.pos_embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop_out_layer = nn.Dropout(self.drop_out)

    def init_weights(self, pretrained=None):
        """ Weight initialization
        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feature):
        """ Forward computation
        Args:
            input_feature (Tensor): in shape of [B x N x L]
        Returns:
            Tensor: in shape of [B x N x L x D], where D is the embedding_dim.
        """
        embed_vector = self.embedding(input_feature)
        embed_vector = self.drop_out_layer(embed_vector)
        return embed_vector

class NodeEmbedding(nn.Module):
    """NodeEmbedding (for each bbox). """
    def __init__(self,
                 pos_embedding=None,
                 merge_type='Sum',
                 dropout_ratio=0.1,
                 sentence_embedding=None
                 ):
        """
        Args:
            pos_embedding (dict): pos embedding module, e.g. PositionEmbedding2D
            merge_type (str): fusion type, e.g. 'Sum', 'Concat'
            dropout_ratio (float): dropout ratio of fusion features
            sentence_embedding (dict): sentence embedding module, e.g. SentenceEmbeddingCNN
        """
        super().__init__()

        # pos embedding
        self.pos_embedding = build_embedding(pos_embedding)
        self.pos_embedding_dim = pos_embedding.get('embedding_dim', 128)

        # sentence_embedding
        self.sentence_embedding = build_embedding(sentence_embedding)
        self.sentence_embedding_dim = sentence_embedding.get('embedding_dim', 128)

        # merge param
        self.merge_type = merge_type
        if self.merge_type == 'Sum':
            assert self.sentence_embedding_dim == self.pos_embedding_dim

            self.layernorm = nn.LayerNorm(self.pos_embedding_dim)
        elif self.merge_type == 'Concat':
            self.layernorm = nn.LayerNorm(self.sentence_embedding_dim + self.pos_embedding_dim)
        else:
            raise "Unknown merge type {}".format(self.merge_type)

        self.dropout = nn.Dropout(dropout_ratio)

    def init_weights(self, pretrained=None):
        """ Weight initialization
        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Node Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, recog_hidden, gt_bboxes):
        """ Forward computation
        Args:
            recog_hidden (list(Tensor)): textual feature maps, in shape of [N x L x C] x B
            gt_bboxes (Tensor): textual feature maps, in shape of [N x 4] x B
        Returns:
            Tensor: fused feature maps, in shape of [B x L x C]
        """
        # shape reform
        recog_hidden = torch.stack(recog_hidden, 0)
        gt_bboxes = torch.stack(gt_bboxes, 0)

        # sentence_embedding
        x_sentence = self.sentence_embedding(recog_hidden)

        # positiion embedding
        sum_position_embedding = self.pos_embedding(gt_bboxes)

        # feature merge
        if self.merge_type == 'Sum':
            x_sentence = x_sentence + sum_position_embedding
        elif self.merge_type == 'Concat':
            x_sentence = torch.cat((x_sentence, sum_position_embedding), -1)

        x_sentence = self.layernorm(x_sentence)
        x_sentence = self.dropout(x_sentence)

        return x_sentence

class SentenceEmbeddingCNN(nn.Module):
    """SentenceEmbeddingCNN (for each text)."""
    def __init__(self,
                 embedding_dim,
                 output_dim=None,
                 kernel_sizes=None):
        """
        Args:
            embedding_dim (int): dim of input features
            output_dim (int or None): dim of output features, if not specified, use embedding_dim as default
            kernel_sizes (list(int): multiple kernels used in CNN to extract sentence embeddings
        """
        super().__init__()
        assert kernel_sizes is not None and isinstance(kernel_sizes, list)
        self.kernel_sizes = kernel_sizes
        self.embedding_dim = embedding_dim
        self.output_dim = self.embedding_dim if output_dim is None else output_dim

        # parallel cnn
        self.sentence_cnn_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=_),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)) for _ in kernel_sizes])
        # fc projection
        self.sentence_input_proj = nn.Linear(
            self.embedding_dim * len(kernel_sizes), self.output_dim)
        self.sentence_input_proj_relu = nn.ReLU()

    def init_weights(self, pretrained=None):
        """ Weight initialization
        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Sentence Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feat, char_nums=None):
        """ Forward computation
        Args:
            input_feat (list(Tensor)): textual feature maps, in shape of [B x N x L x C]
            char_nums (list(int)): valid char nums in each text.
        Returns:
            Tensor: fused feature maps, in shape of [B x N x C]
        """
        feat_x = input_feat
        i_b, i_l, i_n, i_d = feat_x.size()
        img_feat = feat_x.view(-1, i_n, i_d).permute(0, 2, 1)

        conv_feat = []
        for per_cnn in self.sentence_cnn_conv:
            conv_feat.append(per_cnn(img_feat))

        img_feat = torch.cat(conv_feat, 1)
        img_feat = img_feat.squeeze(2).view(i_b, i_l, img_feat.size(1))
        x_sentence = self.sentence_input_proj_relu(self.sentence_input_proj(img_feat))
        return x_sentence