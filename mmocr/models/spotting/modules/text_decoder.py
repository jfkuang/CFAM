#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/1 20:13
# @Author : WeiHua

import math
import copy

import timm.models.layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmocr.models import PositionalEncoding
# from .transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder, TransformerDecoderLayer


def clones(module, N):
    """ Produce N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, drop_path=0.):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        if drop_path > 0:
            self.drop_path = timm.models.layers.DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, sublayer):
        #tmp = self.norm(x)
        #tmp = sublayer(tmp)
        return x + self.drop_path(self.dropout(sublayer(self.norm(x))))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def self_attention(query, key, value, mask=None, dropout=None,
                   attn_map_type=None):
    """
    Compute 'Scale Dot Product Attention'
    """

    d_k = value.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        #score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        score = score.masked_fill(mask == 0, -6.55e4) # for fp16
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    if attn_map_type:
        if attn_map_type == 'origin':
            return torch.matmul(p_attn, value), score
        elif attn_map_type == 'sigmoid':
            return torch.matmul(p_attn, value), torch.sigmoid(score)
        else:
            raise RuntimeError(f"Not support attn-map-type: {attn_map_type}")
    else:
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):

    def __init__(self, headers, d_model, dropout, attn_map_type=None):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.attn_map_type = attn_map_type

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout,
                                      attn_map_type=self.attn_map_type)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, drop_path=0., global_dec=None,
                 ar_mode=True, use_layout=False, custom_cross_mode=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.src_attn = MultiHeadAttention(**src_attn)
        self.sublayer = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 3)
        self.feed_forward = FeedForward(**feed_forward)
        self.global_dec = global_dec
        self.ar_mode = ar_mode
        self.use_layout = use_layout
        self.custom_cross_mode = custom_cross_mode
        if not global_dec:
            self.sublayer = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 3)
        else:
            if use_layout:
                self.layout_mlp = FeedForward(**feed_forward)
            self.merge_mlp = FeedForward(**feed_forward)
            self.glb_pe = PositionalEncoding(n_position=10000)

            assert global_dec in ['VIES_MANNER', "CUSTOM_MANNER"]
            # module for instance-level modeling
            self.conv1ds = nn.ModuleList()
            self.conv1ds.append(nn.Conv1d(in_channels=size, out_channels=size, kernel_size=2, padding=1))
            self.conv1ds.append(nn.Conv1d(in_channels=size, out_channels=size, kernel_size=3, padding=1))
            self.conv1ds.append(nn.Conv1d(in_channels=size, out_channels=size, kernel_size=4, padding=2))
            self.ins_fc = nn.Linear(len(self.conv1ds)*size, size)

            if global_dec == 'VIES_MANNER':
                self.sublayer = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 4)
                self.ins_sublayer = SubLayerConnection(size, dropout, drop_path=drop_path)
                self.ins_attn = MultiHeadAttention(**self_attn)
                self.char_sublayer = SubLayerConnection(size, dropout, drop_path=drop_path)
                self.char_attn = MultiHeadAttention(**self_attn)
            else:
                self.sublayer = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 3)
                self.glb_sublayer = SubLayerConnection(size, dropout, drop_path=drop_path)
                self.glb_attn = MultiHeadAttention(**self_attn)

    def forward(self, x, feature, src_mask, tgt_mask, num_ins=-1,
                pad_mask=None, ins_mask=None, global_seq_mask=None,
                layout_info=None):
        """
        Args:
            x:
            feature:
            src_mask:
            tgt_mask:
            num_ins:
            pad_mask: Tensor with shape (BN, L), True -> Non-pad
            ins_mask: Tensor with shape (B, N), True -> Pad
            layout_info: layout information, Tensor with shape (B, N, C)

        Returns:

        """
        if self.global_dec:
            B = x.shape[0] // num_ins
            L, C = x.shape[1:]
            if self.global_dec == 'VIES_MANNER':
                x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
                if not isinstance(layout_info, type(None)):
                    x = self.layout_mlp(x+layout_info.reshape(-1, C).unsqueeze(1))
                if self.ar_mode:
                    ins_feats = self.global_ins_modeling(x, [B, num_ins, L, C], ins_mask,
                                                         seq_mask=tgt_mask)
                else:
                    ins_feats = self.global_ins_modeling(x, [B, num_ins, L, C], ins_mask,
                                                         pad_mask=pad_mask)
                char_feats = self.global_char_modeling(x, [B, num_ins, L, C], global_seq_mask)
                x = self.sublayer[1](char_feats + ins_feats + x, self.merge_mlp)
                x = self.sublayer[2](x, lambda x: self.src_attn(x, feature, feature, src_mask))
                return self.sublayer[3](x, self.feed_forward)
            else:
                x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
                x = self.global_custom_modeling(x, [B, num_ins, L, C], tgt_mask, ins_mask)
                x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
                return self.sublayer[2](x, self.feed_forward)

        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
            return self.sublayer[2](x, self.feed_forward)

    def global_ins_modeling(self, x, shapes, ins_mask, seq_mask=None, pad_mask=None):
        """
        instance-level global modeling. NAR mode will use every position's info of a instance.
        AR mode will only use appear positions' info of a instance.
        Args:
            x: Tensor, (B x N, L, C)
            shapes: Tuple or List, (B, N, L, C)
            ins_mask: Tensor of bool type, (B, N), True -> Pad
            seq_mask: Tensor of bool type, (B x N, 1, L, L), True -> Non-Pad
            pad_mask: Tensor of bool type, (BN, L), True -> Non-pad

        Returns:
            instance-level global feature: Tensor, (BN, L, C)
        """
        B, N, L, C = shapes
        # x: (BN, L, C)
        if self.ar_mode:
            # BN, L, C -> BN, 1, L, C -> BN, L, L, C -> BNL, L, C -> BNL, C, L
            semantic_feat = x.unsqueeze(1).expand(B*N, L, L, C).masked_fill(
                seq_mask.squeeze(1).unsqueeze(-1) == 0, 0
            ).reshape(-1, L, C).permute(0, 2, 1)
            ins_feats = []
            for conv_ in self.conv1ds:
                feat = conv_(semantic_feat)
                ins_feats.append(torch.max_pool1d(feat, feat.shape[-1]))
            # BNL, C, num_conv1d -> BNL, num_conv1d, C -> BNL, num_conv1d x C
            ins_feats = torch.cat(ins_feats, dim=-1).permute(0, 2, 1).reshape(B * N * L, -1)
            # BNL, num_conv1d x C -> BNL, C -> B, N, L, C -> B, L, N, C -> BL, N, C
            ins_feats = self.ins_fc(ins_feats).reshape(B, N, L, C).permute(0, 2, 1, 3).reshape(-1, N, C)
            ins_feats = self.glb_pe(ins_feats)
            # prepare mask for self-attention
            ins_mask_ = ~ins_mask
            # B, N, N : True -> Non-Pad
            ins_mask_ = ins_mask_.unsqueeze(1).expand(B, N, N) & ins_mask_.unsqueeze(-1).expand(B, N, N)
            # B, N, N -> B, 1, N, N -> B, L, N, N -> BL, N, N
            ins_mask_ = ins_mask_.unsqueeze(1).expand(B, L, N, N).reshape(-1, N, N)
            ins_feats = self.ins_sublayer(ins_feats, lambda x: self.ins_attn(x, x, x, ins_mask_.unsqueeze(1)))
            # BL, N, C -> B, L, N, C -> B, N, L, C -> BN, L, C
            ins_feats = ins_feats.reshape(B, L, N, C).permute(0, 2, 1, 3).reshape(-1, L, C)
        else:
            # BN, L, C -> BN, C, L
            semantic_feat = x.masked_fill(pad_mask.unsqueeze(-1) == 0, 0).permute(0, 2, 1)
            ins_feats = []
            for conv_ in self.conv1ds:
                feat = conv_(semantic_feat)
                ins_feats.append(torch.max_pool1d(feat, feat.shape[-1]))
            # BN, C, num_conv1d -> BN, num_conv1d, C -> BN, num_conv1d x C
            ins_feats = torch.cat(ins_feats, dim=-1).permute(0, 2, 1).reshape(x.shape[0], -1)
            # BN, num_conv1d x C -> BN, C -> B, N, C
            ins_feats = self.ins_fc(ins_feats).reshape(B, N, C)
            ins_feats = self.glb_pe(ins_feats)
            # prepare mask for self-attention
            ins_mask_ = ~ins_mask
            # B, N, N : True -> Non-Pad
            ins_mask_ = ins_mask_.unsqueeze(1).expand(B, N, N) & ins_mask_.unsqueeze(-1).expand(B, N, N)
            ins_feats = self.ins_sublayer(ins_feats, lambda x: self.ins_attn(x, x, x, ins_mask_.unsqueeze(1)))
            # B, N, C -> BN, C -> BN, 1, C -> BN, L, C
            ins_feats = ins_feats.reshape(-1, C).unsqueeze(1).expand(B * N, L, C)
        return ins_feats

    def global_char_modeling(self, x, shapes, global_seq_mask):
        """
        char-level global modeling
        Args:
            x: Tensor, (B x N, L, C)
            shapes: Tuple or List, (B, N, L, C)
            global_char_mask: Tensor of bool type, (B, 1, N x L, N x L), True -> Non-Pad

        Returns:

        """
        B, N, L, C = shapes
        # BN, L, C -> B, NL, C
        char_feats = self.glb_pe(x.reshape(B, N * L, C))
        # prepare mask for self-attention
        char_feats = self.char_sublayer(char_feats, lambda x: self.char_attn(x, x, x, global_seq_mask))
        # B, NL, C -> BN, L, C
        return char_feats.reshape(-1, L, C)

    def global_custom_modeling(self, x, shapes, seq_mask, ins_mask):
        """
        Custom global modeling
        Args:
            x: Tensor, (B x N, L, C)
            shapes: Tuple or List, (B, N, L, C)
            seq_mask: Tensor of bool type, (B x N, 1, L, L), True -> Non-Pad
            ins_mask: Tensor of bool type, (B, N), True -> Pad

        Returns:

        """
        B, N, L, C = shapes
        if self.ar_mode:
            # BN, L, C -> BN, 1, L, C -> BN, L, L, C -> BNL, L, C -> BNL, C, L
            semantic_feat = x.unsqueeze(1).expand(B*N, L, L, C).masked_fill(
                seq_mask.squeeze(1).unsqueeze(-1) == 0, 0
            ).reshape(-1, L, C).permute(0, 2, 1)
            feats = []
            for conv_ in self.conv1ds:
                feat = conv_(semantic_feat)
                feats.append(torch.max_pool1d(feat, feat.shape[-1]))
            # BNL, C, num_conv1d -> BNL, num_conv1d, C -> BNL, num_conv1d x C
            feats = torch.cat(feats, dim=-1).permute(0, 2, 1).reshape(B * N * L, -1)
            # BNL, num_conv1d x C -> BNL, C -> B, N, L, C -> B, L, N, C -> BL, N, C
            feats = self.ins_fc(feats).reshape(B, N, L, C).permute(0, 2, 1, 3).reshape(-1, N, C)
            feats = self.glb_pe(feats)

            # prepare mask for self-attention
            ins_mask_ = ~ins_mask
            # B, N, N : True -> Non-Pad
            ins_mask_ = ins_mask_.unsqueeze(1).expand(B, N, N) & ins_mask_.unsqueeze(-1).expand(B, N, N)
            # B, N, N -> B, 1, N, N -> B, L, N, N -> BL, N, N
            ins_mask_ = ins_mask_.unsqueeze(1).expand(B, L, N, N).reshape(-1, N, N)
            if self.custom_cross_mode:
                # may not converge
                # BN, L, C -> B, N, L, C -> B, L, N, C -> BL, N, C
                query = self.glb_pe(x.reshape(B, N, L, C).permute(0, 2, 1, 3).reshape(-1, N, C))
                query = self.glb_sublayer(query, lambda x: self.glb_attn(x, feats, feats, ins_mask_.unsqueeze(1)))
            else:
                query = self.glb_sublayer(feats, lambda x: self.glb_attn(x, x, x, ins_mask_.unsqueeze(1)))
            # BL, N, C -> B, L, N, C -> B, N, L, C -> BN, L, C
            query = query.reshape(B, L, N, C).permute(0, 2, 1, 3).reshape(-1, L, C)
            return query + x
        else:
            raise NotImplementedError

class EncoderLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, drop_path=0.):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.self_attn_1 = MultiHeadAttention(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout, drop_path=drop_path), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn_1(x, x, x, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)

def build_instance_select_mask(num_boxes_):
    """
    build instance select mask, pick valid instance from batched samples.
    Args:
        num_boxes_: list[int], actual instances num of each image

    Returns:
        selective mask: list[int]
    """
    num_boxes = num_boxes_.copy()
    ins_num = num_boxes.pop()
    select_mask = []
    for idx, num_ in enumerate(num_boxes):
        select_mask.extend(list(range(idx*ins_num, idx*ins_num+num_)))
    return select_mask


'''
class DeformDecoder(nn.Module):
    def __init__(self, input_dim, dim_ff=1024, n_level=1, num_heads=8,
                 n_point=4, dropout=0.1, num_block=4, max_seq_len=512):
        """
        Deformable Transformer Decoder
        """
        super(DeformDecoder, self).__init__()
        assert num_block % 2 == 0, f"For deformable decoder, the number of block should be even."
        num_block = num_block // 2
        deform_decoder_layer = DeformableTransformerDecoderLayer(
            dim=input_dim, dim_ff=dim_ff, n_level=n_level,
            n_head=num_heads, n_point=n_point, dropout=dropout
        )
        decoder_layer = TransformerDecoderLayer(
            dim=input_dim, dim_ff=dim_ff, n_head=num_heads,
            dropout=dropout
        )
        decoder_layers = []
        for _ in range(num_block):
            decoder_layers.append(copy.deepcopy(deform_decoder_layer))
            decoder_layers.append(copy.deepcopy(decoder_layer))
        self.decoder = DeformableTransformerDecoder(
            decoder_layer=decoder_layers, n_layer=-1,
            autoregressive=True
        )
        self.decoder_pos = nn.Parameter(torch.randn(max_seq_len, input_dim) * 0.02)


    def forward(self, x, memory, src_mask, tgt_mask=None):
        # x: (B*N, L, C)
        B, L, C = x.shape
        # expand to match later cuda operation
        if x.shape[0] > 64:
            # print(f"pre shape:{x.shape[0]}")
            valid_batch = x.shape[0]
            expand_batch = (valid_batch // 64 + 1) * 64
            B = expand_batch
            expand_batch -= valid_batch
            x = torch.cat([x, torch.zeros((expand_batch, L, C), device=x.device)], dim=0)
            # print(f"aft shape:{x.shape[0]}")
            memory = torch.cat([memory, torch.zeros((expand_batch, memory.shape[1], C), device=x.device)], dim=0)
        else:
            valid_batch = x.shape[0]

        # learnable position embedding
        decoder_pos = (
            self.decoder_pos[:L].unsqueeze(0).expand(B, -1, -1)
        )
        shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=x.device)
        # level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        level_start = torch.as_tensor([0], device=x.device)
        valid_ratio = self.valid_ratio(torch.zeros((B, H, W), device=x.device).bool()).unsqueeze(1)
        # (B, L, C)
        x = self.layer(x, pos, shapes, level_start, None, valid_ratio)
        if self.equal_dim:
            return x.transpose(1, 2).reshape(B, C, H, W)[:valid_batch]
        x = self.dim_converter(x).permute(0, 2, 1)
        return x.reshape(B, -1, H, W)[:valid_batch]
'''



# rec_layers = []
# for _ in range((rec_layer_num-rec_dual_layer_num) // 2):
#     rec_layers.append(DeformableTransformerDecoderLayer(**rec_decoder_args))
#     rec_layers.append(TransformerDecoderLayer(**rec_decoder_args))
# for _ in range((rec_dual_layer_num) // 2):

class Attn(nn.Module):
    def __init__(self, method, hidden_size, embed_size, onehot_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_size + onehot_size, hidden_size)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B, H*W, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, H*W, hidden_size)
        attn_energies = self.score(
            H, encoder_outputs
        )  # compute attention score (B, H*W)
        return F.softmax(attn_energies, dim=1).unsqueeze(
            1
        )  # normalize with softmax (B, 1, H*W)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(
            self.attn(torch.cat([hidden, encoder_outputs], 2))
        )  # (B, H*W, 2*hidden_size+H+W)->(B, H*W, hidden_size)
        energy = energy.transpose(2, 1)  # (B, hidden_size, H*W)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(
            1
        )  # (B, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (B, 1, H*W)
        return energy.squeeze(1)  # (B, H*W)

#需要参数：hidden_size,embed_size,output_size,n_layers,dropout_p,onehot_size
#input: word_input,last_hidden,encoder_output
#output:
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        embed_size,
        output_size,
        n_layers=1,
        dropout_p=0,
        bidirectional=False,
        onehot_size = (8, 32)
    ):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding.weight.data = torch.eye(embed_size)
        # self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(embed_size, hidden_size)
        self.attn = Attn("concat", hidden_size, embed_size, onehot_size[0] + onehot_size[1])
        self.rnn = nn.GRUCell(2 * hidden_size + onehot_size[0] + onehot_size[1], hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)
        :return
            decoder output
        """
        # Get the embedding of the current input word (last output word)
        word_embedded_onehot = self.embedding(word_input).view(
            1, word_input.size(0), -1
        )  # (1,B,embed_size)
        word_embedded = self.word_linear(word_embedded_onehot)  # (1, B, hidden_size)
        attn_weights = self.attn(last_hidden, encoder_outputs)  # (B, 1, H*W)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1)
        )  # (B, 1, H*W) * (B, H*W, C) = (B,1,C)
        context = context.transpose(0, 1)  # (1,B,C)
        # Combine embedded input word and attended context, run through RNN
        # 2 * hidden_size + W + H: 256 + 256 + 32 + 8 = 552
        rnn_input = torch.cat((word_embedded, context), 2)
        last_hidden = last_hidden.view(last_hidden.size(0), -1)
        rnn_input = rnn_input.view(word_input.size(0), -1)
        hidden = self.rnn(rnn_input, last_hidden)
        if not self.training:
            output = F.softmax(self.out(hidden), dim=1)
        else:
            output = F.log_softmax(self.out(hidden), dim=1)
        # Return final output, hidden state
        # print(output.shape)
        return output, hidden, attn_weights

def build_text_decoder(type, num_block, global_dec=False, use_layout=False, custom_cross_mode=True,**kwargs):
    assert type in ['rnn', 'transformer', 'deform_transformer', 'transformer_encoder'], f"Unsupported decoder type: {type}"
    if type == 'transformer':
        drop_path = kwargs.get('drop_path', 0.)
        kwargs.update(global_dec=global_dec)
        kwargs.update(use_layout=use_layout)
        kwargs.update(custom_cross_mode=custom_cross_mode)
        if drop_path > 0:
            decoder = nn.ModuleList()
            drop_path_list = [x.item() for x in torch.linspace(0, drop_path, num_block)]
            for i in range(num_block):
                kwargs.update(drop_path=drop_path_list[i])
                decoder.append(DecoderLayer(**kwargs))
            return decoder
        else:
            return clones(DecoderLayer(**kwargs), num_block)
    elif type == 'transformer_encoder':
        drop_path = kwargs.get('drop_path', 0.)
        # todo: modify this part
        kwargs.pop("ar_mode", None)
        kwargs.pop("global_dec", None)
        if drop_path > 0:
            decoder = nn.ModuleList()
            drop_path_list = [x.item() for x in torch.linspace(0, drop_path, num_block)]
            for i in range(num_block):
                kwargs.update(drop_path=drop_path_list[i])
                decoder.append(EncoderLayer(**kwargs))
            return decoder
        else:
            return clones(EncoderLayer(**kwargs), num_block)
    elif type == 'deform_transformer':
        raise NotImplementedError(f"Current not support deformable decoder.")
    elif type == 'rnn':
        drop_path = kwargs.get('drop_path', 0.)
        kwargs.update(hidden_size=512)
        kwargs.update(embed_size=512)
        kwargs.update(output_size=512)
        if drop_path > 0:
            decoder = nn.ModuleList()
            drop_path_list = [x.item() for x in torch.linspace(0, drop_path, num_block)]
            for i in range(num_block):
                kwargs.update(drop_path=drop_path_list[i])
                decoder.append(BahdanauAttnDecoderRNN(**kwargs))
            return decoder
        else:
            return clones(DecoderLayer(**kwargs), num_block)

    # elif type == 'deform_transformer':

