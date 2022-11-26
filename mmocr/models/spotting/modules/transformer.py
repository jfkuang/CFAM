import copy
import math

import torch
from torch import nn

from .ops.modules import MSDeformAttn


def inverse_sigmoid(input, eps=1e-5):
    out = input.clamp(min=0, max=1)
    out2 = out.clamp(min=eps)
    out3 = (1 - out).clamp(min=eps)

    return torch.log(out2 / out3)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, in_dim, dim=None, out_dim=None, activation=nn.SiLU, dropout=0):
        dim = in_dim if dim is None else dim
        out_dim = in_dim if out_dim is None else out_dim

        super().__init__(
            nn.Linear(in_dim, dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )

        self.apply(init_weights)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, dim_ff, n_level, n_head, n_point, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = MSDeformAttn(dim, n_level, n_head, n_point)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=nn.ReLU, dropout=dropout
        )

    def add_pos(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, input, pos, reference_point, shape, level_start, mask=None):
        out = self.add_pos(self.self_norm(input), pos)
        # out, _ = self.self_attn(out, reference_point, input, shape, level_start, mask)
        out = self.self_attn(out, reference_point, input, shape, level_start, mask)
        input = input + self.dropout(out)

        out = self.ff_norm(input)
        out = self.ff(out)
        out = input + self.dropout(out)

        return out


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, n_layer):
        super().__init__()

        self.layers = nn.ModuleList(repeat(encoder_layer, n_layer))
        self.norm = nn.LayerNorm(encoder_layer.dim)

    def reference_point(self, shape, valid_ratio, device):
        reference_list = []

        for level, (height, width) in enumerate(shape):
            y, x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, device=device),
                torch.linspace(0.5, width - 0.5, width, device=device),
            )
            y = y.reshape(-1)[None] / (valid_ratio[:, None, level, 1] * height)
            x = x.reshape(-1)[None] / (valid_ratio[:, None, level, 0] * width)
            ref = torch.stack((x, y), -1)
            reference_list.append(ref)

        points = torch.cat(reference_list, 1)
        points = points[:, :, None] * valid_ratio[:, None]

        return points

    def forward(self, input, pos, shape, level_start, mask, valid_ratio):
        out = input
        reference = self.reference_point(shape, valid_ratio, device=input.device)

        for layer in self.layers:
            out = layer(out, pos, reference, shape, level_start, mask)

        out = self.norm(out)

        return out


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, dim, dim_ff, n_level, n_head, n_point, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)

        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = MSDeformAttn(dim, n_level, n_head, n_point)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=nn.ReLU, dropout=dropout
        )

    def add_pos(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        input,
        pos,
        reference_point,
        source,
        source_shape,
        level_start,
        source_mask,
        mask=None,
    ):
        out = self.self_norm(input)
        query = key = self.add_pos(out, pos)
        out = self.self_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            out.transpose(0, 1),
            attn_mask=mask,
        )[0].transpose(0, 1)
        input = input + self.dropout(out)

        out = self.cross_norm(input)
        out, location = self.cross_attn(
            self.add_pos(out, pos),
            reference_point,
            source,
            source_shape,
            level_start,
            source_mask,
        )
        input = input + self.dropout(out)

        out = self.ff_norm(input)
        out = self.ff(out)
        out = input + self.dropout(out)

        return out, location


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)

        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=nn.ReLU, dropout=dropout
        )

    def add_pos(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        input,
        pos,
        reference_point,
        source,
        source_shape,
        level_start,
        source_mask,
        mask=None,
    ):
        out = self.self_norm(input)
        query = key = self.add_pos(out, pos)
        out = self.self_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            out.transpose(0, 1),
            attn_mask=mask,
        )[0].transpose(0, 1)
        input = input + self.dropout(out)

        out = self.cross_norm(input)
        source_t = source.transpose(0, 1)
        out, location = self.cross_attn(
            self.add_pos(out, pos).transpose(0, 1), source_t, source_t
        )
        out = out.transpose(0, 1)
        input = input + self.dropout(out)

        out = self.ff_norm(input)
        out = self.ff(out)
        out = input + self.dropout(out)

        return out, location


class HeadLayer(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=nn.ReLU, dropout=dropout
        )

    def add_pos(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        input,
        pos,
        reference_point,
        source,
        source_shape,
        level_start,
        source_mask,
        mask=None,
    ):
        out = self.cross_norm(input)
        source_t = source.transpose(0, 1)
        out = self.cross_attn(
            self.add_pos(out, pos).transpose(0, 1), source_t, source_t
        )[0].transpose(0, 1)
        input = input + self.dropout(out)

        out = self.ff_norm(input)
        out = self.ff(out)
        out = input + self.dropout(out)

        return out, None


class DeformableHeadLayer(nn.Module):
    def __init__(self, dim, dim_ff, n_level, n_head, n_point, dropout=0.1):
        super().__init__()

        self.dim = dim

        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = MSDeformAttn(dim, n_level, n_head, n_point)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=nn.ReLU, dropout=dropout
        )

    def add_pos(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        input,
        pos,
        reference_point,
        source,
        source_shape,
        level_start,
        source_mask,
        mask=None,
    ):
        out = self.cross_norm(input)
        out, location = self.cross_attn(
            self.add_pos(out, pos),
            reference_point,
            source,
            source_shape,
            level_start,
            source_mask,
        )
        input = input + self.dropout(out)

        out = self.ff_norm(input)
        out = self.ff(out)
        out = input + self.dropout(out)

        return out, location


class PositionEmbedding(nn.Module):
    # HamPerdredes: seems like 2-d embedding ? Verify it later.
    def __init__(self, dim, temperature=10000, normalize=False):
        super().__init__()

        self.dim = dim
        self.normalize = normalize

        self.scale = 2 * math.pi
        freq = torch.arange(self.dim, dtype=torch.float32)
        freq = temperature ** (2 * (freq // 2) / (dim // 2))
        self.register_buffer("freq", freq)

    def forward(self, mask, center=None):
        not_mask = ~mask
        y_embed = torch.arange(
            1, mask.shape[1] + 1, dtype=torch.float32, device=mask.device
        ).view(1, -1, 1)
        x_embed = torch.arange(
            1, mask.shape[2] + 1, dtype=torch.float32, device=mask.device
        ).view(1, 1, -1)

        if center is not None:
            x_embed_center = x_embed - center[:, None, 0:1]
            y_embed_center = y_embed - center[:, None, 1:2]

            y_embed = y_embed * not_mask
            x_embed = x_embed * not_mask

            y_embed_center = y_embed * not_mask
            x_embed_center = x_embed * not_mask

            if self.normalize:
                eps = 1e-6
                y_embed = (
                    (y_embed_center - 0.5)
                    / (y_embed.max(1, keepdim=True).values + eps)
                    * self.scale
                )
                x_embed = (
                    (x_embed_center - 0.5)
                    / (x_embed.max(2, keepdim=True).values + eps)
                    * self.scale
                )

        else:
            y_embed = y_embed * not_mask
            x_embed = x_embed * not_mask

            if self.normalize:
                eps = 1e-6
                y_embed = (
                    (y_embed - 0.5)
                    / (y_embed.max(1, keepdim=True).values + eps)
                    * self.scale
                )
                x_embed = (
                    (x_embed - 0.5)
                    / (x_embed.max(2, keepdim=True).values + eps)
                    * self.scale
                )

        pos_x = x_embed[:, :, :, None] / self.freq
        pos_y = y_embed[:, :, :, None] / self.freq

        pos_x = torch.stack(
            (torch.sin(pos_x[:, :, :, 0::2]), torch.cos(pos_x[:, :, :, 1::2])), 4
        ).flatten(3)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, :, :, 0::2]), torch.cos(pos_y[:, :, :, 1::2])), 4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


def repeat(object, num_repeat):
    return [copy.deepcopy(object) for _ in range(num_repeat)]


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layer, bbox_proj=None, autoregressive=False):
        super().__init__()

        if isinstance(decoder_layer, (list, tuple)):
            self.layers = nn.ModuleList(decoder_layer)

        else:
            self.layers = nn.ModuleList(repeat(decoder_layer, n_layer))

        self.bbox_proj = bbox_proj

        self.norm = nn.LayerNorm(self.layers[0].dim)

        if self.bbox_proj is not None:
            self.norm = nn.ModuleList(repeat(nn.LayerNorm(self.layers[0].dim), n_layer))

        self.autoregressive = autoregressive

    def autoregressive_mask(self, size, device):
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=torch.uint8), diagonal=1
        ).to(torch.uint8)

        return mask != 0

    def forward(
        self,
        input,
        pos,
        reference_point,
        source,
        source_shape,
        level_start,
        source_mask,
        source_valid_ratio,
    ):
        out = input
        intermediate = []
        intermediate_reference_point = []
        locations = []
        mask = None

        if self.autoregressive:
            mask = self.autoregressive_mask(input.shape[1], device=input.device)

        for i, layer in enumerate(self.layers):
            if reference_point.shape[-1] == 4:
                reference_point_input = (
                    reference_point[:, :, None]
                    * torch.cat([source_valid_ratio, source_valid_ratio], -1)[:, None]
                )

            else:
                reference_point_input = (
                    reference_point[:, :, None] * source_valid_ratio[:, None]
                )

            out, location = layer(
                out,
                pos,
                reference_point_input,
                source,
                source_shape,
                level_start,
                source_mask,
                mask=mask,
            )

            out_norm = out

            if self.bbox_proj is not None:
                out_norm = self.norm[i](out)
                bbox = self.bbox_proj[i](out)
                bbox = bbox[..., :2] + inverse_sigmoid(reference_point)
                reference_point = torch.sigmoid(bbox).detach()

            elif i == len(self.layers) - 1:
                out_norm = self.norm(out_norm)

            intermediate.append(out_norm)
            intermediate_reference_point.append(reference_point)
            locations.append(location)

        return intermediate, intermediate_reference_point, locations
