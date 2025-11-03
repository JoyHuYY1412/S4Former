# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.checkpoint as cp
import numpy as np
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed

class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp
        self.num_heads = num_heads
        
    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, attn_mask):
        # if attn_mask is not None:
        #     import pdb; pdb.set_trace()

        def _inner_forward(x, attn_mask):

            x = self.attn(self.norm1(x), identity=x, attn_mask=attn_mask)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x, attn_mask)
        return x


class weight_PatchRelativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.Tensor(1))
        self.sigma.data.fill_(5)


class PatchRelativeAttention(nn.Module):
    def __init__(self, nhead, max_dist=1, grid_size=0.001):
        super().__init__()

        self.max_len = math.ceil(max_dist / grid_size) + 1
        self.grid_size = grid_size

        self.pos_embed = nn.Embedding(self.max_len, nhead)
        self.pos_embed_t = nn.Embedding(self.max_len, nhead)
        nn.init.uniform_(self.pos_embed.weight)
        nn.init.uniform_(self.pos_embed_t.weight)

    def forward(self, rel_pos):
        """
        rel_pos: (..., 3)
        """
        # dist = torch.norm(rel_pos, dim=-1) / self.grid_size  # (...)
        dist = rel_pos[...,0]
        dist = dist / self.grid_size  # (...)
        idx1 = dist.long()
        idx2 = idx1 + 1
        w1 = idx2.type_as(dist) - dist
        w2 = dist - idx1.type_as(dist)
        idx1[idx1 >= self.max_len] = self.max_len - 1
        idx2[idx2 >= self.max_len] = self.max_len - 1
            
        embed1 = self.pos_embed(idx1)
        embed2 = self.pos_embed(idx2)
        embed = embed1 * w1.unsqueeze(-1) + embed2 * w2.unsqueeze(-1)   # (..., nhead)

        dist_t = rel_pos[...,1]
        dist_t = dist_t / self.grid_size  # (...)
        idx1 = dist_t.long()
        idx2 = idx1 + 1
        w1 = idx2.type_as(dist) - dist
        w2 = dist - idx1.type_as(dist)
        idx1[idx1 >= self.max_len] = self.max_len - 1
        idx2[idx2 >= self.max_len] = self.max_len - 1
            
        embed1_t = self.pos_embed_t(idx1)
        embed2_t = self.pos_embed_t(idx2)

        embed_t = embed1_t * w1.unsqueeze(-1) + embed2_t * w2.unsqueeze(-1)   # (..., nhead)

        embed = embed * embed_t

        return embed


@BACKBONES.register_module()
class VisionTransformer(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None,
                 no_pos_embed=False,
                 feature_ps_indices=0,
                 w_PatchRelativeAttention=False):
        super(VisionTransformer, self).__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )
        self.no_pos_embed = no_pos_embed
        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.w_PatchRelativeAttention = w_PatchRelativeAttention
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        
        self.feature_ps_indices = feature_ps_indices
        
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))

        if self.w_PatchRelativeAttention:
            self.weight_attn_adjust_layers = ModuleList()
            for i in range(num_layers):
                self.weight_attn_adjust_layers.append(
                    weight_PatchRelativeAttention())

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        # self.attn_linear = nn.Linear(1024*4, 3)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode,
                        self.no_pos_embed)

            load_state_dict(self, state_dict, strict=False, logger=logger)
        elif self.init_cfg is not None:
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode, no_pos_embed=False):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        if no_pos_embed:
            pos_embed_weight = torch.zeros_like(pos_embed_weight)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs, no_pos_embed=False, avg_pos_emd=False, duplicate_pos_emd=False, use_fdrop=False,
            attn_mask=None, attn_mask_weight=0.0, adaptive_attn_mask=False):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if no_pos_embed:
            pos_embed = torch.zeros_like(self.pos_embed)
            x = self._pos_embeding(x, hw_shape, pos_embed)
        elif avg_pos_emd:
            factor = 4
            if self.with_cls_token:
                pos_embed = self.pos_embed[:, 1:]
            pos_embed = pos_embed.reshape(pos_embed.size(0), int(np.sqrt(pos_embed.size(1))), int(np.sqrt(pos_embed.size(1))), -1).permute(0, 3, 1,2)
            pos_embed = torch.nn.functional.avg_pool2d(pos_embed, factor)
            pos_up = nn.UpsamplingNearest2d(scale_factor=factor)
            pos_embed = pos_up(pos_embed)
            pos_embed = pos_embed.reshape(pos_embed.size(0), pos_embed.size(1), -1).permute(0,2,1)
            pos_embed = torch.cat((self.pos_embed[:, 1].unsqueeze(1), pos_embed), 1)
            x = self._pos_embeding(x, hw_shape, pos_embed)
        elif duplicate_pos_emd:
            factor = 4
            if self.with_cls_token:
                pos_embed = self.pos_embed[:, 1:]
            pos_embed_h = int(np.sqrt(pos_embed.size(1)))
            pos_embed = pos_embed.reshape(pos_embed.size(0), pos_embed_h, pos_embed_h, -1).permute(0, 3, 1,2)
            pos_embed = pos_embed[:,:, :int(pos_embed_h / factor), :int(pos_embed_h / factor)].repeat(1,1,4,4)
            pos_embed = pos_embed.reshape(pos_embed.size(0), pos_embed.size(1), -1).permute(0,2,1)
            pos_embed = torch.cat((self.pos_embed[:, 1].unsqueeze(1), pos_embed), 1)
            x = self._pos_embeding(x, hw_shape, pos_embed)
        else:
            x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if (attn_mask is not None) and self.with_cls_token:
        # if (attn_mask is not None) and self.with_cls_token and (not self.w_PatchRelativeAttention):
            attn_mask = attn_mask.reshape(attn_mask.size(0), -1)
            attn_mask = torch.cat((torch.zeros([attn_mask.size(0), 1]).to(attn_mask.device), attn_mask), -1)
            if adaptive_attn_mask:
                # if the patch is more confident than half (<half), I don't encourage it to be changed
                attn_mask_A = attn_mask.unsqueeze(1).repeat(1, attn_mask.size(-1), 1)
                mask = torch.topk(attn_mask[:, 1:], int(0.5 * (attn_mask.size(-1) -1)), dim=-1, largest=False)[1]
                mask = mask + 1
                attn_mask_A[torch.arange(mask.size(0)).unsqueeze(1), mask, :] = 0 
                attn_mask = attn_mask_A
            else:
                attn_mask = attn_mask.unsqueeze(1).repeat(1, attn_mask.size(-1), 1)
            attn_mask = attn_mask * attn_mask_weight

            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, attn_mask.size(-1), attn_mask.size(-1))

        outs = []
        multi_self_attn = []
        for i, layer in enumerate(self.layers):
            if (attn_mask is not None) and self.w_PatchRelativeAttention:
                attn_mask = attn_mask * self.weight_attn_adjust_layers[i].sigma
                x = layer(x, attn_mask)
            else:
                x = layer(x, attn_mask)

            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                self_attn = layer.attn.self_attn
                if self.with_cls_token:
                    self_attn = self_attn[:,1:,1:]  # [bs, patch_num, patch_num]
                multi_self_attn.append(self_attn)

                if self.with_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape

                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if use_fdrop:
                    out = nn.Dropout2d(0.5)(out)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        self.multi_self_attn = [multi_self_attn, hw_shape]
        return tuple(outs)

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
