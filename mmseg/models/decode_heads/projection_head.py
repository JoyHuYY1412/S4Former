# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.ops import Upsample
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule


@HEADS.register_module()
class ProjectionHead(BaseModule):
    """Projection Head for COntrastive learning

    see SemiSeg_Contrastive Code

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self, in_channels, channels):
        super(ProjectionHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = channels
        assert isinstance(self.in_channels, int)

        self.proj = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

        self.predict = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        self.apply(self._init_weights)
    
    def predict_proj(self, x):
        x = self.predict(x)
        return x
        
    def forward(self, x):
        x = self.proj(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

'''
@HEADS.register_module()
class ProjectionHead(BaseDecodeHead):
    """Projection Head for COntrastive learning

    see SemiSeg_Contrastive Code

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 num_convs=1,
                 up_scale=4,
                 kernel_size=3,
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))
                 ],
                 **kwargs):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(ProjectionHead, self).__init__(init_cfg=init_cfg, **kwargs)

        assert isinstance(self.in_channels, int)

        _, self.norm = build_norm_layer(norm_layer, self.in_channels)

        in_channels = self.in_channels
        out_channels = self.channels

        self.up_convs = nn.ModuleList()
        
        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    Upsample(
                        scale_factor=up_scale,
                        mode='bilinear',
                        align_corners=self.align_corners)))
            in_channels = out_channels

        self.up_convs.append(
            ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
            )
    def forward(self, x, PatchMix_N=0, PatchMixIndex=None):
        x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()

        if PatchMix_N != 0:
            # x [batchsize, patch_num, feature_dim]
            x = self._repatchmix_inputs(x, PatchMix_N, PatchMixIndex)

        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        # out = self.conv(x)
        return x
'''

