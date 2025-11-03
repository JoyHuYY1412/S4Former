# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.ops import Upsample
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ViTCLSHead(nn.Module):
    """Naive ViT classification head
    check https://github.com/lucidrains/vit-pytorch/blob/2fa2b62def9274f604a3ea0cd7edd7b1ddeacf23/vit_pytorch/vit.py#L106-L111
    """

    def __init__(self,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))
                 ],
                 **kwargs):


        super(ViTCLSHead, self).__init__(init_cfg=init_cfg, **kwargs)

        assert isinstance(self.in_channels, int)

        _, self.norm = build_norm_layer(norm_layer, self.in_channels)

        in_channels = self.in_channels
        out_channels_num_classes = self.channels
        self.mlp_head = nn.Linear(in_channels, out_channels_num_classes)
        

    def forward(self, x):
        # if x is [bs, num_patches+1]
        x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        out = self.cls_seg(x)
        return out
