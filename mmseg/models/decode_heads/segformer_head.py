# Modified from
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
#
# This work is licensed under the NVIDIA Source Code License.
#
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
#
#  1. Definitions
#  "Licensor" means any person or entity that distributes its Work.
#  "Software" means the original work of authorship made available under
# this License.
#  "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.
#  The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.
#  Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.
#  2. License Grants
#      2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.
#  3. Limitations
#      3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.
#      3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.
#      3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.
#      3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.
#      3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.
#      3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.
#  4. Disclaimer of Warranty.
#  THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.
#  5. Limitation of Liability.
#  EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


class MLP(nn.Module):
    """Linear Embedding."""
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        if self.get_mean_feat:
            mean_feat = torch.zeros([self.channels, 128, 128])
            self.register_buffer('mean_feat', mean_feat)

    def get_sup_feature(self, inputs, PatchMix_N=0, PatchMixIndex=None, return_last_feat=False):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)  # just formatting
        outs = []
        # [x1, x2, x3, x4] [64-1/4, 128-1/8, 320-1/16, 512-1/32]
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            feat = conv(x)
            if PatchMix_N != 0:
                n, c, h, w = feat.shape
                feat_repatchmix = self._repatchmix_inputs(feat.view(n, c, -1).permute(0,2,1), PatchMix_N, PatchMixIndex, scale=4/(2**idx))
                feat = feat_repatchmix.permute(0,2,1).view(n, c, h, w)
            outs.append(
                resize(
                    input=feat,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def forward(self, inputs, PatchMix_N=0, PatchMixIndex=None, return_last_feat=False):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)  # just formatting
        outs = []
        # [x1, x2, x3, x4] [64-1/4, 128-1/8, 320-1/16, 512-1/32]
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            feat = conv(x)
            if PatchMix_N != 0:
                n, c, h, w = feat.shape
                feat_repatchmix = self._repatchmix_inputs(feat.view(n, c, -1).permute(0,2,1), PatchMix_N, PatchMixIndex, scale=4/(2**idx))
                feat = feat_repatchmix.permute(0,2,1).view(n, c, h, w)
            outs.append(
                resize(
                    input=feat,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        # if PatchMix_N != 0:
        #     # import pdb; pdb.set_trace()
        #     n, c, h, w = out.shape
        #     # import pdb; pdb.set_trace()
        #     feat_repatchmix = self._repatchmix_inputs(
        #         out.view(n, c, -1).permute(0,2,1), PatchMix_N, PatchMixIndex, scale=4)
        #     out = feat_repatchmix.permute(0,2,1).view(n, c, h, w)

        if self.get_mean_feat:
            if self.training:
                self.mean_feat = 0.999 * self.mean_feat + 0.001*(
                    torch.mean(out, 0).detach())
        if return_last_feat:
            return self.cls_seg(out), out
        out = self.cls_seg(out)
        return out

    def get_repatchmix_feat(self, x, PatchMix_N=0, PatchMixIndex=None, scale=1):
        # x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()

        if PatchMix_N != 0:
            # x [batchsize, patch_num, feature_dim]
            x = self._repatchmix_inputs(x, PatchMix_N, PatchMixIndex, scale)

        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
