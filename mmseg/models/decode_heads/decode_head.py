# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import numpy as np
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 class_re_weight=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg')),
                 get_mean_feat=False,
                 decoder_params=None):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.get_mean_feat = get_mean_feat
        
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _repatchmix_inputs(self, inputs, PatchMix_N, PatchMixIndex, scale=1):
        # inputs [bs, patch_num_h*patch_num_w, feature_channel]
        # only support patch_num_h==patch_num_w now !!
        patch_num = int(np.sqrt(inputs.shape[1]))
        PatchMix_N = int(PatchMix_N * scale)
        inputs_ori = inputs.clone()

        inputs = inputs.reshape([inputs.shape[0], patch_num//PatchMix_N, PatchMix_N, patch_num//PatchMix_N, PatchMix_N, inputs.shape[2]])
        inputs_reshape = []
        for i in range(patch_num//PatchMix_N):
            for j in range(patch_num//PatchMix_N):
                inputs_reshape.append(inputs[:, i, :, j, :, :])           #patch_num/PatchMix_N*patch_num/PatchMix_N  [bs, PatchMix_N, PatchMix_N, feature_channel]
        inputs_reshape = torch.stack(inputs_reshape, dim=1)
        unshuf_order = np.zeros([inputs.shape[0], patch_num//PatchMix_N * patch_num//PatchMix_N])
        for i in range(inputs.shape[0]):
            unshuf_order[i][PatchMixIndex[i]] = np.arange(patch_num//PatchMix_N * patch_num//PatchMix_N)
        inputs_repatchmix = torch.stack([inputs_reshape[i][unshuf_order[i]] for i in range(inputs_reshape.shape[0])])

        inputs_repatchmix_reshape = []
        for k in range(patch_num//PatchMix_N):
            for i in range(PatchMix_N):
                for j in range(patch_num//PatchMix_N*k,patch_num//PatchMix_N*(k+1)):
                    inputs_repatchmix_reshape.append(inputs_repatchmix[:,j,i,...])
        
        inputs_repatchmix_reshape = torch.stack(inputs_repatchmix_reshape, dim=1)
        inputs_repatchmix_reshape = inputs_repatchmix_reshape.reshape([inputs_repatchmix_reshape.size(0), patch_num*patch_num, inputs_repatchmix_reshape.size(-1)])
        return inputs_repatchmix_reshape

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    @auto_fp16()
    def get_repatchmix_feat(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if "PatchMix_N" in img_metas[0]:
            # import pdb; pdb.set_trace()
            PatchMixIndex = []
            for img_meta in img_metas:
                PatchMixIndex.append(torch.tensor(img_meta['PatchMixIndex']))
            PatchMixIndex = torch.stack(PatchMixIndex)
            # import pdb; pdb.set_trace()
            seg_logits = self.forward(inputs, PatchMix_N=img_meta['PatchMix_N'], PatchMixIndex=PatchMixIndex)
        else: 
            seg_logits = self.forward(inputs)  # size of the logits is the crop img size
        '''
        if self.CAI_sup:
            seg_logits = seg_logits * self.CAI_sup_p.unsqueeze(1).unsqueeze(2).unsqueeze(0)
        if self.CAI_sup_v2:
            seg_logits = seg_logits * 1/self.CAI_sup_p.unsqueeze(1).unsqueeze(2).unsqueeze(0)
        '''
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_get_logits(self, inputs,  train_cfg, img_metas=None):
        # import pdb; pdb.set_trace()
        if "PatchMix_N" not in img_metas[0]:
            seg_logits = self.forward(inputs)
        else:
            PatchMixIndex = []
            for img_meta in img_metas:
                PatchMixIndex.append(torch.tensor(img_meta['PatchMixIndex']))
            PatchMixIndex = torch.stack(PatchMixIndex)
            seg_logits = self.forward(inputs, PatchMix_N=img_meta['PatchMix_N'], PatchMixIndex=PatchMixIndex)
        return seg_logits

    def forward_get_logits_feat(self, inputs,  train_cfg, img_metas=None):
        if "PatchMix_N" not in img_metas[0]:
            seg_logits, seg_feat = self.forward(inputs, return_last_feat=True)
        else:
            PatchMixIndex = []
            for img_meta in img_metas:
                PatchMixIndex.append(torch.tensor(img_meta['PatchMixIndex']))
            PatchMixIndex = torch.stack(PatchMixIndex)
            seg_logits, seg_feat = self.forward(inputs, PatchMix_N=img_meta['PatchMix_N'], PatchMixIndex=PatchMixIndex, return_last_feat=True)
        return seg_logits, seg_feat

    def get_repatchmix_feat_decode(self, inputs, img_metas=None, scale=1):
        assert "PatchMix_N" in img_metas[0]
        PatchMixIndex = []
        for img_meta in img_metas:
            PatchMixIndex.append(torch.tensor(img_meta['PatchMixIndex']))
        PatchMixIndex = torch.stack(PatchMixIndex)
        repatchmix_feat = self.get_repatchmix_feat(inputs, PatchMix_N=img_meta['PatchMix_N'], PatchMixIndex=PatchMixIndex, scale=scale)
        return repatchmix_feat


    def forward_test(self, inputs, img_metas, test_cfg, return_last_feat=False):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, return_last_feat=return_last_feat)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # import pdb; pdb.set_trace()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # loss['acc_seg'] = accuracy(
        #     seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss
