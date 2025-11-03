# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss
from mmseg.core import add_prefix
from mmseg.ops import resize 
from mmseg.utils import cut_mix_label_adaptive, generate_unsup_cutmix_data_unimatch, generate_unsup_patchmix_data, generate_unsup_cutmix_data, generate_unsup_cutout_data, generate_sup_cutmix_data, generate_unsup_classmix_data, generate_sup_classmix_data, generate_mix_with_labeled_data
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from ..utils.structual_utils import dict_split, weighted_loss
import random

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 # model & configs
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 projection_head=None,
                 backbone_ema=None,
                 decode_head_ema=None,
                 neck_ema=None,
                 auxiliary_head_ema=None,
                 projection_head_ema=None,
                 backbone_pretrain=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 # ema
                 ema=False,
                 sup_ema=False,
                 ema_momentum=0.999,
                 attn_frozen=False,
                 attn_frozen_rate=0.0,
                 momentum_backbone=None,
                 momentum_head=None,
                 momentum_head_dropout=0.0,
                 momentum_head_exp=0.0,
                 momentum_exp=0.0,
                 ema_test=False,
                # supervised 
                 sup_ClassMix=False,
                 sup_cutmix=False,
                 # supervised 
                 unsup_weight=2.0,
                 unsup_confidence=0.75,
                 unsup_soft=False,
                 unsup_temperature=1.0,
                 iter_unsup_start=0,
                 # strong augmentation 
                 strong_aug_prob=0.5,
                 cutout_area=2,
                 use_CutMix=False,
                 use_CutOut=False,
                 use_ClassMix=False,
                 mix_with_labeled=False,
                 patchwise=False,
                 # PatchShuffle
                 use_PatchShuffle=False,
                 PatchMix_N=8,
                 patchmix_ratio=0.5,
                 patchsize=16,
                 use_PatchShuffle_w_Classmix=False,
                 use_PatchShuffle_w_Cutmix=False,
                 # position embedding ablation
                 no_pos_embed=False,
                 avg_pos_emd=False,
                 duplicate_pos_emd=False,
                 # PASA
                 adaptive_attn_mask=False,
                 attn_mask_weight=50,
                 attn_mask_seperate_head=False,  
                 attn_mask_w_fdrop=False,
                 # ncr
                 negative_class_ranking=False,
                 negative_class_ranking_mode='sup_only',
                 # other methds
                 use_fdrop=False,
                 unimatch=False,
                 fdrop_loss_weight=0.5,
                 use_cutmix_adaptive=False,
                 ):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.unsup_weight = unsup_weight
        self.unsup_confidence = unsup_confidence
        self.unsup_soft = unsup_soft
        self.unsup_temperature = unsup_temperature
        self.iter_unsup_start = iter_unsup_start
        self.ema = ema
        self.sup_ema = sup_ema
        self.sup_ClassMix = sup_ClassMix
        self.sup_cutmix = sup_cutmix
        self.momentum = ema_momentum
        self.attn_frozen = attn_frozen
        self.attn_frozen_rate = attn_frozen_rate
        self.use_fdrop = use_fdrop
        self.mix_with_labeled = mix_with_labeled

        if momentum_backbone is not None:
            self.momentum_backbone = momentum_backbone
        else:
            self.momentum_backbone = self.momentum
        if momentum_head is not None:
            self.momentum_head = momentum_head
        else:
            self.momentum_head = self.momentum
        self.momentum_head_dropout = momentum_head_dropout
        self.momentum_head_exp = momentum_head_exp
        self.momentum_exp = momentum_exp
        self.ema_test = ema_test
        self.no_pos_embed = no_pos_embed
        self.avg_pos_emd = avg_pos_emd
        self.duplicate_pos_emd = duplicate_pos_emd
        self.use_CutMix = use_CutMix
        self.use_CutOut = use_CutOut
        self.cutout_area = cutout_area
        self.patchwise = patchwise
        self.unimatch = unimatch
        self.use_ClassMix = use_ClassMix
        self.strong_aug_prob = strong_aug_prob
        self.use_PatchShuffle = use_PatchShuffle
        self.PatchMix_N = PatchMix_N
        self.patchmix_ratio = patchmix_ratio
        self.patchsize = patchsize
        self.use_PatchShuffle_w_Classmix = use_PatchShuffle_w_Classmix
        self.use_PatchShuffle_w_Cutmix = use_PatchShuffle_w_Cutmix
        self.negative_class_ranking = negative_class_ranking
        self.negative_class_ranking_mode = negative_class_ranking_mode
        self.fdrop_loss_weight = fdrop_loss_weight
        self.use_cutmix_adaptive = use_cutmix_adaptive
        if self.ema:
            self._init_ema_model(pretrained, backbone_ema, decode_head_ema, 
                neck_ema, auxiliary_head_ema, projection_head_ema)
        assert self.with_decode_head

        self.attn_mask_weight = attn_mask_weight
        self.adaptive_attn_mask = adaptive_attn_mask
        self.attn_mask_w_fdrop = attn_mask_w_fdrop
        self.attn_mask_seperate_head = attn_mask_seperate_head

    def _init_feature_contrast(self, feature_contrast):
        """Initialize ``feature_contrast``"""
        if feature_contrast is not None:
            if isinstance(feature_contrast, list):
                self.feature_contrast = nn.ModuleList()
                for head_cfg in feature_contrast:
                    self.feature_contrast.append(builder.build_head(head_cfg))
            else:
                self.feature_contrast = builder.build_head(feature_contrast)

    def _init_pretrain_model(self, pretrained, backbone_pretrain):
        self.backbone_pretrain = builder.build_backbone(backbone_pretrain)
        for param in self.backbone_pretrain.parameters():
            param.detach_()

    def _init_ema_model(self, pretrained, backbone_ema, decode_head_ema, 
            neck_ema, auxiliary_head_ema, projection_head_ema):
        if pretrained is not None:
            assert self.backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone_ema.pretrained = pretrained
        self.backbone_ema = builder.build_backbone(backbone_ema)
        for param in self.backbone_ema.parameters():
            param.detach_()

        if neck_ema is not None:
            self.neck_ema = builder.build_neck(neck_ema)
            for param in self.neck_ema.parameters():
                param.detach_()
        
        self.decode_head_ema = builder.build_head(decode_head_ema)
        for param in self.decode_head_ema.parameters():
            param.detach_()

        self.with_auxiliary_head_ema = False
        if auxiliary_head_ema is not None:
            self.with_auxiliary_head_ema = True
            if isinstance(auxiliary_head_ema, list):
                self.auxiliary_head_ema = nn.ModuleList()
                for head_cfg in auxiliary_head_ema:
                    self.auxiliary_head_ema.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head_ema = builder.build_head(auxiliary_head_ema)
            for param in self.auxiliary_head_ema.parameters():
                param.detach_()

        if projection_head_ema is not None:
            self.projection_head_ema = builder.build_head(projection_head_ema)
            for param in self.projection_head_ema.parameters():
                param.detach_()

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_projection_head(self, projection_head):
        """Initialize ``projection_head``"""
        if projection_head is not None:
            if isinstance(projection_head, list):
                self.projection_head = nn.ModuleList()
                for head_cfg in projection_head:
                    self.projection_head.append(builder.build_head(head_cfg))
            else:
                self.projection_head = builder.build_head(projection_head)

    def extract_feat(self, img, no_pos_embed=False, avg_pos_emd=False, duplicate_pos_emd=False, use_fdrop=False, 
            attn_mask=None, attn_mask_weight=5, adaptive_attn_mask=False):
        x = self.backbone(
            img, 
            no_pos_embed=no_pos_embed, 
            avg_pos_emd=avg_pos_emd,
            duplicate_pos_emd=duplicate_pos_emd,
            use_fdrop=use_fdrop,
            attn_mask=attn_mask,
            attn_mask_weight=attn_mask_weight,
            adaptive_attn_mask=adaptive_attn_mask,
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_ema(self, img):
        """Extract features from images."""
        x = self.backbone_ema(img)
        if self.with_neck:
            x = self.neck_ema(x)
        return x

    def encode_decode(self, img, img_metas, adaptive_attn_mask, return_last_feat=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        if return_last_feat and adaptive_attn_mask:
            x = self.extract_feat_ema(img)
            output = self._decode_head_forward_test_ema(x, img_metas)

            max_value = torch.max(F.softmax(output, dim=1), dim=1)[0]

            conf_mask_reshape = max_value.view(max_value.size(0), int(max_value.size(1) / self.patchsize), self.patchsize, int(max_value.size(2) / self.patchsize), self.patchsize)
            conf_mask_reshape = (1 - conf_mask_reshape).permute(0,1,3,2,4)   # [2, 32, 32, 16, 16]
            conf_mask_reshape = conf_mask_reshape.reshape(conf_mask_reshape.size(0), conf_mask_reshape.size(1), conf_mask_reshape.size(2), -1) 
            attn_mask = torch.sum(conf_mask_reshape, -1) / (self.patchsize * self.patchsize)
            x = self.extract_feat(img, attn_mask=attn_mask, adaptive_attn_mask=adaptive_attn_mask)
        else:
            x = self.extract_feat(img)

        out = self._decode_head_forward_test(x, img_metas, return_last_feat=return_last_feat)
        if return_last_feat:
            out_seg = resize(
                input=out[0],
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            return out_seg, self.backbone.multi_self_attn, out[1]
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def encode_decode_ema(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat_ema(img)
        out = self._decode_head_forward_test_ema(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas, return_last_feat=False):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, 
            self.test_cfg, return_last_feat)
        return seg_logits

    def _decode_head_forward_test_ema(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head_ema.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _auxiliary_head_forward_get_logits(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        aux_logits = [] 
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                seg_lgits = aux_head.forward_get_logits(x, self.train_cfg, img_metas)
                aux_logits.append(seg_lgits)
        else:
            seg_lgits = self.auxiliary_head.forward_get_logits(x, self.train_cfg, img_metas)
            aux_logits.append(seg_lgits)

        return aux_logits

    def _ema_auxiliary_head_forward_get_logits(self, x, img_metas):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        aux_logits = [] 
        if isinstance(self.auxiliary_head_ema, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head_ema):
                seg_lgits = aux_head.forward_get_logits(x, self.train_cfg, img_metas)
                aux_logits.append(seg_lgits)
        else:
            seg_lgits = self.auxiliary_head_ema.forward_get_logits(x, self.train_cfg, img_metas)
            aux_logits.append(seg_lgits)

        return aux_logits

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        current_iter = kwargs.pop("iter")
        self.current_iter = current_iter
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")

        x = kwargs['img']
        img_metas = kwargs['img_metas']

        for _, v in data_groups.items():
            v.pop("tag")

        self.losses = dict()
        if self.ema:
            with torch.no_grad():
                self.update_ema_variables(self.backbone, self.backbone_ema, self.momentum_backbone, self.attn_frozen)
                if self.with_neck:
                    self.update_ema_variables(self.neck, self.neck_ema, self.momentum)
                self.update_ema_variables(self.decode_head, self.decode_head_ema, self.momentum_head, self.momentum_head_dropout)
                if self.with_auxiliary_head_ema:
                    self.update_ema_variables(self.auxiliary_head, self.auxiliary_head_ema, self.momentum)

        # 1 - supervised loss for labeled data
        if "sup" in data_groups:
            sup_imgs = data_groups['sup']['img']
            sup_gts = data_groups['sup']['gt_semantic_seg']
            if self.sup_cutmix:
                if np.random.uniform(0, 1) < self.strong_aug_prob:  
                    sup_imgs, sup_gts = generate_sup_cutmix_data(sup_imgs, sup_gts)
            if self.sup_ClassMix:
                if np.random.uniform(0, 1) < 0.5:  
                    sup_imgs, sup_gts = generate_sup_classmix_data(sup_imgs, sup_gts)
            labeled_features = self.extract_feat(sup_imgs)
            loss_decode_sup = self._decode_head_forward_train(labeled_features, data_groups['sup']['img_metas'], sup_gts)
            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(labeled_features, data_groups['sup']['img_metas'], sup_gts)
                self.losses.update(loss_aux)

            self.losses.update(loss_decode_sup)

        if self.negative_class_ranking:
            if self.negative_class_ranking_mode == 'sup_only' or self.negative_class_ranking_mode == 'both':
                loss_ncr = 0
                pdist = nn.PairwiseDistance(p=2)

                labeled_features_ema = self.extract_feat_ema(data_groups['sup']['img'])
                labeled_ema_seg_logits = self.decode_head_ema.forward_get_logits(labeled_features_ema, self.train_cfg, data_groups['sup']['img_metas'])

                sup_strong_imgs = data_groups['sup_student']["img"]
                labeled_features = self.extract_feat(sup_strong_imgs)
                labeled_seg_logits = self.decode_head.forward_get_logits(labeled_features, self.train_cfg, data_groups['sup']['img_metas'])

                if labeled_seg_logits.shape[2:]!= sup_strong_imgs.shape[2:]:
                    labeled_seg_logits = resize(labeled_seg_logits, size=tuple(sup_strong_imgs.shape[2:]), mode='bilinear')
                    labeled_ema_seg_logits = resize(labeled_ema_seg_logits, size=tuple(sup_strong_imgs.shape[2:]), mode='bilinear')
                    
                for class_i in range(self.num_classes):
                    labeled_seg_logits_class_i = labeled_seg_logits.permute(0,2,3,1)[data_groups['sup']['gt_semantic_seg'].squeeze(1)==class_i]
                    labeled_seg_logits_class_i_ncr = torch.cat((labeled_seg_logits_class_i[:, :class_i], labeled_seg_logits_class_i[:,class_i+1:]),dim=1)
                    labeled_seg_logits_class_i_ncr = nn.functional.softmax(labeled_seg_logits_class_i_ncr, dim=1)
                    
                    labeled_ema_seg_logits_class_i = labeled_ema_seg_logits.permute(0,2,3,1)[data_groups['sup']['gt_semantic_seg'].squeeze(1)==class_i]
                    labeled_ema_seg_logits_class_i_ncr = torch.cat((labeled_ema_seg_logits_class_i[:, :class_i], labeled_ema_seg_logits_class_i[:,class_i+1:]),dim=1)
                    labeled_ema_seg_logits_class_i_ncr = nn.functional.softmax(labeled_ema_seg_logits_class_i_ncr, dim=1)
                    ## KL loss for NCR ##
                    if len(labeled_seg_logits_class_i_ncr) == 0:
                        loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                    else:
                        loss_ncr = loss_ncr + F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')
                    loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                loss_ncr = loss_ncr / (labeled_ema_seg_logits.size(0)*labeled_ema_seg_logits.size(2)*labeled_ema_seg_logits.size(3))
                self.losses.update({'loss_ncr_sup': loss_ncr})

        if self.sup_ema:
            assert self.ema
            labeled_features_ema = self.extract_feat_ema(data_groups['sup']['img'])
            with torch.no_grad():
                labeled_ema_seg_logits = self.decode_head_ema.forward_get_logits(labeled_features_ema, self.train_cfg, data_groups['sup']['img_metas'])
                labeled_ema_labels = nn.functional.softmax(labeled_ema_seg_logits, dim=1)
            labeled_ema_labels = resize(labeled_ema_labels, size=tuple(data_groups['sup']['gt_semantic_seg'].shape[2:]))
            _, labeled_ema_labels = torch.max(labeled_ema_labels, dim=1, keepdim=True)
            loss_decode_sup_ema = self._decode_head_forward_train(labeled_features, data_groups['sup']['img_metas'], labeled_ema_labels)
            self.losses.update({'loss_decode_sup_ema': loss_decode_sup_ema['decode.loss_ce']})

        # 2 - consistency loss between two augmented unlabeled images  
        if ("unsup_student" in data_groups) and self.unsup_weight != 0:
            if 'sup_student' in data_groups: 
                sup_imgs = data_groups['sup_student']['img']
            if self.unimatch:
                unsup_loss = weighted_loss(
                    self.foward_unsup_train_unimatch(
                        data_groups["unsup_teacher"], data_groups["unsup_teacher_mix"],
                        data_groups["unsup_student"], data_groups["unsup_student_2"],
                        data_groups["unsup_student_mix"], data_groups["unsup_student_2_mix"],
                        sup_imgs, sup_gts
                    ),
                    weight=self.unsup_weight,   
                )
            else:
                unsup_loss = weighted_loss(
                    self.foward_unsup_train(
                        data_groups["unsup_teacher"], data_groups["unsup_student"], sup_imgs, sup_gts
                    ),
                    weight=self.unsup_weight,   
                )
            if self.iter_unsup_start != 0:
                if current_iter > self.iter_unsup_start:
                    self.losses.update(unsup_loss) 
            else:
                self.losses.update(unsup_loss)

        return self.losses

    def foward_unsup_train(self, teacher_data, student_data, sup_imgs, sup_gts):
        # sort the teacher and student input to avoid some bugs
        loss_unsup = {}
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]

        with torch.no_grad():
            self.set_eval(self.ema)
            if not self.ema:
                teacher_info = self.extract_teacher_info(
                    teacher_data["img"][
                        torch.Tensor(tidx).to(teacher_data["img"].device).long()
                    ],
                    [teacher_data["img_metas"][idx] for idx in tidx],
                )
            else:
                teacher_info = self.extract_teacher_info_ema(
                    teacher_data["img"][
                        torch.Tensor(tidx).to(teacher_data["img"].device).long()
                    ],
                    [teacher_data["img_metas"][idx] for idx in tidx],
                )
            self.set_train(self.ema)
        
        if "conf_mask" in teacher_info:
            teacher_info["hard_seg_label"][teacher_info["conf_mask"]==0]=255
        
        student_info = self.extract_student_info(**student_data)
        student_imgs = student_data['img'].clone()

        if self.attn_mask_seperate_head:
            if teacher_info["conf_mask"].shape[-1] == student_info["img"].shape[-1]:  # VIT Style
                attn_mask_patch_size = self.patchsize
            else:
                attn_mask_patch_size = 8  # segformer 
            conf_mask_reshape = teacher_info["conf_mask"].view(teacher_info["conf_mask"].size(0), int(teacher_info["conf_mask"].size(1) / attn_mask_patch_size), attn_mask_patch_size, int(teacher_info["conf_mask"].size(1) / attn_mask_patch_size), attn_mask_patch_size)
            conf_mask_reshape = (1 - conf_mask_reshape).permute(0,1,3,2,4)   # [2, 32, 32, 16, 16]
            conf_mask_reshape = conf_mask_reshape.reshape(conf_mask_reshape.size(0), conf_mask_reshape.size(1), conf_mask_reshape.size(2), -1) 
            attn_mask = torch.sum(conf_mask_reshape, -1) / (attn_mask_patch_size * attn_mask_patch_size)
            unlabled_feat = self.extract_feat(
                student_info["img"],
                no_pos_embed=self.no_pos_embed, 
                avg_pos_emd=self.avg_pos_emd,
                duplicate_pos_emd=self.duplicate_pos_emd,
                use_fdrop=self.attn_mask_w_fdrop,
                attn_mask=attn_mask,
                attn_mask_weight=self.attn_mask_weight,
                adaptive_attn_mask=self.adaptive_attn_mask,
            )   
            student_info["backbone_feature"] = unlabled_feat
            loss_unsup['loss_seg_unsup_attn_mask'] = self.compute_pseudo_loss(student_info, teacher_info)['loss_seg_unsup'] * 0.5

        if self.use_fdrop:
            unlabled_feat = self.extract_feat(
                student_info["img"],
                no_pos_embed=self.no_pos_embed, 
                avg_pos_emd=self.avg_pos_emd,
                duplicate_pos_emd=self.duplicate_pos_emd,
                use_fdrop=self.use_fdrop
            )
            student_info["backbone_feature"] = unlabled_feat
            loss_unsup['loss_seg_unsup_fdrop'] = self.compute_pseudo_loss(student_info, teacher_info)['loss_seg_unsup'] * 0.5

        if 'valid' in student_info['img_metas'][0]:
            valid_mask = resize(input=student_info['valid_mask'].float().unsqueeze(1), size=teacher_info["hard_seg_label"].shape[1:], mode='nearest').squeeze(1)
            teacher_info["hard_seg_label"][valid_mask == 0] = 255

        if self.mix_with_labeled:
            conf_mask_reshape = teacher_info["conf_mask"].view(teacher_info["conf_mask"].size(0), int(teacher_info["conf_mask"].size(1) / self.patchsize), self.patchsize, int(teacher_info["conf_mask"].size(1) / self.patchsize), self.patchsize)
            conf_mask_reshape = conf_mask_reshape.permute(0,1,3,2,4) #[2, 32, 32, 16, 16]
            labeled_mix_mask = torch.zeros_like(conf_mask_reshape)

            conf_mask_reshape = conf_mask_reshape.reshape(conf_mask_reshape.size(0), conf_mask_reshape.size(1), conf_mask_reshape.size(2), -1)
            conf_mask_sum = torch.sum(conf_mask_reshape, -1)
            labeled_mix_mask[conf_mask_sum == 0] = 1 
            labeled_mix_mask = labeled_mix_mask.permute(0,1,3,2,4).reshape(teacher_info["conf_mask"].size(0), teacher_info["conf_mask"].size(1), teacher_info["conf_mask"].size(2))            
            teacher_info, student_info = generate_mix_with_labeled_data(
                    teacher_info, student_info, sup_imgs, sup_gts, labeled_mix_mask)
        if self.use_CutMix:  
            if np.random.uniform(0, 1) < self.strong_aug_prob:  
                teacher_info, student_info = generate_unsup_cutmix_data(
                    teacher_info, student_info, ratio=self.cutout_area, patchwise=self.patchwise)
        if self.use_CutOut:  
            if np.random.uniform(0, 1) < 0.5: 
                teacher_info, student_info = generate_unsup_cutout_data(
                    teacher_info, student_info, ratio=self.cutout_area, patchwise=self.patchwise)
        if self.use_ClassMix:
            if np.random.uniform(0, 1) < 0.5: 
                teacher_info, student_info = generate_unsup_classmix_data(teacher_info, student_info, patchwise=self.patchwise)
        if self.use_cutmix_adaptive:
            pred_u = F.softmax(teacher_info['seg_logits'], dim=1)
            # obtain pseudos
            logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)
            # obtain confidence
            entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
            entropy /= np.log(self.num_classes)
            confidence = 1.0 - entropy
            confidence = confidence * logits_u_aug
            confidence = confidence.mean(dim=[1,2])  # 1*C
            confidence = confidence.cpu().numpy().tolist()

            student_imgs, label_u_aug, logits_u_aug = cut_mix_label_adaptive(
                student_imgs,
                label_u_aug,
                logits_u_aug, 
                sup_imgs,
                sup_gts.squeeze(1), 
                confidence
            )
            student_info["img"] = student_imgs
            label_u_aug[logits_u_aug<self.unsup_confidence]=255
            teacher_info['hard_seg_label'] = label_u_aug
        if self.use_PatchShuffle:
            student_info, teacher_info = generate_unsup_patchmix_data(student_info, teacher_info, 
                PatchMix_N=self.PatchMix_N, patchmix_ratio=self.patchmix_ratio)

        if self.use_PatchShuffle_w_Cutmix:
            if np.random.uniform(0, 1) < self.strong_aug_prob:
                teacher_info, student_info = generate_unsup_cutmix_data(
                    teacher_info, student_info, ratio=self.cutout_area, patchwise=False)
            student_info, teacher_info = generate_unsup_patchmix_data(student_info, teacher_info, PatchMix_N=self.PatchMix_N,
                patchmix_ratio=self.patchmix_ratio)

        if self.use_PatchShuffle_w_Classmix:
            if np.random.uniform(0, 1) < 0.5:
                teacher_info, student_info = generate_unsup_classmix_data(
                    teacher_info, 
                    student_info, 
                    patchwise=self.patchwise,
                    patchsize=16 * self.PatchMix_N)

            student_info = generate_unsup_patchmix_data(student_info, PatchMix_N=self.PatchMix_N)

        if not self.attn_mask_seperate_head:
            if teacher_info["conf_mask"].shape[-1] == student_info["img"].shape[-1]:  # VIT Style
                attn_mask_patch_size = self.patchsize
            else:
                attn_mask_patch_size = 8  # segformer 

            conf_mask_reshape = teacher_info["conf_mask"].view(teacher_info["conf_mask"].size(0), int(teacher_info["conf_mask"].size(1) / attn_mask_patch_size), attn_mask_patch_size, int(teacher_info["conf_mask"].size(1) / attn_mask_patch_size), attn_mask_patch_size)
            conf_mask_reshape = (1 - conf_mask_reshape).permute(0,1,3,2,4)   # [2, 32, 32, 16, 16]
            conf_mask_reshape = conf_mask_reshape.reshape(conf_mask_reshape.size(0), conf_mask_reshape.size(1), conf_mask_reshape.size(2), -1) 
            attn_mask = torch.sum(conf_mask_reshape, -1) / (attn_mask_patch_size * attn_mask_patch_size)

            unlabled_feat = self.extract_feat(
                student_info["img"],
                no_pos_embed=self.no_pos_embed, 
                avg_pos_emd=self.avg_pos_emd,
                duplicate_pos_emd=self.duplicate_pos_emd,
                use_fdrop=self.attn_mask_w_fdrop,
                attn_mask=attn_mask,
                attn_mask_weight=self.attn_mask_weight,
                adaptive_attn_mask=self.adaptive_attn_mask
            )
        else:
            unlabled_feat = self.extract_feat(
                student_info["img"],
                no_pos_embed=self.no_pos_embed, 
                avg_pos_emd=self.avg_pos_emd,
                duplicate_pos_emd=self.duplicate_pos_emd
            )

        student_info["backbone_feature"] = unlabled_feat

        if self.use_fdrop or self.attn_mask_seperate_head:
            losses = self.compute_pseudo_loss(student_info, teacher_info)
            if self.negative_class_ranking:
                loss_unsup['loss_ncr_unsup'] = losses['loss_ncr_unsup'] * 0.5
            loss_unsup['loss_seg_unsup'] = losses['loss_seg_unsup'] * self.fdrop_loss_weight
        
        return loss_unsup

    def foward_unsup_train_unimatch(
        self, 
        teacher_data, teacher_mix_data, 
        student_data, student_2_data,
        student_mix_data, student_2_mix_data,
        sup_imgs, sup_gts
    ):
        # sort the teacher and student input to avoid some bugs
        loss_unsup = {}
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]

        # redo for u_t_mix
        tnames = [meta["filename"] for meta in teacher_mix_data["img_metas"]]
        snames = [meta["filename"] for meta in student_mix_data["img_metas"]]
        tidx_mix = [tnames.index(name) for name in snames]
        with torch.no_grad():
            self.set_eval(self.ema)
            if not self.ema:
                teacher_info = self.extract_teacher_info(
                    teacher_data["img"][
                        torch.Tensor(tidx).to(teacher_data["img"].device).long()
                    ],
                    [teacher_data["img_metas"][idx] for idx in tidx],
                )

                teacher_mix_info = self.extract_teacher_info(
                    teacher_mix_data["img"][
                        torch.Tensor(tidx_mix).to(teacher_mix_data["img"].device).long()
                    ],
                    [teacher_mix_data["img_metas"][idx] for idx in tidx_mix],
                )
            else:
                # import pdb; pdb.set_trace()
                teacher_info = self.extract_teacher_info_ema(
                    teacher_data["img"][
                        torch.Tensor(tidx).to(teacher_data["img"].device).long()
                    ],
                    [teacher_data["img_metas"][idx] for idx in tidx],
                )

                teacher_mix_info = self.extract_teacher_info_ema(
                    teacher_mix_data["img"][
                        torch.Tensor(tidx_mix).to(teacher_mix_data["img"].device).long()
                    ],
                    [teacher_mix_data["img_metas"][idx] for idx in tidx_mix],
                )
            self.set_train(self.ema)
        
        # data preparation
        if "conf_mask" in teacher_info:
            teacher_info["hard_seg_label"][teacher_info["conf_mask"]==0]=255
            teacher_mix_info["hard_seg_label"][teacher_mix_info["conf_mask"]==0]=255

        student_info = self.extract_student_info(**student_data)
        student_2_info = self.extract_student_info(**student_2_data)

        student_mix_info = self.extract_student_info(**student_mix_data)
        student_2_mix_info = self.extract_student_info(**student_2_mix_data)
        
        if 'valid' in student_info['img_metas'][0]:
            valid_mask = resize(input=student_info['valid_mask'].float().unsqueeze(1), 
                size=teacher_info["hard_seg_label"].shape[1:], mode='nearest').squeeze(1)
            teacher_info["hard_seg_label"][valid_mask == 0] = 255
            
            valid_mask = resize(input=student_mix_info['valid_mask'].float().unsqueeze(1), 
                size=teacher_mix_info["hard_seg_label"].shape[1:], mode='nearest').squeeze(1)
            teacher_mix_info["hard_seg_label"][valid_mask == 0] = 255

        if self.attn_mask_seperate_head:
            if teacher_info["conf_mask"].shape[-1] == student_info["img"].shape[-1]:  # VIT Style
                attn_mask_patch_size = self.patchsize
            else:
                attn_mask_patch_size = 8  # segformer 
            conf_mask_reshape = teacher_info["conf_mask"].view(teacher_info["conf_mask"].size(0), int(teacher_info["conf_mask"].size(1) / attn_mask_patch_size), attn_mask_patch_size, int(teacher_info["conf_mask"].size(1) / attn_mask_patch_size), attn_mask_patch_size)
            conf_mask_reshape = (1 - conf_mask_reshape).permute(0,1,3,2,4)   # [2, 32, 32, 16, 16]
            conf_mask_reshape = conf_mask_reshape.reshape(conf_mask_reshape.size(0), conf_mask_reshape.size(1), conf_mask_reshape.size(2), -1) 
            attn_mask = torch.sum(conf_mask_reshape, -1) / (attn_mask_patch_size * attn_mask_patch_size)
            unlabled_feat = self.extract_feat(
                student_info["img"],
                no_pos_embed=self.no_pos_embed, 
                avg_pos_emd=self.avg_pos_emd,
                duplicate_pos_emd=self.duplicate_pos_emd,
                use_fdrop=self.attn_mask_w_fdrop,
                attn_mask=attn_mask,
                attn_mask_weight=self.attn_mask_weight,
                adaptive_attn_mask=self.adaptive_attn_mask,
            )   
            student_info["backbone_feature"] = unlabled_feat
            loss_unsup['loss_seg_unsup_attn_mask'] = self.compute_pseudo_loss(student_info, teacher_info)['loss_seg_unsup'] * 0.5

        else:
            # head1 : use fdrop
            student_imgs = student_data['img'].clone()
            unlabled_feat = self.extract_feat(
                student_imgs,
                use_fdrop=True
            )
            student_info["backbone_feature"] = unlabled_feat
            loss_unsup['loss_seg_unsup_fdrop'] = self.compute_pseudo_loss(student_info, teacher_info)['loss_seg_unsup'] * 0.5

        # head2: augmentation1
        teacher_info_mixed_1 = teacher_info.copy()
        if np.random.uniform(0, 1) < self.strong_aug_prob:  
            teacher_info_mixed_1, student_info = generate_unsup_cutmix_data_unimatch(
                teacher_info, teacher_mix_info, student_info, student_mix_info, 
                ratio=self.cutout_area, patchwise=self.patchwise)
        if self.use_PatchShuffle:
            student_info, teacher_info_mixed_1 = generate_unsup_patchmix_data(
                student_info, teacher_info_mixed_1, 
                PatchMix_N=self.PatchMix_N,
                patchmix_ratio=self.patchmix_ratio
            )
        unlabled_feat = self.extract_feat(student_info["img"])
        student_info["backbone_feature"] = unlabled_feat
        losses1 = self.compute_pseudo_loss(student_info, teacher_info_mixed_1)
        loss_unsup['loss_seg_unsup_1'] = losses1['loss_seg_unsup'] * 0.25
        if self.negative_class_ranking:
            loss_unsup['loss_ncr_unsup_1'] = losses1['loss_ncr_unsup'] * 0.25

        # head2: augmentation2
        teacher_info_mixed_2 = teacher_info.copy()
        if np.random.uniform(0, 1) < self.strong_aug_prob:  
            teacher_info_mixed_2, student_2_info = generate_unsup_cutmix_data_unimatch(
                teacher_info, teacher_mix_info, student_2_info, student_2_mix_info, 
                ratio=self.cutout_area, patchwise=self.patchwise)

        if self.use_PatchShuffle:
            student_2_info, teacher_info_mixed_2 = generate_unsup_patchmix_data(
                student_2_info, teacher_info_mixed_2, 
                PatchMix_N=self.PatchMix_N,
                patchmix_ratio=self.patchmix_ratio
            )
        unlabled_feat = self.extract_feat(student_2_info["img"])
        student_2_info["backbone_feature"] = unlabled_feat
        losses2 = self.compute_pseudo_loss(student_2_info, teacher_info_mixed_2)
        loss_unsup['loss_seg_unsup_2'] = losses2['loss_seg_unsup'] * 0.25
        if self.negative_class_ranking:
            loss_unsup['loss_ncr_unsup_2'] = losses2['loss_ncr_unsup'] * 0.25
            
        return loss_unsup
    
    def extract_student_info(self, img, img_metas, gt_semantic_seg, **kwargs):
        student_info = {}
        student_info["img"] = img
        # if self.student.with_rpn:
        #     rpn_out = self.student.rpn_head(feat)
        #     student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["gt_semantic_seg"] = gt_semantic_seg[0]
        if 'valid' in student_info['img_metas'][0]:
            valid_mask = []
            for img_meta in student_info['img_metas']:
                valid_mask.append(img_meta['valid'])
            valid_mask = torch.stack(valid_mask)
            student_info['valid_mask'] = valid_mask
        # student_info["transform_matrix"] = [
        #     torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
        #     for meta in img_metas
        # ]
        return student_info

    def extract_teacher_info(self, img, img_metas, unsup_confidence=None):
        teacher_info = {}
        feat = self.extract_feat(img)
        teacher_info["backbone_feature"] = feat 
        seg_logits = self.decode_head.forward_get_logits(feat, self.train_cfg, img_metas)
        teacher_info['seg_logits'] = seg_logits
        if self.unsup_temperature != 1.0:
            seg_logits = torch.pow(seg_logits, 1.0/self.unsup_temperature)
        seg_logits_after_softmax = nn.functional.softmax(seg_logits, dim=1)
        if self.unsup_soft:
            teacher_info["soft_seg_label"] = seg_logits_after_softmax
        max_value, teacher_info["hard_seg_label"] = torch.max(nn.functional.softmax(teacher_info['seg_logits'], dim=1), dim=1)
        
        if unsup_confidence is not None:
            conf_mask = max_value > unsup_confidence
            teacher_info["conf_mask"] = conf_mask * 1
        if unsup_confidence is None and self.unsup_confidence != 0:
            conf_mask = max_value > self.unsup_confidence
            teacher_info["conf_mask"] = conf_mask * 1

        teacher_info["img_metas"] = img_metas
        return teacher_info

    def extract_teacher_info_ema(self, img, img_metas, unsup_confidence=None):
        teacher_info = {}
        feat = self.extract_feat_ema(img)
        teacher_info["backbone_feature"] = feat 
        seg_logits = self.decode_head_ema.forward_get_logits(feat, self.train_cfg, img_metas)
        teacher_info['seg_logits'] = seg_logits
        
        if self.unsup_temperature != 1.0 or self.unsup_soft:
            seg_logits_soft = torch.pow(seg_logits, 1.0 / self.unsup_temperature)
        if self.unsup_soft:
            seg_logits_after_softmax = nn.functional.softmax(seg_logits_soft, dim=1)
            teacher_info["soft_seg_label"] = seg_logits_after_softmax
        else:
            seg_logits_after_softmax = nn.functional.softmax(seg_logits, dim=1)

        max_value, teacher_info["hard_seg_label"] = torch.max(nn.functional.softmax(teacher_info['seg_logits'], dim=1), dim=1)
        
        if self.with_auxiliary_head_ema:
            aux_logits = self._ema_auxiliary_head_forward_get_logits(teacher_info["backbone_feature"], img_metas)
            teacher_info['aux_logits'] = aux_logits

        if unsup_confidence is not None:
            conf_mask = max_value > unsup_confidence
            teacher_info["conf_mask"] = conf_mask * 1
        if unsup_confidence is None and self.unsup_confidence != 0:
            conf_mask = max_value > self.unsup_confidence
            teacher_info["conf_mask"] = conf_mask * 1
    
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_pseudo_loss(self, student_info, teacher_info):
        loss_unsup = {}
        if not self.unsup_soft:
            loss_unsup_criterion = CrossEntropyLoss(reduction='none', ignore_index=255)
            students_prediction = self.decode_head.forward_get_logits(
                student_info['backbone_feature'], self.train_cfg, student_info["img_metas"])
            loss = loss_unsup_criterion(students_prediction, teacher_info["hard_seg_label"])            
        else:
            loss_unsup_criterion = CrossEntropyLoss(reduction='none')
            students_prediction = self.decode_head.forward_get_logits(student_info['backbone_feature'], self.train_cfg, student_info["img_metas"])
            loss = loss_unsup_criterion(students_prediction, teacher_info["soft_seg_label"])
        mask = torch.ones_like(loss)

        if self.unsup_confidence != 0:
            if self.unsup_soft:
                mask = mask * teacher_info["conf_mask"]   
            # import pdb; pdb.set_trace()
            mask_ratio = torch.sum(
                teacher_info["conf_mask"]).float() / torch.sum(torch.ones_like(loss))
            loss_unsup['mask_ratio'] = mask_ratio
            if self.momentum_head_exp != 0:
                self.momentum_head = mask_ratio**self.momentum_head_exp
                loss_unsup['momentum_head'] = self.momentum_head
            if self.momentum_exp != 0:
                self.momentum_head = mask_ratio**self.momentum_exp
                self.momentum_backbone = mask_ratio**self.momentum_exp
                loss_unsup['momentum_head'] = self.momentum_head
        loss = torch.mean(loss * mask.to(student_info["img"].device))
        loss_unsup['loss_seg_unsup'] = loss

        if self.negative_class_ranking:
            if self.negative_class_ranking_mode == 'unsup_only' or self.negative_class_ranking_mode == 'both':
                loss_ncr = 0
                pdist = nn.PairwiseDistance(p=2)

                labeled_ema_seg_logits = teacher_info['seg_logits']
                labeled_seg_logits = students_prediction

                for class_i in range(self.num_classes):
                    labeled_seg_logits_class_i = labeled_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_seg_logits_class_i_ncr = torch.cat((labeled_seg_logits_class_i[:, :class_i], labeled_seg_logits_class_i[:, class_i + 1:]),dim=1)
                    labeled_seg_logits_class_i_ncr = nn.functional.softmax(labeled_seg_logits_class_i_ncr, dim=1)
                    
                    labeled_ema_seg_logits_class_i = labeled_ema_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_ema_seg_logits_class_i_ncr = torch.cat((labeled_ema_seg_logits_class_i[:, :class_i], labeled_ema_seg_logits_class_i[:,class_i + 1:]),dim=1)
                    labeled_ema_seg_logits_class_i_ncr = nn.functional.softmax(labeled_ema_seg_logits_class_i_ncr, dim=1)
                    loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                loss_ncr = loss_ncr / (labeled_ema_seg_logits.size(0) * labeled_ema_seg_logits.size(2) * labeled_ema_seg_logits.size(3))
                loss_unsup.update({'loss_ncr_unsup': loss_ncr})
            
            if self.negative_class_ranking_mode == 'all':
                loss_ncr = 0
                pdist = nn.PairwiseDistance(p=2)

                labeled_ema_seg_logits = teacher_info['seg_logits']
                labeled_seg_logits = students_prediction

                for class_i in range(self.num_classes):
                    labeled_seg_logits_class_i = labeled_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_seg_logits_class_i_ncr = labeled_seg_logits_class_i
                    labeled_seg_logits_class_i_ncr = nn.functional.softmax(labeled_seg_logits_class_i_ncr, dim=1)
                    
                    labeled_ema_seg_logits_class_i = labeled_ema_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_ema_seg_logits_class_i_ncr = labeled_ema_seg_logits_class_i
                    labeled_ema_seg_logits_class_i_ncr = nn.functional.softmax(labeled_ema_seg_logits_class_i_ncr, dim=1)
                    loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                loss_ncr = loss_ncr / (labeled_ema_seg_logits.size(0) * labeled_ema_seg_logits.size(2) * labeled_ema_seg_logits.size(3))
                loss_unsup.update({'loss_ncr_unsup': loss_ncr})

            if self.negative_class_ranking_mode == 'kl':
                loss_ncr = 0
                pdist = nn.PairwiseDistance(p=2)

                labeled_ema_seg_logits = teacher_info['seg_logits']
                labeled_seg_logits = students_prediction

                for class_i in range(self.num_classes):
                    labeled_seg_logits_class_i = labeled_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_seg_logits_class_i_ncr = labeled_seg_logits_class_i
                    labeled_seg_logits_class_i_ncr = nn.functional.softmax(labeled_seg_logits_class_i_ncr, dim=1)
                    
                    labeled_ema_seg_logits_class_i = labeled_ema_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_ema_seg_logits_class_i_ncr = labeled_ema_seg_logits_class_i
                    labeled_ema_seg_logits_class_i_ncr = nn.functional.softmax(labeled_ema_seg_logits_class_i_ncr, dim=1)
                    if len(labeled_seg_logits_class_i_ncr) == 0 or F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')>1e6:
                        loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                    else:
                        loss_ncr = loss_ncr + F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')
                loss_ncr = loss_ncr / (labeled_ema_seg_logits.size(0) * labeled_ema_seg_logits.size(2) * labeled_ema_seg_logits.size(3))
                loss_unsup.update({'loss_ncr_unsup': loss_ncr})

            if self.negative_class_ranking_mode == 'unsup_only_kl':
                loss_ncr = 0
                pdist = nn.PairwiseDistance(p=2)

                labeled_ema_seg_logits = teacher_info['seg_logits']
                labeled_seg_logits = students_prediction

                for class_i in range(self.num_classes):
                    labeled_seg_logits_class_i = labeled_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_seg_logits_class_i_ncr = torch.cat((labeled_seg_logits_class_i[:, :class_i], labeled_seg_logits_class_i[:, class_i + 1:]),dim=1)
                    labeled_seg_logits_class_i_ncr = nn.functional.softmax(labeled_seg_logits_class_i_ncr, dim=1)
                    
                    labeled_ema_seg_logits_class_i = labeled_ema_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_ema_seg_logits_class_i_ncr = torch.cat((labeled_ema_seg_logits_class_i[:, :class_i], labeled_ema_seg_logits_class_i[:,class_i + 1:]),dim=1)
                    labeled_ema_seg_logits_class_i_ncr = nn.functional.softmax(labeled_ema_seg_logits_class_i_ncr, dim=1)
                    if len(labeled_seg_logits_class_i_ncr) == 0 or F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')>1e6:
                    # if len(labeled_seg_logits_class_i_ncr) == 0:
                        loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                    else:
                        loss_ncr = loss_ncr + F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')
                loss_ncr = loss_ncr / (labeled_ema_seg_logits.size(0) * labeled_ema_seg_logits.size(2) * labeled_ema_seg_logits.size(3))
                loss_unsup.update({'loss_ncr_unsup': loss_ncr})

            if self.negative_class_ranking_mode == 'reweight_unsup_only_kl':
                loss_ncr = 0
                pdist = nn.PairwiseDistance(p=2)

                labeled_ema_seg_logits = teacher_info['seg_logits']
                labeled_seg_logits = students_prediction

                for class_i in range(self.num_classes):
                    labeled_seg_logits_class_i = labeled_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_seg_logits_class_i_ncr = torch.cat((labeled_seg_logits_class_i[:, :class_i], labeled_seg_logits_class_i[:, class_i + 1:]),dim=1)
                    labeled_seg_logits_class_i_ncr = nn.functional.softmax(labeled_seg_logits_class_i_ncr, dim=1)
                    
                    labeled_ema_seg_logits_class_i = labeled_ema_seg_logits.permute(0,2,3,1)[teacher_info["hard_seg_label"].squeeze(1) == class_i]
                    labeled_ema_seg_logits_class_i_ncr = torch.cat((labeled_ema_seg_logits_class_i[:, :class_i], labeled_ema_seg_logits_class_i[:,class_i + 1:]),dim=1)
                    labeled_ema_seg_logits_class_i_ncr = nn.functional.softmax(labeled_ema_seg_logits_class_i_ncr, dim=1)
                    if len(labeled_seg_logits_class_i_ncr) == 0 or F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')>1e6:
                        loss_ncr = loss_ncr + torch.sum(pdist(labeled_seg_logits_class_i_ncr, labeled_ema_seg_logits_class_i_ncr))
                    else:
                        loss_ncr = loss_ncr + F.kl_div(torch.log(labeled_seg_logits_class_i_ncr), labeled_ema_seg_logits_class_i_ncr, reduction='sum')
                loss_ncr = 0.5 * loss_ncr / (labeled_ema_seg_logits.size(0) * labeled_ema_seg_logits.size(2) * labeled_ema_seg_logits.size(3))
                loss_unsup.update({'loss_ncr_unsup': loss_ncr})
                        
        return loss_unsup

    def update_ema_variables(self, model, ema_model, momentum=0.999, dropout=0.0, attn_frozen=False):

        # Use the true average until the exponential average is more correct
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.named_parameters(), ema_model.named_parameters()
        ):                
            if dropout != 0:
                if random.random() < dropout:
                    # print(src_name)
                    continue
            if attn_frozen:
                if 'attn' in 'tgt_name':
                    tgt_parm.data = src_parm.data
                else:
                    tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
            else:
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

        for (src_name, src_buffer), (tgt_name, tgt_buffer) in zip(
            model.named_buffers(), ema_model.named_buffers()):
            if 'bn' in src_name and 'num_batches_tracked' not in src_name:
                # print(src_name)
                tgt_buffer.data.mul_(momentum).add_(src_buffer.data, alpha=1 - momentum)
    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                if self.ema_test:
                    crop_seg_logit = self.encode_decode_ema(crop_img, img_meta)
                else:
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale, tde, return_last_feat=False, use_attn_mask=None):
        """Inference with full image."""
        if self.ema_test:
            seg_logit = self.encode_decode_ema(img, img_meta)
        else:
            if return_last_feat:
                seg_logit, attn_mask, feat = self.encode_decode(img, img_meta, return_last_feat=True, use_attn_mask=use_attn_mask)
            else:
                seg_logit = self.encode_decode(img, img_meta, return_last_feat=False)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            if tde:
                outputs_bias = self.decode_head.conv_seg(self.decode_head.dropout(self.decode_head.mean_feat.unsqueeze(0)))
                outputs_bias = resize(
                    outputs_bias,
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            if tde:
                # import pdb; pdb.set_trace()
                feat = resize(
                    feat.unsqueeze(0),
                    size=self.decode_head.mean_feat.shape[1:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
                c, h, w = feat.shape
                feat = feat.view(c, h*w).permute(1, 0)
                # cos_val = 
                # import pdb; pdb.set_trace()
                # cos_val, sin_val = get_cos_sin(feat, self.decode_head.mean_feat)
                # outputs_bias = cos_val * outputs_bias
                # mask_back = torch.argmax(seg_logit, 1)==0
                # mask_fore = torch.argmax(seg_logit, 1)!=0
                # outputs_bias[:,:,mask_back.squeeze(0)] = 0.0
                # outputs_bias[:,0,mask_fore.squeeze(0)] = 1e+4
                seg_logit = seg_logit - 0.0*outputs_bias
  
        if return_last_feat:
            return seg_logit, attn_mask, feat  
        return seg_logit

    def inference(self, img, img_meta, rescale, tde=False, return_last_feat=False, use_attn_mask=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            if return_last_feat:
                seg_logit, attn_mask, last_feat = self.whole_inference(img, img_meta, rescale, tde, 
                    return_last_feat=return_last_feat, use_attn_mask=use_attn_mask)
            else:
                seg_logit = self.whole_inference(img, img_meta, rescale, tde)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        if return_last_feat:
            return output, attn_mask, last_feat
        return output

    def simple_test(self, img, img_meta, rescale=True, tde=False, return_last_feat=False):
        """Simple test with single image."""
        if return_last_feat:
            seg_logit, feat = self.inference(img, img_meta, rescale, tde, return_last_feat)
        else:
            seg_logit = self.inference(img, img_meta, rescale, tde)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        
        if return_last_feat:
            return seg_pred, feat
        return seg_pred

    def simple_test_with_logits(self, img, img_meta, rescale=True, tde=False, return_last_feat=True, use_attn_mask=None):
        # return_last_feat = False
        """Simple test with single image."""
        if return_last_feat:
            seg_logit, attn_mask, feat = self.inference(img, img_meta, rescale, tde, return_last_feat, use_attn_mask)
        else:
            seg_logit = self.inference(img, img_meta, rescale)
        # seg_pred = seg_logit.argmax(dim=1)
        max_value, seg_pred = torch.max(seg_logit, dim=1)

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        if return_last_feat:
            return seg_pred, max_value, attn_mask, feat
        return seg_pred, max_value

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def set_eval(self, ema=False):
        if not ema:
            self.backbone.eval()
            if self.with_neck:
                self.neck.eval()
            self.decode_head.eval()
            if self.with_auxiliary_head:
                self.auxiliary_head.eval()
        else:
            self.backbone_ema.eval()
            if self.with_neck:
                self.neck_ema.eval()
            self.decode_head_ema.eval()
            if self.with_auxiliary_head_ema:
                self.auxiliary_head_ema.eval()

    def set_train(self, ema=False):
        if not ema:
            self.backbone.train()
            if self.with_neck:
                self.neck.train()
            self.decode_head.train()
            if self.with_auxiliary_head:
                self.auxiliary_head.train()
        else:
            self.backbone_ema.train()
            if self.with_neck:
                self.neck_ema.train()
            self.decode_head_ema.train()
            if self.with_auxiliary_head_ema:
                self.auxiliary_head_ema.train()
