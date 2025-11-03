# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_norm_layer
from mmseg.ops import Upsample
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule


@HEADS.register_module()
class FeatureContrastV2(BaseModule):
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
        channels,
        dataset, 
        num_samples,  
        num_classes,
        memory_per_class=2048, 
        feature_size=256, 
        n_classes=19,
        ignore_label=255,
        negative=False):
        super(FeatureContrastV2, self).__init__()

        self.channels = channels
        assert isinstance(self.channels, int)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.negative = negative
        if dataset == 'cityscapes': # usually all classes in one image
            self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
        elif dataset == 'pascal_voc': # usually only around 3 classes on each image, except background class
            self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))

        self.register_buffer('memory_saved', torch.zeros(num_classes).long())
        self.register_buffer('memory_bank', torch.zeros(num_classes, memory_per_class, feature_size))

        # for class_c in range(num_classes):
        #     # self.register_buffer('memory_'+str(class_c), memory[class_c])
        #     self.register_buffer('memory_'+str(class_c), torch.tensor(1.0))
        #     self.register_buffer('memory_saved'+str(class_c), torch.zeros(num_classes))
        
    def add_features_from_sample_learned(self, features, class_labels, batch_size):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_Selectors_head)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach()

        elements_per_class = batch_size * self.per_class_samples_per_image
        # for each class, save [elements_per_class]
        '''
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            features_c = features[mask_c, :] # get features from class c
            features_c = F.normalize(features_c, dim=1) # normalize
            if features_c.shape[0] > 0:
                # if features_c.shape[0] > elements_per_class:
                if self.memory_saved[c] + features_c.shape[0] >= self.memory_per_class:
                    # use herding to select
                    new_features = []
                    current_feature = torch.cat(
                        (self.memory_bank[c, :self.memory_saved[c]], features_c), dim=0)
                    mu = torch.mean(current_feature, dim=0)
                    
                    w_t = mu
                    id_herding = 0
                    while id_herding < self.memory_per_class:
                        tmp_t = torch.mm(w_t.unsqueeze(0), current_feature.T)
                        ind_max = torch.argmax(tmp_t)
                        id_herding += 1
                        new_features.append(current_feature[ind_max].unsqueeze(0))
                        w_t = w_t + mu - current_feature[ind_max]
                    self.memory_bank[c] = torch.cat(new_features, dim=0)
                    self.memory_saved[c] = self.memory_per_class
                else:
                    if self.memory_saved[c] == 0:
                        self.memory_bank[c,:features_c.shape[0]] = features_c
                    else:
                        self.memory_bank[c, :(self.memory_saved[c]+features_c.shape[0])] = torch.cat(
                            (self.memory_bank[c][:self.memory_saved[c]], features_c),0)
                    self.memory_saved[c] += features_c.shape[0]

                # self.memory_saved[c] = min(self.memory_saved[c], self.memory_per_class)
        '''
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            features_c = features[mask_c, :]  # get features from class c
            features_c = F.normalize(features_c, dim=1)  # normalize
            if features_c.shape[0] > 0:
                mean_features_c = torch.mean(features_c, 0)
                # if features_c.shape[0] > elements_per_class:
                if self.memory_saved[c] + 1 > self.memory_per_class:

                    self.memory_bank[c] = torch.cat((self.memory_bank[c, 1:], mean_features_c.unsqueeze(0)), dim=0)
                    assert self.memory_bank[c].size() == torch.Size([self.memory_per_class, features_c.size(1)])
                    self.memory_saved[c] = self.memory_per_class
                else:
                    self.memory_bank[c, self.memory_saved[c]:self.memory_saved[c] + 1] = mean_features_c
                    self.memory_saved[c] += 1
                # self.memory_saved[c] = min(self.memory_saved[c], self.memory_per_class)

    def forward(self, features, class_labels):
       
        """

        Args:
            features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
            class_labels: N corresponding class labels for every feature vector
            num_classes: number of classesin the dataet
            memory: memory bank [List]

        Returns:
            returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
        """
        loss = 0
        n_c = 0

        for c in range(self.n_classes):
            # get features of an specific class
            mask_c = class_labels == c
            features_c = features[mask_c,:]
            memory_saved_c = self.memory_saved[c]
            memory_c = self.memory_bank[c, :memory_saved_c] # N, 256
            if memory_saved_c > 1 and features_c.shape[0] > 1:

                # L2 normalize vectors
                memory_c = F.normalize(memory_c, dim=1) # N, 256
                features_c_norm = F.normalize(features_c, dim=1) # M, 256

                # # compute similarity. All elements with all elements
                similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
                distances = 1 - similarities  # values between [0, 2] where 0 means same vectors
                # # M (elements), N (memory)
                loss = loss + distances.mean()
                
                # BYOL MSE loss
                # import pdb; pdb.set_trace()
                # distances = 2 - 2 * (features_c_norm * memory_c).sum() / features_c_norm.size(0)
                # loss = loss + distances

                n_c += 1
                if self.negative:
                    n_c_cnt = 0
                    distances_nc_all = 0
                    for n_c in range(self.n_classes):
                        if n_c == c:
                            continue
                        if self.memory_saved[n_c] > 0:
                            memory_nc = self.memory_bank[n_c, :self.memory_saved[n_c]] # N_C, 256
                            similarities_nc = torch.mm(features_c_norm, memory_nc.transpose(1, 0))  # MxN
                            distances_nc = 1 + similarities_nc  # values between [0, 2] where 2 means same vectors
                            distances_nc_all = distances_nc_all + distances_nc.mean()
                            n_c_cnt += 1
                    if n_c_cnt > 0:
                        loss = loss + distances_nc_all / n_c_cnt

        if self.negative:
            return loss / n_c
        return loss / self.num_classes
