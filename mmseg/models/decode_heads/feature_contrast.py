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
class FeatureContrast(BaseModule):
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
        ignore_label=255):
        super(FeatureContrast, self).__init__()

        self.channels = channels
        assert isinstance(self.channels, int)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.ignore_label = ignore_label
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
        
        self.Selectors_head = nn.ModuleList()
        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(channels, channels),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(channels, 1)
            )
            self.Selectors_head.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(channels, channels),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(channels, 1)
            )
            self.Selectors_head.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

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
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            selector = self.Selectors_head.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
            features_c = features[mask_c, :] # get features from class c
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        selector.eval()
                        # get ranking scores
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        # sort them
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        # indices = indices.cpu().numpy()
                        # features_c = features_c.cpu().numpy()
                        # get features with highest rankings
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                        selector.train()

                else:
                    new_features = features_c

                if self.memory_saved[c] == 0:
                    self.memory_bank[c,:new_features.shape[0]] = new_features
                else:
                    self.memory_bank[c, :(self.memory_saved[c]+new_features.shape[0])] = torch.cat((new_features,self.memory_bank[c][:self.memory_saved[c]]),0)[:self.memory_per_class, :]
                self.memory_saved[c] += new_features.shape[0]

                self.memory_saved[c] = min(self.memory_saved[c], self.memory_per_class)


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

        for c in range(self.n_classes):
            # get features of an specific class
            mask_c = class_labels == c
            features_c = features[mask_c,:]
            memory_saved_c = self.memory_saved[c]
            memory_c = self.memory_bank[c, :memory_saved_c] # N, 256

            # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
            selector = self.Selectors_head.__getattr__(
                'contrastive_class_selector_' + str(c))
            selector_memory = self.Selectors_head.__getattr__(
                'contrastive_class_selector_memory' + str(c))

            if memory_saved_c > 1 and features_c.shape[0] > 1:

                # L2 normalize vectors
                memory_c = F.normalize(memory_c, dim=1) # N, 256
                features_c_norm = F.normalize(features_c, dim=1) # M, 256

                # compute similarity. All elements with all elements
                similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
                distances = 1 - similarities # values between [0, 2] where 0 means same vectors
                # M (elements), N (memory)


                # now weight every sample

                learned_weights_features = selector(features_c.detach()) # detach for trainability
                learned_weights_features_memory = selector_memory(memory_c)

                # self-atention in the memory featuers-axis and on the learning contrsative featuers-axis
                learned_weights_features = torch.sigmoid(learned_weights_features)
                rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)) * learned_weights_features
                rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
                distances = distances * rescaled_weights


                learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
                learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
                rescaled_weights_memory = (learned_weights_features_memory.shape[0] / learned_weights_features_memory.sum(dim=0)) * learned_weights_features_memory
                rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
                distances = distances * rescaled_weights_memory


                loss = loss + distances.mean()

        return loss / self.num_classes


