# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss

class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, 
        ignore_index=255, 
        thresh=0.7, 
        min_kept=256, 
        use_weight=False, 
        reduction='mean'
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            # weight = torch.FloatTensor(
            #     [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
            #      0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
            #      1.0865, 1.1529, 1.0507]).cuda()
            weight = torch.FloatTensor(
                [
                    0.4762,
                    0.5,
                    0.4762,
                    1.4286,
                    1.1111,
                    0.4762,
                    0.8333,
                    0.5,
                    0.5,
                    0.8333,
                    0.5263,
                    0.5882,
                    1.4286,
                    0.5,
                    3.3333,
                    5.0,
                    10.0,
                    2.5,
                    0.8333,
                ]
            ).cuda()
            print(f"OHEM weight {weight.cpu().numpy()}")

            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)



@LOSSES.register_module()
class OHEM_CrossEntropyLoss(nn.Module):
    def __init__(self,
                 thres=0.7,
                 reduction='mean',
                 min_kept=100000,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 use_sigmoid=False,
                 loss_name='loss_ohem_ce',
                 avg_non_ignore=False,
                 use_weight=False):
        super(OHEM_CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.cls_criterion = OhemCrossEntropy2dTensor(
            ignore_index, thres, min_kept, use_weight, reduction
        )

        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self, cls_score, label, weight=None, ignore_index=None):
        """Forward function."""
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
