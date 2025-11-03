from __future__ import division

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler, WeightedRandomSampler


class DistributedSemiBalanceSampler(Sampler):
    def __init__(
        self,
        dataset,
        by_prob=False,
        max_iter_size=None,
        sample_ratio=None,
        samples_per_gpu=1,
        num_replicas=None,
        rank=None,
        **kwargs
    ):
        # import pdb; pdb.set_trace()
        # check to avoid some problem
        assert samples_per_gpu > 1, "samples_per_gpu should be greater than 1."
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.by_prob = by_prob
        self.max_iter_size = max_iter_size
        self.num_samples = 0
        self.cumulative_sizes = dataset.cumulative_sizes  # size of supervised and unsupervised data [#sup, #unsup]
        
        # decide the frequency to sample each kind of datasets
        if not isinstance(sample_ratio, list):
            sample_ratio = [sample_ratio] * len(self.cumulative_sizes)
        self.sample_ratio = sample_ratio
        self.sample_ratio = [
            int(sr / min(self.sample_ratio)) for sr in self.sample_ratio
        ]
        self.size_of_dataset = []
        
        for i in range(len(self.cumulative_sizes)):
            size_of_dataset = np.ceil(self.cumulative_sizes[i] / self.sample_ratio[i])

            self.size_of_dataset.append(
                int(np.ceil(size_of_dataset / self.samples_per_gpu / self.num_replicas))
                * self.samples_per_gpu
            )
        for j in range(len(self.cumulative_sizes)):
            self.num_samples += self.size_of_dataset[-1] * self.sample_ratio[j]
        # import pdb; pdb.set_trace()
        self.total_size = self.num_samples * self.num_replicas
        # group_factor = [g / sum(self.group_sizes) for g in self.group_sizes]
        # self.epoch_length = np.ceil(self.max_iter_size*self.samples_per_gpu/self.total_size)
        # self.epoch_length[-1] = epoch_length - sum(self.epoch_length[:-1])

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        cumulative_sizes = [0] + self.cumulative_sizes
        indice_per_dataset = []
        
        for i in range(len(self.cumulative_sizes)):
            indice_per_dataset.append(np.array(range(cumulative_sizes[i], cumulative_sizes[i+1])))

            shuffled_indice_per_dataset = [
                s[list(torch.randperm(int(s.shape[0]), generator=g).numpy())]
                for s in indice_per_dataset
            ]

        # split into
        total_indice = []
        batch_idx = 0
        while batch_idx < self.max_iter_size * self.num_replicas:
            ratio = [x / sum(self.sample_ratio) for x in self.sample_ratio]

            # num of each dataset
            ratio = [int(r * self.samples_per_gpu) for r in ratio]

            ratio[-1] = self.samples_per_gpu - sum(ratio[:-1])
            selected = []
            # import pdb; pdb.set_trace()
            for i in range(len(shuffled_indice_per_dataset)):
                if len(shuffled_indice_per_dataset[i]) < ratio[i]:
                    shuffled_indice_per_dataset[i] = np.concatenate(
                        (
                            shuffled_indice_per_dataset[i],
                            indice_per_dataset[i][
                                list(
                                    torch.randperm(
                                        int(indice_per_dataset[i].shape[0]),
                                        generator=g,
                                    ).numpy()
                                )
                            ],
                        )
                    )

                selected.append(shuffled_indice_per_dataset[i][: ratio[i]])
                shuffled_indice_per_dataset[i] = shuffled_indice_per_dataset[i][
                    ratio[i] :
                ]
            selected = np.concatenate(selected)
            total_indice.append(selected)
            batch_idx += 1
            # print(self.size_of_dataset)
        indices = np.concatenate(total_indice)
        # indices.append(indice)
        # indices = np.concatenate(indices)  # k
        indices = [
            indices[j]
            for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu,
                    generator=g,
                )
            )
            for j in range(
                i * self.samples_per_gpu,
                (i + 1) * self.samples_per_gpu,
            )
        ]

        offset = len(self) * self.rank
        indices = indices[offset : offset + len(self)]
        # print(indices)
        assert len(indices) == len(self)
        # print('indices:', indices)
        return iter(indices)

    def __len__(self):
        return self.max_iter_size * self.samples_per_gpu

    def set_epoch(self, epoch):
        self.epoch = epoch

    # duplicated, implement it by weight instead of sampling
    # def update_sample_ratio(self):
    #     if self.dynamic_step is not None:
    #         self.sample_ratio = [d(self.epoch) for d in self.dynamic]
