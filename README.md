## Introduction

This is the repo for our CVPR2024 paper "Training Vision Transformers for Semi-Supervised Semantic Segmentation".

## Setup

Please refer to [get_started.md](docs/en/get_started.md#installation) for MMSegmentation environments installation and [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets) for Pascal VOC 2012 and Cityscapes datasets preparation. Datasplits for labeled/unlabeled data division have been given under the `data` folder.

## Pre-Training Weight

#### 1. Download Pre-trained Weights

We recommend using the pre-trained weights provided by OpenMMLab, which are already adapted for MMCV/MMSegmentation.

- **DeiT-Base (Patch16, ImageNet-1k)**  
  Download link:  [deit-base_pt-16xb64_in1k_20220216-db63c16c.pth](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth)

**After downloading, rename it to:**

```bash
deit_base_p16.pth
```

#### 2. Place the Weights

Put the weight file under the `pretrain/` directory:

~~~
project_root/
│── pretrain/
│    └── deit_base_p16.pth
│── configs/
│── mmseg/
│── tools/
~~~

#### 3. Convert the Weight Format (if needed)

In some cases, the key names of the pre-trained weights may not fully match the current implementation.
 If you encounter `unexpected key(s) in state_dict` or `missing key(s)` errors,
 please run the following script to convert the weight format:

~~~python
import torch

# 1. Load mmcls DeiT weights
ckpt = torch.load("pretrain/deit_base_p16.pth", map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)

new_state_dict = {}
for k, v in state_dict.items():
    new_k = k
    # remove "backbone." prefix
    if new_k.startswith("backbone."):
        new_k = new_k.replace("backbone.", "")
    # attention qkv mapping
    if "attn.qkv.weight" in new_k:
        new_k = new_k.replace("attn.qkv.weight", "attn.attn.in_proj_weight")
    if "attn.qkv.bias" in new_k:
        new_k = new_k.replace("attn.qkv.bias", "attn.attn.in_proj_bias")
    if "attn.proj.weight" in new_k:
        new_k = new_k.replace("attn.proj.weight", "attn.attn.out_proj.weight")
    if "attn.proj.bias" in new_k:
        new_k = new_k.replace("attn.proj.bias", "attn.attn.out_proj.bias")
    new_state_dict[new_k] = v

# 2. Save the converted weights
torch.save(new_state_dict, "pretrain/deit_base_p16.pth")
print("Converted and saved: pretrain/deit_base_p16.pth")
~~~




## Running Example
We run our S<sup>4</sup>Former based on the segmentation network of SegFormer([paper](https://arxiv.org/abs/2105.15203)|[project](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer)). The batch size is set to 8 for both labeled images and unlabeled images. Here we take 1/8 labeled data protocal on Pascal VOC 2012 as the example.

### Supervised-Only 
```bash
# use torch.distributed.launch
sh ./tools/dist_train.sh \
configs/setr/setr_deit-base_pup_bs_8_512x512_80k_pascal_1over16_split_classic_sup.py 2 \
--seed 1999 
```
### Mean Teacher Baseline
```bash
# use torch.distributed.launch
sh ./tools/dist_train.sh \
configs/setr/setr_deit-base_pup_bs_8_512x512_80k_pascal_1over16_split_classic_semi_beta_1_th_0.95_MT.py 2 \
--seed 1999 
```
### S<sup>4</sup>Former (Ours)
```bash
# use torch.distributed.launch
sh ./tools/dist_train.sh \
configs/setr/setr_deit-base_pup_bs_8_512x512_80k_pascal_1over16_split_classic_semi_beta_1_th_0.95_MT_w_ours.py 2 \
--seed 1999 
```

We thank so much for the feedbacks and updates made by Shun Zuo. 