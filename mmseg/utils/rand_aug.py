import numpy as np
import torch
from mmseg.ops import resize
import random

def CutOut(teacher_info, student_info, area_ratio):
    data = student_info["img"]
    target = teacher_info["seg_logits"]

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, 
                size=tuple(data.shape[2:]), 
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
    target = F.softmax(target, dim=1)

    import pdb; pdb.set_trace()
    new_data = []
    new_target = []
    for i in range(batch_size):
        mix_mask = generate_cutout_mask([im_h, im_w], area_ratio).to(device)
        target[i][0][(1 - mix_mask).bool()] = 255
        new_data.append((data[i] * mix_mask).unsqueeze(0))
        new_target.append(target[i].unsqueeze(0))
        
    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)
    
    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long()
    return teacher_info, student_info


def CutMix(teacher_info, student_info, area_ratio):
    data = student_info["img"]
    target = teacher_info["seg_logits"]

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, 
                size=tuple(data.shape[2:]), 
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
    target = F.softmax(target, dim=1)

    import pdb; pdb.set_trace()
    new_data = []
    new_target = []
    if "valid_mask" in student_info:
        valid_mask = student_info["valid_mask"]
        new_valid_mask = []
    # new_logits = []
    for i in range(batch_size):
        mix_mask = generate_cutout_mask([im_h, im_w], area_ratio).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        if "valid_mask" in student_info:
            valid_mask = valid_mask.to(device)
            new_valid_mask.append(
                (
                    valid_mask[i] * mix_mask + valid_mask[(i + 1) % batch_size] * (1 - mix_mask)
                ).unsqueeze(0)
            )

    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)
    if "valid_mask" in student_info:
        new_valid_mask = torch.cat(new_valid_mask)
    
    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long()
    if "valid_mask" in student_info:
        student_info["valid_mask"] = new_valid_mask
    return teacher_info, student_info


def ClassMix(teacher_info, student_info, ratio):
    data = student_info["img"]
    target = teacher_info["seg_logits"]

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, 
                size=tuple(data.shape[2:]), 
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
    target = F.softmax(target, dim=1)

    import pdb; pdb.set_trace()

    new_data = []
    new_target = []

    for i in range(batch_size):
        mix_mask = generate_class_mask(target[i], ratio).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        
    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest')
    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long().squeeze(1)
    return teacher_info, student_info


def PatchShuffle(teacher_info, student_info, block_size, block_num):
    data = student_info["img"]
    target = teacher_info["seg_logits"]

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, 
                size=tuple(data.shape[2:]), 
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
    target = F.softmax(target, dim=1)

    import pdb; pdb.set_trace()

    new_data = []
    new_target = []
    
    PatchMix_Size =  block_size * block_num
    h, w, c = np.shape(data[0])
    assert h % PatchMix_Size == 0
    assert w % PatchMix_Size == 0
    PatchMix_Num = (h // PatchMix_Size) * (w//PatchMix_Size)
    index_patches = np.arange(PatchMix_Num)

    for img_i in range(batch_size):
        random.shuffle(index_patches)
        ImgPatches = []
        for i in range(h // PatchMix_Size):
            for j in range(w//PatchMix_Size):
                ImgPatches.append(img[PatchMix_Size*i:PatchMix_Size*(i+1), PatchMix_Size*j:PatchMix_Size*(j+1), :])
        ImgPatches = np.stack(ImgPatches, axis=0)   #[PatchMix_Num, PatchMix_Size, PatchMix_Size, 3]   
        MixedImgPatches = ImgPatches[index_patches, ...]
        # ForkedPdb().set_trace()
        PatchMixedImg = np.zeros_like(img)
        for i in range(h // PatchMix_Size):
            for j in range(w//PatchMix_Size):
                PatchMixedImg[PatchMix_Size*i:PatchMix_Size*(i+1), PatchMix_Size*j:PatchMix_Size*(j+1), :] = MixedImgPatches[h // PatchMix_Size*i+j]

def generate_class_mask(pseudo_labels, ratio):
    rv_ratio = int(1/ratio)
    labels = torch.unique(pseudo_labels)  # all unique labels

    # randomly select several (default: half of) labels
    if 255 in labels and len(labels) > 1:
        labels = labels[labels != 255]
    labels_select = labels[torch.randperm(len(labels))][
        : int(((len(labels) - len(labels) % rv_ratio) / rv_ratio) + 1)
    ]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.long()

def generate_cutout_mask(img_size, ratio=0.5):
    cutout_area = img_size[0] * img_size[1] * ratio

    w = np.random.randint(img_size[1] * ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)
    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()

def format_patch_input(inputs, patchmix_num):
    patchmix_num_hw = int(np.sqrt(patchmix_num))
    b, c, h, w = inputs.shape
    inputs_reshape = inputs.reshape([b, c, 
        patchmix_num_hw, h//patchmix_num_hw, 
        patchmix_num_hw, w//patchmix_num_hw])
    return inputs_reshape


def generate_new_data(data, img_metas, to_cutmix_id):
    batch_size = data.shape[0]
    new_data = []
    for i in range(batch_size):
        data_i = data[i].clone()  # [c, 4, 128, 4, 128]
        PatchMixIndex_i = img_metas[i]['PatchMixIndex'] 
        data_next_i = data[(i+1) % batch_size].clone()
        PatchMixIndex_next_i = img_metas[(i+1) % batch_size]['PatchMixIndex']

        for p in to_cutmix_id:
            index_h_i = PatchMixIndex_i.tolist().index(p)//int(np.sqrt(len(PatchMixIndex_i)))
            index_w_i = PatchMixIndex_i.tolist().index(p)%int(np.sqrt(len(PatchMixIndex_i)))
            index_h_next_i = PatchMixIndex_next_i.tolist().index(p)//int(np.sqrt(len(PatchMixIndex_next_i)))
            index_w_next_i = PatchMixIndex_next_i.tolist().index(p)%int(np.sqrt(len(PatchMixIndex_next_i)))
            
            data_i[:, index_h_i, :, index_w_i, :] = data_next_i[:, index_h_next_i, :, index_w_next_i, :]
        new_data.append(data_i.unsqueeze(0))
    return new_data

def generate_new_target(data, img_metas, to_cutmix_id):
    batch_size = data.shape[0]
    new_data = []
    for i in range(batch_size):
        data_i = data[i].clone()  # [c, 4, 128, 4, 128]
        PatchMixIndex_i = img_metas[i]['PatchMixIndex'] 
        data_next_i = data[(i+1) % batch_size].clone()
        PatchMixIndex_next_i = img_metas[(i+1) % batch_size]['PatchMixIndex']

        for p in to_cutmix_id:
            index_h_i = p // int(np.sqrt(len(PatchMixIndex_i)))
            index_w_i = p % int(np.sqrt(len(PatchMixIndex_i)))
            
            data_i[:, index_h_i, :, index_w_i, :] = data_next_i[:, index_h_i, :, index_w_i, :]
        new_data.append(data_i.unsqueeze(0))
    return new_data

def generate_unsup_patchcutmix_data(teacher_info, student_info):
    patchmix_num = len(student_info['img_metas'][0]['PatchMixIndex']) 
    to_cutmix_id = random.sample(range(patchmix_num), patchmix_num//2)
    # start_idx = random.sample(range(patchmix_num//2), 1)
    # to_cutmix_id =  [start_idx : start_idx + 8]

    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()
    
    batch_size, im_c, im_h, im_w = data.shape
    batch_size, tgt_c, tgt_h, tgt_w = target.shape
    # target = resize(
    #             target, size=tuple(data.shape[2:]), mode='nearest')

    data = format_patch_input(data, patchmix_num)
    target = format_patch_input(target, patchmix_num)

    new_data = generate_new_data(data, student_info['img_metas'], to_cutmix_id)
    new_target = generate_new_target(target, student_info['img_metas'], to_cutmix_id)
    new_data = torch.cat(new_data).view(batch_size, im_c, im_h, im_w)
    new_target = torch.cat(new_target).view(batch_size, tgt_c, tgt_h, tgt_w).squeeze(1)
    # new_target = resize(
    #     new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)

    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long()

    return teacher_info, student_info

def generate_sup_cutmix_data(data, target):

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []

    for i in range(batch_size):
        mix_mask = generate_cutout_mask([im_h, im_w]).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        
    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest')

    return new_data, new_target



def generate_new_data_v2(data, img_metas, to_cutmix_id):
    batch_size = data.shape[0]
    new_data = []
    for i in range(batch_size):
        data_i = data[i].clone()  # [c, 4, 128, 4, 128]
        PatchMixIndex_i = img_metas[i]['PatchMixIndex'] 
        data_next_i = data[(i+1) % batch_size].clone()
        PatchMixIndex_next_i = img_metas[(i+1) % batch_size]['PatchMixIndex']

        for p in to_cutmix_id[i]:
            index_h_i = PatchMixIndex_i.tolist().index(p)//int(np.sqrt(len(PatchMixIndex_i)))
            index_w_i = PatchMixIndex_i.tolist().index(p)%int(np.sqrt(len(PatchMixIndex_i)))
            index_h_next_i = PatchMixIndex_next_i.tolist().index(p)//int(np.sqrt(len(PatchMixIndex_next_i)))
            index_w_next_i = PatchMixIndex_next_i.tolist().index(p)%int(np.sqrt(len(PatchMixIndex_next_i)))
            
            data_i[:, index_h_i, :, index_w_i, :] = data_next_i[:, index_h_next_i, :, index_w_next_i, :]
        new_data.append(data_i.unsqueeze(0))
    return new_data

def generate_new_data_v3(data, img_metas, to_cutout_id):
    batch_size = data.shape[0]
    new_data = []
    for i in range(batch_size):
        data_i = data[i].clone()  # [c, 4, 128, 4, 128]
        PatchMixIndex_i = img_metas[i]['PatchMixIndex'] 

        for p in to_cutout_id[i]:
            index_h_i = PatchMixIndex_i.tolist().index(p)//int(np.sqrt(len(PatchMixIndex_i)))
            index_w_i = PatchMixIndex_i.tolist().index(p)%int(np.sqrt(len(PatchMixIndex_i)))

            data_i[:, index_h_i, :, index_w_i, :] = 0

        new_data.append(data_i.unsqueeze(0))
    return new_data

def generate_new_target_v2(data, img_metas, to_cutmix_id):
    batch_size = data.shape[0]
    new_data = []
    for i in range(batch_size):
        data_i = data[i].clone()  # [c, 4, 128, 4, 128]
        PatchMixIndex_i = img_metas[i]['PatchMixIndex'] 
        data_next_i = data[(i+1) % batch_size].clone()
        PatchMixIndex_next_i = img_metas[(i+1) % batch_size]['PatchMixIndex']

        for p in to_cutmix_id[i]:
            index_h_i = p // int(np.sqrt(len(PatchMixIndex_i)))
            index_w_i = p % int(np.sqrt(len(PatchMixIndex_i)))
            
            data_i[:, index_h_i, :, index_w_i, :] = data_next_i[:, index_h_i, :, index_w_i, :]
        new_data.append(data_i.unsqueeze(0))
    return new_data
    
def generate_new_target_v3(data, img_metas, to_cutout_id):
    batch_size = data.shape[0]
    new_data = []
    for i in range(batch_size):
        data_i = data[i].clone()  # [c, 4, 128, 4, 128]
        PatchMixIndex_i = img_metas[i]['PatchMixIndex'] 

        for p in to_cutout_id[i]:
            index_h_i = p // int(np.sqrt(len(PatchMixIndex_i)))
            index_w_i = p % int(np.sqrt(len(PatchMixIndex_i)))
            
            data_i[:, index_h_i, :, index_w_i, :] = 255
        new_data.append(data_i.unsqueeze(0))
    return new_data

def generate_unsup_patchcutmix_v2_data(teacher_info, student_info):
    patchmix_num = len(student_info['img_metas'][0]['PatchMixIndex']) 
    # to_cutmix_id = random.sample(range(patchmix_num), patchmix_num//2)

    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()
    
    batch_size, im_c, im_h, im_w = data.shape
    batch_size, tgt_c, tgt_h, tgt_w = target.shape
    to_cutmix_id = []
    for i in range(batch_size):
        start_idx = random.sample(range(patchmix_num//2), 1)
        to_cutmix_id.append(student_info['img_metas'][i]['PatchMixIndex'][start_idx[0] : start_idx[0] + patchmix_num//2])
        # to_cutmix_id.append(random.sample(range(patchmix_num), patchmix_num//2))
    data = format_patch_input(data, patchmix_num)
    target = format_patch_input(target, patchmix_num)

    new_data = generate_new_data_v2(data, student_info['img_metas'], to_cutmix_id)
    new_target = generate_new_target_v2(target, student_info['img_metas'], to_cutmix_id)
    new_data = torch.cat(new_data).view(batch_size, im_c, im_h, im_w)
    new_target = torch.cat(new_target).view(batch_size, tgt_c, tgt_h, tgt_w).squeeze(1)
    # new_target = resize(
    #     new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)

    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long()

    return teacher_info, student_info

def generate_unsup_patchcutout_data(teacher_info, student_info):
    patchmix_num = len(student_info['img_metas'][0]['PatchMixIndex']) 
    # to_cutmix_id = random.sample(range(patchmix_num), patchmix_num//2)

    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()
    
    batch_size, im_c, im_h, im_w = data.shape
    batch_size, tgt_c, tgt_h, tgt_w = target.shape
    to_cutout_id = []
    for i in range(batch_size):
        cutout_idx = random.sample(range(patchmix_num//2), 1)
        to_cutout_id.append(random.sample(range(patchmix_num), cutout_idx[0]))
        # to_cutout_id.append(random.sample(range(patchmix_num), patchmix_num//2))
    data = format_patch_input(data, patchmix_num)
    target = format_patch_input(target, patchmix_num)

    new_data = generate_new_data_v3(data, student_info['img_metas'], to_cutout_id)
    new_target = generate_new_target_v3(target, student_info['img_metas'], to_cutout_id)
    new_data = torch.cat(new_data).view(batch_size, im_c, im_h, im_w)
    new_target = torch.cat(new_target).view(batch_size, tgt_c, tgt_h, tgt_w).squeeze(1)
    # new_target = resize(
    #     new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)

    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long()

    return teacher_info, student_info






def generate_sup_classmix_data(data, target):
    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []

    for i in range(batch_size):
        mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        
    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest')
    return new_data, new_target
