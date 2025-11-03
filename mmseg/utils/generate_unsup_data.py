import numpy as np
import torch
from mmseg.ops import resize
import random


def generate_cutout_mask(img_size, ratio=2):
    if isinstance(ratio, int):
        cutout_area = img_size[0] * img_size[1] / ratio
    elif isinstance(ratio, tuple):
        assert ratio[0] < ratio[1]
        ratio = random.uniform(ratio[0], ratio[1])
        cutout_area = int(img_size[0] * img_size[1] / ratio)

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()


def format_patch_input(inputs, patchnum):
    b, c, h, w = inputs.shape

    patchsize = int(h / np.sqrt(patchnum))

    inputs_reshape = inputs.reshape([b, c, 
        h // patchsize, patchsize, 
        w // patchsize, patchsize])
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


def generate_unsup_data(teacher_info, student_info, ratio=2, patchwise=False, patchsize=16 * 4):
    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []
    if "valid_mask" in student_info:
        valid_mask = student_info["valid_mask"]
        new_valid_mask = []
    # new_logits = []
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
        if "valid_mask" in student_info:
            valid_mask = valid_mask.to(device)
            new_valid_mask.append(
                (
                    valid_mask[i] * mix_mask + valid_mask[(i + 1) % batch_size] * (1 - mix_mask)
                ).unsqueeze(0)
            )
        # new_logits.append(
        #     (
        #         logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
        #     ).unsqueeze(0)
        # )
    # new_data, new_target, new_logits = (
    #     torch.cat(new_data),
    #     torch.cat(new_target),
    #     torch.cat(new_logits),
    # )
    # return new_data, new_target.long(), new_logits

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

'''
def generate_cutout_mask_patchwise(img_size, ratio=2):
    if isinstance(ratio, int):
        cutout_area = img_size[0] * img_size[1] / ratio
    elif isinstance(ratio, tuple):
        assert ratio[0] < ratio[1]
        ratio = random.uniform(ratio[0], ratio[1])
        cutout_area = int(img_size[0] * img_size[1] / ratio)

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()
'''


def generate_patchwise_cutout_mask(img_size, patchsize, ratio):
    im_h, im_w = img_size[0], img_size[1]
    patchmix_num = (im_h // patchsize) * (im_w // patchsize)
    mask = torch.ones(patchmix_num, patchsize, patchsize)
    if isinstance(ratio, int):
        cut_patch_num = patchmix_num // ratio
    elif isinstance(ratio, tuple):
        assert ratio[0] < ratio[1]
        ratio = random.uniform(ratio[0], ratio[1])
        cut_patch_num = patchmix_num // ratio
    to_cutout_id = random.sample(range(patchmix_num), cut_patch_num)
    mask[to_cutout_id,:,:] = 0
    mask = mask.reshape(im_h // patchsize, im_w // patchsize, patchsize, patchsize).permute(
        0,2,1,3).reshape(img_size)
    return mask.long()


def generate_unsup_cutout_data(teacher_info, student_info, ratio=2, patchwise=False, patchsize=16 * 4):
    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []

    for i in range(batch_size):
        if patchwise:
            mix_mask = generate_patchwise_cutout_mask([im_h, im_w], patchsize=patchsize, ratio=ratio).to(device)

        else:
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=ratio).to(device)
        target[i][0][(1 - mix_mask).bool()] = 255
        new_data.append((data[i] * mix_mask).unsqueeze(0))
        new_target.append(target[i].unsqueeze(0))
        
    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)

    return teacher_info, student_info


def generate_unsup_cutmix_data(teacher_info, student_info, ratio=2, patchwise=False, patchsize=16 * 8):
    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []

    # new_logits = []
    for i in range(batch_size):
        if patchwise:
            mix_mask = generate_patchwise_cutout_mask([im_h, im_w], patchsize=patchsize, ratio=ratio).to(device)
       
        else:
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=ratio).to(device)
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

        # if 'conf_mask' in teacher_info:
        #     teacher_info['conf_mask'][i] = torch.ones_like(
        #         teacher_info['conf_mask'][i])
        # new_logits.append(
        #     (
        #         logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
        #     ).unsqueeze(0)
        # )
    # new_data, new_target, new_logits = (
    #     torch.cat(new_data),
    #     torch.cat(new_target),
    #     torch.cat(new_logits),
    # )
    # return new_data, new_target.long(), new_logits

    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest').squeeze(1)
    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long()
    return teacher_info, student_info


def generate_unsup_cutmix_data_unimatch(teacher_info, teacher_mix_info, student_info, student_mix_info, ratio=2, patchwise=False, patchsize=16 * 8):
    teacher_info_mixed = teacher_info.copy()
    batch_size, _, im_h, im_w = student_info["img"].shape
    cutmix_box = torch.zeros_like(student_info["img"])[:, 0, :, :]

    for i in range(batch_size):
        cutmix_box[i] = generate_cutout_mask([im_h, im_w], ratio=ratio)

    target_cutmixed = teacher_info["hard_seg_label"].clone()
    target_cutmixed_mix = teacher_mix_info["hard_seg_label"].clone()
    _, tgt_h, tgt_w = target_cutmixed.shape

    if tgt_w != im_w:
        target_cutmixed = resize(
            target_cutmixed.unsqueeze(1).float(), size=(im_h, im_w), mode='nearest').squeeze(1)
        target_cutmixed_mix = resize(
            target_cutmixed_mix.unsqueeze(1).float(), size=(im_h, im_w), mode='nearest').squeeze(1)

    img_cutmixed = student_info["img"].clone()

    img_cutmixed[cutmix_box.unsqueeze(1).expand(img_cutmixed.shape) == 1] = \
        student_mix_info["img"][cutmix_box.unsqueeze(1).expand(img_cutmixed.shape) == 1]

    target_cutmixed[cutmix_box == 1] = target_cutmixed_mix[cutmix_box == 1]

    if tgt_w != im_w:
        target_cutmixed = resize(
            target_cutmixed.unsqueeze(1), size=(tgt_h, tgt_w), mode='nearest').squeeze(1).long()

    teacher_info_mixed["hard_seg_label"] = target_cutmixed
    student_info["img"] = img_cutmixed

    return teacher_info_mixed, student_info


def generate_patchwise_class_mask(pseudo_labels, img_size, patchsize):
    im_h, im_w = img_size[0], img_size[1]
    patchmix_num = (im_h // patchsize) * (im_w // patchsize)
    mask = torch.zeros(patchmix_num, patchsize, patchsize).to(pseudo_labels.device)
    patch_pseudo_labels = pseudo_labels.reshape(1, (im_h // patchsize), patchsize, (im_w // patchsize), patchsize).permute(0,1,3,2,4).reshape(1, patchmix_num, patchsize, patchsize)
    for i in range(patchmix_num):
        patch_labels = torch.unique(patch_pseudo_labels[0,i])
        if 255 in patch_labels:
            mask[i] = (patch_pseudo_labels[0,i].unsqueeze(-1) == 255).any(-1)
            # import pdb; pdb.set_trace()
            patch_labels = patch_labels[patch_labels != 255]

        if len(patch_labels) > 1:
            # if len(patch_labels) > 0:
            #     patch_labels_select = patch_labels[torch.randperm(len(patch_labels))][
            #         : int(((len(labels) - len(labels) % 2) / 2) + 1)]
            #     mask[i] = (patch_pseudo_labels[0,i].unsqueeze(-1) == patch_labels_select).any(-1)
            labels_select = patch_labels[torch.randperm(len(patch_labels))][
                    : int(((len(patch_labels) - len(patch_labels) % 2) / 2) + 1)]  # randomly select half of labels
            mask[i] = (mask[i] + (patch_pseudo_labels[0,i].unsqueeze(-1) == labels_select).any(-1))

    mask = mask.reshape(im_h // patchsize, im_w // patchsize, patchsize, patchsize).permute(
        0,2,1,3).reshape(img_size).unsqueeze(0)
    # import pdb; pdb.set_trace()
    return mask.long()


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    # labels_select = labels[torch.randperm(len(labels))][
    #     : len(labels) // 2
    # ]  # randomly select half of labels
    # '''
    if 255 in labels and len(labels) > 1:
        labels = labels[labels != 255]
    labels_select = labels[torch.randperm(len(labels))][
        : int(((len(labels) - len(labels) % 2) / 2) + 1)
    ]  # randomly select half of labels
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    
    # '''
    '''
    mask = torch.zeros_like(pseudo_labels).squeeze(0)
    if 255 in labels:
        mask = (pseudo_labels.unsqueeze(-1) == 255).any(-1)
        labels = labels[labels != 255]
    if len(labels) > 1:
        labels_select = labels[torch.randperm(len(labels))][
                    : int(((len(labels) - len(labels) % 2) / 2) + 1)]
        mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    '''
    return mask.long()


def generate_mix_with_labeled_data(teacher_info, student_info, sup_imgs, sup_gts, labeled_mix_mask, patchsize=16):
    
    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()

    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []

    for i in range(batch_size):
        new_data.append(
            (
                sup_imgs[i] * labeled_mix_mask[i] + data[i] * (1 - labeled_mix_mask[i])
            ).unsqueeze(0)
        )
        new_target.append(
            (
                sup_gts[i] * labeled_mix_mask[i] + target[i] * (1 - labeled_mix_mask[i])
            ).unsqueeze(0)
        )
    new_data = torch.cat(new_data)
    new_target = torch.cat(new_target)
    new_target = resize(
        new_target, size=(tgt_h, tgt_w), mode='nearest')
    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long().squeeze(1)
    return teacher_info, student_info


def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix_label_adaptive(unlabeled_image, unlabeled_mask, unlabeled_logits, 
        labeled_image, labeled_mask, lst_confidences):
    reshape = False
    
    if unlabeled_mask.size(-1) != unlabeled_image.size(-1):
        reshape = True
        batch_size, tgt_h, tgt_w = unlabeled_mask.shape

        unlabeled_mask = resize(unlabeled_mask.unsqueeze(1).float(), size=tuple(unlabeled_image.shape[2:]), mode='nearest').squeeze(1)
        unlabeled_logits = resize(unlabeled_logits.unsqueeze(1).float(), size=tuple(unlabeled_image.shape[2:]), mode='bilinear').squeeze(1)

    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 2) get box
    l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 3) labeled adaptive
    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
        
            mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            
            mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
    # 4) copy and paste
    for i in range(0, unlabeled_image.shape[0]):
        unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            
        unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits

    if reshape:  
        unlabeled_mask = resize(unlabeled_mask.unsqueeze(1), size=(tgt_h, tgt_w), mode='nearest').squeeze(1).long()
        unlabeled_logits = resize(unlabeled_logits.unsqueeze(1), size=(tgt_h, tgt_w), mode='bilinear').squeeze(1).float()

    return unlabeled_image, unlabeled_mask, unlabeled_logits 
 

def generate_unsup_classmix_data(teacher_info, student_info, patchwise=False, patchsize=16 * 8):
    data = student_info["img"]
    target = teacher_info["hard_seg_label"].unsqueeze(1).float()
    
    batch_size, _, im_h, im_w = data.shape
    batch_size, _, tgt_h, tgt_w = target.shape
    device = data.device

    target = resize(
                target, size=tuple(data.shape[2:]), mode='nearest')

    new_data = []
    new_target = []

    for i in range(batch_size):
        if patchwise:
            mix_mask = generate_patchwise_class_mask(target[i], [im_h,im_w], patchsize=patchsize).to(device)
        else:
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
    student_info["img"] = new_data
    teacher_info["hard_seg_label"] = new_target.long().squeeze(1)
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



def generate_unsup_patchmix_data(results, teacher_info=None, patchmix_ratio=0.5, patch_size=16, 
        PatchMix_N=1, use_mask=False, ratio=2):
    data = results["img"]
    new_data = data.clone()
    # batch_size, c, h, w = imgs.shape
    batch_size, c, h, w = results['img'].shape

    for batch_index in range(batch_size):
        img = new_data[batch_index]
        # if teacher_info is not None:
        #     conf_mask = teacher_info['conf_mask'][batch_index]
        # img = imgs[batch_index]
        img = img.permute(1,2,0)
        h, w, c = img.shape
        random_patch = np.random.rand()
        patchmix = True if random_patch < patchmix_ratio else False
        PatchMix_Size = patch_size * PatchMix_N
        assert h % PatchMix_Size == 0
        assert w % PatchMix_Size == 0
        PatchMix_Num = (h // PatchMix_Size) * (w // PatchMix_Size)
        index_patches = torch.arange(PatchMix_Num)
        if not patchmix:  
            # import pdb; pdb.set_trace()
            ## torch.tensor(index_patches)
            results['img_metas'][batch_index]['PatchMixIndex'] = index_patches.clone().detach()
            results['img_metas'][batch_index]['PatchMix_N'] = PatchMix_N
            # # random.shuffle(index_patches)
            # perm_indices = torch.randperm(index_patches.size(0))
            # index_patches = index_patches[perm_indices]
        else:
            # start re-arrange the images
            # random.shuffle(index_patches)
            if use_mask:
                if isinstance(ratio, int):
                    to_shuffle_num = PatchMix_Num // ratio
                elif isinstance(ratio, tuple):
                    assert ratio[0] < ratio[1]
                    ratio = random.uniform(ratio[0], ratio[1])
                    to_shuffle_num = PatchMix_Num // ratio

                to_shuffle_id = random.sample(range(PatchMix_Num), int(to_shuffle_num))
                index_patches[to_shuffle_id] = index_patches[to_shuffle_id][torch.randperm(
                    index_patches[to_shuffle_id].size(0))]
            else:    
                perm_indices = torch.randperm(index_patches.size(0))
                index_patches = index_patches[perm_indices]
            # if teacher_info is not None:
            #     ConfPatches =[]
            ImgPatches = []
            for i in range(h // PatchMix_Size):
                for j in range(w//PatchMix_Size):
                    ImgPatches.append(img[PatchMix_Size*i:PatchMix_Size*(i+1), PatchMix_Size*j:PatchMix_Size*(j+1), :])
                    # if teacher_info is not None:
                    #     ConfPatches.append(conf_mask[PatchMix_Size*i:PatchMix_Size*(i+1), PatchMix_Size*j:PatchMix_Size*(j+1)])
            # import pdb; pdb.set_trace()
            ImgPatches = torch.stack(ImgPatches, axis=0)   #[PatchMix_Num, PatchMix_Size, PatchMix_Size, 3]   
            MixedImgPatches = ImgPatches[index_patches, ...]
            # ForkedPdb().set_trace()
            PatchMixedImg = torch.zeros_like(img)
            # if teacher_info is not None:
            #     ConfPatches = torch.stack(ConfPatches, axis=0)   #[PatchMix_Num, PatchMix_Size, PatchMix_Size, 3]   
            #     MixedConfPatches = ConfPatches[index_patches, ...]
            #     PatchMixedConf = torch.zeros_like(conf_mask)
            for i in range(h // PatchMix_Size):
                for j in range(w // PatchMix_Size):
                    PatchMixedImg[PatchMix_Size*i:PatchMix_Size*(i+1), PatchMix_Size*j:PatchMix_Size*(j+1), :] = MixedImgPatches[h // PatchMix_Size*i+j]
                    # if teacher_info is not None:
                    #     PatchMixedConf[PatchMix_Size*i:PatchMix_Size*(i+1), PatchMix_Size*j:PatchMix_Size*(j+1)] = MixedConfPatches[h // PatchMix_Size*i+j]
            # ForkedPdb().set_trace()
            # import pdb; pdb.set_trace()
            imgf = PatchMixedImg.permute(2,0,1)
            new_data[batch_index] = imgf
            results['img_metas'][batch_index]['PatchMixIndex'] = index_patches
            results['img_metas'][batch_index]['PatchMix_N'] = PatchMix_N
            # if (teacher_info is not None) and ('conf_mask' in teacher_info):
            #     teacher_info['conf_mask'][batch_index] = torch.ones_like(
            #         teacher_info['conf_mask'][batch_index])
    results["img"] = new_data

    if teacher_info is not None:
        return results, teacher_info
    
    return results
