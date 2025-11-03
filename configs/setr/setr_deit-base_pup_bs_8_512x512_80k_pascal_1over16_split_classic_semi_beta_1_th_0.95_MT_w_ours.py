_base_ = [
    '../_base_/models/setr_pup.py',
    '../_base_/datasets/pascal_voc12_aug_1over16_split_classic_semi.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_pascal_1over8.py'
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)

img_scale = (2048, 512)
crop_size = (512, 512)

# semi setting
labeled_ratio = 1 / 16
split = 'classic'

use_EMA = True
ema_momentum = 0.999

# semi-branch parameters
beta = 1.0  # semi-branch weight
thres = 0.95

# PatchMix Parameters
PatchMix_N = 8

samples_per_gpu_sup = 4
samples_per_gpu_unsup = 4
samples_per_gpu = samples_per_gpu_sup + samples_per_gpu_unsup

workers_per_gpu = 4
unconfident_attn_weight = True


negative_class_ranking = True
negative_class_ranking_mode = 'unsup_only'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect', 
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 
            'ori_filename', 
            'ori_shape', 
            'img_shape', 
            'pad_shape', 
            'scale_factor', 
            'flip', 
            'flip_direction', 
            'img_norm_cfg',
            'tag',)
    ),
]


strong_pipeline = [
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect', 
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 
        'ori_filename', 
        'ori_shape', 
        'img_shape', 
        'pad_shape', 
        'scale_factor', 
        'flip', 
        'flip_direction', 
        'img_norm_cfg',
        'tag',)
    ),
]

weak_pipeline = [
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect', 
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 
        'ori_filename', 
        'ori_shape', 
        'img_shape', 
        'pad_shape', 
        'scale_factor', 
        'flip', 
        'flip_direction', 
        'img_norm_cfg',
        'tag',)
    ),
]


unsup_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_seg=False), # unlabled data has no anno 
    dict(type='LoadAnnotations'), # for debugging
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=False),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

backbone = dict(
        type='VisionTransformer',
        img_size=crop_size, 
        patch_size=16,
        in_channels=3,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bilinear',
        drop_rate=0.,
        embed_dims=768, 
        num_heads=12, 
        num_layers=12,
        out_indices=(4, 7, 9, 11),
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/deit_base_p16.pth')
)

decode_head = dict(
        type='SETRUPHead',
        align_corners=False, 
        num_convs=4, 
        in_channels=768,
        num_classes=21,
        channels=256,
        in_index=3,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        up_scale=2,
        kernel_size=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

auxiliary_head = [
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=0,
            num_classes=21,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=1,
            num_classes=21,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=2,
            num_classes=21,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=768,
            channels=256,
            in_index=3,
            num_classes=21,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
]

model = dict(
    pretrained=None,
    backbone=backbone,
    backbone_ema=backbone,
    auxiliary_head=auxiliary_head,
    # auxiliary_head_ema=auxiliary_head, 
    decode_head=decode_head,
    decode_head_ema=decode_head,
    ema=use_EMA,
    ema_momentum=ema_momentum,
    unsup_weight=beta,  
    unsup_confidence=thres,
    test_cfg=dict(mode='whole'),
    attn_mask_seperate_head=True,
    attn_mask_weight=5,
    adaptive_attn_mask=True,
    use_PatchShuffle_w_Cutmix=True,
    PatchMix_N=PatchMix_N,
    negative_class_ranking=negative_class_ranking,
    negative_class_ranking_mode=negative_class_ranking_mode,
)


optimizer = dict(
    lr=0.001, 
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

data = dict(
    train=dict(sup=dict(pipeline=train_pipeline),
        unsup=dict(pipeline=unsup_train_pipeline)),
    # unsup_train=dict(pipeline=unsup_train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    samples_per_gpu=samples_per_gpu, 
    sampler=dict(train=dict(sample_ratio=[
        samples_per_gpu_sup,samples_per_gpu_unsup]))
)