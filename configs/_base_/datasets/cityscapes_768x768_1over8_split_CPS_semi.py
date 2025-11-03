# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=train_pipeline,
            split='datasplits/city_splits_CPS/372_train_supervised.txt',
        ),
        unsup=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',  #unsupervised data have no lables 
            pipeline=train_pipeline,
            split='datasplits/city_splits_CPS/372_train_unsupervised.txt',
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 1],   # the frequency to sample each kind of datasets
            # by_prob=True,
            by_prob=False,
            # at_least_one=True,
            # epoch_length=7330,
            max_iter_size=40000
        )
    )
)
