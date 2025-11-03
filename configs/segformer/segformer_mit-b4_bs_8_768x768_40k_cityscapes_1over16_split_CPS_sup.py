_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_768x768_1over16_split_CPS_sup.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]
# _base_ = ['./segformer_mit-b0_bs_8_512x512_80k_pascal_1over8_split_CPS_sup.py']

checkpoint = './pretrain/segformer_mit_b4.pth'  # noqa

crop_size = (768, 768)

labeled_ratio = 1/16
split = 'CPS'

samples_per_gpu_sup = 4
workers_per_gpu = 4

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=19),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512))
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=samples_per_gpu_sup, workers_per_gpu=workers_per_gpu)
