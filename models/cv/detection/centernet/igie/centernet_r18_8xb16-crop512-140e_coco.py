# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

auto_scale_lr = dict(base_batch_size=128, enable=False)
backend_args = None
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 28
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(type='BN'),
        norm_eval=False,
        type='ResNet'),
    bbox_head=dict(
        feat_channels=64,
        in_channels=64,
        loss_center_heatmap=dict(loss_weight=1.0, type='GaussianFocalLoss'),
        loss_offset=dict(loss_weight=1.0, type='L1Loss'),
        loss_wh=dict(loss_weight=0.1, type='L1Loss'),
        num_classes=80,
        type='CenterNetHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=512,
        num_deconv_filters=(
            256,
            128,
            64,
        ),
        num_deconv_kernels=(
            4,
            4,
            4,
        ),
        type='CTResNetNeck',
        use_dcn=False),
    test_cfg=dict(local_maximum_kernel=3, max_per_img=100, topk=100),
    train_cfg=None,
    type='CenterNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=28,
        gamma=0.1,
        milestones=[
            18,
            24,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='images/val2017/'),
        data_root='/home/peng.yang/Datasets/coco',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(
                border=None,
                mean=[
                    0,
                    0,
                    0,
                ],
                ratios=None,
                std=[
                    1,
                    1,
                    1,
                ],
                test_mode=True,
                test_pad_add_pix=1,
                test_pad_mode=[
                    'logical_or',
                    31,
                ],
                to_rgb=True,
                type='RandomCenterCropPad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'border',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/peng.yang/Datasets/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(
        border=None,
        mean=[
            0,
            0,
            0,
        ],
        ratios=None,
        std=[
            1,
            1,
            1,
        ],
        test_mode=True,
        test_pad_add_pix=1,
        test_pad_mode=[
            'logical_or',
            31,
        ],
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'border',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=28, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train2017.json',
            backend_args=None,
            data_prefix=dict(img='train2017/'),
            data_root='data/coco/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    brightness_delta=32,
                    contrast_range=(
                        0.5,
                        1.5,
                    ),
                    hue_delta=18,
                    saturation_range=(
                        0.5,
                        1.5,
                    ),
                    type='PhotoMetricDistortion'),
                dict(
                    crop_size=(
                        512,
                        512,
                    ),
                    mean=[
                        0,
                        0,
                        0,
                    ],
                    ratios=(
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                        1.1,
                        1.2,
                        1.3,
                    ),
                    std=[
                        1,
                        1,
                        1,
                    ],
                    test_pad_mode=None,
                    to_rgb=True,
                    type='RandomCenterCropPad'),
                dict(keep_ratio=True, scale=(
                    512,
                    512,
                ), type='Resize'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='PhotoMetricDistortion'),
    dict(
        crop_size=(
            512,
            512,
        ),
        mean=[
            0,
            0,
            0,
        ],
        ratios=(
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
        ),
        std=[
            1,
            1,
            1,
        ],
        test_pad_mode=None,
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(
                border=None,
                mean=[
                    0,
                    0,
                    0,
                ],
                ratios=None,
                std=[
                    1,
                    1,
                    1,
                ],
                test_mode=True,
                test_pad_add_pix=1,
                test_pad_mode=[
                    'logical_or',
                    31,
                ],
                to_rgb=True,
                type='RandomCenterCropPad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'border',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './'
