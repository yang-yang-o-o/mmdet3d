########### default_runtime.py
checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
###########

########### fcos3d
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True

point_cloud_range = [-100, -50, -5.0, 120, 50, 3.0]
post_center_range = [-120, -70, -10.0, 140, 70, 10.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['VEHICLE_CAR','VEHICLE_TRUCK','BIKE_BICYCLE','PEDESTRIAN']
sensors = ['center_camera_fov30','center_camera_fov120', 'left_front_camera', 'left_rear_camera',
           'rear_camera', 'right_rear_camera', "right_front_camera"]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)


model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        ),
    # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
    # stage_with_dcn=(False, False, True, True)
    # img_neck=dict(),
    pts_bbox_head=dict(
        type='PETRv2Head',
        num_classes=4,
        in_channels=2048,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        position_range = point_cloud_range,
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=post_center_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=4), 
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost_custom', weight=2.0),
            reg_cost=dict(type='mmdet.BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost_custom', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))
###########

###########

dataset_type = 'InternalDatasetSweep'
data_root = 'data/internal/'
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         'cla-datasets/': 'sh1984:s3://sh1984_datasets/cla-datasets/',
#         'detr3d/': 'sh1984:s3://sh1984_datasets/detr3d/',
#         'Pilot/': 'sh1424:s3://sh1424_datasets/Pilot/',
#     }))
file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.4, 0.7),
        "final_dim": (540, 960),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 1080,
        "W": 1920,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_petrv2', to_float32=True, file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles_petrv2', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[0,2], sensors=sensors, file_client_args=file_client_args),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='mmdet3d.ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='mmdet3d.ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage_petrv2', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage_petrv2',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage_petrv2', **img_norm_cfg),
    dict(type='PadMultiViewImage_petrv2', size_divisor=32),
    dict(type='mmdet3d.DefaultFormatBundle3D', class_names=class_names),
    dict(type='mmdet3d.Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_petrv2', to_float32=True, file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles_petrv2', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[0,2], sensors=sensors, file_client_args=file_client_args),
    dict(type='ResizeCropFlipImage_petrv2', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage_petrv2', **img_norm_cfg),
    dict(type='PadMultiViewImage_petrv2', size_divisor=32),
    dict(
        type='mmdet3d.MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='mmdet3d.DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='mmdet3d.Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
        ])
]

###########

########### dataset

# ann_file_client_args = dict(backend='petrel')
ann_file_client_args = dict(backend='disk')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        point_cloud_range=point_cloud_range,
        file_client_args=ann_file_client_args,
        # ann_file='s3://sh1424_datasets/datasets_release/v1.0/detr3d_with_sweeps_7cam_train_full.json',
        ann_file='F:\GitHub\mmdet3d\z_custom\Petrv2\data/0.json',
        # ann_file='sh1424:s3://sh1424_datasets/datasets_release/v1.0/detr3d_with_sweeps_7cam_train_full_48.json',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        point_cloud_range=point_cloud_range,
        file_client_args=ann_file_client_args,
        ann_file='s3://sh1424_datasets/datasets_release/v1.0/detr3d_with_sweeps_7cam_val_full.json',
        # ann_file='sh1424:s3://sh1424_datasets/datasets_release/v1.0/detr3d_with_sweeps_7cam_val_full_48.json',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        point_cloud_range=point_cloud_range,
        file_client_args=ann_file_client_args,
        ann_file='s3://sh1424_datasets/datasets_release/v1.0/detr3d_with_sweeps_7cam_val_full.json',
        # ann_file='sh1424:s3://sh1424_datasets/datasets_release/v1.0/detr3d_with_sweeps_7cam_val_full_48.json',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

# test_setting = dict(
#     repo='pillar',
#     single_gpu_test=dict(show=False),
#     multi_gpu_test=dict())

# evaluation = dict(interval=250, pipeline=eval_pipeline)
# evaluation = dict(
#     type='mme.EvalHook',
#     dataset=data['val'],
#     dataloader=dict(samples_per_gpu=4, workers_per_gpu=2),
#     test_setting=test_setting,
#     interval=4,
#     by_epoch=True)
evaluation = dict(interval=2)
###########

# resume_from = 'F:\GitHub\mmdet3d\checkpoints\\v0.3_fov120_r18_no_norm_unmerge_attr_scale_epoch_24.pth'


########### mmdet_schedule_1x.py
# optimizer
optimizer = dict(
    type='mmcv.AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )
momentum_config = None

total_epochs = 1

find_unused_parameters=False # when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
runner = dict(type='mmdet3d.EpochBasedRunner', max_epochs=total_epochs)
load_from='F:\GitHub\mmdet3d\z_custom\Petrv2\data/fcos3d_r50_new.pth'
resume_from=None
###########