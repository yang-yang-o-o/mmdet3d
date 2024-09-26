

point_cloud_range = [-100, -22.4, -5.0, 72.8, 22.4, 3.0]
# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
class_names = [
    'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
]

data_config={
    # 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    #          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    # 'Ncams': 6,
    # 'input_size': (256, 704),
    # 'src_size': (900, 1600),
    'cams': ['center_camera_fov120', \
            'left_front_camera', 'left_rear_camera',\
            'rear_camera', 'right_rear_camera', \
            'right_front_camera'],
    'Ncams': 6,
    'input_size': (540, 960),
    'src_size': (1080, 1920),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

# Model
grid_config={
        'xbound': [-100, 72.8, 0.8],
        'ybound': [-22.4, 22.4, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 100.0, 1.0],}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans=64

log_config=dict(
    interval=50,
    hooks=[
        dict(type='mmcv.TextLoggerHook'),
        dict(type='mmcv.TensorboardLoggerHook')
    ])

model = dict(
    type='BEVDet',
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='/mnt/lustre/sunyecheng/zhanghongyu/fcos3d_r50_new.pth', prefix='img_backbone')),
        # init_cfg=dict(type='Pretrained', checkpoint='data/pretrain_models/fcos3d_r50_new.pth', prefix='img_backbone')),
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrain_models/resnet50-0676ba61.pth')),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(type='ViewTransformerLiftSplatShoot',
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans),
    img_bev_encoder_backbone = dict(type='ResNetForBEVDet', numC_input=numC_Trans),
    img_bev_encoder_neck = dict(type='FPN_LSS',
                                in_channels=numC_Trans*8+numC_Trans*2,
                                out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['VEHICLE_CAR']),
            dict(num_class=1, class_names=['VEHICLE_TRUCK']),
            dict(num_class=1, class_names=['BIKE_BICYCLE']),
            dict(num_class=1, class_names=['PEDESTRIAN']),
            # dict(num_class=1, class_names=['car']),
            # dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            # dict(num_class=2, class_names=['bus', 'trailer']),
            # dict(num_class=1, class_names=['barrier']),
            # dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            # dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='mmdet3d.SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1728, 448, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2,

            # Scale-NMS
            # nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            # nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            # nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]]
            nms_type=['rotate',  'rotate', 'rotate', 'rotate'],
            nms_thr=[0.2,  0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, 0.7, 1.0, 4.5]
        )))


dataset_type = 'InternalBEVDetDataset'
data_root = 'data/internal/'
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         'detr3d/': 's3://sh1984_datasets/detr3d/',
#         'cla-datasets/': 's3://sh1984_datasets/cla-datasets/',
#         'Pilot/': 's3://sh1424_datasets/Pilot/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_Internal', is_train=True, data_config=data_config,
         file_client_args=file_client_args,data_name='pilot'),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True
        ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True
        ),
    dict(type='mmdet3d.ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='mmdet3d.ObjectNameFilter', classes=class_names),
    dict(type='mmdet3d.DefaultFormatBundle3D', class_names=class_names),
    dict(type='mmdet3d.Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_Internal', data_config=data_config, 
        file_client_args=file_client_args,data_name='pilot'),
    # load lidar points for --show in test.py only
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
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
            dict(type='mmdet3d.Collect3D', keys=['points','img_inputs'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_Internal', data_config=data_config, 
        file_client_args=file_client_args,data_name='pilot'),
    dict(
        type='mmdet3d.DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='mmdet3d.Collect3D', keys=['img_inputs'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# ann_file_client_args = dict(backend='petrel')
ann_file_client_args = dict(backend='disk')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type, # 'InternalBEVDetDataset'
        data_root=data_root,
        point_cloud_range =point_cloud_range,
        file_client_args=ann_file_client_args,
        ann_file='F:\GitHub\mmdet3d\z_custom\Petrv2\data/0.json',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        modality=input_modality,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        img_info_prototype='bevdet',
        dir_type='local'),
    val=dict(
        type=dataset_type, # 'InternalBEVDetDataset'
        data_root=data_root,
        point_cloud_range =point_cloud_range,
        file_client_args=ann_file_client_args,
        ann_file='/mnt/lustre/yangyang14/data_json/pilot_intrinsic_extrinsic/CN_007_intrinsic_1_extrinsic_2_val.json',
        # ann_file='/mnt/lustre/yangyang14/data_json/pilot_intrinsic_extrinsic/CN_007_intrinsic_1_extrinsic_1.json',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet',
        dir_type='local'),
    test=dict(
        type=dataset_type, # 'InternalBEVDetDataset'
        data_root=data_root,
        point_cloud_range =point_cloud_range,
        file_client_args=ann_file_client_args,
        ann_file='/mnt/lustre/yangyang14/data_json/pilot_intrinsic_extrinsic/CN_007_intrinsic_1_extrinsic_2_val.json',
        # ann_file='/mnt/lustre/yangyang14/data_json/pilot_intrinsic_extrinsic/CN_007_intrinsic_1_extrinsic_1.json',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet',
        dir_type='local'))

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

# Optimizer
optimizer = dict(type='mmcv.AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(5, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
runner = dict(type='mmcv.EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
# load_from = None
load_from='F:\GitHub\mmdet3d\z_custom\Petrv2\data/fcos3d_r50_new.pth'
resume_from = None
workflow = [('train', 1)]
