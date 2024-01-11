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
model = dict(
    type='FCOSMono3D_faw',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./checkpoints/resnet18-f37072fd.pth')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSMono3DHead_faw',
        num_classes=3,
        num_attrs=12,
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=False,
        pred_bbox2d=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        strides=[8, 16, 32, 64, 128],
        bbox_code_size=7,
        group_reg_dims=(2, 1, 3, 1),
        cls_branch=(64, ),
        reg_branch=((64, ), (64, ), (64, ), (64, )),
        dir_branch=(64, ),
        attr_branch=(64, ),
        bbox2d_branch=(64, ),
        loss_cls=dict(
            # type='mmdet.FocalLoss',
            type='FocalLoss_faw',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            # type='mmdet.SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0),
            type='SmoothL1Loss_faw', beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(
            # type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            type='CrossEntropyLoss_faw', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            # type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            type='CrossEntropyLoss_faw', use_sigmoid=False, loss_weight=1.0),
        # loss_bbox2d=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_bbox2d=dict(type='IoULoss_faw', loss_weight=1.0),
        loss_centerness=dict(
            # type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            type='CrossEntropyLoss_faw', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(
            type='FCOS3DBBoxCoderDxScale_faw',
            code_size=7,
            rescale_depth=True,
            scale_depth=True,
            scale_deltax=True,
            scale_factor=10,
            global_yaw=False),
        global_yaw=False,
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=False,
        norm_cfg=None,
        rescale_depth=True,
        is_deploy=False),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.05,
        score_thr=0.2,
        min_bbox_size=0,
        max_per_img=200))
###########


########### dataset
INF = 100000000.0
class_names = ['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN']
attr_names = [
    'VEHICLE_CAR', 'VEHICLE_SUV', 'VEHICLE_MPV', 'SMALL_TRUCK', 'SMALL_VAN',
    'HUGE_TRUCK', 'VEHICLE_BUS', 'SHEDLESS_TRIKE', 'COVERED_TRIKE',
    'SPECIAL_VEHICLE', 'BIKE_BICYCLE', 'PEDESTRIAN'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=2,
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
            type='SenseautoMonoDataset_faw',
            data_root=None, # 不为None的话，会加到其他非绝对路径前面
            ann_file= # 通过self.load_annotations函数加载，生成self.data_infos
            'F:\GitHub\mmdet3d\z_custom\data\\test_data.json',
            # img_prefix 是图片路径前缀，在LoadImageFromFileMono3D_faw类的__call__函数中使用
            img_prefix='F:\GitHub\mmdet3d\z_custom\data',
            # classes 作为 self.CLASSES
            classes=['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN'],
            # pipeline 作为 self.pipeline
            pipeline=[
                dict(type='LoadImageFromFileMono3D_faw'),
                dict(
                    type='mmdet3d.LoadAnnotations3D',
                    with_bbox=True,
                    with_label=True,
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox_depth=True,
                    with_attr_label=True),
                dict(type='mmdet3d.RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='Mono3DResize_faw',
                    multiscale_mode='value',
                    # img_scale=[(960, 540), (1024, 576), (1088, 612),
                    #            (1152, 648), (1216, 684), (1280, 720),
                    #            (1344, 756), (1408, 792)],
                    img_scale=[(1024, 576)],
                    keep_ratio=False),
                dict(
                    type='mmdet.Normalize',
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='mmdet.Pad', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D_faw',
                    class_names=['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN']),
                dict(
                    type='Collect3D_faw',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                        'gt_labels_3d', 'centers2d', 'depths', 'attr_labels'
                    ])
            ],
            # modality 应该是没有用到，暂时没有发现哪里使用了。?
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            test_mode=False, # 在__gititem__函数中控制选择训练和测试
            # depth_filter 用来过滤car的gt-box，如果cliped-2dbox的高和图像高的比值小于阈值（16 / 720），就丢弃这个gt-box
            depth_filter=False,
            # center2d_filter 用来根据2d-gt-box的中心图像坐标来过滤2d-gt-box
            center2d_filter=False,
            # 用于获取 box_type_3d 和 box_mode_3d
            box_type_3d='Camera',
            # 使用细分类作为标签
            use_sub_type_as_label=False),
    val=dict(
        type='SenseautoMonoDataset_faw',
        data_root=None,
        ann_file=
        '/mnt/lustre/yangyang14/pillar-fcos3d/functions/mono/VDC_Release_doc/data/test_data.json',
        img_prefix='/mnt/lustre/yangyang14/pillar-fcos3d/functions/mono/VDC_Release_doc/data',
        classes=['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D_faw'),
            dict(
                type='mmdet3d.MultiScaleFlipAug',
                img_scale=(1024, 576),
                flip=False,
                transforms=[
                    dict(
                        type='Mono3DResize_faw',
                        multiscale_mode='value',
                        keep_ratio=False),
                    dict(
                        type='mmdet.Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='mmdet.Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D_faw',
                        class_names=[
                            'VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN'
                        ],
                        with_label=False),
                    dict(type='Collect3D_faw', keys=['img'])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True, # 在__gititem__函数中控制选择训练和测试
        box_type_3d='Camera',
        use_sub_type_as_label=False    
    ),
    test=dict(
        type='SenseautoMonoDataset_faw',
        data_root=None,
        ann_file=
        'F:\GitHub\mmdet3d\z_custom\data\\test_data.json',
        img_prefix='F:\GitHub\mmdet3d\z_custom\data',
        classes=['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN'],
        pipeline=[
            dict(type='LoadImageFromFileMono3D_faw'),
            dict(
                type='mmdet3d.MultiScaleFlipAug',
                # img_scale=(1408, 792),
                img_scale=[(1024, 576)],
                flip=False,
                transforms=[
                    dict(
                        type='Mono3DResize_faw',
                        multiscale_mode='value',
                        keep_ratio=False),
                    dict(
                        type='mmdet.Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='mmdet.Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D_faw',
                        class_names=[
                            'VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN'
                        ],
                        with_label=False),
                    dict(type='Collect3D_faw', keys=['img'])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True, # 在__gititem__函数中控制选择训练和测试
        box_type_3d='Camera',
        use_sub_type_as_label=False,
        eval_pipeline= [
            dict(type='LoadImageFromFileMono3D_faw'),
            dict(
                type='DefaultFormatBundle3D_faw',
                class_names=['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN'],
                with_label=False),
            dict(type='Collect3D_faw', keys=['img'])
        ]
        ))

evaluation = dict(interval=2)
###########

resume_from = 'F:\GitHub\mmdet3d\checkpoints\\v0.3_fov120_r18_no_norm_unmerge_attr_scale_epoch_24.pth'


########### mmdet_schedule_1x.py
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
total_epochs = 12
evaluation = dict(interval=2)
###########