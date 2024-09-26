# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import build_dataloader
from .builder import DATASETS, PIPELINES, build_dataset
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .kitti_dataset import KittiDataset
from .kitti_mono_dataset import KittiMonoDataset
from .lyft_dataset import LyftDataset
from .nuscenes_dataset import NuScenesDataset
from .nuscenes_mono_dataset import NuScenesMonoDataset
# yapf: disable
from .pipelines import (AffineResize, BackgroundPointsFilter, GlobalAlignment,
                        GlobalRotScaleTrans, IndoorPatchPointSample,
                        IndoorPointSample, LoadAnnotations3D,
                        LoadPointsFromDict,
                        LoadPointsFromMultiSweeps, NormalizePointsColor,
                        ObjectNameFilter, ObjectNoise, ObjectRangeFilter,
                        ObjectSample, PointSample, PointShuffle,
                        PointsRangeFilter, RandomDropPointsColor, RandomFlip3D,
                        RandomJitterPoints, RandomShiftScale,
                        VoxelBasedPointSampler)
from .pipelines import (LoadMultiViewImageFromFiles_petrv2, LoadAnnotations3D_petrv2, LoadPointsFromFile_petrv2,
                      LoadMultiViewImageFromFiles_Internal_petrv2, LoadMultiViewImageFromMultiSweepsFiles_petrv2)
from .pipelines import (ResizeCropFlipImage_petrv2, GlobalRotScaleTransImage_petrv2,
                                   NormalizeMultiviewImage_petrv2, PadMultiViewImage_petrv2)

# yapf: enable
from .s3dis_dataset import S3DISDataset, S3DISSegDataset
from .scannet_dataset import (ScanNetDataset, ScanNetInstanceSegDataset,
                              ScanNetSegDataset)
from .semantickitti_dataset import SemanticKITTIDataset
from .sunrgbd_dataset import SUNRGBDDataset
from .utils import get_loading_pipeline
from .waymo_dataset import WaymoDataset
from .senseauto_mono_dataset_faw import SenseautoMonoDataset_faw
from .internal_dataset_sweep import InternalDatasetSweep

__all__ = [
    'KittiDataset', 'KittiMonoDataset', 'build_dataloader', 'DATASETS',
    'build_dataset', 'NuScenesDataset', 'NuScenesMonoDataset', 'LyftDataset',
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter',
    'S3DISSegDataset', 'S3DISDataset',
    'NormalizePointsColor', 'IndoorPatchPointSample', 'IndoorPointSample',
    'PointSample', 'LoadAnnotations3D', 'GlobalAlignment', 'SUNRGBDDataset',
    'ScanNetDataset', 'ScanNetSegDataset', 'ScanNetInstanceSegDataset',
    'SemanticKITTIDataset', 'Custom3DDataset', 'Custom3DSegDataset',
    'LoadPointsFromMultiSweeps', 'WaymoDataset', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'get_loading_pipeline', 'RandomDropPointsColor',
    'RandomJitterPoints', 'ObjectNameFilter', 'AffineResize',
    'RandomShiftScale', 'LoadPointsFromDict', 'PIPELINES',
    'SenseautoMonoDataset_faw',
    'InternalDatasetSweep',

    'LoadPointsFromFile_petrv2', 'LoadMultiViewImageFromFiles_petrv2', 'LoadAnnotations3D_petrv2',
    'LoadMultiViewImageFromFiles_Internal_petrv2', 'LoadMultiViewImageFromMultiSweepsFiles_petrv2',
    'ResizeCropFlipImage_petrv2', 'GlobalRotScaleTransImage_petrv2', 'NormalizeMultiviewImage_petrv2',
    'PadMultiViewImage_petrv2'
]
