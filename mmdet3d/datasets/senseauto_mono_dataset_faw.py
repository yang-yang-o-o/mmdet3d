# Copyright (c) OpenMMLab. All rights reserved.
import os
import math
import copy
import json
import shutil
import tempfile
import warnings
from os import path as osp
from tqdm import tqdm
from multiprocessing import Process, Queue

import skvideo.io
import cv2
import numpy as np
import torch
import mmcv
from mmcv.utils import print_log
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

# from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from .pipelines import Compose
from mmdet3d.datasets.utils import extract_result_dict, get_loading_pipeline
from mmdet3d.core.bbox import CameraInstance3DBoxes, get_box_type, points_cam2img
# from ..core import NuScenesEval_faw, show_multi_modality_result_faw, show_mono_result_faw
from ..core import NuScenesEval_faw, show_mono_result_faw
from .builder import DATASETS


@DATASETS.register_module()
class SenseautoMonoDataset_faw(CocoDataset):
    r"""Monocular 3D detection on NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        data_root (str): Path of dataset root.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Camera' in this class. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
    """
    CLASSES = ('VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN')
    CLASS_MAPPING = ['VEHICLE_CAR', 'BIKE_BICYCLE', 'PEDESTRIAN']

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root,
                 load_interval=1,
                 with_velocity=False,
                 modality=None,
                 box_type_3d='Camera',
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 version='v1.0-trainval',
                 eval_data_root='./work_dirs/',
                 is_side=False,
                 valid_range=None,
                 shelter_ratio_range=None,
                 depth_filter=False,
                 use_sub_type_as_label=False,
                 classes=None,
                 classes_mapping=None,
                 center2d_filter=False,
                 result_path=None,
                 # add for test quant
                 use_vis_quant_mode=False,
                 quant_git_file=None,
                 eval_pipeline=None,
                 save_path='./work_dirs/mono/test',
                 file_client_args=None,
                 **kwargs):
        self.file_client_args = file_client_args
        super().__init__(ann_file=ann_file, pipeline=[], **kwargs)
        self.pipeline = Compose(pipeline)
        self.use_sub_type_as_label = use_sub_type_as_label
        if classes:
            self.CLASSES = classes
        if classes_mapping:
            self.CLASS_MAPPING = classes_mapping
        else:
            self.CLASS_MAPPING = self.CLASSES
        self.data_root = data_root
        self.eval_data_root = eval_data_root
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.modality = modality
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.eval_version = eval_version
        self.use_valid_flag = use_valid_flag
        self.bbox_code_size = 7
        self.version = version
        self.is_side = is_side
        self.valid_range = valid_range
        self.shelter_ratio_range = shelter_ratio_range
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory
            self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False)
        self.depth_filter = depth_filter
        self.center2d_filter = center2d_filter
        self.use_vis_quant_mode = use_vis_quant_mode
        if self.use_vis_quant_mode:
            self.quant_git_file = quant_git_file
            self.quant_gt = mmcv.load(quant_git_file)
        self.eval_pipeline = eval_pipeline

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        if self.file_client_args is not None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
            with self.file_client.get_local_path(ann_file) as local_path:
                self.coco = COCO(local_path)
        else:
            self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        default_labels = [{'id': 0,'name': 'VEHICLE_CAR'}, {'id': 1,'name': 'VEHICLE_TRUCK'}, {'id': 2,'name': 'BIKE_BICYCLE'}, {'id': 3,'name': 'PEDESTRIAN'}]
        self.cat_ids = [cat['id'] for cat in self.coco.dataset['categories']] #[0,1,2]
        self.attr_ids = [cat['id'] for cat in self.coco.dataset.get('sub_types', default_labels)] #[0,1,2...11]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.attr2label = {attr_id: i for i, attr_id in enumerate(self.attr_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        # print('img_ids num:',len(self.img_ids))
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.
            "aws_path""file_name""image_id""area""category_name"
            "category_id""bbox""iscrowd""bbox_cam3d""center2d"
            "segmentation""id"

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map 
        """
        gt_bboxes = []
        gt_labels = []
        attr_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        shelter_ratio = []
        if self.depth_filter:
            # cam_intrinsic = np.array(img_info['cam_intrinsic']).astype(np.float32)
            # min_h = 0
            # if cam_intrinsic.size > 0:
            #     min_h = 1.6 / 50 * cam_intrinsic[0][0]
            # elif img_info['width'] == 3840:
            #     min_h = 64
            # elif img_info['width'] == 1920:
            #     min_h = 32
            h_filter_ratio = 16 / 720
        for i, ann in enumerate(ann_info):
            # if ann.get('ignore', False):
            #     continue
            # cx, cy, w, h
            x1, y1, w, h = ann['bbox']
            x1 = x1 - w * 0.5  #left top x
            y1 = y1 - h * 0.5  #left top y
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if self.depth_filter:
                # depth filter and 2d bbox height ranging filter
                # bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                # if bbox_cam3d.size > 0 and bbox_cam3d.squeeze()[2] > 70:
                #     continue
                # if bbox_cam3d.size <= 0 and ann['category_id'] == 0 and inter_h < min_h:
                #     continue
                if (ann['category_id'] == 0) and (inter_h / img_info['height'] <= h_filter_ratio):
                    continue

            if self.center2d_filter:
                cx, cy = ann['center2d'][:2]
                if cx < (-1 / 2) * img_info['width'] or cx > (3 / 2) * img_info['width']:
                    continue
                if cy < (-1 / 2) * img_info['height'] or cy > (3 / 2) * img_info['height']:
                    continue

            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                print(ann['category_id'], " not in cat_ids!")
                continue
            if ann['sub_type_id'] not in self.attr_ids:
                print(ann['sub_type_id'], " not in cat_ids!")
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                if not self.use_sub_type_as_label:
                    gt_labels.append(self.cat2label[ann['category_id']])
                else:
                    gt_labels.append(self.attr2label[ann['sub_type_id']])
                attr_labels.append(self.attr2label[ann['sub_type_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                # ann['bbox_cam3d'][-1] -= math.pi / 2
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                if bbox_cam3d.size <= 0:
                    bbox_cam3d = np.ones((1, 7)) * -1.
                # velo_cam3d = np.array([0.0, 0.0]).reshape(1, 2)
                # bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                # TODO(Ang): for truncation object
                # truncation_thresh = 15
                # if center2d[0] < 0:
                #     if x1 < truncation_thresh:
                #         center2d = [x1, y1 + h / 2]
                #     elif abs(img_info['width'] - (x1 + w)) < truncation_thresh:
                #         center2d = [x1 + w, y1 + h / 2]
                # elif center2d[0] > w:
                #     if x1 < truncation_thresh:
                #         center2d = [x1, y1 + h / 2]
                #     elif abs(img_info['width'] - (x1 + w)) < truncation_thresh:
                #         center2d = [x1 + w, y1 + h / 2]
                # depth = ann['center2d'][2]
                # depth = ann['bbox_cam3d'][2]
                depth = bbox_cam3d.squeeze()[2]
                centers2d.append(center2d)
                depths.append(depth)
                shelter_ratio.append(ann.get('shelter_ratio', 0))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            attr_labels = np.array(attr_labels, dtype=np.int64)
        else:
            # return None
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        if shelter_ratio:
            shelter_ratio = np.array(shelter_ratio, dtype=np.float32)
        else:
            shelter_ratio = np.zeros((0), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, # 2dbox ，nx4 ，左上右下
            labels=gt_labels, # 2dbox 大类别，n
            gt_bboxes_3d=gt_bboxes_cam3d, # 3dbox ，nx7
            gt_labels_3d=gt_labels_3d, # 3dbox 大类别，n
            attr_labels=attr_labels, # 2dbox 细分类，n
            centers2d=centers2d, # 3d中心点投影2d点，nx2
            depths=depths, # 3d中心点相机系下z坐标值，n
            bboxes_ignore=gt_bboxes_ignore, # 0x4，全0
            masks=gt_masks_ann, # 13，全空
            seg_map=seg_map, # seg图片路径
            shelter_ratio=shelter_ratio) # n，全0

        return ann

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)


        
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info == None:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        img_info = self.data_infos[index]
        input_dict = dict(img_info=img_info)

        if load_annos:
            ann_info = self.get_ann_info(index)
            input_dict.update(dict(ann_info=ann_info))

        # if self.vis_3d:
        #     pts_info = self.get_data_info(index)
        #     if 'img_info' in pts_info.keys():
        #         _ = pts_info.pop('img_info')
        #     input_dict.update(pts_info)

        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]

        return data

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'img_bbox' in result.keys():
                result = result['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            img, img_metas = self._extract_data(i, pipeline,
                                                ['img', 'img_metas'])
            # need to transpose channel to first dim
            img = img.numpy().transpose(1, 2, 0)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes = result['boxes_3d']
            pred_bbox2d = result['bboxes_2d']
            show_multi_modality_result_faw(
                img,
                gt_bboxes,
                pred_bboxes,
                pred_bbox2d,
                img_metas['cam2img'],
                out_dir,
                file_name,
                box_mode='camera',
                show=show)

    # KITTI
    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        # print('outputs:',outputs)
        # print('pklfile_prefix:',pklfile_prefix)
        # print('tmp_dir:',tmp_dir)

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES,
                                                    pklfile_prefix,
                                                    submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0] or \
                'img_bbox2d' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                # print('name:',name)
                results_ = [out[name] for out in outputs]
                # print('results_:',results_)
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if '2d' in name:
                    result_files_ = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir

    '''
    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import senseauto_eval, senseauto_eval_coco_style
        gt_annos = []
        for data_info in self.data_infos:
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info)
            gt_anno = self.bbox2gt_kitti(ann, data_info)
            gt_annos.append(gt_anno)

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bbox', 'bev', '3d']
                if '2d' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = senseauto_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox2d':
                ap_result_str, ap_dict = senseauto_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = senseauto_eval(
                    gt_annos, result_files, self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict
    '''

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        
        # save_path = os.environ['WORK_DIR'] + '/' + os.environ['TAG']
        save_path = "F:\GitHub\mmdet3d\work_dirs\my_fcos3d"

        self.evaluate_internal_data(results, save_path, **kwargs)
        # self.evaluate_pillar(results, save_path, **kwargs)

        show_res = True
        vis_rate = 1
        if show_res and vis_rate != 0:
            try:
                # import pdb;pdb.set_trace()
                matched_results = mmcv.load(os.path.join(save_path,'eval_results.json'))
                matched_gts = matched_results['matched_gt']
                matched_results = matched_results['matched_results']
            except:
                matched_gts = None
                matched_results = None
        
            self.show_results(results, save_path, pipeline=self.eval_pipeline, sample_rate=vis_rate, 
                                    matched_results=matched_results, matched_gts=matched_gts)
        
        return {'evaluation': 'finished'}

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos)
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            # print('idx:',idx)
            # print('pred_dicts:',pred_dicts)
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['id']
            image_shape = [info['height'], info['width']]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                label_preds = box_dict['label_preds']

                for box, bbox, score, label in zip(box_preds, box_2d_preds,
                                                   scores, label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                # print('anno_pre:',anno)
                annos.append(anno)

            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)
        # print('det_annos:',det_annos)
        return det_annos

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.anno_infos)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.anno_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-10)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.anno_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.
                - boxes_3d (:obj:`CameraInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.
                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['id']
        if 'bboxes_2d' in box_dict:
            bbox2d_preds = box_dict['bboxes_2d']

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        cam_intrinsic = np.array(info['cam_intrinsic']).astype(np.float32)
        img_shape = np.array([info['height'], info['width']])
        cam_intrinsic = box_preds.tensor.new_tensor(cam_intrinsic)

        box_preds_camera = box_preds
        # box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
        #                                        np.linalg.inv(rect @ Trv2c))

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, cam_intrinsic)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            if 'bboxes_2d' in box_dict:
                # use bbox2d results
                return dict(
                    bbox=bbox2d_preds[valid_inds].numpy(),
                    box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                    scores=scores[valid_inds].numpy(),
                    label_preds=labels[valid_inds].numpy(),
                    sample_idx=sample_idx)
            return dict(
                # use projection bbox2d results
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                # box3d_lidar=box_preds_lidar[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                # box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

    def bbox2gt_kitti(self, gt_annos, data_info):
        """Convert 3D gt to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            
        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        anno = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': []
        }
        if len(gt_annos['bboxes']) > 0:
            box_2d = gt_annos['bboxes']
            box_3d = gt_annos['gt_bboxes_3d']
            labels = gt_annos['labels']

            for box, bbox, label in zip(box_3d, box_2d, labels):
                # bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                # bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno['name'].append(self.CLASSES[int(label)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                anno['bbox'].append(bbox)
                anno['dimensions'].append(box[3:6])
                anno['location'].append(box[:3])
                anno['rotation_y'].append(box[6])

            anno = {k: np.stack(v) for k, v in anno.items()}
            # print('gt_anno:',anno)

        else:
            anno = {
                'name': np.array([]),
                'truncated': np.array([]),
                'occluded': np.array([]),
                'alpha': np.array([]),
                'bbox': np.zeros([0, 4]),
                'dimensions': np.zeros([0, 3]),
                'location': np.zeros([0, 3]),
                'rotation_y': np.array([]),
            }
        return anno


    def evaluate_internal_data(self, results, save_path, **kwargs):
        self.save_path = save_path

        self.gts_label = [] # 没用到
        self.gts_ignore = []
        self.preds_ignore = []

        if self.valid_range is not None: # 不
            max_pos = np.array([self.valid_range[2],self.valid_range[3]])
            min_pos = np.array([self.valid_range[0],self.valid_range[1]])

        # 加载 self.gts_label，self.gts_ignore，self.preds_ignore
        # 后续只用到了 self.gts_ignore，self.preds_ignore
        frame_id = 0
        for data_info in self.data_infos: # 加载每张图的 gts_label，gts_ignore，preds_ignore
            img_id = data_info['id'] # 根据images中的id去coco格式的annotation里找标签
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info) # 预处理annotation，一些修改和过滤逻辑

            if self.use_vis_quant_mode:
                gt_results = self.quant_gt[frame_id]
                gt_labels = gt_results['img_bbox']['labels_3d'].numpy()
                self.gts_label.append(gt_labels)
                gt_bboxes = gt_results['img_bbox']['boxes_3d'].tensor.numpy()
            else:
                gt_labels = ann['labels']
                self.gts_label.append(gt_labels)
                gt_bboxes = ann['gt_bboxes_3d']

            # print("results: ",results)
            pred = results[frame_id]
            pred_bboxes = pred['img_bbox']['boxes_3d'].tensor.numpy()
            # pred = results['img_bbox'][frame_id]
            # pred_bboxes = pred['boxes_3d'].tensor.numpy()
            bbox_shelter_ratio = ann['shelter_ratio']

            gt_ignore = np.zeros(len(gt_bboxes))
            pred_ignore = np.zeros(len(pred_bboxes))

            if self.valid_range is not None:
                # car to cam
                extrinsic = np.array(data_info['extrinsic'])
                # cam to car rotation
                R = extrinsic[:3, :3].T
                # T
                T = extrinsic[:3, 3]

                # gt
                idx = 0
                for gt_bbox in gt_bboxes:
                    cam_pos = gt_bbox[:3]
                    car_pos = np.dot(R, (cam_pos - T))
                    car_pos = car_pos[:2]
                    flag = np.sum(car_pos > min_pos) + np.sum(car_pos < max_pos)
                    if flag != 4:
                        gt_ignore[idx] = 1
                    if self.is_side and abs(car_pos[1]) <= 1.875:
                        gt_ignore[idx] = 1
                    idx += 1

                # pred
                pred_idx = 0
                for pred_bbox in pred_bboxes:
                    cam_pos = pred_bbox[:3]
                    car_pos = np.dot(R, (cam_pos - T))
                    car_pos = car_pos[:2]
                    flag = np.sum(car_pos > min_pos) + np.sum(car_pos < max_pos)
                    if flag != 4:
                        pred_ignore[pred_idx] = 1
                    if self.is_side and abs(car_pos[1]) <= 1.875:
                        pred_ignore[pred_idx] = 1
                    pred_idx += 1

            if self.shelter_ratio_range is not None:
                # gt
                idx = 0
                for ratio in bbox_shelter_ratio:
                    if ratio < self.shelter_ratio_range[0] or ratio > self.shelter_ratio_range[1]:
                        gt_ignore[idx] = 1
                    idx += 1
            self.gts_ignore.append(gt_ignore)
            self.preds_ignore.append(pred_ignore)
            frame_id += 1

        # self.convert_results(results['img_bbox'])
        self.convert_results(results) # 获取 self.predictions 和 self.gts , self.predictions 比 self.gts多最后一列scores
        # import pdb;pdb.set_trace()

        # run evaluation
        # import pdb;pdb.set_trace()
        ########### 没用到
        all_gt_xyz = [gt_info[:, 1:4] for gt_info in self.gts]
        all_gt_xyz = np.concatenate(all_gt_xyz, axis=0).astype(np.float32) # 当前batch的所有gt的xyz，nx3
        xyz_min = np.min(all_gt_xyz, axis=0)
        xyz_max = np.max(all_gt_xyz, axis=0)
        ###########
        # all_gt_xyz = np.array(all_gt_xyz).astype(np.float32)
        # all_pred_xyz = [gt_info[0, 1:4] for gt_info in self.predictions]
        # all_pred_xyz = np.array(all_gt_xyz).astype(np.float32)
        # import pdb;pdb.set_trace()
        # point_cloud_range = [max(0.0, xyz_min[0]), xyz_min[1] - 5, -5, xyz_max[0] + 10, xyz_max[1] + 5, 5]
        point_cloud_range = [0.0, -60, -5, 100, 60, 5]
        # point_cloud_range = [0.0, -8, -5, 250, 8, 5]
        # point_cloud_range = None

        # self.predictions：class x y z l w h r x1 y1 x1 y2 score
        # self.gts：class x y z l w h r x1 y1 x1 y2
        evaluator = NuScenesEval_faw(self.predictions,
                                 self.gts,
                                 'class x y z l w h r x1 y1 x1 y2 score', # self.predictions 的列含义
                                 self.save_path, 
                                 gts_ignore=self.gts_ignore, preds_ignore = self.preds_ignore,
                                 classes=sorted(list(set(self.CLASS_MAPPING)), key=self.CLASS_MAPPING.index),
                                 point_cloud_range=point_cloud_range)
        eval_results = evaluator.get_metric_results()
        mmcv.dump(eval_results, os.path.join(self.save_path,
                                             'eval_results.json'))

        # bad cases
        matched_gts = eval_results['matched_gt']
        matched_results = eval_results['matched_results']
        bad_cases = []

        cates = [{
            'id': 0,
            'name': 'VEHICLE_CAR'
        }, {
            'id': 1,
            'name': 'BIKE_BICYCLE'
        }, {
            'id': 2,
            'name': 'PEDESTRIAN'
        }]
        bad_cases_json = {'images': [], 'annotations': [], 'categories': cates}
            
        for i in range(len(matched_gts)): # 遍历每张图片
            matched_gt = np.array(matched_gts[i])[np.where(np.array(matched_gts[i])!= -1)[0]] # 当前图片中被gt是否被匹配，被匹配为true，未被匹配为false
            matched_result = np.array(matched_results[i])[np.where(np.array(matched_results[i])!= -1)[0]] # 当前图片中pre是否被匹配，被匹配为true，未被匹配为false
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            if (np.sum(matched_gt) != len(matched_gt)) or (np.sum(matched_result) != len(matched_result)): # 存在gt未被匹配 或者 pre未被匹配
                bad_cases.append(img_path) # 当前图片存在 漏检 或 误检
                img_id = data_info['id']
                ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
                ann_info = self.coco.load_anns(ann_ids)

                bad_cases_json['images'].extend([data_info]) # 加入当前图片的信息
                bad_cases_json['annotations'].extend(ann_info) # 加入当前图片的标签信息

        bad_cases_dict={}
        bad_cases_dict['bad_cases_file_name']=bad_cases # 所有漏检或误检的图片路径
        bad_cases_save_path = os.path.join(self.save_path,'bad_cases.json')
        mmcv.dump(bad_cases_dict, bad_cases_save_path)
        bad_cases_json_path = os.path.join(self.save_path,'bad_cases.coco.json')
        mmcv.dump(bad_cases_json, bad_cases_json_path)       
        print('bad cases num:',len(bad_cases))
        return


    def convert_results(self, results):
        self.predictions = [] # 比 self.gts多最后一列scores
        self.gts = []

        frame_id = 0
        for data_info in self.data_infos: # 获取 self.gts
            img_id = data_info['id'] # 根据images中的id去coco格式的annotation里找标签
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids) # 当前图像的所有annotation
            ann = self._parse_ann_info(data_info, ann_info) # 预处理annotation，一些修改和过滤逻辑

            if self.use_vis_quant_mode:
                gt_results = self.quant_gt[frame_id]
                gt_labels = gt_results['img_bbox']['labels_3d'].numpy()
                self.gts_label.append(gt_labels)
                gt_bboxes = gt_results['img_bbox']['boxes_3d'].tensor.numpy()
                gt_bboxes_2d = gt_results['img_bbox']['bboxes_2d'].numpy()
            else:
                gt_bboxes = ann['gt_bboxes_3d']
                gt_labels = ann['labels']
                gt_bboxes_2d = ann['bboxes']

            # prepare gt
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            x1, y1, x2, y2 = [], [], [], []
            for gt_label, gt_bbox, gt_bbox2d in zip(gt_labels, gt_bboxes, gt_bboxes_2d):
                classes.append(str(self.CLASS_MAPPING[gt_label]))
                x.append(gt_bbox[2])
                y.append(gt_bbox[0])
                z.append(gt_bbox[1])
                # x.append(gt_bbox[0])
                # y.append(gt_bbox[1])
                # z.append(gt_bbox[2])
                l.append(gt_bbox[3])
                h.append(gt_bbox[4])
                w.append(gt_bbox[5])
                r.append(gt_bbox[6])
                x1.append(gt_bbox2d[0])
                y1.append(gt_bbox2d[1])
                x2.append(gt_bbox2d[2])
                y2.append(gt_bbox2d[3])

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1),
                 np.array(x1).reshape(-1, 1), np.array(y1).reshape(-1, 1),
                 np.array(x2).reshape(-1, 1), np.array(y2).reshape(-1, 1)))
            self.gts.append(final_array)
            frame_id += 1

        for pred in results: # 获取 self.predictions
            pred_bboxes = pred['img_bbox']['boxes_3d'].tensor.numpy()
            pred_scores = pred['img_bbox']['scores_3d'].numpy()
            pred_labels = pred['img_bbox']['labels_3d'].numpy()
            pred_bboxes2d = pred['img_bbox']['bboxes_2d'].numpy()

            # pred_bboxes = pred['boxes_3d'].tensor.numpy()
            # pred_scores = pred['scores_3d'].numpy()
            # pred_labels = pred['labels_3d'].numpy()
            # pred_bboxes2d = pred['bboxes_2d'].numpy()

            # prepare prediction
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            x1, y1, x2, y2 = [], [], [], []
            for pred_label, pred_bbox, pred_score, pred_bbox2d in zip(
                    pred_labels, pred_bboxes, pred_scores, pred_bboxes2d):
                classes.append(str(self.CLASS_MAPPING[pred_label]))
                x.append(pred_bbox[2])
                y.append(pred_bbox[0])
                z.append(pred_bbox[1])
                # x.append(pred_bbox[0])
                # y.append(pred_bbox[1])
                # z.append(pred_bbox[2])
                l.append(pred_bbox[3])
                h.append(pred_bbox[4])
                w.append(pred_bbox[5])
                r.append(pred_bbox[6])
                score.append(pred_score)
                x1.append(pred_bbox2d[0])
                y1.append(pred_bbox2d[1])
                x2.append(pred_bbox2d[2])
                y2.append(pred_bbox2d[3])

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1),
                 np.array(x1).reshape(-1, 1), np.array(y1).reshape(-1, 1),
                 np.array(x2).reshape(-1, 1), np.array(y2).reshape(-1, 1)))
            final_array = np.hstack(
                (final_array, np.array(score).reshape(-1, 1)))
            self.predictions.append(final_array)

    def evaluate_internal_data_by_class(self, results, save_path):
        self.save_path = save_path
        print('***************save_path:',save_path)
        # import pdb;pdb.set_trace()
        self.predictions = {}
        self.gts = {}
        self.gts_label = []
        
        self.gts_ignore = {}
        self.preds_ignore = {}

        frame_id = 0
        for data_info in self.data_infos:
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info)
            gt_labels = ann['labels']
            self.gts_label.append(gt_labels)


        for class_label in self.CLASS_MAPPING:
            print('***************process:',class_label)
            class_save_path = os.path.join(self.save_path,class_label)
            if os.path.exists(class_save_path):
                shutil.rmtree(class_save_path)
            os.makedirs(class_save_path, exist_ok=True)

            self.convert_results_by_class(results,class_label)
            # run evaluation
            point_cloud_range = [0.0, -60, -5, 100, 60, 5]
            evaluator = NuScenesEval_faw(self.predictions[class_label],
                                 self.gts[class_label],
                                 'class x y z l w h r x1 y1 x1 y2 score',
                                 class_save_path, gts_ignore=self.gts_ignore[class_label],
                                 preds_ignore = self.preds_ignore[class_label],
                                 classes=[class_label], point_cloud_range=point_cloud_range)
            eval_results = evaluator.get_metric_results()
            mmcv.dump(eval_results, os.path.join(class_save_path,
                                             class_label+'_eval_results.json'))
        
        all_class_matched_results = {}
        all_class_matched_gt = {}
        eval_results = {}
        for class_label in self.CLASS_MAPPING:
            try:
                matched_results = mmcv.load(os.path.join(self.save_path,class_label,
                                                    class_label+'_eval_results.json'))
                # import pdb;pdb.set_trace()
                eval_results[class_label] = matched_results[class_label]
                matched_gt = matched_results['matched_gt']
                matched_results = matched_results['matched_results']
                
            except:
                print(class_label,' not find!')
                matched_results = None
                matched_gt = None
                
            all_class_matched_results[class_label]=matched_results
            all_class_matched_gt[class_label]=matched_gt

        # import pdb;pdb.set_trace()
        matched_results = []
        matched_gts = []
        for i in range(len(results)):
            matched_result=[]
            matched_gt = []
       
            for label in results[i]['img_bbox']['labels_3d'].tolist():
                matched_result.append(all_class_matched_results[self.CLASS_MAPPING[label]][i][0])
                all_class_matched_results[self.CLASS_MAPPING[label]][i] = all_class_matched_results[self.CLASS_MAPPING[label]][i][1:]
           
            for label in self.gts_label[i]:
                matched_gt.append(all_class_matched_gt[self.CLASS_MAPPING[label]][i][0])
                all_class_matched_gt[self.CLASS_MAPPING[label]][i] = all_class_matched_gt[self.CLASS_MAPPING[label]][i][1:]
         

            matched_results.append(matched_result)
            matched_gts.append(matched_gt)
        eval_results['matched_results'] = matched_results
        eval_results['matched_gt'] = matched_gts
        mmcv.dump(eval_results, os.path.join(self.save_path,'eval_results.json'))
        
        # bad cases
        bad_cases = []
            
        for i in range(len(matched_gts)):
            matched_gt = np.array(matched_gts[i])[np.where(np.array(matched_gts[i])!= -1)[0]]
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            if np.sum(matched_gt) != len(matched_gt):
                bad_cases.append(img_path)
        bad_cases_dict={}
        bad_cases_dict['bad_cases_file_name']=bad_cases
        bad_cases_save_path = os.path.join(self.save_path,'bad_cases.json')
        mmcv.dump(bad_cases_dict, bad_cases_save_path)       
        print('bad cases num:',len(bad_cases))
        return

    def convert_results_by_class(self,results,class_label):
        self.predictions[class_label] = []
        self.gts[class_label] = []
        self.gts_ignore[class_label] = []
        self.preds_ignore[class_label] = []

        if self.valid_range is not None:
            max_pos = np.array([self.valid_range[2],self.valid_range[3]])
            min_pos = np.array([self.valid_range[0],self.valid_range[1]])

        for data_info in self.data_infos:
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info)

            gt_bboxes = ann['gt_bboxes_3d']
            # car to cam
            extrinsic = np.array(data_info['extrinsic'])
            # cam to car rotation
            R = extrinsic[:3,:3].T
            # T
            T = extrinsic[:3,3]

            gt_bboxes = ann['gt_bboxes_3d']
            gt_labels = ann['labels']
            gt_bboxes_2d = ann['bboxes']

            # prepare gt
            gt_ignore=[]
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            x1, y1, x2, y2 = [], [], [], []
            for gt_label, gt_bbox, gt_bbox2d in zip(gt_labels, gt_bboxes, gt_bboxes_2d):
                if self.CLASS_MAPPING[gt_label]==class_label:
                    classes.append(str(self.CLASS_MAPPING[gt_label]))
                    x.append(gt_bbox[2])
                    y.append(gt_bbox[0])
                    z.append(gt_bbox[1])
                    # x.append(gt_bbox[0])
                    # y.append(gt_bbox[1])
                    # z.append(gt_bbox[2])
                    l.append(gt_bbox[3])
                    h.append(gt_bbox[4])
                    w.append(gt_bbox[5])
                    r.append(gt_bbox[6])
                    x1.append(gt_bbox2d[0])
                    y1.append(gt_bbox2d[1])
                    x2.append(gt_bbox2d[2])
                    y2.append(gt_bbox2d[3])

                    cam_pos = gt_bbox[:3]
                    car_pos = np.dot(R,(cam_pos-T))
                    car_pos = car_pos[:2]
                    if self.valid_range is not None:
                        flag = np.sum(car_pos>min_pos)+np.sum(car_pos<max_pos)
                        if flag == 4:
                            if self.is_side and abs(car_pos[1]) <= 1.875:
                                gt_ignore.append(1)
                            else:
                                gt_ignore.append(0)
                        else:
                            gt_ignore.append(1)
                    else:
                        gt_ignore.append(0)

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1),
                 np.array(x1).reshape(-1, 1), np.array(y1).reshape(-1, 1),
                 np.array(x2).reshape(-1, 1), np.array(y2).reshape(-1, 1)))
            self.gts[class_label].append(final_array)

            gt_ignore = np.array(gt_ignore)
            self.gts_ignore[class_label].append(gt_ignore)

        frame_id = 0
        for pred in results:
            pred_bboxes = pred['img_bbox']['boxes_3d'].tensor.numpy()
            pred_scores = pred['img_bbox']['scores_3d'].numpy()
            pred_labels = pred['img_bbox']['labels_3d'].numpy()
            pred_bboxes2d = pred['img_bbox']['bboxes_2d'].numpy()

            data_info = self.data_infos[frame_id]
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info)

            # car to cam
            extrinsic = np.array(data_info['extrinsic'])
            # cam to car rotation
            R = extrinsic[:3,:3].T
            # T
            T = extrinsic[:3,3]

            # prepare prediction
            pred_ignore = []
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            x1, y1, x2, y2 = [], [], [], []
            for pred_label, pred_bbox, pred_score, pred_bbox2d in zip(
                    pred_labels, pred_bboxes, pred_scores, pred_bboxes2d):
                if self.CLASS_MAPPING[pred_label]==class_label:
                    classes.append(str(self.CLASS_MAPPING[pred_label]))
                    x.append(pred_bbox[2])
                    y.append(pred_bbox[0])
                    z.append(pred_bbox[1])
                    # x.append(pred_bbox[0])
                    # y.append(pred_bbox[1])
                    # z.append(pred_bbox[2])
                    l.append(pred_bbox[3])
                    h.append(pred_bbox[4])
                    w.append(pred_bbox[5])
                    r.append(pred_bbox[6])
                    score.append(pred_score)
                    x1.append(pred_bbox2d[0])
                    y1.append(pred_bbox2d[1])
                    x2.append(pred_bbox2d[2])
                    y2.append(pred_bbox2d[3])

                    cam_pos = pred_bbox[:3]
                    car_pos = np.dot(R,(cam_pos-T))
                    car_pos = car_pos[:2]
                    if self.valid_range is not None:
                        flag = np.sum(car_pos>min_pos)+np.sum(car_pos<max_pos)
                        if flag == 4:
                            if self.is_side and abs(car_pos[1]) <= 1.875:
                                pred_ignore.append(1)
                            else:
                                pred_ignore.append(0)
                        else:
                            pred_ignore.append(1)
                    else:
                        pred_ignore.append(0)

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1),
                 np.array(x1).reshape(-1, 1), np.array(y1).reshape(-1, 1),
                 np.array(x2).reshape(-1, 1), np.array(y2).reshape(-1, 1)))
            final_array = np.hstack(
                (final_array, np.array(score).reshape(-1, 1)))
            self.predictions[class_label].append(final_array)

            pred_ignore = np.array(pred_ignore)
            self.preds_ignore[class_label].append(pred_ignore)
            frame_id += 1

    def show_bbox2d(self, img, bbox2ds, color=(0, 0, 255), labels=None, bboxes3d=None):
        i=0
        for bbox in bbox2ds:
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            if labels is not None:
                cv2.putText(img, str(labels[i]), (left, top+5), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1, cv2.LINE_AA)
            if bboxes3d is not None:
                tmp = bboxes3d.tensor.numpy()
                text = str(round(tmp[i][0], 2))+', '+str(round(tmp[i][1], 2))+', '+str(round(tmp[i][2], 2))
                cv2.putText(img, text, (left+10, top+20), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1, cv2.LINE_AA)
            i+=1
        # from IPython import embed; embed()
        return img

    def show_bev(self, img, pred_bev_corners, gt_bev_corners, pred_color=(241, 101, 72), gt_color = (61, 102, 255)):
        # bev_size = 1600
        # scale = 8
        # if img is None:
        #     img = np.zeros((bev_size, bev_size, 3))
        # draw circle
        # for i in range(bev_size // (20 * scale)):
        #     cv2.circle(img, (bev_size // 2, bev_size // 2),
        #                (i + 1) * 10 * scale, (125, 217, 233), 2)
        bev_size_h = 1500
        bev_size_w = 1000
        scale = 10
        if img is None:
            img = np.zeros((bev_size_h, bev_size_w, 3))
        # draw grad
        for i in range(bev_size_h//(10*scale)+1):
            cv2.line(img,(0,i*10*scale),(bev_size_w,i*10*scale),(125, 217, 233), 2)
        for i in range(bev_size_h//(50*scale)+1):
            cv2.line(img,(0,i*50*scale),(bev_size_w,i*50*scale),(125, 217, 233), 8)

        for i in range(bev_size_w//(10*scale)+1):
            cv2.line(img,(i*10*scale,0),(i*10*scale,bev_size_h),(125, 217, 233), 2)
        for i in range(bev_size_w//(50*scale)+1):
            cv2.line(img,(i*50*scale,0),(i*50*scale,bev_size_h),(125, 217, 233), 8)

        if pred_bev_corners is not None:

            pred_bev_corners[:, :,
                            0] = pred_bev_corners[:, :, 0] * scale + bev_size_w // 2
            pred_bev_corners[:, :,
                            1] = -pred_bev_corners[:, :,
                                                    1] * scale + bev_size_h
            # pred_color = (241, 101, 72)
            for corners in pred_bev_corners:
                cv2.line(img, (int(corners[0, 0]), int(corners[0, 1])),
                        (int(corners[1, 0]), int(corners[1, 1])), pred_color, 3)
                cv2.line(img, (int(corners[1, 0]), int(corners[1, 1])),
                        (int(corners[2, 0]), int(corners[2, 1])), pred_color, 3)
                cv2.line(img, (int(corners[2, 0]), int(corners[2, 1])),
                        (int(corners[3, 0]), int(corners[3, 1])), pred_color, 3)
                cv2.line(img, (int(corners[3, 0]), int(corners[3, 1])),
                        (int(corners[0, 0]), int(corners[0, 1])), pred_color, 3)

        if gt_bev_corners is not None:
            gt_bev_corners[:, :,
                        0] = gt_bev_corners[:, :, 0] * scale + bev_size_w // 2
            gt_bev_corners[:, :,
                        1] = -gt_bev_corners[:, :, 1] * scale + bev_size_h

            # gt_color = (61, 102, 255)
            for corners in gt_bev_corners:
                cv2.line(img, (int(corners[0, 0]), int(corners[0, 1])),
                        (int(corners[1, 0]), int(corners[1, 1])), gt_color, 4)
                cv2.line(img, (int(corners[1, 0]), int(corners[1, 1])),
                        (int(corners[2, 0]), int(corners[2, 1])), gt_color, 4)
                cv2.line(img, (int(corners[2, 0]), int(corners[2, 1])),
                        (int(corners[3, 0]), int(corners[3, 1])), gt_color, 4)
                cv2.line(img, (int(corners[3, 0]), int(corners[3, 1])),
                        (int(corners[0, 0]), int(corners[0, 1])), gt_color, 4)  

        return img

    def bev_to_corners(self, bev):
        n = bev.shape[0]

        # modify heading
        # bev[:, -1] = -bev[:, -1]

        corners = torch.stack((
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) -
            0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) +
            0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) +
            0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) -
            0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) +
            0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) -
            0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) -
            0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) +
            0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
        ))
        corners = corners.reshape(4, 2, n).permute(2, 0, 1)

        return corners

    def show_demo(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        bar = mmcv.ProgressBar(len(results))
        for i, result in enumerate(results):
            if 'img_bbox' in result.keys():
                result = result['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            # if self.vis_3d:
            #     points, img_metas, img = self._extract_data(
            #         i, pipeline, ['points', 'img_metas', 'img'])
            # else:
            img, img_metas = self._extract_data(i, pipeline,
                                                ['img', 'img_metas'])
            # need to transpose channel to first dim
            img = img.numpy().transpose(1, 2, 0)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes = result['boxes_3d']
            pred_bbox2d = result['bboxes_2d']
            pred_img = show_multi_modality_result_faw(
                img,
                gt_bboxes,
                pred_bboxes,
                pred_bbox2d,
                img_metas['cam2img'],
                out_dir,
                file_name,
                box_mode='camera',
                show=show,
                thickness=2,
                return_flag=True)

            ### bev test
            gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)
            bev_img = None
            bev_img = self.show_bev(bev_img, pred_bev_corners, gt_bev_corners)

            ### img
            img_size = (1080, 2640, 3)
            demo = np.zeros(img_size, np.uint8)
            bev_img = cv2.resize(bev_img, (720,1080))
            demo[:1080, 1920:2640] = bev_img

            pred_img = cv2.resize(pred_img, (1920, 1080))
            demo[:1080, :1920] = pred_img
 
            save_path = osp.join(out_dir[:-1] + '_demo', f'{file_name}.png')
            if img is not None:
                mmcv.imwrite(demo, save_path)
            bar.update()

    def show_demo_runner(self,
                         records,
                         start,
                         end,
                         num_workers,
                         out_dir,
                         pipeline=None):
        records = records[start:end]
        nr_records = len(records)
        nr_per_thread = math.ceil(nr_records / num_workers)
        pbar = tqdm(total=nr_records)
        print("\n")
        result_queue = Queue(1000)
        procs = []
        pipeline = self._get_pipeline(pipeline)
        for i in range(num_workers):
            pstart = i * nr_per_thread
            pend = min(pstart + nr_per_thread, nr_records)
            split_records = records[pstart:pend]
            proc = Process(
                target=self.show_demo_worker,
                args=(split_records, pstart, out_dir, result_queue, pipeline))
            print("process:%d, start:%d, end:%d" % (i, pstart, pend))
            proc.start()
            procs.append(proc)

        all_results = []
        for i in range(nr_records):
            t = result_queue.get()
            all_results.append(t)
            pbar.update(1)

        for p in procs:
            p.join()

        img_size = (1080, 2640, 3)
        output_path = out_dir
        fps = 5
        output_video_path = osp.join(output_path, "demo.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(output_video_path, fourcc, fps,
                                      (img_size[1], img_size[0]))
        image_list = [f for f in os.listdir(output_path) if f.endswith(".png")]
        image_list = sorted(image_list)
        for i in tqdm(range(0, len(image_list))):
            frame = cv2.imread(os.path.join(output_path, image_list[i]))
            videoWriter.write(frame)
        videoWriter.release()

        import glob
        os.chdir(output_path)
        for file_name in glob.glob("*.{}".format("png")):
            os.remove(file_name)
        import random
        os.system("ffmpeg -i demo.avi demo_{}.mp4".format(
            int(random.random() * 1000)))

    def show_demo_worker(self, results, start_index, out_dir, result_queue,
                         pipeline):
        for index, result in enumerate(results):
            ret = self.process_per_img(start_index + index, result, out_dir,
                                       pipeline)
            result_queue.put_nowait(ret)

    def process_per_img(self, index, result, out_dir, pipeline):
        output_path = out_dir
        mmcv.mkdir_or_exist(output_path)
        img_size = (1080, 2640, 3)
        if 'img_bbox' in result.keys():
            result = result['img_bbox']
        data_info = self.data_infos[index]
        img_path = data_info['file_name']
        file_name = osp.split(img_path)[-1].split('.')[0]
        img, img_metas = self._extract_data(index, pipeline,
                                            ['img', 'img_metas'])
        # need to transpose channel to first dim
        img = img.numpy().transpose(1, 2, 0)
        gt_bboxes = self.get_ann_info(index)['gt_bboxes_3d']
        pred_bboxes = result['boxes_3d']
        pred_bbox2d = result['bboxes_2d']
        pred_img = show_multi_modality_result_faw(
            img,
            gt_bboxes,
            pred_bboxes,
            pred_bbox2d,
            img_metas['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=False,
            thickness=2,
            return_flag=True)

        ### bev test
        gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
        pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)
        bev_img = None
        bev_img = self.show_bev(bev_img, pred_bev_corners, gt_bev_corners)

        ### img
        demo = np.zeros(img_size, np.uint8)
        bev_img = cv2.resize(bev_img, (720, 1080))
        demo[:1080, 1920:2640] = bev_img

        pred_img = self.show_bbox2d(pred_img, pred_bbox2d)
        pred_img = cv2.resize(pred_img, (1920, 1080))
        demo[:1080, :1920] = pred_img

        save_path = osp.join(output_path, f'{file_name}.png')
        if img is not None:
            cv2.imwrite(save_path, demo)
        return file_name

    def show_dataset_runner(self,
                         num_workers,
                         out_dir,
                         pipeline=None):
        nr_records = len(self.img_ids)
        nr_per_thread = math.ceil(nr_records / num_workers)
        pbar = tqdm(total=nr_records)
        print("\n")
        result_queue = Queue(1000)
        procs = []
        pipeline = self._get_pipeline(pipeline)
        for i in range(num_workers):
            pstart = i * nr_per_thread
            pend = min(pstart + nr_per_thread, nr_records)
            # split_records = records[pstart:pend]
            proc = Process(
                target=self.show_dataset_worker,
                args=(pstart, out_dir, result_queue, pipeline))
            print("process:%d, start:%d, end:%d" % (i, pstart, pend))
            proc.start()
            procs.append(proc)

        all_results = []
        for i in range(nr_records):
            t = result_queue.get()
            all_results.append(t)
            pbar.update(1)

        for p in procs:
            p.join()

        img_size = (1080, 2640, 3)
        output_path = out_dir
        fps = 5
        output_video_path = osp.join(output_path, "demo.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(output_video_path, fourcc, fps,
                                      (img_size[1], img_size[0]))
        image_list = [f for f in os.listdir(output_path) if f.endswith(".png")]
        image_list = sorted(image_list)
        for i in tqdm(range(0, len(image_list))):
            frame = cv2.imread(os.path.join(output_path, image_list[i]))
            videoWriter.write(frame)
        videoWriter.release()

        import glob
        os.chdir(output_path)
        for file_name in glob.glob("*.{}".format("png")):
            os.remove(file_name)
        import random
        os.system("ffmpeg -i demo.avi demo_{}.mp4".format(
            int(random.random() * 1000)))

    def show_dataset_worker(self, start_index, out_dir, result_queue,
                         pipeline):
        for index, i in enumerate(self.img_ids):
            ret = self.process_dataset_img(start_index + index, out_dir,
                                       pipeline)
            result_queue.put_nowait(ret)

    def process_dataset_img(self, index, out_dir, pipeline):
        output_path = out_dir
        mmcv.mkdir_or_exist(output_path)
        img_size = (1080, 2640, 3)

        data_info = self.data_infos[index]
        img_path = data_info['file_name']
        file_name = osp.split(img_path)[-1].split('.')[0]
        img, img_metas = self._extract_data(index, pipeline,
                                            ['img', 'img_metas'])
        # need to transpose channel to first dim
        img = img.numpy().transpose(1, 2, 0)
        # print(self.get_ann_info(index))
        gt_bboxes = self.get_ann_info(index)['gt_bboxes_3d']
        gt_bbox2d = torch.tensor(self.get_ann_info(index)['bboxes'])
        # print(gt_bboxes)
        # print(gt_bbox2d)
   
        pred_img = show_multi_modality_result_faw(
            img,
            None,
            gt_bboxes,
            gt_bbox2d,
            img_metas['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=False,
            thickness=2,
            return_flag=True)

        ### bev test
        gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
        bev_img = None
        bev_img = self.show_bev(bev_img, None, gt_bev_corners)

        ### img
        demo = np.zeros(img_size, np.uint8)
        bev_img = cv2.resize(bev_img, (720, 1080))
        demo[:1080, 1920:2640] = bev_img

        pred_img = self.show_bbox2d(pred_img, gt_bbox2d)
        pred_img = cv2.resize(pred_img, (1920, 1080))
        demo[:1080, :1920] = pred_img

        save_path = osp.join(output_path, f'{file_name}.png')
        if img is not None:
            cv2.imwrite(save_path, demo)
        return file_name

    def show_dataset(self, out_dir, video_name, pipeline=None, 
                    sample_rate=1, show_3d=True, show_2d=True,
                    show_labels=True, show_scores=False):
        """Dataset visualization.

        Args:
            out_dir (str): Output directory of visualization result.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'

        # init video
        video_writer = self.init_videos(out_dir,video_name=video_name)
        # sample_rate = 100

        pipeline = self._get_pipeline(pipeline)

        for i, data_info in tqdm(list(enumerate(self.data_infos))):

            if i % sample_rate != 0:
                continue

            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]

            img, img_metas = self._extract_data(i, pipeline,
                                                ['img', 'img_metas'])
            # need to transpose channel to first dim
            img = img.numpy().transpose(1, 2, 0)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bbox2d = torch.tensor(self.get_ann_info(i)['bboxes']).numpy()
            depths = self.get_ann_info(i)['depths']
            if show_labels:
                gt_labels = self.get_ann_info(i)['labels']
            else:
                gt_labels = None
            
            # import pdb;pdb.set_trace()
            pred_img=img.copy()

            if show_3d:
                if len(depths[np.where(depths!=-1)[0]]):
                    pred_img = show_mono_result_faw(
                        pred_img,
                        gt_bboxes,
                        None,
                        None,
                        gt_labels,
                        None,
                        None,
                        img_metas['cam2img'],
                        out_dir,
                        file_name,
                        box_mode='camera',
                        thickness=2)
            if show_2d:
                pred_img = self.show_bbox2d(pred_img, gt_bbox2d, color=(61, 102, 255))

            
            ### bev test
            gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
            bev_img = None
            bev_img = self.show_bev(bev_img, None, gt_bev_corners)

            cv2.putText(pred_img, 'frame_id: {}'.format(i), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # img_path = '/'.join(img_path.split('/')[-2:])
            cv2.putText(pred_img, img_path, (50, 150), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)
            
            ### img
            img_size = (1080, 2640, 3)
            demo = np.zeros(img_size, np.uint8)
            bev_img = cv2.resize(bev_img, (720,1080))
            demo[:1080, 1920:2640] = bev_img

            pred_img = cv2.resize(pred_img, (1920, 1080))
            demo[:1080, :1920] = pred_img
            # import pdb;pdb.set_trace()

            demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
            video_writer.writeFrame(demo)
        video_writer.close()
    
    def show_results(self, results, out_dir, show=False, pipeline=None, 
                    sample_rate=1, matched_results=None, matched_gts=None,
                    show_3d=True, show_2d=False, show_labels=True, show_scores=False, only_show_fn_fp=False,
                    keep_all_frame=False):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'

        # init video
        os.makedirs(out_dir, exist_ok=True)
        video_writer = self.init_videos(out_dir)
        # sample_rate = 100

        pipeline = self._get_pipeline(pipeline) # eval pipeline
        # bar = mmcv.ProgressBar(len(results))
        # for i, result in tqdm(list(enumerate(results['img_bbox']))):
        for i, result in tqdm(list(enumerate(results))): # 遍历每张图

            # only bad case: fn
            # import pdb;pdb.set_trace()
            if matched_gts is not None:
                matched_gt = np.array(matched_gts[i])[np.where(np.array(matched_gts[i])!= -1)[0]]
            # bad cases: fp
            if matched_results is not None:
                matched_result = np.array(matched_results[i])[np.where(np.array(matched_results[i])!= -1)[0]]

            if not keep_all_frame:
                if matched_gts is not None:
                    if np.sum(matched_gt) == len(matched_gt): # 如果没有漏检
                        if matched_results is not None:
                            if np.sum(matched_result) == len(matched_result): # 如果没有误检
                                continue
            
            fn_nums = len(np.array(matched_gts[i])[np.where(np.array(matched_gts[i]) == 0)[0]]) # 漏检个数
            fp_nums = len(np.array(matched_results[i])[np.where(np.array(matched_results[i]) == 0)[0]]) # 误检个数
            if not keep_all_frame:
                if fn_nums == 0 and fp_nums == 0 and only_show_fn_fp:
                    continue
                if i % sample_rate != 0 and not only_show_fn_fp: # 不满足采样率就continue
                    continue
            # 处理存在误检或者漏检的帧
            if 'img_bbox' in result.keys():
                result = result['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]

            # 获取原图及其img_meta
            img, img_metas = self._extract_data(i, pipeline,
                                                ['img', 'img_metas'])
            # need to transpose channel to first dim
            img = img.numpy().transpose(1, 2, 0)
            if self.use_vis_quant_mode:  # false
                gt_results = self.quant_gt[i]
                gt_bboxes = gt_results['img_bbox']['boxes_3d']
                gt_bbox2d = gt_results['img_bbox']['bboxes_2d'].numpy()
                if show_labels:
                    gt_labels = gt_results['img_bbox']['labels_3d'].numpy()
                    pred_labels = result['labels_3d']
                else:
                    gt_labels = None
                    pred_labels = None
            else: # true
                gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'] # 标签3d框
                gt_bbox2d = torch.tensor(self.get_ann_info(i)['bboxes']).numpy() # 标签2d框
                if show_labels: # true
                    gt_labels = self.get_ann_info(i)['labels'] # 标签类别
                    pred_labels = result['labels_3d'] # 预测类别
                else:
                    gt_labels = None
                    pred_labels = None

            pred_bboxes = result['boxes_3d'] # 预测3d框
            pred_bbox2d = result['bboxes_2d'] # 预测2d框
            if show_scores: # false
                pred_scores = result['scores_3d']
            else:
                pred_scores = None
            pred_img=img.copy() # 拷贝图像用于画框

            # 标签3d框和标签2d框画到图像pred_img内
            if show_3d:
                pred_img = show_mono_result_faw(
                    pred_img,
                    gt_bboxes,
                    None,
                    None,
                    gt_labels,
                    None,
                    None,
                    img_metas['cam2img'],
                    out_dir,
                    file_name,
                    box_mode='camera',
                    thickness=2)
            if show_2d:
                pred_img = self.show_bbox2d(pred_img, gt_bbox2d, color=(61, 102, 255), labels=gt_labels)
            
            ### bev test ，生成bev_img，并且把标签3d框和预测3d框的bev box 分别画在bev_img上
            gt_bev_corners = self.bev_to_corners(gt_bboxes.bev) # 拿出bev 2d box，然后乘以一个绕z轴旋转的矩阵
            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)
            bev_img = None
            bev_img = self.show_bev(bev_img, pred_bev_corners, gt_bev_corners)

            # 把图片id和路径写到图像pred_img上
            cv2.putText(pred_img, 'frame_id: {}'.format(i), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            img_path = '/'.join(img_path.split('/')[-2:])
            cv2.putText(pred_img, img_path, (50, 150), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)

            # 把 FN（漏检）画到 pred_img 和 bev_img 上
            if matched_gts is not None:
                import copy
                # draw FN: blue 漏检
                gt_bboxes_fn = copy.deepcopy(gt_bboxes)
                gt_bboxes_fn.tensor = gt_bboxes_fn.tensor[np.where(np.array(matched_gts[i]) == 0)[0]]
                gt_bbox2d_fn = gt_bbox2d[np.where(np.array(matched_gts[i]) == 0)[0]]
                if show_labels:
                    gt_labels_fn = gt_labels[np.where(np.array(matched_gts[i]) == 0)[0]]
                else:
                    gt_labels_fn = None
                gt_bbox_color=(255, 0, 0)
                if show_3d:
                    pred_img = show_mono_result_faw(
                        pred_img,
                        gt_bboxes_fn,
                        None,
                        None,
                        gt_labels_fn,
                        None,
                        None,
                        img_metas['cam2img'],
                        out_dir,
                        file_name,
                        box_mode='camera',
                        thickness=2, gt_bbox_color=gt_bbox_color)
                #2d bbox
                if show_2d:
                    pred_img = self.show_bbox2d(pred_img, gt_bbox2d_fn, color=gt_bbox_color, labels=gt_labels_fn)
                gt_bev_corners = self.bev_to_corners(gt_bboxes_fn.bev)
                bev_img = self.show_bev(bev_img, None, gt_bev_corners, gt_color=gt_bbox_color)
            
            # 把 TP（正确检出或者说被匹配的预测，matched_results为1）、FP（误检，matched_results为0）、ignore（matched_results为-1） 画到 pred_img 和 bev_img 上
            if matched_results is not None:
                import copy
                # draw TP: green 正确检出
                pred_bboxes_tp = copy.deepcopy(pred_bboxes)
                pred_bboxes_tp.tensor = pred_bboxes_tp.tensor[np.where(np.array(matched_results[i]) == 1.0)[0]]
                pred_bbox2d_tp = copy.deepcopy(pred_bbox2d)
                pred_bbox2d_tp = pred_bbox2d_tp[np.where(np.array(matched_results[i]) == 1.0)[0]]
                if show_labels:
                    pred_labels_tp = copy.deepcopy(pred_labels)
                    pred_labels_tp = np.array(pred_labels_tp[np.where(np.array(matched_results[i]) == 1.0)[0]])
                else:
                    pred_labels_tp = None
                if show_scores:
                    pred_scores_tp = copy.deepcopy(pred_scores)
                    pred_scores_tp = np.array(pred_scores_tp[np.where(np.array(matched_results[i]) == 1.0)[0]])
                else:
                    pred_scores_tp = None
                pred_bbox_color=(0, 255, 0)
                if show_3d:
                    pred_img = show_mono_result_faw(
                        pred_img,
                        None,
                        pred_bboxes_tp,
                        None,
                        None,
                        pred_labels_tp,
                        pred_scores_tp,
                        img_metas['cam2img'],
                        out_dir,
                        file_name,
                        box_mode='camera',
                        thickness=2, pred_bbox_color=pred_bbox_color)
                if show_2d:
                    pred_img = self.show_bbox2d(pred_img, pred_bbox2d_tp, color=pred_bbox_color, labels=pred_labels_tp, bboxes3d=pred_bboxes_tp)
                pred_bev_corners = self.bev_to_corners(pred_bboxes_tp.bev)
                bev_img = self.show_bev(bev_img, pred_bev_corners, None, pred_color=pred_bbox_color)

                # draw FP: red 误检
                pred_bboxes_fp = copy.deepcopy(pred_bboxes)
                pred_bboxes_fp.tensor = pred_bboxes_fp.tensor[np.where(np.array(matched_results[i]) == 0)[0]]
                # only draw fp
                if not keep_all_frame:
                    if pred_bboxes_fp.tensor.shape[0] == 0:
                        continue
                pred_bbox2d_fp = copy.deepcopy(pred_bbox2d)
                pred_bbox2d_fp = pred_bbox2d_fp[np.where(np.array(matched_results[i]) == 0)[0]]
                if show_labels:
                    pred_labels_fp = copy.deepcopy(pred_labels)
                    pred_labels_fp = np.array(pred_labels_fp[np.where(np.array(matched_results[i]) == 0)[0]])
                else:
                    pred_labels_fp = None
                if show_scores:
                    pred_scores_fp = copy.deepcopy(pred_scores)
                    pred_scores_fp = np.array(pred_scores_fp[np.where(np.array(matched_results[i]) == 0)[0]])
                else:
                    pred_scores_fp = None
                pred_bbox_color=(0, 0, 255)
                if show_3d:
                    pred_img = show_mono_result_faw(
                        pred_img,
                        None,
                        pred_bboxes_fp,
                        None,
                        None,
                        pred_labels_fp,
                        pred_scores_fp,
                        img_metas['cam2img'],
                        out_dir,
                        file_name,
                        box_mode='camera',
                        thickness=2, pred_bbox_color=pred_bbox_color)
                if show_2d:
                    pred_img = self.show_bbox2d(pred_img, pred_bbox2d_fp, color=pred_bbox_color, labels=pred_labels_fp, bboxes3d=pred_bboxes_fp)
                pred_bev_corners = self.bev_to_corners(pred_bboxes_fp.bev)
                bev_img = self.show_bev(bev_img, pred_bev_corners, None, pred_color=pred_bbox_color)

                # draw ignore: white
                pred_bboxes_ignore = copy.deepcopy(pred_bboxes)
                pred_bboxes_ignore.tensor = pred_bboxes_ignore.tensor[np.where(np.array(matched_results[i]) == -1)[0]]
                pred_bbox2d_ignore = copy.deepcopy(pred_bbox2d)
                pred_bbox2d_ignore = pred_bbox2d_ignore[np.where(np.array(matched_results[i]) == -1)[0]]
                if show_labels:
                    pred_labels_ignore = copy.deepcopy(pred_labels)
                    pred_labels_ignore = np.array(pred_labels_ignore[np.where(np.array(matched_results[i]) == -1)[0]])
                else:
                    pred_labels_ignore = None
                if show_scores:
                    pred_scores_ignore = copy.deepcopy(pred_scores)
                    pred_scores_ignore = np.array(pred_scores_ignore[np.where(np.array(matched_results[i]) == -1)[0]])
                else:
                    pred_scores_ignore = None
                pred_bbox_color=(255, 255, 255)
                if show_3d:
                    pred_img = show_mono_result_faw(
                        pred_img,
                        None,
                        pred_bboxes_ignore,
                        None,
                        None,
                        pred_labels_ignore,
                        pred_scores_ignore,
                        img_metas['cam2img'],
                        out_dir,
                        file_name,
                        box_mode='camera',
                        thickness=2, pred_bbox_color=pred_bbox_color)
                if show_2d:
                    pred_img = self.show_bbox2d(pred_img, pred_bbox2d_ignore, color=pred_bbox_color, labels=pred_labels_ignore, bboxes3d=pred_bboxes_ignore)
                pred_bev_corners = self.bev_to_corners(pred_bboxes_ignore.bev)
                bev_img = self.show_bev(bev_img, pred_bev_corners, None, pred_color=pred_bbox_color)

            # 如果预测的结果没有匹配，预测就全画在 pred_img 和 bev_img 上
            else:
                # draw : green
                pred_bbox_color=(0, 255, 0)
                if show_3d:
                    pred_img = show_mono_result_faw(
                        pred_img,
                        None,
                        pred_bboxes,
                        None,
                        pred_labels,
                        pred_scores,
                        None,
                        img_metas['cam2img'],
                        out_dir,
                        file_name,
                        box_mode='camera',
                        thickness=2, pred_bbox_color=pred_bbox_color)
                if show_2d:
                    pred_img = self.show_bbox2d(pred_img, pred_bbox2d, color=pred_bbox_color, labels=pred_labels, bboxes3d=pred_bboxes)
                pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)
                bev_img = self.show_bev(bev_img, pred_bev_corners, None, pred_color=pred_bbox_color)

            # 拼接 pred_img 和 bev_img，到 demo
            ### img
            img_size = (1080, 2640, 3)
            demo = np.zeros(img_size, np.uint8)
            bev_img = cv2.resize(bev_img, (720,1080))
            demo[:1080, 1920:2640] = bev_img

            pred_img = cv2.resize(pred_img, (1920, 1080))
            demo[:1080, :1920] = pred_img
            # import pdb;pdb.set_trace()
            
            save_path = osp.join(out_dir, f'{file_name}.png')
            # cv2.imwrite(save_path, demo)
            demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
            video_writer.writeFrame(demo)
        video_writer.close()

    def show_results_only(self, results, out_dir, show=False, pipeline=None, 
                    sample_rate=1, matched_results=None, matched_gts=None,
                    show_3d=True, show_2d=True, show_labels=True, show_scores=False):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'

        # init video
        os.makedirs(out_dir, exist_ok=True)
        video_writer = self.init_videos(out_dir)
        # sample_rate = 100

        pipeline = self._get_pipeline(pipeline)
        # bar = mmcv.ProgressBar(len(results))
        for i, result in tqdm(list(enumerate(results))):

            # only bad case: fn
            # import pdb;pdb.set_trace()
            if matched_gts is not None:
                matched_gt = np.array(matched_gts[i])[np.where(np.array(matched_gts[i])!= -1)[0]]
            # bad cases: fp
            if matched_results is not None:
                matched_result = np.array(matched_results[i])[np.where(np.array(matched_results[i])!= -1)[0]]

            if matched_gts is not None:
                if np.sum(matched_gt) == len(matched_gt):
                    if matched_results is not None:
                        if np.sum(matched_result) == len(matched_result):
                            continue


            if i % sample_rate != 0:
                continue

            if 'img_bbox' in result.keys():
                result = result['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]

            img, img_metas = self._extract_data(i, pipeline,
                                                ['img', 'img_metas'])
            # need to transpose channel to first dim
            img = img.numpy().transpose(1, 2, 0)
            # gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            # gt_bbox2d = torch.tensor(self.get_ann_info(i)['bboxes']).numpy()
            if show_labels:
                # gt_labels = self.get_ann_info(i)['labels']
                # pred_labels = result['labels_3d']
                pred_labels = result['attrs_3d'].numpy()
            else:
                # gt_labels = None
                pred_labels = None
            pred_bboxes = result['boxes_3d']
            pred_bbox2d = result['bboxes_2d']
            if show_scores:
                pred_scores = result['scores_3d']
            else:
                pred_scores = None
            pred_img=img.copy()

            if show_3d:
                pred_img = show_mono_result_faw(
                    pred_img,
                    None,
                    pred_bboxes,
                    None,
                    None,
                    pred_labels,
                    None,
                    img_metas['cam2img'],
                    out_dir,
                    file_name,
                    box_mode='camera',
                    thickness=2)
            if show_2d:
                pred_img = self.show_bbox2d(pred_img, pred_bbox2d, color=(255, 255, 255), labels=None)
            
            ### bev test
            # gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)
            bev_img = None
            bev_img = self.show_bev(bev_img, pred_bev_corners, None)

            cv2.putText(pred_img, 'frame_id: {}'.format(i), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            img_path = '/'.join(img_path.split('/')[-2:])
            cv2.putText(pred_img, img_path, (50, 100), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)
            
            ### img
            img_size = (1080, 2640, 3)
            demo = np.zeros(img_size, np.uint8)
            bev_img = cv2.resize(bev_img, (720,1080))
            demo[:1080, 1920:2640] = bev_img

            pred_img = cv2.resize(pred_img, (1920, 1080))
            demo[:1080, :1920] = pred_img
            # import pdb;pdb.set_trace()
            
            save_path = osp.join(out_dir, f'{file_name}.png')
            # cv2.imwrite(save_path, demo)
            demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
            video_writer.writeFrame(demo)
        video_writer.close()

    def init_videos(self, save_videos_path, video_name = 'internal_results.mp4'):
        save_video_name = os.path.join(save_videos_path, video_name)
        if os.path.exists(save_video_name):
            os.remove(save_video_name)
        video_fps = 10
        video_writer = skvideo.io.FFmpegWriter(
            save_video_name,
            inputdict={
                '-r': str(video_fps),
                '-s': '{}x{}'.format(int(2640), int(1080))
            },
            outputdict={
                '-r': str(video_fps),
                '-vcodec': 'libx264'
            })
        return video_writer

    def bboxes3d_corners_project(self, bboxes3d, cam2img):
        from mmdet3d.core.bbox import points_cam2img
        corners_3d = bboxes3d.corners
        num_bbox = corners_3d.shape[0]
        points_3d = corners_3d.reshape(-1, 3)
        if not isinstance(cam2img, torch.Tensor):
            cam2img = torch.from_numpy(np.array(cam2img))

        assert (cam2img.shape == torch.Size([3, 3])
                or cam2img.shape == torch.Size([4, 4]))
        cam2img = cam2img.float().cpu()

        # project to 2d to get image coords (uv)
        uv_origin = points_cam2img(points_3d, cam2img)
        uv_origin = (uv_origin - 1).round()
        imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()
        return imgfov_pts_2d

    def calculate_shelter(self, bbox1, bbox2, depth1, depth2):
        lx = max(bbox1[0], bbox2[0])
        ly = max(bbox1[1], bbox2[1])
        rx = min(bbox1[2], bbox2[2])
        ry = min(bbox1[3], bbox2[3])
        if lx > rx or ly > ry:
            return 0
        inter = (rx - lx + 1) * (ry - ly + 1)
        self_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        obj_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        if self_area + obj_area - inter <= 1e-4:
            return 0
        elif depth1 >= depth2:
            return inter / self_area
        else:
            return -(inter / obj_area)

    def judge_overlap(self, gt_2d_bboxes, depths, imgfov_pts_2d, img_height,
                      img_width):
        bbox_num = gt_2d_bboxes.shape[0]
        shelter_ratio_dict = {}
        for i in range(bbox_num):
            shelter_ratio_dict.setdefault(i, 0)
        # cal overlap ratio
        for i in range(bbox_num - 1):
            for j in range(i + 1, bbox_num):
                shelter = self.calculate_shelter(gt_2d_bboxes[i],
                                                 gt_2d_bboxes[j], depths[i],
                                                 depths[j])
                if shelter > 0:
                    shelter_ratio_dict[i] = max(shelter_ratio_dict[i], shelter)
                else:
                    shelter_ratio_dict[j] = max(shelter_ratio_dict[j],
                                                -shelter)

        # cal truncation ratio
        for idx, pts_2d in enumerate(imgfov_pts_2d):
            pts_lt = pts_2d.min(axis=0)
            pts_rb = pts_2d.max(axis=0)
            if pts_lt[0] < 0 or pts_lt[1] < 0 or pts_rb[
                    0] > img_width or pts_rb[1] > img_height:
                lx = max(pts_lt[0], 0)
                ly = max(pts_lt[1], 0)
                rx = min(pts_rb[0], img_width)
                ry = min(pts_rb[1], img_height)
                if lx > rx or ly > ry:
                    continue
                inter = (rx - lx + 1) * (ry - ly + 1)
                self_area = (pts_rb[0] - pts_lt[0] + 1) * (
                    pts_rb[1] - pts_lt[1] + 1)
                trun_ratio = 1 - inter / self_area
                if shelter_ratio_dict[idx] != 0:
                    shelter_ratio_dict[idx] *= (1 - trun_ratio)
                shelter_ratio_dict[idx] += trun_ratio
        return shelter_ratio_dict

    def save_shelter_ratio(self, save_path):
        print(len(self.data_infos))
        ann_shelter_ratio_dict = {}

        for idx, data_info in tqdm(enumerate(self.data_infos)):
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info)
            gt_2d_bboxes = ann['bboxes']
            gt_3d_bboxed = ann['gt_bboxes_3d']
            depths = ann['depths']
            cam2img = data_info['cam_intrinsic']
            if len(cam2img) <= 0:
                cam2img = np.eye(3, 3).tolist()
            imgfov_pts_2d = self.bboxes3d_corners_project(
                gt_3d_bboxed, cam2img)
            img_height, img_width = data_info['height'], data_info['width']
            shelter_ratio_dict = self.judge_overlap(gt_2d_bboxes, depths,
                                                    imgfov_pts_2d, img_height,
                                                    img_width)
            for idx, ann_id in enumerate(ann_ids):
                ann_shelter_ratio_dict[ann_id] = shelter_ratio_dict[idx]
        cates = [{
            'id': 0,
            'name': 'VEHICLE_CAR'
        }, {
            'id': 1,
            'name': 'BIKE_BICYCLE'
        }, {
            'id': 2,
            'name': 'PEDESTRIAN'
        }]
        crowd_cases_dict = {
            'images': [],
            'annotations': [],
            'categories': cates
        }
        print('start to save:')
        for data_info in tqdm(self.data_infos):
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            for ann_id in ann_ids:
                ann_info = self.coco.load_anns(ann_id)
                ann_info[0]['shelter_ratio'] = ann_shelter_ratio_dict[ann_id]
                crowd_cases_dict['annotations'].extend(ann_info)
            crowd_cases_dict['images'].extend([data_info])
        with open(save_path, 'w') as f:
            json.dump(crowd_cases_dict, f)

    def evaluate_pillar(self, results, save_path, **kwargs):
        self.save_path = save_path
        self.gts_label = []
    
        frame_id = 0
        for data_info in self.data_infos:
            img_id = data_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann = self._parse_ann_info(data_info, ann_info)
            gt_labels = ann['labels']
            self.gts_label.append(gt_labels)

        for class_label in self.CLASS_MAPPING:
            print('***************process:',class_label)
            class_save_path = os.path.join(self.save_path, class_label)
            if os.path.exists(class_save_path):
                shutil.rmtree(class_save_path)
            os.makedirs(class_save_path, exist_ok=True)

            predictions, gts, gts_ignore, preds_ignore = \
                self.convert_results_by_class(results, class_label)
            # run evaluation
            point_cloud_range = [0.0, -60, -5, 100, 60, 5]
            evaluator = NuScenesEval_faw(predictions,
                                 gts,
                                 'class x y z l w h r score',
                                 class_save_path, gts_ignore=gts_ignore,
                                 preds_ignore=preds_ignore,
                                 classes=[class_label], point_cloud_range=point_cloud_range)
            eval_results = evaluator.get_metric_results()
            mmcv.dump(eval_results, os.path.join(class_save_path,
                                             class_label+'_eval_results.json'))
        
        all_class_matched_results = {}
        all_class_matched_gt = {}
        eval_results = {}
        for class_label in self.CLASS_MAPPING:
            try:
                matched_results = mmcv.load(os.path.join(self.save_path, class_label,
                                                    class_label+'_eval_results.json'))
                eval_results[class_label] = matched_results[class_label]
                matched_gt = matched_results['matched_gt']
                matched_results = matched_results['matched_results']
                
            except:
                print(class_label,' not find!')
                matched_results = None
                matched_gt = None
                
            all_class_matched_results[class_label]=matched_results
            all_class_matched_gt[class_label]=matched_gt

        matched_results = []
        matched_gts = []

        img_bboxes = results['img_bbox']
        for i in range(len(img_bboxes)):
            matched_result=[]
            matched_gt = []
       
            for label in img_bboxes[i]['labels_3d'].tolist():
                matched_result.append(all_class_matched_results[self.CLASS_MAPPING[label]][i][0])
                all_class_matched_results[self.CLASS_MAPPING[label]][i] = all_class_matched_results[self.CLASS_MAPPING[label]][i][1:]
           
            for label in self.gts_label[i]:
                matched_gt.append(all_class_matched_gt[self.CLASS_MAPPING[label]][i][0])
                all_class_matched_gt[self.CLASS_MAPPING[label]][i] = all_class_matched_gt[self.CLASS_MAPPING[label]][i][1:]
         

            matched_results.append(matched_result)
            matched_gts.append(matched_gt)
        eval_results['matched_results'] = matched_results
        eval_results['matched_gt'] = matched_gts
        mmcv.dump(eval_results, os.path.join(self.save_path, 'eval_results.json'))
        
        # bad cases
        bad_cases = []
            
        for i in range(len(matched_gts)):
            matched_gt = np.array(matched_gts[i])[np.where(np.array(matched_gts[i])!= -1)[0]]
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            if np.sum(matched_gt) != len(matched_gt):
                bad_cases.append(img_path)
        bad_cases_dict={}
        bad_cases_dict['bad_cases_file_name']=bad_cases
        bad_cases_save_path = os.path.join(self.save_path, 'bad_cases.json')
        mmcv.dump(bad_cases_dict, bad_cases_save_path)       
        print('bad cases num:',len(bad_cases))
        return
