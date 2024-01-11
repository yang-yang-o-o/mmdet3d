# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import cv2
import torch
import mmcv
import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
# from mmdet.datasets.builder import PIPELINES
# from pillar.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
# from ..builder import OBJECTSAMPLERS
# from .data_augment_utils import noise_per_object_v3_
# from ...utils.pit_transform import PIT_module
from ..builder import PIPELINES



@PIPELINES.register_module()
class Mono3DResize_faw:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False,
                 scale_depth_by_focal_lengths_factor=500,
                 scale_cam=True):
        self.scale_depth_by_focal_lengths_factor = scale_depth_by_focal_lengths_factor
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

        self.scale_cam = scale_cam

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            # print(img)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_cam2img(self, results):
        # factor: 1.2, 2.4 (w, h)
        cam2img = np.array(results['cam2img'], dtype=np.float32)
        cam2img[0, :] = cam2img[0, :] * results['scale_factor'][0]
        cam2img[1, :] = cam2img[1, :] * results['scale_factor'][1]
        results['cam2img'] = cam2img.tolist()

    def _resize_center2d(self, results):
        if 'centers2d' in results.keys():
            results['centers2d'] = results['centers2d'] * results[
                'scale_factor'][:2]

    def _resize_depth(self, results):
        cam2img = np.array(results['cam2img'], dtype=np.float32)
        cam2img_inv = np.linalg.inv(cam2img)
        cam2img_inv = torch.tensor(cam2img_inv, dtype=torch.float32)
        pixel_size = torch.norm(
            torch.stack([cam2img_inv[0, 0], cam2img_inv[1, 1]], dim=-1),
            dim=-1)
        depth_factors = 1 / (
            pixel_size * self.scale_depth_by_focal_lengths_factor)
        results['depth_factors'] = depth_factors.tolist()
        
    def _resize_mask(self, results):
        if 'gt_masks' in results.keys():
            results['gt_masks'], _, _ = mmcv.imresize(
                results['gt_masks'],
                results['scale'],
                return_scale=True,
                backend=self.backend)
            results['gt_masks'] = results['gt_masks']

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        if self.scale_cam:
            self._resize_cam2img(results)
        self._resize_center2d(results)
        self._resize_depth(results)
        self._resize_mask(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class PITMonoTransform_faw(object):
    """Apply PIT Transform for Input.
        Note: Insert after M.S.
    """

    def __init__(self, is_train=True):
        self.is_train = is_train

    def apply_pit_transform(self, results):
        # img
        height, width = results['img'].shape[:2]
        cam2img = results['ori_cam2img']
        fx = cam2img[0][0]
        fy = cam2img[1][1]
        fovx = 2 * np.arctan2(width, 2 * fx)
        fovy = 2 * np.arctan2(height, 2 * fy)
        proj = PIT_module(width, height, fovx=fovx, fovy=fovy, device='cpu')
        img = results['img']
        img2 = np.ascontiguousarray(img)
        t = torch.from_numpy(img2).permute(2, 0, 1)[None, ...]
        t_new = proj.pit(t, interpolation=2, reverse=False)
        im_new = t_new[0, ...].permute(1, 2, 0)
        im_new = im_new.numpy().astype('uint8')
        results['img'] = im_new

        # annotations
        new_height, new_width = im_new.shape[:2]
        results['img_info']['height'] = new_height
        results['img_info']['width'] = new_width
        results['ori_shape'] = im_new.shape
        # 2d bboxes transform
        if self.is_train:
            gt_bboxes = results['gt_bboxes']
            new_gt_bboxes = []
            for gt_box in gt_bboxes:
                xmin, ymin, xmax, ymax = gt_box
                xmin, ymin = proj.coord_plain_to_arc_scalar(xmin, ymin)
                xmax, ymax = proj.coord_plain_to_arc_scalar(xmax, ymax)
                if xmax > new_width:
                    xmax = new_width
                if ymax > new_height:
                    ymax = new_height
                new_gt_bboxes.append([xmin, ymin, xmax, ymax])
            new_gt_bboxes = np.array(new_gt_bboxes, dtype=np.float32)
            results['gt_bboxes'] = new_gt_bboxes

        # centers2d transform
        if self.is_train:
            centers2d = results['centers2d']
            new_centers2d = []
            for center2d in centers2d:
                center2d_new_x, center2d_new_y = proj.coord_plain_to_arc_scalar(
                    center2d[0], center2d[1])
                # if center2d_new_x > new_width:
                #     center2d_new_x = new_width
                # if center2d_new_y > new_height:
                #     center2d_new_y = new_height
                new_centers2d.append([center2d_new_x, center2d_new_y])
            new_centers2d = np.array(new_centers2d, dtype=np.float32)
            results['centers2d'] = new_centers2d

    def __call__(self, results):
        """Call function to apply pit transform.
        """
        # dict_keys(['img_info', 'ann_info', 'img_prefix', 'seg_prefix',
        # 'proposal_file', 'img_fields', 'bbox3d_fields', 'pts_mask_fields',
        # 'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields', 'box_type_3d',
        # 'box_mode_3d', 'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape',
        # 'cam2img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_3d',
        # 'centers2d', 'depths', 'gt_labels_3d', 'scale', 'scale_idx',
        # 'pad_shape', 'scale_factor', 'keep_ratio', 'depth_factors'])

        if 'ori_cam2img' not in results.keys():
            results['ori_cam2img'] = results['cam2img']
        self.apply_pit_transform(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

