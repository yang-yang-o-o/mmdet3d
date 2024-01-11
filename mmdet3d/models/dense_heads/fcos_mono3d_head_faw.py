# Copyright (c) OpenMMLab. All rights reserved.
from logging import warning
import os
import mmcv
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init, ConvModule
from mmcv.runner import force_fp32
from mmdet3d.core import limit_period, points_img2cam, xywhr2xyxyr
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.core.bbox.builder import build_bbox_coder

# from ...utils.pit_transform import PIT_module
from ...core import box3d_multiclass_nms
from .anchor_free_mono3d_head import AnchorFreeMono3DHead
from ..builder import HEADS

INF = 1e8


@HEADS.register_module()
class FCOSMono3DHead_faw(AnchorFreeMono3DHead):
    """Anchor-free head used in FCOS3D.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]], optional): Regress range of multiple
            level points.
        center_sampling (bool, optional): If true, use center sampling. Default: True.
        center_sample_radius (float, optional): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool, optional): If true, normalize the regression targets
            with FPN strides. Default: True.
        centerness_on_reg (bool, optional): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: True.
        centerness_alpha (int, optional): Parameter used to adjust the intensity
            attenuation from the center to the periphery. Default: 2.5.
        loss_cls (dict, optional): Config of classification loss.
        loss_bbox (dict, optional): Config of localization loss.
        loss_dir (dict, optional): Config of direction classification loss.
        loss_attr (dict, optional): Config of attribute classification loss.
        loss_centerness (dict, optional): Config of centerness loss.
        norm_cfg (dict, optional): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        centerness_branch (tuple[int], optional): Channels for centerness branch.
            Default: (64, ).
    """  # noqa: E501

    def __init__(self,
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                                 (384, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 center_sample_radius_for_large_object=1.5,
                 norm_on_bbox=True,
                 centerness_on_reg=True,
                 centerness_alpha=2.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_dir=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_attr=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox2d=dict(type='IoULoss', loss_weight=1.0),
                 loss_homography=None,
                 bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 centerness_branch=(64, ),
                 bbox2d_branch=(128, 64),
                 init_cfg=None,
                 rescale_depth=False,
                 use_bbox2d_reg_feat=True,
                 is_deploy=False,
                 centerness_use_bbox2d_reg_feat=True,
                 is_pit_transform=False,
                 vru_dir_weight=1.0,
                 debug=False,
                 infer_use_onnx=False,
                 global_yaw=False,
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.center_sample_radius_for_large_object = center_sample_radius_for_large_object
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha
        self.centerness_branch = centerness_branch
        self.bbox2d_branch = bbox2d_branch
        self.rescale_depth = rescale_depth
        self.is_deploy = is_deploy
        self.is_pit_transform = is_pit_transform
        self.use_bbox2d_reg_feat = use_bbox2d_reg_feat
        self.centerness_use_bbox2d_reg_feat = centerness_use_bbox2d_reg_feat
        self.vru_dir_weight = vru_dir_weight
        self.debug = debug
        self.infer_use_onnx = infer_use_onnx

        super().__init__(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = HEADS.build(loss_centerness)
        self.loss_bbox2d = HEADS.build(loss_bbox2d)
        self.loss_homography = None
        if loss_homography is not None:
            self.loss_homography = HEADS.build(loss_homography)
            self.max_iou_assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5)
        bbox_coder['code_size'] = self.bbox_code_size
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.bbox2d_coder = build_bbox_coder(
            dict(type='DistancePointBBoxCoder_faw'))
        self.global_yaw = global_yaw

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3
        if self.pred_bbox2d:
            self.conv_bbox2d_prev = self._init_branch(
                conv_channels=self.bbox2d_branch,
                conv_strides=(1, ) * len(self.bbox2d_branch))
            self.conv_bbox2d = nn.Conv2d(self.bbox2d_branch[-1], 4, 1)
            self.scale_dim = 4  # only for offset, depth and size regression, bbox2d
            if self.use_bbox2d_reg_feat:
                self._init_bbox2d_reg_convs()

        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])

    def _init_bbox2d_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.bbox2d_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.bbox2d_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized init_weights because the default
        init of DCN triggered by the init_cfg will init conv_offset.weight,
        which mistakenly affects the training stability.
        """
        super().init_weights()
        for m in self.conv_centerness_prev:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        if self.pred_bbox2d:
            for n in self.conv_bbox2d_prev:
                if isinstance(n.conv, nn.Conv2d):
                    normal_init(n.conv, std=0.01)
            normal_init(self.conv_bbox2d, std=0.01)
            if self.use_bbox2d_reg_feat:
                for m in self.bbox2d_reg_convs:
                    if isinstance(m.conv, nn.Conv2d):
                        normal_init(m.conv, std=0.01)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_3d (list[Tensor]): 3D ground truth bboxes of the image,
                shape (num_gts, self.bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D ground truth labels of each box,
                shape (num_gts,).
            centers2d (list[Tensor]): Projected 3D center of each box,
                shape (num_gts, 2).
            depths (list[Tensor]): Depth of projected 3D center of each box,
                shape (num_gts,).
            attr_labels (list[Tensor]): Attribute labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        # len(6): cls_scores, bbox_preds, dir_cls_preds, attr_preds, centernesses, bbox2d_preds,
        # each is len(5) list: 
        # torch.Size([4, 3, 112, 200]), torch.Size([4, 3, 56, 100]), torch.Size([4, 3, 28, 50]), torch.Size([4, 3, 14, 25]), torch.Size([4, 3, 7, 13])
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_bboxes_3d, centers2d, depths,
                                  attr_labels, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                                  gt_labels_3d, centers2d, depths, attr_labels,
                                  img_metas)
        losses = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list




    def forward(self, feats, img_metas):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2).
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        if self.is_deploy:
            return multi_apply(self.forward_single, feats, self.scales,
                               self.strides)
        if self.rescale_depth:
            depth_factors = [
                img_metas[idx]['depth_factors']
                for idx in range(len(img_metas))
            ]
            depth_factors = torch.tensor(depth_factors, device=feats[0].device)
            depth_factors_list = [depth_factors] * len(self.scales)
            return multi_apply(self.forward_single, feats, self.scales,
                               self.strides, depth_factors_list)[:6]

        # Note: we use [:6] to filter feats and only return predictions
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)[:6]

    def forward_single(self, x, scale, stride, depth_factors_list=None):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox and direction class
                predictions, centerness predictions of input feature maps.
        """
        # torch.Size([4, 3, 112, 200]); torch.Size([4, 7, 112, 200]); torch.Size([4, 2, 112, 200]); 
        # None; torch.Size([4, 256, 112, 200]); torch.Size([4, 256, 112, 200])
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = \
            super().forward_single(x)

        if self.use_bbox2d_reg_feat: # True
            bbox2d_reg_feat = x
            for bbox2d_reg_layer in self.bbox2d_reg_convs:
                bbox2d_reg_feat = bbox2d_reg_layer(bbox2d_reg_feat)
        else:
            bbox2d_reg_feat = reg_feat.clone()

        if self.centerness_on_reg: # True
            # use bbox 2d feature to regress 2d
            if self.centerness_use_bbox2d_reg_feat: # True
                clone_reg_feat = bbox2d_reg_feat.clone()
            else:    
                clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)

        if self.is_deploy:
            bbox2d_pred = None
            for conv_bbox2d_prev_layer in self.conv_bbox2d_prev:
                bbox2d_reg_feat = conv_bbox2d_prev_layer(bbox2d_reg_feat)
            bbox2d_pred = self.conv_bbox2d(bbox2d_reg_feat)
            return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, bbox2d_pred, scale

        bbox2d_pred = None
        if self.pred_bbox2d:
            for conv_bbox2d_prev_layer in self.conv_bbox2d_prev:
                bbox2d_reg_feat = conv_bbox2d_prev_layer(bbox2d_reg_feat)
            bbox2d_pred = self.conv_bbox2d(bbox2d_reg_feat)
        
        if self.debug:
            outputs= {}
            # cls_num = cls_score.shape[1]
            # bbox_num = bbox_pred.shape[1]
            # bbox2d_num = bbox2d_pred.shape[1]
            fpns={8:0,16:1,32:2,64:3,128:4}   
            # cls score
            # clvl_cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            #     -1, cls_num).sigmoid()
            # yaw
            # clvl_dir_cls_pred = dir_cls_pred.permute(0, 2, 3,
            #                                             1).reshape(-1, 2)
            # centerness
            # clvl_centerness = centerness.permute(0, 2, 3,
            #                                         1).reshape(-1,
            #                                                 1).sigmoid()
            # bbox 3d
            # clvl_bbox_pred = bbox_pred.permute(0, 2, 3,
            #                                     1).reshape(-1, bbox_num)
            # bbox 2d
            # clvl_bbox2d_pred = bbox2d_pred.permute(0, 2, 3, 1).reshape(
            #     -1, bbox2d_num)
            
            
            # similarity calculation
            # # [B*H*W, 7]
            # outputs['fpn{}.bbox_pred'.format(fpns[stride])] = bbox_pred.permute(0, 2, 3, 1)
            # # [B*H*W, 4]
            # outputs['fpn{}.bbox2d_pred'.format(fpns[stride])] = bbox2d_pred.permute(0, 2, 3, 1)
            # # [B*H*W, 2]
            # outputs['fpn{}.dir_cls_pred'.format(fpns[stride])] = dir_cls_pred.permute(0, 2, 3, 1)
            # # [B*H*W, 1]
            # outputs['fpn{}.centerness'.format(fpns[stride])] = centerness.permute(0, 2, 3, 1)
            # # [B*H*W, 4]
            # outputs['fpn{}.cls_score'.format(fpns[stride])] = cls_score.permute(0, 2, 3, 1)
            # out_path = '/mnt/lustre/share_data/chenxuepan/workspace/Hozon/other/hozon_deploy_align/torch_output/'
            # mmcv.dump(outputs, out_path+'fpn{}net_out.pkl'.format(fpns[stride]))
            
            # dump head output to sdk
            out_path = '/mnt/lustre/share_data/chenxuepan/workspace/Hozon/other/hozon_deploy_align/torch_output/net_out.json'
            if os.path.exists(out_path):
                outputs = mmcv.load(out_path)
            # [B*H*W, 13]
            outputs['fpn{}.bbox_pred'.format(fpns[stride])] = torch.cat([bbox_pred, bbox2d_pred, dir_cls_pred], dim=1).cpu().numpy().tolist()
            # [B*H*W, 4]
            # outputs['fpn{}.bbox2d_pred'.format(fpns[stride])] = bbox2d_pred
            # [B*H*W, 2]
            # outputs['fpn{}.dir_cls_pred'.format(fpns[stride])] = dir_cls_pred
            # [B*H*W, 1]
            outputs['fpn{}.centerness'.format(fpns[stride])] = centerness.cpu().numpy().tolist()
            # [B*H*W, 4]
            outputs['fpn{}.cls_score'.format(fpns[stride])] = cls_score.cpu().numpy().tolist()
            mmcv.dump(outputs, out_path.format(fpns[stride]))
        
        if self.pred_bbox2d:
            # scale the bbox_pred of different level
            # float to avoid overflow when enabling FP16
            scale_bbox2d = scale[3]
            bbox2d_pred = scale_bbox2d(bbox2d_pred).float()
            if self.norm_on_bbox:
                # bbox_pred needed for gradient computation has been modified
                # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
                # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
                bbox2d_pred = bbox2d_pred.clamp(min=0)
                if not self.training:
                    bbox2d_pred *= stride
            else:
                bbox2d_pred = bbox2d_pred.exp()

        bbox_pred = self.bbox_coder.decode(bbox_pred, scale, stride,
                                           self.training, cls_score,
                                           depth_factors_list)

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                bbox2d_pred, cls_feat, reg_feat

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(reg_targets,
                             dir_offset=0,
                             dir_limit_offset=0.0,
                             num_bins=2,
                             one_hot=True):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int, optional): Direction offset. Default to 0.
            dir_limit_offset (float, optional): Offset to set the direction
                range. Default to 0.0.
            num_bins (int, optional): Number of bins to divide 2*PI.
                Default to 2.
            one_hot (bool, optional): Whether to encode as one hot.
                Default to True.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset,
                                  2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot /
                                      (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device)
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
                  'centernesses', 'bbox2d_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             attr_preds,
             centernesses,
             bbox2d_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): 3D boxes ground truth with shape of
                (num_gts, code_size).
            gt_labels_3d (list[Tensor]): same as gt_labels
            centers2d (list[Tensor]): 2D centers on the image with shape of
                (num_gts, 2).
            depths (list[Tensor]): Depth ground truth with shape of
                (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(
            attr_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        # generate points and weights(pit)
        if gt_masks is not None:
            all_level_points, all_level_weights = self.get_points(
                featmap_sizes,
                bbox_preds[0].dtype,
                bbox_preds[0].device,
                gt_masks=gt_masks)
        else:
            all_level_weights = None
            all_level_points = self.get_points(
                featmap_sizes,
                bbox_preds[0].dtype,
                bbox_preds[0].device,
                gt_masks=None)
       
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets, bbox2d_targets, targets_3d_mask = \
            self.get_targets(
                all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d,
                gt_labels_3d, centers2d, depths, attr_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, dir_cls_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        if gt_masks is not None:
            flatten_weights = torch.cat(all_level_weights)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)


        flatten_targets_3d_mask = torch.cat(targets_3d_mask)
        pos_inds_3d = (
            (flatten_labels_3d >= 0) &
            (flatten_labels_3d < bg_class_ind)) * flatten_targets_3d_mask
        pos_inds_3d = pos_inds_3d.nonzero().reshape(-1)
        num_pos_3d = len(pos_inds_3d)

        pos_inds_vru = ((flatten_labels_3d > 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        vru_idx_in_pos_inds = []
        # car_idx_in_pos_inds = []
        for idx, val in enumerate(pos_inds_3d):
            if val in pos_inds_vru:
                vru_idx_in_pos_inds.append(idx)
            # else:
            #     car_idx_in_pos_inds.append(idx)
        vru_idx_in_pos_inds = pos_inds_3d.new_tensor(vru_idx_in_pos_inds)
        # car_idx_in_pos_inds = pos_inds.new_tensor(car_idx_in_pos_inds)
        
        if gt_masks is not None:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels_3d,
                weight=flatten_weights,
                avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores.cpu(),
                flatten_labels_3d.cpu(),
                avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds_3d]
        pos_centerness = flatten_centerness[pos_inds]
        
        if self.use_direction_classifier:
            flatten_dir_cls_preds = [
                dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
                for dir_cls_pred in dir_cls_preds
            ]
            flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
            pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds_3d]

        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            flatten_attr_targets = torch.cat(attr_targets)
            pos_attr_preds = flatten_attr_preds[pos_inds_3d]

        if self.pred_bbox2d:
            flatten_bbox2d_preds = [
                bbox2d_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                for bbox2d_pred in bbox2d_preds
            ]
            flatten_bbox2d_preds = torch.cat(flatten_bbox2d_preds)
            flatten_bbox2d_targets = torch.cat(bbox2d_targets)
            pos_bbox2d_preds = flatten_bbox2d_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds_3d]
            pos_centerness_targets = flatten_centerness_targets[pos_inds_3d]
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds_3d]
            if self.pred_bbox2d:
                pos_bbox2d_targets = flatten_bbox2d_targets[pos_inds]
                pos_centerness_2d_targets = self.centerness_target(
                    pos_bbox2d_targets)
                # centerness weighted iou loss
                centerness_denorm = max(
                    reduce_mean(pos_centerness_2d_targets.sum().detach()),
                    1e-6)
            
            bbox_weights = pos_centerness_2d_targets.new_ones(
                len(pos_centerness_2d_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_2d_targets.new_ones(
                pos_centerness_2d_targets.shape)
            
            # centerness
            if gt_masks is not None:
                loss_centerness = self.loss_centerness(
                    pos_centerness,
                    pos_centerness_2d_targets,
                    weight=flatten_weights[pos_inds],
                    avg_factor=equal_weights.sum())
            else:
                loss_centerness = self.loss_centerness(
                    pos_centerness, pos_centerness_2d_targets)
            
            loss_bbox2d = None
            if self.pred_bbox2d:
                pos_points = flatten_points[pos_inds]
                pos_decoded_bbox_preds = self.bbox2d_coder.decode(
                    pos_points, pos_bbox2d_preds)
                pos_decoded_target_preds = self.bbox2d_coder.decode(
                    pos_points, pos_bbox2d_targets)
                loss_bbox2d = self.loss_bbox2d(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_2d_targets,
                    avg_factor=centerness_denorm)
                # if gt_masks is not None:
                #     loss_bbox2d = self.loss_bbox2d(
                #         pos_decoded_bbox_preds,
                #         pos_decoded_target_preds,
                #         weight=pos_centerness_2d_targets *
                #         flatten_weights[pos_inds],
                #         avg_factor=centerness_denorm)
                # else:
                #     loss_bbox2d = self.loss_bbox2d(
                #         pos_decoded_bbox_preds,
                #         pos_decoded_target_preds,
                #         weight=pos_centerness_2d_targets,
                #         avg_factor=centerness_denorm)
            
            # 3d loss
            if num_pos_3d > 0:
                bbox_weights_3d = pos_bbox_targets_3d.new_ones(
                    len(pos_bbox_targets_3d), sum(self.group_reg_dims))
                equal_weights_3d = pos_bbox_targets_3d.new_ones(
                    pos_bbox_targets_3d.shape[0])

                dir_weight_vru = bbox_weights_3d.new_full((len(vru_idx_in_pos_inds),), self.vru_dir_weight)
                index_col = vru_idx_in_pos_inds.new_full((len(vru_idx_in_pos_inds),), 6)
                bbox_idx = (vru_idx_in_pos_inds, index_col)
                bbox_weights_3d = bbox_weights_3d.index_put(bbox_idx, dir_weight_vru)

                code_weight = self.train_cfg.get('code_weight', None)
                if code_weight:
                    assert len(code_weight) == sum(self.group_reg_dims)
                    if gt_masks is None:
                        bbox_weights = bbox_weights * bbox_weights.new_tensor(
                            code_weight)
                        bbox_weights_3d = bbox_weights_3d * bbox_weights_3d.new_tensor(
                        code_weight)
                    else:
                        loss_weight = bbox_weights.new_tensor(code_weight)  # c
                        pos_score_targets = flatten_weights[pos_inds]  # n
                        weight = pos_score_targets.new_ones(
                            len(pos_score_targets),
                            sum(self.group_reg_dims))  # n * c
                        weight = torch.einsum('nc,n->nc', weight,
                                            flatten_weights[pos_inds])
                        weight = torch.einsum('nc,c->nc', weight, loss_weight)
                        bbox_weights = weight

                if self.use_direction_classifier:
                    pos_dir_cls_targets = self.get_direction_target(
                        pos_bbox_targets_3d,
                        self.dir_offset, # 4/pi
                        self.dir_limit_offset, # 0
                        one_hot=False)

                if self.diff_rad_by_sin:
                    pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                        pos_bbox_preds, pos_bbox_targets_3d)

                loss_offset = self.loss_bbox(
                    pos_bbox_preds[:, :2],
                    pos_bbox_targets_3d[:, :2],
                    weight=bbox_weights_3d[:, :2],
                    avg_factor=equal_weights_3d.sum())
                loss_depth = self.loss_bbox(
                    pos_bbox_preds[:, 2],
                    pos_bbox_targets_3d[:, 2],
                    weight=bbox_weights_3d[:, 2],
                    avg_factor=equal_weights_3d.sum())
                loss_size = self.loss_bbox(
                    pos_bbox_preds[:, 3:6],
                    pos_bbox_targets_3d[:, 3:6],
                    weight=bbox_weights_3d[:, 3:6],
                    avg_factor=equal_weights_3d.sum())
                loss_rotsin = self.loss_bbox(
                    pos_bbox_preds[:, 6],
                    pos_bbox_targets_3d[:, 6],
                    weight=bbox_weights_3d[:, 6],
                    avg_factor=equal_weights_3d.sum())
                
                # direction classification loss
                loss_dir = None
                if self.use_direction_classifier:
                    if gt_masks is not None:
                        loss_dir = self.loss_dir(
                            pos_dir_cls_preds,
                            pos_dir_cls_targets,
                            # equal_weights * flatten_weights[pos_inds],
                            bbox_weights_3d[:, 6] * flatten_weights[pos_inds],
                            avg_factor=equal_weights.sum())
                    else:
                        loss_dir = self.loss_dir(
                            pos_dir_cls_preds,
                            pos_dir_cls_targets,
                            # equal_weights,
                            bbox_weights_3d[:, 6],
                            avg_factor=equal_weights_3d.sum())
                
                loss_velo = None
                if self.pred_velo:
                    loss_velo = self.loss_bbox(
                        pos_bbox_preds[:, 7:9],
                        pos_bbox_targets_3d[:, 7:9],
                        weight=bbox_weights_3d[:, 7:9],
                        avg_factor=equal_weights_3d.sum())

                # attribute classification loss
                loss_attr = None
                if self.pred_attrs:
                    if gt_masks is not None:
                        loss_attr = self.loss_attr(
                            pos_attr_preds,
                            pos_attr_targets,
                            pos_centerness_targets * flatten_weights[pos_inds_3d],
                            avg_factor=pos_centerness_targets.sum())
                    else:
                        loss_attr = self.loss_attr(
                            pos_attr_preds,
                            pos_attr_targets,
                            pos_centerness_targets,
                            avg_factor=pos_centerness_targets.sum())
            else:
                loss_offset = pos_bbox_preds[:, :2].sum()
                loss_depth = pos_bbox_preds[:, 2].sum()
                loss_size = pos_bbox_preds[:, 3:6].sum()
                loss_rotsin = pos_bbox_preds[:, 6].sum()
                loss_velo = None
                if self.pred_velo:
                    loss_velo = pos_bbox_preds[:, 7:9].sum()
                loss_dir = None
                if self.use_direction_classifier:
                    loss_dir = pos_dir_cls_preds.sum()
                loss_attr = None
                if self.pred_attrs:
                    loss_attr = pos_attr_preds.sum()
                
        else:
            # need absolute due to possible negative delta x/y
            loss_offset = pos_bbox_preds[:, :2].sum()
            loss_depth = pos_bbox_preds[:, 2].sum()
            loss_size = pos_bbox_preds[:, 3:6].sum()
            loss_rotsin = pos_bbox_preds[:, 6].sum()
            loss_velo = None
            if self.pred_velo:
                loss_velo = pos_bbox_preds[:, 7:9].sum()
            loss_centerness = pos_centerness.sum()
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()
            loss_attr = None
            if self.pred_attrs:
                loss_attr = pos_attr_preds.sum()
            loss_bbox2d = None
            if self.pred_bbox2d:
                loss_bbox2d = pos_bbox2d_preds.sum()

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size=loss_size,
            loss_rotsin=loss_rotsin,
            loss_centerness=loss_centerness)

        if loss_velo is not None:
            loss_dict['loss_velo'] = loss_velo

        if loss_dir is not None:
            loss_dict['loss_dir'] = loss_dir

        if loss_attr is not None:
            loss_dict['loss_attr'] = loss_attr

        if loss_bbox2d is not None:
            loss_dict['loss_bbox2d'] = loss_bbox2d

        if self.loss_homography is not None:
        # if self.loss_homography is None:
            pred_bboxes = self.get_bboxes(cls_scores, bbox_preds, dir_cls_preds, attr_preds,
                            centernesses, bbox2d_preds, img_metas)
            img_nums = len(img_metas)
            match_gt_bev_points = []
            homo_gt_bev_points = []
            loss_homo = None
            for idx in range(img_nums):
                if pred_bboxes[idx][0].tensor.shape[0] == 0 or gt_bboxes_3d[idx].tensor.shape[0] == 0:
                    continue
                pred_bev = pred_bboxes[idx][0].nearest_bev
                gt_bev = gt_bboxes_3d[idx].nearest_bev.to(pred_bev.device)
                assign_result = self.max_iou_assigner.assign(pred_bev, gt_bev)
                match_gt_idx = assign_result.gt_inds[assign_result.gt_inds > 0] - 1
                gt_corners_center = self.gt_bottom_corners_center(
                    gt_bboxes_3d[idx], img_metas[idx]['cam2img']).to(pred_bev.device)
                match_gt_uv = gt_corners_center[match_gt_idx, :]
                img_h, img_w = img_metas[idx]['ori_shape'][0], img_metas[idx]['ori_shape'][1]
                no_trunc_idx = torch.where((match_gt_uv[:,:,0]>0) & (match_gt_uv[:,:,0]<img_w)
                                           & (match_gt_uv[:,:,1]>0) & (match_gt_uv[:,:,1]<img_h))
                match_gt_uv = match_gt_uv[no_trunc_idx]
                gt_ones = match_gt_uv.new_ones(match_gt_uv.shape[0], 1)
                match_gt_uv = torch.cat([match_gt_uv, gt_ones], dim = -1).reshape(-1, 3)

                pred_corners_center = self.bev_to_corners_center(pred_bboxes[idx][0].bev)
                match_pred_bev = pred_corners_center[assign_result.gt_inds > 0, :][no_trunc_idx]
                pred_ones = match_pred_bev.new_ones(match_pred_bev.shape[0], 1)
                match_pred_bev = torch.cat([match_pred_bev, pred_ones], dim = -1).reshape(-1, 3)
                if match_pred_bev.shape[0] < 4:
                    continue
                
                gt_bev = self.bev_to_corners_center(gt_bboxes_3d[idx].bev).to(pred_bev.device)
                match_gt_bev = gt_bev[match_gt_idx, :][no_trunc_idx]
                gt_bev_ones = match_gt_bev.new_ones(match_gt_bev.shape[0], 1)
                match_gt_bev = torch.cat([match_gt_bev, gt_bev_ones], dim = -1).reshape(-1, 3)
                match_gt_bev_points.append(match_gt_bev)
                
                H_ = match_pred_bev @ torch.pinverse(match_gt_uv)
                homo_gt_bev = H_ @ match_gt_uv
                homo_gt_bev_points.append(homo_gt_bev)
            # print(len(match_gt_bev_points), len(homo_gt_bev_points))
            if len(match_gt_bev_points) != 0 and len(homo_gt_bev_points) != 0:
                match_gt_bev_points = torch.cat(match_gt_bev_points)
                homo_gt_bev_points = torch.cat(homo_gt_bev_points)
                loss_homo = self.loss_homography(match_gt_bev_points, homo_gt_bev_points)
            if loss_homo is not None:
                loss_dict['loss_homo'] = loss_homo * loss_homo.new_tensor(0.1)
            else:
                loss_dict['loss_homo'] = pred_bboxes[0][0].tensor.new_tensor(0)
                for idx in range(img_nums):
                    loss_dict['loss_homo'] += pred_bboxes[idx][0].tensor.sum()
            # print(loss_dict['loss_homo'])
        return loss_dict

    def bev_to_corners_center(self, bev):
        n = bev.shape[0]
        # modify heading
        # bev[:, -1] = -bev[:, -1]
        corners_center = torch.stack((
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
            bev[:, 0],
            bev[:, 1]
        ))
        corners_center = corners_center.reshape(5, 2, n).permute(2, 0, 1)
        return corners_center

    def gt_bottom_corners_center(self, bboxes3d, cam2img):
        from mmdet3d.core.bbox import points_cam2img

        bottom_corners_center_3d = bboxes3d.bottom_corners_center
        num_bbox = bottom_corners_center_3d.shape[0]
        points_3d = bottom_corners_center_3d.reshape(-1, 3)
        if not isinstance(cam2img, torch.Tensor):
            cam2img = torch.from_numpy(np.array(cam2img)).to(points_3d.device)

        assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
        cam2img = cam2img.float().cpu()

        # project to 2d to get image coords (uv)
        uv_origin = points_cam2img(points_3d, cam2img)
        uv_origin = (uv_origin - 1).round()
        imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 5, 2)
        return imgfov_pts_2d

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
                  'centernesses', 'bbox2d_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   attr_preds,
                   centernesses,
                   bbox2d_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs:
                attr_pred_list = [
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            if self.pred_bbox2d:
                bbox2d_pred_list = [
                    bbox2d_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                bbox2d_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [4, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                attr_pred_list, centerness_pred_list, bbox2d_pred_list,
                mlvl_points, input_meta, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           attr_preds,
                           centernesses,
                           bbox2d_preds,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels and attributes.
        """
        view = np.array(input_meta['cam2img'])
        # scale_factor = input_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_bboxes2d = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, bbox2d_pred, \
                points in zip(cls_scores, bbox_preds, dir_cls_preds,
                              attr_preds, centernesses, bbox2d_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            if self.infer_use_onnx:
                scores = cls_score.permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels)
                centerness = centerness.permute(1, 2, 0).reshape(-1)
            else:
                scores = cls_score.permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels).sigmoid()
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            bbox2d_pred = bbox2d_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred = bbox_pred[:, :self.bbox_code_size]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                bbox2d_pred = bbox2d_pred[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            # if rescale:
            #     print('scale_factor:',scale_factor)
            #     bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)

            bbox2d_pred = self.bbox2d_coder.decode(points, bbox2d_pred)
            scale_factor = input_meta['scale_factor']
            scale_factor = torch.tensor(
                scale_factor, device=bbox2d_pred.device)
            bbox2d_pred = bbox2d_pred / scale_factor

            ### TODO: reverse pit here
            if self.is_pit_transform:
                view = np.array(input_meta['ori_cam2img'])
                bbox_pred[:, :2] /= scale_factor[:2]
                cam_intrinsic = input_meta['ori_cam2img']
                fx = cam_intrinsic[0][0]
                fy = cam_intrinsic[1][1]

                # input_meta['img_shape'] # 374, 1238
                fovx = input_meta['img_info']['width']/ fx
                fovy = input_meta['img_info']['height']/ fy
                proj = PIT_module(
                    input_meta['img_info']['width'],
                    input_meta['img_info']['height'],
                    fovx=fovx,
                    fovy=fovy,
                    isPITedSize=True)
                bbox_pred_x, bbox_pred_y = proj.coord_arc_to_plain_tensor(
                    bbox_pred[:, 0], bbox_pred[:, 1])
                bbox_pred[:, 0] = bbox_pred_x
                bbox_pred[:, 1] = bbox_pred_y
                bbox2d_pred_x1, bbox2d_pred_y1 = proj.coord_arc_to_plain_tensor(
                    bbox2d_pred[:, 0], bbox2d_pred[:, 1])
                bbox2d_pred_x2, bbox2d_pred_y2 = proj.coord_arc_to_plain_tensor(
                    bbox2d_pred[:, 2], bbox2d_pred[:, 3])
                bbox2d_pred[:, 0] = bbox2d_pred_x1
                bbox2d_pred[:, 1] = bbox2d_pred_y1
                bbox2d_pred[:, 2] = bbox2d_pred_x2
                bbox2d_pred[:, 3] = bbox2d_pred_y2

            pred_center2d = bbox_pred[:, :3].clone()
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_bboxes2d.append(bbox2d_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        # change local yaw to global yaw for 3D nms
        cam2img = mlvl_centers2d.new_zeros((4, 4))
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1) # 增加一个背景类别score的全0列
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]

        if self.debug:
            post_objs={}
            post_objs['scores_3d']=mlvl_nms_scores[:,:3].cpu().numpy().tolist()
            post_objs['boxes_3d']=mlvl_bboxes.to('cpu').numpy().tolist()
            post_objs['boxes_2d']=mlvl_bboxes2d.cpu().numpy().tolist()
            out_path = '/mnt/lustre/share_data/chenxuepan/workspace/Hozon/other/hozon_deploy_align/torch_output/'
            mmcv.dump(post_objs, out_path+'post_objs.json')

            post_objs_0_2={}
            score_idx = np.where(mlvl_nms_scores.max(dim=1)[0].cpu().numpy()>0.2)[0]
            post_objs_0_2['scores_3d']=mlvl_nms_scores[score_idx,:][:,:3].cpu().numpy().tolist()
            post_objs_0_2['boxes_3d']=mlvl_bboxes[score_idx,:].to('cpu').numpy().tolist()
            post_objs_0_2['boxes_2d']=mlvl_bboxes2d[score_idx,:].cpu().numpy().tolist()
            post_objs_0_2['obj_0_2'] = np.concatenate((post_objs_0_2['boxes_3d'], post_objs_0_2['boxes_2d'], post_objs_0_2['scores_3d']), axis=-1).tolist()
            mmcv.dump(post_objs_0_2, out_path+'post_objs_0.2.json')
            # return mlvl_nms_scores[:,:3], mlvl_bboxes, mlvl_bboxes2d

        if self.pred_bbox2d:
            results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                           mlvl_nms_scores, cfg.score_thr,
                                           cfg.max_per_img, cfg,
                                           mlvl_dir_scores, mlvl_attr_scores,
                                           mlvl_bboxes2d)
            bboxes, scores, labels, dir_scores, attrs, bboxes2d = results
        else:
            results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                           mlvl_nms_scores, cfg.score_thr,
                                           cfg.max_per_img, cfg,
                                           mlvl_dir_scores, mlvl_attr_scores)
            bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = input_meta['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        if not self.pred_attrs:
            attrs = None
        if not self.pred_bbox2d:
            bboxes2d = None
        
        # if self.debug:
        #     outputs_topk={}
        #     outputs_topk['bboxes_2d']=bboxes2d.cpu().numpy()
        #     outputs_topk['boxes_3d']=mlvl_bboxes.to('cpu').numpy()
        #     outputs_topk['scores_3d']=mlvl_nms_scores[:,:3].cpu().numpy()
        #     mmcv.dump(outputs_topk, 'outputs_topk.pkl')

        return bboxes, scores, labels, attrs, bboxes2d

    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3],
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera intrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3],
                3 corresponds with x, y, z in 3D space.
        """
        warning.warn('DeprecationWarning: This static method has been moved '
                     'out of this class to mmdet3d/core. The function '
                     'pts2Dto3D will be deprecated.')

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogeneous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False,
                           gt_masks=None):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2

        if gt_masks is not None:
            #N C Hi Wi, N Ho Wo 2 
            y_s = ((y * stride + stride // 2) - (gt_masks.shape[2] / 2)) / (gt_masks.shape[2] / 2)
            x_s = ((x * stride + stride // 2) - (gt_masks.shape[3] / 2)) / (gt_masks.shape[3] / 2)
            sample_points = torch.stack((y_s, x_s), dim=-1)
            sample_points = sample_points.unsqueeze(dim=0).repeat(gt_masks.shape[0], 1, 1, 1)
            weigths = F.grid_sample(gt_masks, sample_points)
            return points, weigths

        else:
            return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list, attr_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        if attr_labels_list is None:
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]

        # get labels and bbox_targets of each image
        _, bbox2d_targets_list, labels_3d_list, bbox_targets_3d_list, centerness_targets_list, \
            attr_targets_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,
                centers2d_list,
                depths_list,
                attr_labels_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]
        attr_targets_list = [
            attr_targets.split(num_points, 0)
            for attr_targets in attr_targets_list
        ]
        bbox2d_targets_list = [
            bbox2d_targets.split(num_points, 0)
            for bbox2d_targets in bbox2d_targets_list
        ]
        # if 2d, is_3d_data_flag = 0, else if 3d, is_3d_data_flag = 1
        is_3d_data_flag = []
        
        for depths in depths_list:
            if bool(torch.prod(depths == -1)):
                is_3d_data_flag.append(0)
            else:
                is_3d_data_flag.append(1)

        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []
        concat_lvl_bbox2d_targets = []
        concat_lvl_targets_3d_mask = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            bbox2d_targets = torch.cat(
                [bbox2d_targets[i] for bbox2d_targets in bbox2d_targets_list])
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :
                                2] = bbox_targets_3d[:, :2] / self.strides[i]
                bbox2d_targets[:, :] = bbox2d_targets[:, :] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
            concat_lvl_bbox2d_targets.append(bbox2d_targets)

            targets_3d_mask = torch.cat([
                torch.ones(
                    bbox_targets_3d[i].shape[0],
                    device=bbox_targets_3d[i].device) * is_3d_data_flag[j]
                for j, bbox_targets_3d in enumerate(bbox_targets_3d_list)
            ])
            concat_lvl_targets_3d_mask.append(targets_3d_mask)
        
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets, \
            concat_lvl_bbox2d_targets, concat_lvl_targets_3d_mask

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths, attr_labels,
                           points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \
                   attr_labels.new_full(
                       (num_points,), self.attr_background_label)

        # change orientation to local yaw
        if not self.global_yaw:
            gt_bboxes_3d[..., 6] = -torch.atan2(
                gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]) + gt_bboxes_3d[..., 6]

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        large_object_radius = self.center_sample_radius_for_large_object
        center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        # center_xs = centers2d[..., 0]
        # center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            if (self.strides[lvl_idx] == 64) or (self.strides[lvl_idx] == 128):
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * large_object_radius
            else:
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # (Ang) condition: limit point in real gt box
        gt_dist_left = xs - gt_bboxes[..., 0]
        gt_dist_right = gt_bboxes[..., 2] - xs
        gt_dist_top = ys - gt_bboxes[..., 1]
        gt_dist_bottom = gt_bboxes[..., 3] - ys
        center_bbox = torch.stack(
            (gt_dist_left, gt_dist_top, gt_dist_right, gt_dist_bottom), -1)
        inside_real_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        inside_gt_bbox_mask *= inside_real_gt_bbox_mask

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)
        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels

    def get_points(self,
                   featmap_sizes,
                   dtype,
                   device,
                   flatten=False,
                   gt_masks=None):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        if gt_masks is not None:
            mlvl_weights = []

        mlvl_points = []
        for i in range(len(featmap_sizes)):
            if gt_masks is not None:
                points, weights = self._get_points_single(
                    featmap_sizes[i],
                    self.strides[i],
                    dtype,
                    device,
                    flatten,
                    gt_masks=gt_masks)
                mlvl_points.append(points)
                mlvl_weights.append(weights)

            else:
                mlvl_points.append(
                    self._get_points_single(
                        featmap_sizes[i],
                        self.strides[i],
                        dtype,
                        device,
                        flatten,
                        gt_masks=None))

        if gt_masks is not None:
            loss_weights = []
            for level in range(len(mlvl_weights)):
                single_weights = mlvl_weights[level]
                batch_weights = []
                for bs_index in range(single_weights.shape[0]):

                    # debug
                    # image = single_weights[bs_index] * 100
                    # mmcv.imwrite(image.permute(1, 2, 0).detach().cpu().numpy(), "./test_{}_{}.jpg".format(level, bs_index))

                    per_batch_weights = single_weights[bs_index].squeeze(
                    )  # squeeze "channel" dim
                    batch_weights.append(per_batch_weights.view(-1))
                weights = torch.cat(batch_weights)
                loss_weights.append(weights)

            return mlvl_points, loss_weights
        else:
            return mlvl_points

    def forward_adela(self, bbox_preds_lvl_data, cls_scores_lvl_data, centerness_lvl_data, img_metas):
        image_shapes = [
            img_metas[idx]['img_shape']
            for idx in range(len(img_metas))
        ]
        image_shapes_list = image_shapes * len(self.strides)

        if self.rescale_depth:
            depth_factors = [
                img_metas[idx]['depth_factors']
                for idx in range(len(img_metas))
            ]

            depth_factors = torch.tensor(depth_factors, device=bbox_preds_lvl_data[0].device)
            depth_factors_list = [depth_factors] * len(self.scales)

            return multi_apply(self.forward_single_adela,
                               bbox_preds_lvl_data, cls_scores_lvl_data, centerness_lvl_data,
                               image_shapes_list,
                               self.scales,
                               self.strides, depth_factors_list)[:6]

        # Note: we use [:6] to filter feats and only return predictions
        return multi_apply(self.forward_single_adela,
                           bbox_preds_lvl_data, cls_scores_lvl_data, centerness_lvl_data,
                           image_shapes_list,
                           self.scales,
                           self.strides)[:6]


    def forward_single_adela(self, raw_bbox_preds, raw_cls_scores, raw_centerness, image_shape,
                             scale, stride, depth_factors_list=None):
        w, h = int(np.ceil(image_shape[1]/stride)), int(np.ceil(image_shape[0]/stride))

        if raw_cls_scores.shape[0] == 3:
            raw_cls_scores = raw_cls_scores.reshape(3, h, w).unsqueeze(0)
            cls_score = raw_cls_scores
            attr_pred = None
        elif raw_cls_scores.shape[0] == 15:
            raw_cls_scores = raw_cls_scores.reshape(15, h, w).unsqueeze(0)
            cls_score = raw_cls_scores[:, :3, :, :]
            attr_pred = raw_cls_scores[:, 3:, :, :]

        raw_centerness = raw_centerness.reshape(1, h, w).unsqueeze(0)
        raw_bbox_preds = raw_bbox_preds.reshape(13, h, w).unsqueeze(0)


        bbox_pred = raw_bbox_preds[:, :7, :, :]
        bbox2d_pred = raw_bbox_preds[:, 7:11, :, :]
        dir_cls_pred = raw_bbox_preds[:, 11:, :, :]
        centerness = raw_centerness

        if self.pred_bbox2d:
            # scale the bbox_pred of different level
            # float to avoid overflow when enabling FP16
            scale_bbox2d = scale[3]
            bbox2d_pred = scale_bbox2d(bbox2d_pred).float()
            if self.norm_on_bbox:
                # bbox_pred needed for gradient computation has been modified
                # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
                # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
                bbox2d_pred = bbox2d_pred.clamp(min=0)
                if not self.training:
                    bbox2d_pred *= stride
            else:
                bbox2d_pred = bbox2d_pred.exp()

        bbox_pred = self.bbox_coder.decode(bbox_pred, scale, stride,
                                           self.training, cls_score,
                                           depth_factors_list)

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
               bbox2d_pred