# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet3d.core.bbox.structures import limit_period



@BBOX_CODERS.register_module()
class FCOS3DBBoxCoderDxScale_faw(BaseBBoxCoder):
    """Bounding box coder for FCOS3D.

    Args:
        base_depths (tuple[tuple[float]]): Depth references for decode box
            depth. Defaults to None.
        base_dims (tuple[tuple[float]]): Dimension references for decode box
            dimension. Defaults to None.
        code_size (int): The dimension of boxes to be encoded. Defaults to 7.
        norm_on_bbox (bool): Whether to apply normalization on the bounding
            box 2D attributes. Defaults to True.
    """

    def __init__(self,
                 base_depths=None,
                 base_dims=None,
                 code_size=7,
                 norm_on_bbox=True,
                 rescale_depth=False,
                 scale_depth=False,
                 scale_deltax=False,
                 scale_angle=False,
                 scale_factor=10,
                 global_yaw=False):
        super(FCOS3DBBoxCoderDxScale_faw, self).__init__()
        self.base_depths = base_depths
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.norm_on_bbox = norm_on_bbox
        self.rescale_depth = rescale_depth

        self.scale_depth = scale_depth
        self.scale_deltax = scale_deltax
        self.scale_angle = scale_angle
        self.scale_factor = scale_factor
        self.global_yaw = global_yaw


    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        # TODO: refactor the encoder in the FCOS3D and PGD head
        pass

    def decode(self,
               bbox,
               scale,
               stride,
               training,
               cls_score=None,
               depth_factors_list=None):
        """Decode regressed results into 3D predictions.

        Note that offsets are not transformed to the projected 3D centers.

        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            cls_score (torch.Tensor): Classification score map for deciding
                which base depth or dim is used. Defaults to None.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # scale the bbox of different level
        # only apply to offset, depth and size prediction
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox = bbox.clone()
        bbox[:, :2] = scale_offset(clone_bbox[:, :2]).float()
        bbox[:, 2] = scale_depth(clone_bbox[:, 2]).float()
        bbox[:, 3:6] = scale_size(clone_bbox[:, 3:6]).float()

        # scale deltax
        if self.scale_deltax:
            bbox[:, 0] = bbox[:, 0] * self.scale_factor

        if self.scale_angle:
            bbox[:, -1] /= self.scale_factor

        if self.base_depths is None:
            # scale depth
            if self.scale_depth:
                bbox[:, 2] = bbox[:, 2] * self.scale_factor
            else:
                bbox[:, 2] = bbox[:, 2].exp()
        elif len(self.base_depths) == 1:  # only single prior
            mean = self.base_depths[0][0]
            std = self.base_depths[0][1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std
        else:  # multi-class priors
            assert len(self.base_depths) == cls_score.shape[1], \
                'The number of multi-class depth priors should be equal to ' \
                'the number of categories.'
            indices = cls_score.max(dim=1)[1]
            depth_priors = cls_score.new_tensor(
                self.base_depths)[indices, :].permute(0, 3, 1, 2)
            mean = depth_priors[:, 0]
            std = depth_priors[:, 1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std

        if self.rescale_depth:
            bbox[:, 2] = torch.einsum('bhw,b->bhw', bbox[:, 2],
                                      depth_factors_list)

        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None:
            assert len(self.base_dims) == cls_score.shape[1], \
                'The number of anchor sizes should be equal to the number ' \
                'of categories.'
            indices = cls_score.max(dim=1)[1]
            size_priors = cls_score.new_tensor(
                self.base_dims)[indices, :].permute(0, 3, 1, 2)
            bbox[:, 3:6] = size_priors * bbox.clone()[:, 3:6]

        assert self.norm_on_bbox is True, 'Setting norm_on_bbox to False '\
            'has not been thoroughly tested for FCOS3D.'
        if self.norm_on_bbox:
            if not training:
                # Note that this line is conducted only when testing
                bbox[:, :2] *= stride

        return bbox

    def decode_yaw(self, bbox, centers2d, dir_cls, dir_offset, cam2img):
        """Decode yaw angle and change it from local to global.i.

        Args:
            bbox (torch.Tensor): Bounding box predictions in shape
                [N, C] with yaws to be decoded.
            centers2d (torch.Tensor): Projected 3D-center on the image planes
                corresponding to the box predictions.
            dir_cls (torch.Tensor): Predicted direction classes.
            dir_offset (float): Direction offset before dividing all the
                directions into several classes.
            cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].

        Returns:
            torch.Tensor: Bounding boxes with decoded yaws.
        """
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi) # 把 θ - π/4 限定到 [0,π] 之间
            bbox[..., 6] = \
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype) # 考虑朝向二分类，加0或π，再把offset加回去（π/4）
        if not self.global_yaw:
            bbox[:, 6] = torch.atan2(centers2d[:, 0] - cam2img[0, 2],
                                    cam2img[0, 0]) + bbox[:, 6] # θ 加上 atan2(x,z)

        return bbox