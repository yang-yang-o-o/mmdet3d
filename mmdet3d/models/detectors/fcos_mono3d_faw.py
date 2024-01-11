# Copyright (c) OpenMMLab. All rights reserved.
from .single_stage_mono3d_faw import SingleStageMono3DDetector_faw
# from ...core import bbox3d2result_faw
from ..builder import DETECTORS
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import DataContainer as DC
import os
import json
import base64
import struct
from mmcv.cnn import Scale, normal_init, ConvModule


@DETECTORS.register_module()
class FCOSMono3D_faw(SingleStageMono3DDetector_faw):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSMono3D_faw, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)


# data utils
def load_res_data_faw(data_path):
    dict = {}
    with open(data_path, 'r+') as f:
        for line in f.readlines():
            target = {}
            item = json.loads(line)
            id = item['id']
            tensors = item['tensors']
            for net_name, item in tensors.items():
                for blob_name, tensor_value in item.items():
                    dict_key = net_name + "_" + blob_name
                    target[dict_key] = tensor_value
            dict[id] = target
    return dict


def load_name_data_faw(data_path, sdk_mode=False):
    dict = {}
    with open(data_path, 'r+') as f:
        for line in f.readlines():
            item = json.loads(line)
            id = item['id']
            aws_path = item['tensors']['net']['data']['blob_data']
            aws_path = "/".join(aws_path.split('/')[1:])
            aws_path = aws_path.replace(".npy", ".jpg")
            dict[id] = aws_path
    return dict


def load_data_faw(results_path, gt_mode=False, sdk_mode=False):
    if not gt_mode:
        resfile = os.path.join(results_path, 'runner_result.txt')
    else:
        resfile = os.path.join(results_path, 'ground_truth.txt')
    namefile = os.path.join(results_path, 'input.txt')
    res = load_res_data_faw(resfile)
    names = load_name_data_faw(namefile, sdk_mode=sdk_mode)
    data = {}
    for key in res.keys():
        data[names[key]] = res[key]
    return data


@DETECTORS.register_module()
class FCOSMono3D_adela_faw(SingleStageMono3DDetector_faw):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 # add for adela quant test
                 adela_result=None,
                 deploy_files=None,
                 with_attr=False,
                 gt_mode=False,
                 ):
        super(FCOSMono3D_adela_faw, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
        # load data
        self.adela_data = load_data_faw(adela_result, gt_mode)

        self.reload_head_scale(deploy_files)
        self.with_attr = with_attr
        self.gt_mode = gt_mode


    def reload_head_scale(self, deploy_files):
        params_json = os.path.join(deploy_files, 'parameters.json')
        with open(params_json, 'r') as f:
            params = json.load(f)

        scale_offset = params['scale_offset']
        scale_depth = params['scale_depth']
        scale_size = params['scale_size']
        scale_bbox2d = params['scale_bbox2d']

        scale_list = []
        # scale_offset, scale_depth, scale_size, scale_bbox = scales[i]
        for i in range(len(scale_offset)):
            lvl_scale_list = nn.ModuleList([
                Scale(scale_offset[i]), Scale(scale_depth[i]), Scale(scale_size[i]), Scale(scale_bbox2d[i])
            ])
            scale_list.append(lvl_scale_list)
        load_scales = nn.ModuleList(scale_list)
        self.bbox_head.scales = load_scales
        print("scales loaded !")


    def load_adela_data(self, aws_path, device):

        raw_data = self.adela_data['sh1424_datasets/'+aws_path]

        bbox_preds_lvl_data = []
        cls_scores_lvl_data = []
        centerness_lvl_data = []

        bbox_keys = ['net_fpn0.bbox_pred', 'net_fpn1.bbox_pred', 'net_fpn2.bbox_pred',
                     'net_fpn3.bbox_pred', 'net_fpn4.bbox_pred']
        cls_keys = ['net_fpn0.cls_score', 'net_fpn1.cls_score', 'net_fpn2.cls_score',
                     'net_fpn3.cls_score', 'net_fpn4.cls_score']
        cen_keys = ['net_fpn0.centerness', 'net_fpn1.centerness', 'net_fpn2.centerness',
                     'net_fpn3.centerness', 'net_fpn4.centerness']
        for key in bbox_keys:
            feature_map_bin = base64.b64decode(raw_data[key]['feature_map'])
            bin_len = len(feature_map_bin)
            feature_map = np.array(struct.unpack(
                ('%df' % (bin_len / 4)), feature_map_bin))
            feature_map = feature_map.reshape(13, -1)
            feature_map = torch.tensor(feature_map, dtype=torch.float32).to(device)
            bbox_preds_lvl_data.append(feature_map)

        for key in cls_keys:
            feature_map_bin = base64.b64decode(raw_data[key]['feature_map'])
            bin_len = len(feature_map_bin)
            feature_map = np.array(struct.unpack(
                ('%df' % (bin_len / 4)), feature_map_bin))
            if self.with_attr:
                feature_map = feature_map.reshape(15, -1)
            else:
                feature_map = feature_map.reshape(3, -1)
            feature_map = torch.tensor(feature_map, dtype=torch.float32).to(device)
            cls_scores_lvl_data.append(feature_map)

        for key in cen_keys:
            feature_map_bin = base64.b64decode(raw_data[key]['feature_map'])
            bin_len = len(feature_map_bin)
            feature_map = np.array(struct.unpack(
                ('%df' % (bin_len / 4)), feature_map_bin))
            feature_map = feature_map.reshape(1, -1)
            feature_map = torch.tensor(feature_map, dtype=torch.float32).to(device)
            centerness_lvl_data.append(feature_map)

        return bbox_preds_lvl_data, cls_scores_lvl_data, centerness_lvl_data

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)

        filename = img_metas[0]['filename']
        bbox_preds_lvl_data, cls_scores_lvl_data, centerness_lvl_data = self.load_adela_data(filename, img.device)
        outs = self.bbox_head.forward_adela(bbox_preds_lvl_data, cls_scores_lvl_data, centerness_lvl_data, img_metas)

        bbox_outputs = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

        bbox_img = [
                bbox3d2result_faw(bboxes, scores, labels, attrs, bboxes2d)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox

        return bbox_list