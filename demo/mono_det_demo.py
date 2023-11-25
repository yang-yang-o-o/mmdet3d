# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)


def main():
    parser = ArgumentParser()
    parser.add_argument('--image',default='F:\GitHub\mmdetection3d-1.0.0rc3/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', help='image file')
    parser.add_argument('--ann',default='F:\GitHub\mmdetection3d-1.0.0rc3/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json', help='ann file')
    parser.add_argument('--config',default='F:\GitHub\mmdetection3d-1.0.0rc3/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py', help='Config file')
    parser.add_argument('--checkpoint',default='F:\GitHub\mmdetection3d-1.0.0rc3/checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_mono_3d_detector(model, args.image, args.ann)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='mono-det')


if __name__ == '__main__':
    main()
