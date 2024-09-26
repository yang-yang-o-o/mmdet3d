import os
import shutil
import argparse

import numpy as np
import mmcv
from mmcv import Config, DictAction

from .lidar_det_eval.nuscenes_eval_core import NuScenesEval


CLASS_MAPPING = ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']
CLASS_MAPPING_GT = {'VEHICLE_CAR':'VEHICLE_CAR', 
'VEHICLE_TRUCK': 'VEHICLE_TRUCK', 
'BIKE_BICYCLE':'PEDESTRIAN', 
'PEDESTRIAN': 'PEDESTRIAN'}


class PRDEvaluator():

    def __init__(self, data_infos, pred, point_cloud_range, save_path='work_dirs/nview', score_threshold=0.0) -> None:
        self.point_cloud_range = point_cloud_range
        self.score_threshold = score_threshold
        self.gt = data_infos
        self.pred = pred
        self.save_path = save_path

    def nusenseeval(self, point_cloud_range, distance_threshold):
        print("point_cloud_range: ", point_cloud_range)
        print("distance_threshold: ", distance_threshold)
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 distance_threshold=distance_threshold,
                                 point_cloud_range=point_cloud_range)
        return evaluator

    def evaluate_internal_data(self):
        gt_save_path = os.path.join(self.save_path, 'gt')
        pred_save_path = os.path.join(self.save_path, 'pred')
        self.convert_results(gt_save_path, pred_save_path)

        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        print('==================== standard evaluation =====================')
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        # run evaluation
        print("self.point_cloud_range: ", self.point_cloud_range)
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 distance_threshold=0.1,
                                 score_threshold=self.score_threshold,
                                 classes=['VEHICLE_CAR', 'VEHICLE_TRUCK'],
                                 point_cloud_range=self.point_cloud_range)
        eval_results = evaluator.get_metric_results()
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 distance_threshold=0.2,
                                 score_threshold=self.score_threshold,
                                 classes=['BIKE_BICYCLE', 'PEDESTRIAN'],
                                 point_cloud_range=self.point_cloud_range)
        eval_results.update(evaluator.get_metric_results())
        mmcv.dump(eval_results, os.path.join(self.save_path,
                                             'eval_results.json'))

        # run internal evaluation
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        print('========= A zone evaluation (distance_threshold=0.2) =========')
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        zone = [-0, -1.5, -5.0, 100, 1.5, 3.0] # [back, left_side, down, front, right_side, up]
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 distance_threshold=0.1,
                                 classes=['VEHICLE_CAR', 'VEHICLE_TRUCK'],
                                 point_cloud_range=zone)
        eval_results = evaluator.get_metric_results()
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 distance_threshold=0.2,
                                 classes=['BIKE_BICYCLE', 'PEDESTRIAN'],
                                 point_cloud_range=zone)
        eval_results.update(evaluator.get_metric_results())
        mmcv.dump(eval_results, os.path.join(self.save_path,
                                             'eval_results_A_zone.json'))

        simplified_results = {}
        for single_class in ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']:
            simplified_results.update({
                single_class : {
                    key: value 
                    for key, value in eval_results[single_class].items() if isinstance(value, np.ndarray) == False
                }
            })
        return simplified_results

    def run_nuscene_evaluation(self):
        gt_save_path = os.path.join(self.save_path, 'gt')
        pred_save_path = os.path.join(self.save_path, 'pred')

        # run evaluation
        NuScenesEval(self.predictions,
                     self.gts,
                     'class x y z l w h r score',
                     self.save_path,
                     max_range=50)
        return

    def convert_results(self, gt_save_path, pred_save_path):
        if os.path.exists(gt_save_path):
            shutil.rmtree(gt_save_path)
        if os.path.exists(pred_save_path):
            shutil.rmtree(pred_save_path)
        os.makedirs(gt_save_path, exist_ok=True)
        os.makedirs(pred_save_path, exist_ok=True)

        self.predictions = []
        self.gts = []

        for anno, pred in zip(self.gt, self.pred['pts_bbox']):
            gt_bboxes = anno['gt_boxes']
            gt_names = anno['gt_names']

            pred_bboxes = pred['boxes_3d'].tensor.numpy()
            pred_scores = pred['scores_3d'].numpy()
            pred_labels = pred['labels_3d'].numpy()

            valid_idx = pred_scores >= 0.3
            pred_bboxes = pred_bboxes[valid_idx]
            pred_scores = pred_scores[valid_idx]
            pred_labels = pred_labels[valid_idx]

            # prepare gt
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            for gt_name, gt_bbox in zip(gt_names, gt_bboxes):
                # classes.append(str(CLASS_MAPPING_GT[gt_name]))
                classes.append(str(gt_name))
                x.append(gt_bbox[0])
                y.append(gt_bbox[1])
                z.append(gt_bbox[2])
                l.append(gt_bbox[3])
                w.append(gt_bbox[4])
                h.append(gt_bbox[5])
                r.append(gt_bbox[6])

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1)))
            self.gts.append(final_array)

            # prepare prediction
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            for pred_label, pred_bbox, pred_score in zip(
                    pred_labels, pred_bboxes, pred_scores):
                classes.append(str(CLASS_MAPPING[pred_label]))
                x.append(pred_bbox[0])
                y.append(pred_bbox[1])
                z.append(pred_bbox[2])
                l.append(pred_bbox[3])
                w.append(pred_bbox[4])
                h.append(pred_bbox[5])
                r.append(pred_bbox[6])
                score.append(pred_score)

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1)))
            final_array = np.hstack(
                (final_array, np.array(score).reshape(-1, 1)))
            self.predictions.append(final_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--config', help='config file in pickle format')
    parser.add_argument('--pred', help='output result file in pickle format')
    parser.add_argument('--eval_only',
                        action='store_true',
                        help='only do evaluation')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    pred = args.pred
    eval_only = args.eval_only
    internal_evaluator = PRDEvaluator(cfg, pred)
    if eval_only:
        internal_evaluator.run_nuscene_evaluation()
    else:
        internal_evaluator.evaluate_internal_data()
