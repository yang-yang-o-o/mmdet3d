import glob
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from .label_parser import LabelParser
from .kuhn_munkres import KMMatcher


class NuScenesEval_faw:
    def __init__(self, pred_label_path, gt_label_path, label_format, save_loc, 
                 distance_threshold=0.1, gts_ignore=None, preds_ignore=None, iou_threshold = 0.5,
                 classes=['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'],
                 score_threshold=0.0, point_cloud_range=None, run=True, area_name="full_zone"):
        
        # Initialize
        self.save_loc = save_loc
        self.iou_threshold = iou_threshold
        self.distance_threshold_sq = distance_threshold # distance_threshold**2
        self.distance_threshold_ate = 0.2  # used for ate calculation
        self.score_threshold = score_threshold
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.total_N_pos = 0
        self.results_dict = {}
        self.area_name = area_name
        self.gts_ignore = gts_ignore
        self.preds_ignore = preds_ignore

        os.makedirs(os.path.join(self.save_loc, self.area_name), exist_ok=True)
        for single_class in classes:
            class_dict = {}
            class_dict['class'] = single_class
            class_dict['T_p'] = np.empty((0, 12))
            class_dict['gt'] = np.empty((0, 11))
            class_dict['total_N_pos'] = 0
            class_dict['result'] = np.empty((0, 5))
            class_dict['precision'] = []
            class_dict['recall'] = []
            class_dict['T_p_ate'] = np.empty((0, 12))
            class_dict['gt_ate'] = np.empty((0, 11))
            class_dict['result_ate'] = np.empty((0, 5))
            self.results_dict[single_class] = class_dict

        # metric results
        self.metric_results = {}
        for single_class in classes:
            self.metric_results[single_class] = {}
        
        # matched_results, -1: ignore; 1: TP; 0: FP
        
        # 键matched_results的值是一个list，元素个数为图片数量，每个元素是一个array，array的元素个数是一张图里的预测数量，array里每个元素初始为-1
        self.metric_results['matched_results'] = [-1 * np.ones(len(prediction)) for prediction in pred_label_path]
        # self.matched_results = [-1 * np.ones(len(prediction)) for prediction in pred_label_path]
        # 键matched_gt的值是一个list，元素个数为图片数量，每个元素是一个array，array的元素个数是一张图里的gt数量，array里每个元素初始为-1
        self.metric_results['matched_gt'] = [-1 * np.ones(len(gt)) for gt in gt_label_path]

        # Run
        if run:
            self.time = time.time()
            self.evaluate(pred_label_path, gt_label_path, label_format)

    def get_metric_results(self):
        return self.metric_results


    def evaluate(self, all_predictions, all_gts, label_format):
        num_examples = len(all_predictions)
        print("Starting evaluation for {} file predictions".format(num_examples))
        print("--------------------------------------------")
        print('eval range: ', self.point_cloud_range)

        ## Check missing files
        print("Confirmation prediction ground truth file pairs.")

        ## Evaluate matches，匹配gt和pre
        print("Evaluation examples")
        frame_index = 0
        for predictions, ground_truth in tqdm(zip(all_predictions, all_gts)): # 遍历每张图，匹配图里的gt和pre

            if self.gts_ignore is not None:
                self.gt_ignore = self.gts_ignore[frame_index] # 拿出当前图片的 gt_ignore
            else:
                self.gt_ignore = np.zeros(len(ground_truth))
            
            if self.preds_ignore is not None:
                self.pred_ignore = self.preds_ignore[frame_index] # 拿出当前图片的 pred_ignore
            else:
                self.pred_ignore = np.zeros(len(predictions))

            if self.point_cloud_range is not None: # 使用ground_truth计算一个range，然后过滤掉不在range之内的predictions（设置pred_ignore），self.point_cloud_range没用到
                self.ignore_by_gt_range(predictions, ground_truth, point_range=self.point_cloud_range)
                  
            matched_result, matched_gt = self.eval_pair(predictions, ground_truth) # 对gt和pre做匈牙利匹配
            # import pdb;pdb.set_trace()
            # 存储匈牙利匹配结果
            self.metric_results['matched_results'][frame_index] = matched_result
            self.metric_results['matched_gt'][frame_index] = matched_gt
            frame_index += 1

        print("\nDone!")
        print("----------------------------------")
        ## Calculate，计算
        indicator_res = [[] for i in range(len(self.classes))]
        for idx, single_class in enumerate(self.classes): # 分别计算每个类别
            class_dict = self.results_dict[single_class]
            print("Calculating metrics for {} class".format(single_class))
            print("----------------------------------")
            print("Number of ground truth labels: ", class_dict['total_N_pos'])
            print("Number of detections: ", np.sum(class_dict['result'][:, 0] != -1)) # 不是漏检的个数
            print("Number of true positives: ", np.sum(class_dict['result'][:, 0] == 1))
            print("Number of false positives: ", np.sum(class_dict['result'][:, 0] == 0))
            self.metric_results[single_class]["gt_num"] = class_dict[
                'total_N_pos']
            self.metric_results[single_class]["pred_num"] = np.sum(
                class_dict['result'][:, 0] != -1)
            self.metric_results[single_class]["tp_num"] = np.sum(
                class_dict['result'][:, 0] == 1)
            self.metric_results[single_class]["fp_num"] = np.sum(
                class_dict['result'][:, 0] == 0)
            self.metric_results[single_class]["match_pair"] = class_dict['result']
            if class_dict['total_N_pos'] == 0:
                print("No detections for this class!")
                print(" ")
                continue
            ## Recall Precision
            self.compute_recall_precision(single_class)
            print('Recall: %.3f ' % self.metric_results[single_class]["recall_range"][-1])
            print('Precision: %.3f ' % self.metric_results[single_class]["precision_range"][-1])
            if self.metric_results[single_class]["recall_range"][-1] < 1e-3 or \
                self.metric_results[single_class]["precision_range"][-1] < 1e-3:
                print('cur class eval failure!')
                continue
            ## AP：某个类别 准确率-召回率曲线 下方的面积
            # MAP：所有类别的 AP 的均值，代码这里的meap_ap只是某个类别的AP，不是MAP
            self.compute_ap_curve(class_dict)
            mean_ap = self.compute_mean_ap(class_dict['precision'], class_dict['recall'])
            print('Mean AP: %.3f ' % mean_ap)
            self.metric_results[single_class]["ap"] = mean_ap
            # F1 score
            f1 = self.compute_f1_score(class_dict['precision'], class_dict['recall'])
            print('F1 Score: %.3f ' % f1)
            self.metric_results[single_class]["f1_score"] = f1
            print(' ')
            ## Positive Thresholds
            # ATE 2D，bev平面上的欧氏距离
            ate2d, ate2d_pct = self.compute_ate2d(
                class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average 2D Translation Error[m]: {:.4f}, {:.2f}%'.format(
                ate2d, ate2d_pct*100))
            self.metric_results[single_class]["ate_2d"] = ate2d
            self.metric_results[single_class]["ate_2d_pct"] = ate2d_pct*100

            self.compute_ate2d_by_distance(
                class_dict['T_p_ate'], class_dict['gt_ate'])

            self.compute_ate2d_by_distance_plt(
                class_dict['T_p_ate'], class_dict['gt_ate'],single_class)
            
            # ATE 3D，3D空间中的欧式距离
            ate3d, ate3d_pct = self.compute_ate3d(
                class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average 3D Translation Error [m]:  {:.4f}, {:.2f}%'.format(
                ate3d, ate3d_pct*100))
            self.metric_results[single_class]["ate_3d"] = ate3d
            self.metric_results[single_class]["ate_3d_pct"] = ate3d_pct*100
            # ASE： 3d box 体积的比值，作为3d IOU
            # class_dict['T_p_ate']: x y z l w h r x1 y1 x1 y2 score
            # class_dict['gt_ate']: x y z l w h r x1 y1 x1 y2
            ase = self.compute_ase(class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average Scale Error:  %.4f ' % ase)
            self.metric_results[single_class]["ase"] = ase
            # AOE： yaw 角误差的均值
            aoe = self.compute_aoe(class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average Orientation Error [rad]:  {:.4f}, [degree] {:.4f}'.format(
                aoe, aoe * 180 / np.pi))
            self.metric_results[single_class]["aoe"] = aoe * 180 / np.pi
            print(" ")
            indicator_res[idx].append(self.metric_results[single_class]["recall_range"][-1])
            indicator_res[idx].append(self.metric_results[single_class]["precision_range"][-1])
            indicator_res[idx].append(mean_ap)
            indicator_res[idx].append(ate3d)
            indicator_res[idx].append(ate3d_pct*100)
            indicator_res[idx].append(aoe)
            indicator_res[idx].append(aoe * 180 / np.pi)

        # 打印计算结果
        self.time = float(time.time() - self.time)
        print("Total evaluation time: %.5f " % self.time)
        for idx, indicator in enumerate(indicator_res):
            single_class_res = ''
            for ind in indicator:
                single_class_res += ' {:.3f}'.format(ind)
            if single_class_res == '':
                print(self.classes[idx], "no evaluation results")
            else:
                print(self.classes[idx], ' :', single_class_res, '%', sep='')

    def compute_recall_precision(self, single_class):
        match_pair = self.metric_results[single_class]['match_pair']
        tp = match_pair[match_pair[:, 0] == 1]
        fp = match_pair[match_pair[:, 0] == 0]
        gt = match_pair[(match_pair[:, 0] == 1) + (match_pair[:, 0] == -1)]
        pred = match_pair[match_pair[:, 0] >= 0]
        self.metric_results[single_class]["fp_pred"] = np.array([sum((fp[:, 2] > i*10) * (fp[:, 2] <= (i+1)*10)) for i in range(20)]) # 统计[0,200]米间10米间隔的个数            
        self.metric_results[single_class]["fp_gt"] = np.array([sum((fp[:, 3] > i*10) * (fp[:, 3] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["tp_pred"] = np.array([sum((tp[:, 2] > i*10) * (tp[:, 2] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["tp_gt"] = np.array([sum((tp[:, 3] > i*10) * (tp[:, 3] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["pred"] = np.array([sum((pred[:, 2] > i*10) * (pred[:, 2] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["gt"] = np.array([sum((gt[:, 3] > i*10) * (gt[:, 3] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["recall_ring"] = self.metric_results[single_class]["tp_gt"] / (self.metric_results[single_class]["gt"] + 1e-5)
        self.metric_results[single_class]["recall_range"] = np.array([sum(tp[:, 3] < i*10) / (sum(gt[:, 3] < i*10) + 1e-5)for i in range(20)])
        self.metric_results[single_class]["precision_ring"] = self.metric_results[single_class]["tp_pred"] / (self.metric_results[single_class]["pred"] + 1e-5)
        self.metric_results[single_class]["precision_range"] = np.array([sum(tp[:, 2] < i*10) / (sum(pred[:, 2] < i*10) + 1e-5) for i in range(20)])
        # import pdb;pdb.set_trace()
        plt.figure()
        l1, = plt.plot([10*i for i in range(self.metric_results[single_class]["recall_ring"].shape[0])], 
                self.metric_results[single_class]["recall_ring"])

        l3, = plt.plot([10*i for i in range(self.metric_results[single_class]["precision_ring"].shape[0])],
                self.metric_results[single_class]["precision_ring"])

        plt.legend(handles=[l1, l3],
                labels=['recall_range', 'precision_range'],
                loc='best')
        plt.title(single_class + "_pr_ring")
        plt.xlabel("distance")
        plt.ylabel("percentage")
        plt.savefig(os.path.join(self.save_loc, self.area_name, single_class + "_pr_ring.png"))
        plt.close()
        
        plt.figure()
        l2, = plt.plot([10*i for i in range(self.metric_results[single_class]["recall_range"].shape[0])],
                self.metric_results[single_class]["recall_range"])
        l4, = plt.plot([10*i for i in range(self.metric_results[single_class]["precision_range"].shape[0])],
                self.metric_results[single_class]["precision_range"])
        plt.legend(handles=[l2, l4],
                labels=['recall_range', 'precision_range'],
                loc='best')
        plt.title(single_class + "_pr_range")
        plt.xlabel("distance")
        plt.ylabel("percentage")
        plt.savefig(os.path.join(self.save_loc, self.area_name, single_class + "_pr_range.png"))
        plt.close()

    def compute_ap_curve(self, class_dict):
        t_pos = 0
        class_dict['precision'] = np.ones(class_dict['result'].shape[0]+2)
        class_dict['recall'] = np.zeros(class_dict['result'].shape[0]+2)
        # 按照pred的score降序排序
        sorted_detections = class_dict['result'][(-class_dict['result'][:, 1]).argsort(), :]
        for i, (result_bool, result_score, _, _, _) in enumerate(sorted_detections):
            if result_bool == 1: # TP
                t_pos += 1
            # t_pos 表示当预测个数为i+1时，最多可能有多少个TP
            class_dict['precision'][i+1] = t_pos / (i + 1) # 当预测个数为i+1个时的 precision
            class_dict['recall'][i+1] = t_pos / class_dict['total_N_pos'] # 当预测个数为i+1个时的 recall
        class_dict['precision'][i+2] = 0
        class_dict['recall'][i+2] = class_dict['recall'][i+1]

        ## Plot
        plt.figure()
        plt.plot(class_dict['recall'], class_dict['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve for {} Class'.format(class_dict['class']))
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.savefig(os.path.join(self.save_loc, self.area_name, class_dict['class'] + "_pr_curve.png"))
        plt.close()

    def compute_f1_score(self, precision, recall):
        p, r = precision[(precision+recall) > 0], recall[(precision+recall) > 0]
        f1_scores = 2 * p * r / (p + r) # 根据公式计算 F1 score
        return np.max(f1_scores)

    def compute_mean_ap(self, precision, recall, precision_threshold=0.0, recall_threshold=0.0):
        mean_ap = 0
        threshold_mask = np.logical_and(precision > precision_threshold,
                                        recall > recall_threshold)
        # calculate mean AP
        precision = precision[threshold_mask]
        recall = recall[threshold_mask]
        recall_diff = np.diff(recall)
        precision_diff = np.diff(precision)
        # Square area under curve based on i+1 precision, then linear difference in precision
        # precision[1:]*recall_diff 是用precision[i]和recall[i+1]-recall[i]计算的矩形面积
        # recall_diff*precision_diff/2 是用precision[i+1]-precision[i] 和 recall[i+1]-recall[i] 计算的矩形上方的小三角形的面积
        # 总的来说，这里的mean_ap只是一个类别的PR曲线下方的面积
        mean_ap = np.sum(precision[1:]*recall_diff + recall_diff*precision_diff/2)
        # We need to divide by (1-recall_threshold) to make the max possible mAP = 1. In practice threshold by the first
        # considered recall value (threshold = 0.1 -> first considered value may be = 0.1123)
        mean_ap = mean_ap/(1-recall[0]) # 归一化，原理暂时没有深入分析
        return mean_ap

    def compute_ate2d(self, predictions, ground_truth):
        # euclidean distance 3d
        mean_ate2d = np.mean(np.linalg.norm( # pred和gt在bev平面的欧式距离 均值
            predictions[:, :2] - ground_truth[:, :2], axis=-1))
        mean_ate2d_pct = np.mean(np.linalg.norm( # pred和gt在bev平面的欧式距离占gt到原点距离的百分百 均值
            predictions[:, :2] - ground_truth[:, :2], axis=-1) / np.linalg.norm(ground_truth[:, :2], axis=-1))
        return mean_ate2d, mean_ate2d_pct

    def compute_ate2d_by_distance(self, predictions, ground_truth):
        # euclidean distance 2d
        all_gt_xyz = [gt_info[0:3] for gt_info in ground_truth]
        all_gt_xyz = np.array(all_gt_xyz).astype(np.float32)
        xyz_max = np.max(all_gt_xyz, axis=0)
        # x
        max_x_range = int(xyz_max[0]/10)+1
        range_x_inds = [[i for i in range(max_x_range)], [i+1 for i in range(max_x_range)]*2]
        range_x_inds[0].extend([0 for i in range(max_x_range)])
        for i in range(len(range_x_inds[0])):
            valid_idx = np.where((ground_truth[:,0]> range_x_inds[0][i]*10)&(ground_truth[:,0]<= range_x_inds[1][i]*10))[0]
            if len(valid_idx):
                pred=predictions[valid_idx]
                gt=ground_truth[valid_idx]
                mean_ate2d = np.mean(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1))
                mean_ate2d_pct = np.mean(np.linalg.norm(
                    pred[:, :2] - gt[:, :2], axis=-1) / np.linalg.norm(gt[:, :2], axis=-1))

                x_err = np.mean(pred[:, 0] - gt[:, 0])
                x_err_pct = np.mean((pred[:, 0] - gt[:, 0]) / gt[:, 0])

                y_err = np.mean(pred[:, 1] - gt[:, 1])
                y_err_pct = np.mean((pred[:, 1] - gt[:, 1]) / gt[:, 1])

                x_err_abs = np.mean(np.abs(pred[:, 0] - gt[:, 0]))
                x_err_abs_pct = np.mean(np.abs(pred[:, 0] - gt[:, 0]) / np.abs(gt[:, 0]))

                y_err_abs = np.mean(np.abs(pred[:, 1] - gt[:, 1]))
                y_err_abs_pct = np.mean(np.abs(pred[:, 1] - gt[:, 1]) / np.abs(gt[:, 1]))

                print('eval range: x(', range_x_inds[0][i]*10,',', range_x_inds[1][i]*10,'] ATE[m]: {:.4f}, {:.2f}%, x_err[m]: {:.4f}, {:.2f}%, y_err[m]: {:.4f}, {:.2f}%, x_err_abs[m]: {:.4f}, {:.2f}%, y_err_abs[m]: {:.4f}, {:.2f}% '.format(
                    mean_ate2d, mean_ate2d_pct*100, x_err, x_err_pct*100, y_err, y_err_pct*100, x_err_abs, x_err_abs_pct*100, y_err_abs, y_err_abs_pct*100))
        # y
        # max_y_range = int(xyz_max[1]/10)+1
        # range_y_inds = [[i for i in range(max_y_range)], [i+1 for i in range(max_y_range)]*2]
        # range_y_inds[0].extend([0 for i in range(max_y_range)])
        # for i in range(len(range_y_inds[0])):
        #     valid_idx = np.where((ground_truth[:,1]> range_y_inds[0][i]*10)&(ground_truth[:,1]<= range_y_inds[1][i]*10))[0]
        #     if len(valid_idx):
        #         pred=predictions[valid_idx]
        #         gt=ground_truth[valid_idx]
        #         mean_ate2d = np.mean(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1))
        #         mean_ate2d_pct = np.mean(np.linalg.norm(
        #             pred[:, :2] - gt[:, :2], axis=-1) / np.linalg.norm(gt[:, :2], axis=-1))

        #         x_err = np.mean(pred[:, 0] - gt[:, 0])
        #         x_err_pct = np.mean((pred[:, 0] - gt[:, 0]) / gt[:, 0])

        #         y_err = np.mean(pred[:, 1] - gt[:, 1])
        #         y_err_pct = np.mean((pred[:, 1] - gt[:, 1]) / gt[:, 1])

        #         x_err_abs = np.mean(np.abs(pred[:, 0] - gt[:, 0]))
        #         x_err_abs_pct = np.mean(np.abs(pred[:, 0] - gt[:, 0]) / np.abs(gt[:, 0]))

        #         y_err_abs = np.mean(np.abs(pred[:, 1] - gt[:, 1]))
        #         y_err_abs_pct = np.mean(np.abs(pred[:, 1] - gt[:, 1]) / np.abs(gt[:, 1]))

        #         print('eval range: y(', range_y_inds[0][i]*10,',', range_y_inds[1][i]*10,'] ATE[m]: {:.4f}, {:.2f}%, x_err[m]: {:.4f}, {:.2f}%, y_err[m]: {:.4f}, {:.2f}%, x_err_abs[m]: {:.4f}, {:.2f}%, y_err_abs[m]: {:.4f}, {:.2f}% '.format(
        #             mean_ate2d, mean_ate2d_pct*100, x_err, x_err_pct*100, y_err, y_err_pct*100, x_err_abs, x_err_abs_pct*100, y_err_abs, y_err_abs_pct*100))
        
    
    def compute_ate2d_by_distance_plt(self, predictions, ground_truth, single_class):
        # x
        all_x_err = predictions[:, 0] - ground_truth[:, 0]
        all_x_err = all_x_err.tolist()
        plt.figure() 
        plt.scatter(ground_truth[:,0].tolist(),all_x_err,s=1,alpha=0.5)
        plt.grid(alpha=0.5,linestyle='-.') 
        plt.xlabel('gt x(m)')  
        plt.ylabel('x err(m)')  
        plt.title(single_class+' x error') 
        plt.savefig(os.path.join(self.save_loc, self.area_name, single_class + "_x_err_distance.png"))
        # y
        all_y_err = predictions[:, 1] - ground_truth[:, 1]
        all_y_err = all_y_err.tolist()
        plt.figure() 
        plt.scatter(ground_truth[:,1].tolist(),all_y_err,s=1,alpha=0.5)
        plt.grid(alpha=0.5,linestyle='-.') 
        plt.xlabel('gt y(m)')  
        plt.ylabel('y err(m)')  
        plt.title(single_class+' y error') 
        plt.savefig(os.path.join(self.save_loc, self.area_name, single_class + "_y_err_distance.png"))
    
    
    def compute_ate3d(self, predictions, ground_truth):
        # euclidean distance 2d
        mean_ate3d = np.mean(np.linalg.norm( # pred和gt在3D空间的欧式距离 均值
            predictions[:, :3] - ground_truth[:, :3], axis=-1))
        mean_ate3d_pct = np.mean(np.linalg.norm( # pred和gt在3D空间的欧式距离占gt到原点距离的百分百 均值
            predictions[:, :3] - ground_truth[:, :3], axis=-1) / np.linalg.norm(ground_truth[:, :3], axis=-1))
        return mean_ate3d, mean_ate3d_pct

    def compute_ase(self, predictions, ground_truth):
        # simplified iou where boxes are centered and aligned with eachother
        pred_vol = predictions[:, 3]*predictions[:, 4]*predictions[:, 5] # w*h*l，体积
        gt_vol = ground_truth[:, 3]*ground_truth[:, 4]*ground_truth[:, 5] # w*h*l，体积
        iou3d = np.mean(1 - np.minimum(pred_vol, gt_vol)/np.maximum(pred_vol, gt_vol)) # 1减去最小体积与最大体积的比值，再求均值
        return iou3d

    def compute_aoe(self, predictions, ground_truth):
        err = ground_truth[:,6] - predictions[:,6]
        aoe = np.mean(np.abs((err + np.pi) % (2*np.pi) - np.pi))
        return aoe

    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (x1, y1, x2, y2), which reflects
                (left, top, right, bottom)
        :param rec2: (x1, y1, x2, y2)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
    
        # find the each edge of intersect rectangle
        left_line = max(rec1[0], rec2[0])
        right_line = min(rec1[2], rec2[2])
        top_line = max(rec1[1], rec2[1])
        bottom_line = min(rec1[3], rec2[3])
    
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0

    
    def compute_ious(self, predictions, gt_bbox):
        ious = np.zeros((len(predictions), 1))
        for idx in range(len(predictions)):
            pred_bbox = predictions[idx,:]
            ious[idx,:] = self.compute_iou(pred_bbox,gt_bbox)
        return ious
    
    def eval_pair(self, pred_label, gt_label):
        # pred_label: class x y z l w h r x1 y1 x1 y2 score
        # gt_label: class x y z l w h r x1 y1 x1 y2
        ## Check
        assert pred_label.shape[1] == 13
        assert gt_label.shape[1] == 12
        assert len(pred_label) == len(self.pred_ignore)
        assert len(gt_label) == len(self.gt_ignore)

        matched_result = -1 * np.ones(len(pred_label))
        matched_gt = -1 * np.ones(len(gt_label))

        ## Threshold score
        score_index = []
        if pred_label.shape[0] > 0: # 挑出大于阈值 self.score_threshold 的 pred
            score_index = np.where(pred_label[:, 12].astype(np.float) > self.score_threshold)[0]
            pred_label = pred_label[pred_label[:, 12].astype(np.float) > self.score_threshold, :]
            self.pred_ignore = self.pred_ignore[score_index]
           
        for single_class in self.classes: # 每个类别分别匹配
            # get all pred labels, order by score
            # 取出当前类别的pred，并按score进行降序排序
            valid_idx = np.where(pred_label[:, 0].astype(str) == single_class)[0]
            # import pdb;pdb.set_trace()
            class_pred_label = pred_label[valid_idx, 1:]
            score = class_pred_label[:, 11].astype(np.float)
            score_order = (-score).argsort()
            class_pred_label = class_pred_label[score_order, :].astype(np.float) # sort decreasing
            # import pdb;pdb.set_trace()
            self.pred_ignore_single_class = self.pred_ignore[valid_idx][score_order] # 


            # add gt label length to total_N_pos
            # 取出当前类别的gt，计算total_N_pos
            valid_gt_idx = np.where(gt_label[:, 0].astype(str) == single_class)[0]
            class_gt_label = gt_label[gt_label[:, 0].astype(str) == single_class, 1:].astype(np.float)
            self.gt_ignore_single_class = self.gt_ignore[valid_gt_idx]
            # total_N_pos就是过滤后的gt的个数
            self.results_dict[single_class]['total_N_pos'] += np.sum(self.gt_ignore_single_class == 0)


            # match pairs for ap
            # pred_array: TP
            # gt_array: 与TP匹配的gt
            # result_score_pair: [n,5]，第一列取1,0,-1，分别对应TP,FP,FN，后四列为 匹配的pred的score，bev下匹配的pred中心到光心的欧式距离，bev下匹配的gt中心到光心的欧式距离，匹配的gt和pred之间的IOU
            # matched_pred_index: 匹配了的pred的下标
            # matched_gt_index: 与pred对应的匹配gt的下标
            pred_array, gt_array, result_score_pair, matched_pred_index, matched_gt_index = self.match_pairs_km_bbox2d(class_pred_label, class_gt_label, self.iou_threshold)
            if len(score_index) > 0:
                matched_result[score_index[valid_idx]] = 0 # 没有被匹配的pred
                matched_result[score_index[valid_idx[score_order[matched_pred_index]]]] = 1 # 匹配了的pred
                matched_result[score_index[valid_idx[score_order[np.where(self.pred_ignore_single_class == 1)[0]]]]] = -1 # ignore 的 pre 设为-1
            matched_gt[valid_gt_idx] = 0 # 没有被匹配的gt
            matched_gt[valid_gt_idx[matched_gt_index]] = 1 # 被匹配了的gt
            matched_gt[valid_gt_idx[np.where(self.gt_ignore_single_class == 1)[0]]] = -1 # ignore 的 gt 设为-1
            # import pdb;pdb.set_trace()

            # add to existing results
            self.results_dict[single_class]['T_p'] = np.vstack((self.results_dict[single_class]['T_p'], pred_array))
            self.results_dict[single_class]['gt'] = np.vstack((self.results_dict[single_class]['gt'], gt_array))
            self.results_dict[single_class]['result'] = np.vstack((self.results_dict[single_class]['result'],
                                                                   result_score_pair))

            # match pairs for ATE
            # pred_array, gt_array, result_score_pair, _ , _ = self.match_pairs_km(class_pred_label, class_gt_label, self.distance_threshold_ate)

            # add to existing results
            # 和上面的 'T_p'、'gt'、'result' 是一样的
            self.results_dict[single_class]['T_p_ate'] = np.vstack((self.results_dict[single_class]['T_p_ate'], pred_array))
            self.results_dict[single_class]['gt_ate'] = np.vstack((self.results_dict[single_class]['gt_ate'], gt_array))
            self.results_dict[single_class]['result_ate'] = np.vstack((self.results_dict[single_class]['result_ate'],
                                                                   result_score_pair))
        return matched_result, matched_gt

    def match_pairs(self, pred_label, gt_label):
        true_preds = np.empty((0, 8))
        corresponding_gt = np.empty((0, 7))
        result_score = np.empty((0, 2))
        # Initialize matching loop
        match_incomplete = True
        while match_incomplete and gt_label.shape[0] > 0:
            match_incomplete = False
            for gt_idx, single_gt_label in enumerate(gt_label):
                # Check is any prediction is in range
                distance_sq_array = (single_gt_label[0] - pred_label[:, 0])**2 + (single_gt_label[1] - pred_label[:, 1])**2
                # If there is a prediction in range, pick closest
                if np.any(distance_sq_array < self.distance_threshold_sq):
                    min_idx = np.argmin(distance_sq_array)
                    # Store true prediction
                    true_preds = np.vstack((true_preds, pred_label[min_idx, :].reshape(-1, 1).T))
                    corresponding_gt = np.vstack((corresponding_gt, gt_label[gt_idx]))

                    # Store score for mAP
                    result_score = np.vstack((result_score, np.array([[1, pred_label[min_idx, 7]]])))

                    # Remove prediction and gt then reset loop
                    pred_label = np.delete(pred_label, obj=min_idx, axis=0)
                    gt_label = np.delete(gt_label, obj=gt_idx, axis=0)
                    match_incomplete = True
                    break

        # If there were any false detections, add them.
        if pred_label.shape[0] > 0:
            false_positives = np.zeros((pred_label.shape[0], 2))
            false_positives[:, 1] = pred_label[:, 11]
            result_score = np.vstack((result_score, false_positives))
        return true_preds, corresponding_gt, result_score

    def match_pairs_km(self, pred_label, gt_label, distance_threshold=0.1):
        assert len(self.gt_ignore_single_class) == len(gt_label)
        assert len(self.pred_ignore_single_class) == len(pred_label)
        
        true_preds = np.empty((0, 12))
        corresponding_gt = np.empty((0, 11))
        result_score = np.empty((0, 5))
        # calculate similarity between predictions and gts
        score_metric = np.zeros((len(pred_label), len(gt_label)))
        gts_visited = np.ones(len(gt_label)) * -1
        # distance
        for gt_idx in range(len(gt_label)):
            distance = np.linalg.norm(
                gt_label[gt_idx, :2] - pred_label[:, :2], axis=-1)
            score_metric[:, gt_idx] = 1. / distance
        
        # km
        km_matcher = KMMatcher()
        score_metric = score_metric.transpose()
        km_matcher.setInformationMatrix(score_metric)
        km_matcher.processKM()
        matched_result = km_matcher.getMatchedResult()

        # process matched results
        matched_pred_index = []
        matched_gt_index = []
        for pred_idx in range(matched_result.size):
            gt_idx = matched_result[pred_idx]
            pred_distance = np.linalg.norm(pred_label[pred_idx, :2])
            if gt_idx >= 0:
                gts_visited[gt_idx] = 1
                gt_distance = np.linalg.norm(gt_label[gt_idx, :2])
                if distance_threshold == "small_object":
                    if gt_distance < 50:
                        threshold = -0.0006389*gt_distance**2 + 0.22733382*gt_distance + 0.27651913
                        threshold = 1/threshold
                    else:
                        threshold = 1. / (10 + 0.2 * (gt_distance - 50))
                elif distance_threshold == 0:
                    threshold = 0.0
                else:
                    threshold = 1. / (distance_threshold * gt_distance)
                    
                
                if score_metric[gt_idx][pred_idx] > threshold:
                    if self.gt_ignore_single_class[gt_idx] != 1:
                        true_preds = np.vstack(
                            (true_preds, pred_label[pred_idx, :].reshape(-1, 1).T))
                        corresponding_gt = np.vstack(
                            (corresponding_gt, gt_label[gt_idx]))
                        matched_pred_index.append(pred_idx)
                        matched_gt_index.append(gt_idx)

                        # Store score for mAP
                        result_score = np.vstack(
                            (result_score, np.array([[1, pred_label[pred_idx, 11], pred_distance, gt_distance, 1/score_metric[gt_idx][pred_idx]]])))
                    else:
                        self.pred_ignore_single_class[pred_idx] = 1
                else:
                    if self.gt_ignore_single_class[gt_idx] != 1:
                        # FP
                        result_score = np.vstack(
                            (result_score, np.array([[0, pred_label[pred_idx, 11], pred_distance, gt_distance, 1/score_metric[gt_idx][pred_idx]]])))
                    else:
                        self.pred_ignore_single_class[pred_idx] = 1
            else:
                # FP
                if self.pred_ignore_single_class[pred_idx] != 1:
                    result_score = np.vstack(
                        (result_score, np.array([[0, pred_label[pred_idx, 11], pred_distance, -1, -1]])))

        for gt_idx in range(len(gts_visited)):
            if gts_visited[gt_idx] == 1:
                continue
            if self.gt_ignore_single_class[gt_idx] == 1:
                continue
            gt_distance = np.linalg.norm(gt_label[gt_idx, :2])
            result_score = np.vstack(
                    (result_score, np.array([[-1, -1, -1, gt_distance, -1]])))



        return true_preds, corresponding_gt, result_score, matched_pred_index, matched_gt_index

    def match_pairs_km_bbox2d(self, pred_label, gt_label, iou_threshold=0.1):
        assert len(self.gt_ignore_single_class) == len(gt_label)
        assert len(self.pred_ignore_single_class) == len(pred_label)
        
        true_preds = np.empty((0, 12)) # [n,12]
        corresponding_gt = np.empty((0, 11)) # [n,11]
        result_score = np.empty((0, 5)) # [n,5]
        # calculate similarity between predictions and gts
        score_metric = np.zeros((len(pred_label), len(gt_label))) # [n_pre, n_gt]，每行是一个pred和所有gt的iou
        gts_visited = np.ones(len(gt_label)) * -1 # [n_gt]
        
        # bbox2d iou
        for gt_idx in range(len(gt_label)):
            iou = self.compute_ious(pred_label[:, 7:11], gt_label[gt_idx, 7:11])
            score_metric[:, gt_idx] = iou.reshape(-1)
        
        # km
        km_matcher = KMMatcher()
        score_metric = score_metric.transpose() # X集合是gt，Y集合是pred，(行是gt，列式pred)
        km_matcher.setInformationMatrix(score_metric)
        km_matcher.processKM() # 为每个gt都找到一个匹配的pred
        matched_result = km_matcher.getMatchedResult() # 匹配结果 matched_result[i] 表示下标为i的pred匹配了下标为 matched_result[i] 的 gt

        # process matched results
        matched_pred_index = []
        matched_gt_index = []
        num_tp = 0
        num_fp = 0
        num_fn = 0
        for pred_idx in range(matched_result.size): # 遍历匹配结果
            gt_idx = matched_result[pred_idx] # 取出匹配结果
            pred_distance = np.linalg.norm(pred_label[pred_idx, :2])
            
            if gt_idx >= 0:
                # gts_visited[gt_idx] = 1
                gt_distance = np.linalg.norm(gt_label[gt_idx, :2])

                if score_metric[gt_idx][pred_idx] > iou_threshold: # 如果IOU大于阈值
                    if self.gt_ignore_single_class[gt_idx] != 1: # gt没有被过滤
                        # 收集匹配结果
                        true_preds = np.vstack(
                            (true_preds, pred_label[pred_idx, :].reshape(-1, 1).T))
                        corresponding_gt = np.vstack(
                            (corresponding_gt, gt_label[gt_idx]))
                        matched_pred_index.append(pred_idx)
                        matched_gt_index.append(gt_idx)
                        gts_visited[gt_idx] = 1

                        # Store score for mAP
                        num_tp += 1 # 正确检出
                        result_score = np.vstack( # 五个值分别为 1，匹配的pred的score，bev下匹配的pred中心到光心的欧式距离，bev下匹配的gt中心到光心的欧式距离，匹配的gt和pred之间的IOU
                            (result_score, np.array([[1, pred_label[pred_idx, 11], pred_distance, gt_distance, score_metric[gt_idx][pred_idx]]])))
                    else: # gt被过滤，与之匹配的pred也应该被过滤
                        self.pred_ignore_single_class[pred_idx] = 1
                else: # 匹配了gt，但是IOU小于阈值
                    if self.gt_ignore_single_class[gt_idx] != 1: # gt没有被过滤，IOU较小的误检
                        # FP
                        num_fp += 1 # 误检
                        result_score = np.vstack( # 五个值分别为 0，匹配的pred的score，bev下匹配的pred中心到光心的欧式距离，bev下匹配的gt中心到光心的欧式距离，匹配的gt和pred之间的IOU
                            (result_score, np.array([[0, pred_label[pred_idx, 11], pred_distance, gt_distance, score_metric[gt_idx][pred_idx]]])))
                    else: # gt被过滤，与之匹配的pred也应该被过滤
                        self.pred_ignore_single_class[pred_idx] = 1
            else: # 没有gt匹配，误检
                # FP 
                num_fp += 1 # 误检
                if self.pred_ignore_single_class[pred_idx] != 1: # pred 没有被过滤
                    result_score = np.vstack( # 五个值分别为 0，匹配的pred的score，bev下匹配的pred中心到光心的欧式距离，bev下匹配的gt中心到光心的欧式距离，匹配的gt和pred之间的IOU
                        (result_score, np.array([[0, pred_label[pred_idx, 11], pred_distance, -1, -1]])))

        for gt_idx in range(len(gts_visited)): # 遍历每个gt
            if gts_visited[gt_idx] == 1: # gt被匹配
                continue
            if self.gt_ignore_single_class[gt_idx] == 1: # gt被过滤
                continue
            num_fn += 1 # 漏检
            gt_distance = np.linalg.norm(gt_label[gt_idx, :2])
            result_score = np.vstack( # 五个值分别为 -1，匹配的pred的score，bev下匹配的pred中心到光心的欧式距离，bev下匹配的gt中心到光心的欧式距离，匹配的gt和pred之间的IOU
                    (result_score, np.array([[-1, -1, -1, gt_distance, -1]])))
        # true_preds: TP
        # corresponding_gt: 与TP匹配的gt
        # result_score: [n,5]，第一列取1,0,-1，分别对应TP,FP,FN，后四列为 匹配的pred的score，bev下匹配的pred中心到光心的欧式距离，bev下匹配的gt中心到光心的欧式距离，匹配的gt和pred之间的IOU
        # matched_pred_index: 匹配了的pred的下标
        # matched_gt_index: 与pred对应的匹配gt的下标
        return true_preds, corresponding_gt, result_score, matched_pred_index, matched_gt_index
    
    
    # def filter_by_range(self, pred_label, gt_label, range=0):
    #     pred_dist = np.linalg.norm(pred_label[:, 1:4].astype(np.float32), axis=1) < range
    #     gt_dist = np.linalg.norm(gt_label[:, 1:4].astype(np.float32), axis=1) < range
    #     return pred_label[pred_dist, :], gt_label[gt_dist, :]
    def filter_by_range(self,
                        pred_label,
                        gt_label,
                        point_range=[-50, -50, -5.0, 50, 50, 3.0]):
        valid_pred_index = [
            i for i in range(len(pred_label))
            if self.is_inside_point_cloud_range(
                pred_label[i, 1:4].astype(np.float32), point_range)
        ]
        valid_gt_index = [
            i for i in range(len(gt_label)) if self.is_inside_point_cloud_range(
                gt_label[i, 1:4].astype(np.float32), point_range)
        ]
        return pred_label[valid_pred_index, :], gt_label[valid_gt_index, :], valid_pred_index, valid_gt_index

    def filter_by_gt_range(self,
                        pred_label,
                        gt_label,
                        point_range=[-50, -50, -5.0, 50, 50, 3.0]):
        # import pdb;pdb.set_trace()
        all_gt_xyz = [gt_info[1:4] for gt_info in gt_label]
        if len(all_gt_xyz):
            # all_gt_xyz = np.concatenate(all_gt_xyz, axis=0).astype(np.float32)
            all_gt_xyz = np.array(all_gt_xyz).astype(np.float32)
            xyz_min = np.min(all_gt_xyz, axis=0)
            xyz_max = np.max(all_gt_xyz, axis=0)
            point_range = [0, xyz_min[1] - 10, -5, xyz_max[0] + 10, xyz_max[1] + 10, 5]
            # point_cloud_range = [0.0, -60, -5, 100, 60, 5]
            valid_pred_index = [
                i for i in range(len(pred_label))
                if self.is_inside_point_cloud_range(
                    pred_label[i, 1:4].astype(np.float32), point_range)
            ]
            valid_gt_index = [
                i for i in range(len(gt_label)) if self.is_inside_point_cloud_range(
                    gt_label[i, 1:4].astype(np.float32), point_range)
            ]
            return pred_label[valid_pred_index, :], gt_label[valid_gt_index, :], valid_pred_index, valid_gt_index
        
        valid_pred_index = []
        valid_gt_index = []
        return pred_label[valid_pred_index, :], gt_label[valid_gt_index, :], valid_pred_index, valid_gt_index

    def ignore_by_gt_range(self,
                           pred_label,
                           gt_label,
                           point_range=[-50, -50, -5.0, 50, 50, 3.0]):
        # import pdb;pdb.set_trace()
        all_gt_xyz = [gt_info[1:4] for gt_info in gt_label]
        if len(all_gt_xyz):
            # all_gt_xyz = np.concatenate(all_gt_xyz, axis=0).astype(np.float32)
            all_gt_xyz = np.array(all_gt_xyz).astype(np.float32)
            xyz_min = np.min(all_gt_xyz, axis=0)
            xyz_max = np.max(all_gt_xyz, axis=0)
            point_range = [0, xyz_min[1] - 10, -10, xyz_max[0] + 10, xyz_max[1] + 10, 10]
            for i in range(len(pred_label)): # 判断每个预测是否在point_cloud_range中
                if not self.is_inside_point_cloud_range(
                    pred_label[i, 1:4].astype(np.float32), point_range):
                    self.pred_ignore[i] = 1   

    def is_inside_point_cloud_range(self, point, point_range):
        if (point[0] <= point_range[3] and point[0] >= point_range[0]) and (
                point[1] <= point_range[4] and
                point[1] >= point_range[1]) and (point[2] <= point_range[5] and
                                                 point[2] >= point_range[2]):
            return True
        else:
            return False
