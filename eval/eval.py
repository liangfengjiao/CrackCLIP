# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import numpy as np
import data_io
from prf_metrics import cal_prf_ods_metrics, cal_prf_ois_metrics
#from segment_metrics import cal_semantic_metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='prf', help='[prf | sem]')
parser.add_argument('--f1_threshold_mode', type=str, default='ods', help='[ois | ods]')
parser.add_argument('--model_name', type=str, default='crack500_CAM_Location_pseudo-label')
parser.add_argument('--results_dir', type=str, default='../data/crack/pavement/train/crack500/')
parser.add_argument('--gt_dir', type=str, default='../data/crack/pavement/train/crack500/mask')
parser.add_argument('--output', type=str, default='./metrics/image/crack500')
parser.add_argument('--dataset', type=str, default='crack500')
parser.add_argument('--model_type', type=str, default='')
parser.add_argument('--thresh_step', type=float, default=0.01)
parser.add_argument('--post_or', type=str, default='mask_cam_location')
args = parser.parse_args()

if __name__ == '__main__':
    #sigma = 2
    # for i in range(1, 4):
    #     args.post_or = 'masks_eroded_3_'+str(i)
    metric_mode = args.metric_mode
    results_dir = '../data/crack/pavement/train/crack500/mask_cam_thr'
    #results_dir = os.path.join(args.results_dir, args.model_name, args.model_type, 'image', args.dataset, args.post_or)#os.path.join(args.results_dir, args.model_name, 'pred', 'mask', 'crack500', 'image')
    #results_dir = os.path.join(args.results_dir, args.model_name, 'pred_bg', 'mask', args.dataset, 'image')

    gt_dir = os.path.join(args.gt_dir)
    src_img_list, tgt_img_list = data_io.get_image_pairs_test(gt_dir, results_dir)

    final_results = []
    if metric_mode == 'prf':
        if args.f1_threshold_mode == 'ods':
            final_results = cal_prf_ods_metrics(src_img_list, tgt_img_list, args.thresh_step)
        elif args.f1_threshold_mode == 'ois':
            final_results = cal_prf_ois_metrics(src_img_list, tgt_img_list, args.thresh_step)
        else:
            print('Error f1_threshold_mode!')
    #'crack500_cam_thr_train' + args.f1_threshold_mode + '.prf'
    output_file = args.model_type + '_' + args.f1_threshold_mode + '.prf'
    output_path = os.path.join(args.output, args.model_name, args.post_or)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output = os.path.join(output_path, output_file)
    data_io.save_results(final_results, output)
        #sigma += 2
