# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Calculate sensitivity and specificity metrics:
 - Precision
 - Recall
 - F-score
"""

import numpy as np
from data_io import imread


def cal_prf_ods_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        # print(thresh)
        statistics = []

        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))

        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # calculate recall
        r_acc = tp / (tp + fn)
        # calculate f1-score
        f1_score = 2 * p_acc * r_acc / (p_acc + r_acc)
        final_accuracy_all.append([thresh, p_acc, r_acc, f1_score])

    AP = 0
    for thresh in np.arange(1, 100, 1):
        Re_t = final_accuracy_all[thresh][2]
        Re_t_1 = final_accuracy_all[thresh - 1][2]
        Pr_t = final_accuracy_all[thresh][1]
        AP += (Re_t_1 - Re_t) * Pr_t
    print('AP:{}'.format(AP))

    return final_accuracy_all


def cal_prf_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        best_thresh, f1s = find_best_thresh(pred, gt, thresh_step)
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255 > best_thresh).astype('uint8')
        statistics.append(get_statistics(pred_img, gt_img))

    # get tp, fp, fn
    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])

    # calculate precision
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    # calculate recall
    r_acc = tp / (tp + fn)
    # calculate f1-score
    f1_score = 2 * p_acc * r_acc / (p_acc + r_acc)
    final_accuracy_all.append([p_acc, r_acc, f1_score])
    return final_accuracy_all


def find_best_thresh(pred, gt, thresh_step=0.01, index=0):
    statistics = []
    # calculate each image
    for thresh in np.arange(0.0, 1.0, thresh_step):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255 > thresh).astype('uint8')
        p_acc, r_acc, f1_score = get_statistics_prf(pred_img, gt_img, index)
        statistics.append([thresh, p_acc, r_acc, f1_score])

    f1_index = np.argmax([v[3] for v in statistics])  # Only the first occurrence is returned.
    best_thresh = statistics[f1_index][0]
    return best_thresh, statistics[f1_index][3]


def get_statistics_prf(pred, gt, index=0):
    """
    calculate p_acc, r_acc, f1_score for one image
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    # calculate precision
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    # calculate recall
    r_acc = tp / (tp + fn)
    # calculate f1-score
    f1_score = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc, r_acc, f1_score


'''
def cal_prf_AP(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = cal_prf_ods_metrics(pred_list, gt_list, thresh_step=thresh_step)
    sum = 0
    for thresh in np.arange(1, 100, 1):
        Re_t = final_accuracy_all[thresh][2]
        Re_t_1 = final_accuracy_all[thresh-1][2]
        Pr_t = final_accuracy_all[thresh][1]
        sum += (Re_t_1 - Re_t) * Pr_t
    #AP = sum / 100.0
    #print('AP:{}'.format(AP))
    return AP'''


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]
