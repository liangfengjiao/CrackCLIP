import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import color, segmentation
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import gray2rgb
import torchvision.ops as ops

def nms():
    image_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/image/DeepCrack/image'
    result_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/image/DeepCrack/masks_nms_0.05'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        image = cv2.imread(img_path, 0) / 255
        image_probability = (image > 0.05).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_probability, connectivity=8)
        boxes = []
        scores = []

        for i in range(1, num_labels):  # 从1开始，跳过背景
            x, y, w, h, area = stats[i]
            if area > 10:  # 过滤掉小的区域
                boxes.append([x, y, x + w, y + h])
                scores.append(image[y:y + h, x:x + w].mean())  # 使用区域内的平均概率作为得分

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        iou_threshold = 0.5
        keep_indices = ops.nms(torch.as_tensor(boxes), torch.as_tensor(scores), iou_threshold)

        original_image = cv2.imread(img_path, 0)

        mask = np.zeros_like(image_probability)

        for idx in keep_indices:
            x1, y1, x2, y2 = boxes[idx]
            mask[int(y1):int(y2), int(x1):int(x2)] = image_probability[int(y1):int(y2), int(x1):int(x2)]

        result_image = original_image.copy()
        result_image[mask == 0] = 0
        cv2.imwrite(os.path.join(result_path, img), result_image)



        # result_image = cv2.cvtColor(image_probability, cv2.COLOR_GRAY2BGR) * 255
        #
        # for idx in keep_indices:
        #     x1, y1, x2, y2 = boxes[idx]
        #     cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #
        # cv2.imshow('Crack Detection with NMS', result_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('')

#nms()

def calculate_histogram():
    histograms = np.zeros((256,), dtype=np.float64)
    #image_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/image'
    image_path = './data/crack/pavement/train/crack500/image'

    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        image = cv2.imread(img_path, 0)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        histograms += hist.flatten()

    total_histogram = histograms / histograms.sum()
    # cdf = np.cumsum(total_histogram)
    #
    # threshold = np.argmax(cdf >= 0.9)
    # print(threshold)

    plt.figure(figsize=(12,6))
    plt.plot(total_histogram, color='black')
    plt.title('crack500 Histgram')
    plt.xlim([0,256])

    plt.show()

def post_thr():
    image_path = './results/crack500_cam_{pavement}_crf/vit_large_14_518_boundaryloss(-)/image/DeepCrack/masks_eroded_3_3'
    # for i in range(1, 4):
    result_path = './results/crack500_cam_{pavement}_crf/vit_large_14_518_boundaryloss(-)/image/DeepCrack/masks_eroded_3_3_thr_210'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        image = cv2.imread(img_path, 0)
        image[image < 210] = 0
        #mor_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        #mor_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # dilated = cv2.dilate(image, kernel, itrations=10)
        # # cv2.imwrite('./dilated_10.jpg', dilated)
        #eroded = cv2.erode(image, kernel, iterations=3)
        cv2.imwrite(os.path.join(result_path, img), image)

post_thr()

def post_mor():
    image_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/masks/7Q3A9060-1.jpg'
    #img_path = os.path.join(image_path, img)
    image = cv2.imread(image_path, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(image, kernel, iterations=10)
    #cv2.imwrite('./dilated_10.jpg', dilated)
    eroded = cv2.erode(dilated, kernel, iterations=10)

    cv2.imwrite('./mor_image_10.jpg', eroded)

def calculate_mean_std():
    # 数据集路径
    image_path = './data/crack/pavement/test/CFD/image'
    mean, variance = 0, 0
    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        image = cv2.imread(img_path, 0)
        mean += np.mean(image/255)
        variance += np.var(image/255)
    num_images = len(os.listdir(image_path))
    mean /= num_images
    variance /= num_images
    print(f'Mean: {mean}')
    print(f'Variance: {variance}')

    '''
    crack500/train-----
    Mean: 0.4955951925434395
    Variance: 0.028040982368993016
    crack500/test-----
    Mean: 0.491896053860391
    Variance: 0.03246050350602103
    AEL-----
    Mean: 0.4982924089490743
    Variance: 0.019988259164481433
    DeepCrack-----
    Mean: 0.5864024297749976
    Variance: 0.013544908985744236
    CFD-----
    Mean: 0.5221504096651022
    Variance: 0.0033213659224523503
    '''


def visualize_segmentation(gt_mask, pred_mask, cmap=None):
    """
    可视化分割结果，使用不同颜色标识 FP 和 FN 像素。

    :param gt_mask: 真实掩码。
    :param pred_mask: 预测掩码。
    :param cmap: 自定义颜色映射，默认为 None，使用预定义的颜色映射。
    :return: 可视化结果的图像。
    """
    # 将预测结果转换为二进制掩码
    binary_pred_mask = (pred_mask >= 0.5).astype(int)

    # 创建一个新的图像，用于存放可视化结果
    visualization = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    # 真实正例（True Positive）
    tp = (gt_mask == 1) & (binary_pred_mask == 1)
    visualization[tp] = [0, 255, 0]  # 绿色

    # 漏检（False Positive）
    fp = (gt_mask == 0) & (binary_pred_mask == 1)
    visualization[fp] = [255, 0, 0]  # 红色

    # 错检（False Negative）
    fn = (gt_mask == 1) & (binary_pred_mask == 0)
    visualization[fn] = [0, 0, 255]  # 蓝色

    # 背景（True Negative）
    tn = (gt_mask == 0) & (binary_pred_mask == 0)
    visualization[tn] = [255, 255, 255]  # 白色

    # 如果提供了自定义颜色映射，则使用它
    # if cmap is not None:
    #     cmap = ListedColormap(cmap)
    #     plt.imshow(visualization, cmap=cmap)
    # else:
    #     plt.imshow(visualization)
    #
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()

    return visualization



#calculate_mean_std()

#post_mor()
#calculate_histogram()
#post_thr()
def show_error_pred():
    image_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/image'
    mask_path = './data/crack/pavement/test/DeepCrack/mask'
    save_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/save'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        image = cv2.imread(img_path, 0)
        msk_path = os.path.join(mask_path, img).replace('jpg','png')
        mask = cv2.imread(msk_path, 0)

        vis_image = visualize_segmentation(mask/255, image/255)
        cv2.imwrite(os.path.join(save_path, img), vis_image)

def bilateralFilters():
    image_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/image'
    #mask_path = './data/crack/pavement/test/DeepCrack/mask'
    sigma = 15
    for i in range(10):
        sigma += 10
        save_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/bilateralFilter_9_'+str(sigma)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for img in os.listdir(image_path):
            img_path = os.path.join(image_path, img)
            image = cv2.imread(img_path, 0)
            filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)
            cv2.imwrite(os.path.join(save_path, img), filtered_image)


def mean_shift_smoothing(segmentation_result, bandwidth=None):
    """
    使用 MeanShift 算法平滑图像分割结果。

    :param segmentation_result: 分割结果图像，形状为 (H, W)
    :param bandwidth: 均值移位的带宽，如果为 None，则自动估计
    :return: 平滑后的图像，形状为 (H, W)
    """
    # 将分割结果转换为三通道图像
    #segmentation_result_rgb = #gray2rgb(segmentation_result)

    # 将图像展平为 (H * W, 3) 的形状
    flat_image = segmentation_result.reshape(-1, 3)

    # 估计带宽
    if bandwidth is None:
        bandwidth = estimate_bandwidth(flat_image, quantile=0.2, n_samples=500)

    # 应用 MeanShift 算法
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    labels = ms.labels_

    # 将标签重新 reshape 回 (H, W) 的形状
    smoothed_image = labels.reshape(segmentation_result.shape)

    return smoothed_image

def meanShift():
    image_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/image'
    sigma = 0
    for i in range(10):
        sigma += 2
        save_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/meanShiftFilter_15_'+str(sigma)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for img in os.listdir(image_path):
            img_path = os.path.join(image_path, img)
            image = cv2.imread(img_path)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #filtered_image = mean_shift_smoothing(image)
            spatial_bandwidth = 15
            color_bandwidth = sigma
            filtered_image = cv2.pyrMeanShiftFiltering(image_hsv, spatial_bandwidth, color_bandwidth)
            cv2.imwrite(os.path.join(save_path, img), filtered_image)

#meanShift()
#
# img_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/image/7Q3A9060-1.jpg'
# #img_path = os.path.join(image_path, img)
# image = cv2.imread(img_path)
# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #filtered_image = mean_shift_smoothing(image)
# spatial_bandwidth = 10
# color_bandwidth = 32
# filtered_image = cv2.pyrMeanShiftFiltering(image_hsv, spatial_bandwidth, color_bandwidth)
# cv2.imwrite('./mean_filter_10.jpg', filtered_image)





#bilateralFilters()
# 假设我们有一个分割结果图像
# seg_path = './results/crack500_cam_{pavement}_thr/vit_large_14_518/mask/DeepCrack/image/7Q3A9060-1.jpg'
# segmentation_result = cv2.imread(seg_path, 0)#(np.random.rand(256, 256) * 255).astype(np.uint8)

# 应用双边滤波
# filtered_image_3 = cv2.bilateralFilter(segmentation_result, d=3, sigmaColor=75, sigmaSpace=75)
# filtered_image_5 = cv2.bilateralFilter(segmentation_result, d=5, sigmaColor=75, sigmaSpace=75)
# filtered_image_7 = cv2.bilateralFilter(segmentation_result, d=7, sigmaColor=75, sigmaSpace=75)
# filtered_image_9 = cv2.bilateralFilter(segmentation_result, d=9, sigmaColor=75, sigmaSpace=75)
# filtered_image_11 = cv2.bilateralFilter(segmentation_result, d=11, sigmaColor=75, sigmaSpace=75)
# filtered_image_11_25 = cv2.bilateralFilter(segmentation_result, d=11, sigmaColor=25, sigmaSpace=25)
# filtered_image_11_50 = cv2.bilateralFilter(segmentation_result, d=11, sigmaColor=50, sigmaSpace=50)
# filtered_image_11_75 = cv2.bilateralFilter(segmentation_result, d=11, sigmaColor=75, sigmaSpace=75)
# filtered_image_11_100 = cv2.bilateralFilter(segmentation_result, d=11, sigmaColor=100, sigmaSpace=100)
# filtered_image_11_125 = cv2.bilateralFilter(segmentation_result, d=11, sigmaColor=125, sigmaSpace=125)


# 显示原始分割结果和滤波后的结果
# plt.figure(figsize=(30, 6))
# plt.subplot(1, 6, 1)
# plt.imshow(segmentation_result, cmap='gray')
# plt.title('Original Segmentation Result')
# plt.axis('off')
#
# plt.subplot(1, 6, 2)
# plt.imshow(filtered_image_11_25, cmap='gray')
# plt.title('Filtered with Bilateral Filter')
# plt.axis('off')
#
# plt.subplot(1, 6, 3)
# plt.imshow(filtered_image_11_50, cmap='gray')
# plt.title('Filtered with Bilateral Filter')
# plt.axis('off')
#
# plt.subplot(1, 6, 4)
# plt.imshow(filtered_image_11_75, cmap='gray')
# plt.title('Filtered with Bilateral Filter')
# plt.axis('off')
#
# plt.subplot(1, 6, 5)
# plt.imshow(filtered_image_11_100, cmap='gray')
# plt.title('Filtered with Bilateral Filter')
# plt.axis('off')
#
# plt.subplot(1, 6, 6)
# plt.imshow(filtered_image_11_125, cmap='gray')
# plt.title('Filtered with Bilateral Filter')
# plt.axis('off')
# plt.show()