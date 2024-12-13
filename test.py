import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import open_clip
from few_shot import memory
from model import LinearLayer
from dataset import VisaDataset, MVTecDataset, CrackDataset
from prompt_ensemble import encode_text_with_prompt_ensemble, encode_crack_text_with_prompt_ensemble


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])

    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.586], std=[0.014])
        ])
    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test')
    elif args.dataset == 'visa':
        test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform)
    else:
        test_data = CrackDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test', parse='test')

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names()

    # few shot
    if args.mode == 'few_shot':
        mem_features = memory(args.model, model, obj_list, dataset_dir, save_path, preprocess, transform,
                              args.k_shot, few_shot_features, dataset_name, device)

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts = encode_crack_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
        #text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    for items in test_dataloader:
        image = items['img'].to(device)
        cls_name = items['cls_name']
        specie_name = items['specie_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())


        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, features_list)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0)

            # sample
            text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
            results['pr_sp'].append(text_probs[0][1].cpu().item())

            # pixel
            patch_tokens = linearlayer(patch_tokens)
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            anomaly_map = np.sum(anomaly_maps, axis=0)

            # few shot
            if args.mode == 'few_shot':
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                anomaly_maps_few_shot = []
                for idx, p in enumerate(patch_tokens):
                    if 'ViT' in args.model:
                        p = p[0, 1:, :]
                    else:
                        p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                    cos = pairwise.cosine_similarity(mem_features[cls_name[0]][idx].cpu(), p.cpu())
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = np.min((1 - cos), 0).reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                         size=img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                anomaly_map = anomaly_map + anomaly_map_few_shot
            
            results['anomaly_maps'].append(anomaly_map)
            #anomaly_map = gaussian_filter(anomaly_map[0], sigma=3)
            #anomaly_map = cv2.bilateralFilter(anomaly_map[0], 3, 75, 75)
            #anomaly_map = cv2.medianBlur(anomaly_map[0], 5)
            # visualization
            path = items['img_path']
            cls = path[0].split('/')[-2]
            filename = path[0].split('/')[-1]
            # vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            mask = normalize(anomaly_map[0])
            #vis = apply_ad_scoremap(vis, mask)
            vis = mask * 255
            # print(path[0])
            # print(cv2.imread(path[0], 0).shape)
            #vis[vis < 50] = 0
            mask_shape = cv2.imread(path[0], 0).shape
            vis = cv2.resize(vis, dsize=(mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
            # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            save_vis = os.path.join(save_path, 'image', specie_name[0], cls)
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)
            cv2.imwrite(os.path.join(save_vis, filename), vis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/crack", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/crack500_cam_{pavement}_fsv/vit_large_14_518_boundaryloss(-)',
                        help='path to save results')
    parser.add_argument("--checkpoint_path", type=str,
                        default='./exps/crack500_cam_{pavement}_fsv/vit_large_14_518_boundaryloss(-)/epoch_3.pth',
                        help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json',
                        help="model configs")  # ViT-B-16.json
    # model
    parser.add_argument("--dataset", type=str, default='crack', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")  # ViT-B-16
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24],
                        help="features used")  # 6, 12, 18, 24/3, 6, 9, 12
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9],
                        help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    # few shot
    parser.add_argument("--k_shot", type=int, default=0, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)
