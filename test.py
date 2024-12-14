import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
from matplotlib import image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from os import listdir
from torchvision import transforms
import torch
from resnet import wide_resnet50_2
import math
from scipy.stats import gaussian_kde
import os
import time
from sklearn import metrics


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def evaluation_ATTA(encoder, bn, decoder, dataloader,device, type_of_test, img_size, lamda=0.5, dataset_name='mnist', _class_=None, project=None):
    encoder_train, _ = wide_resnet50_2(pretrained=True)
    encoder_train = encoder_train.to(device)
    encoder_train.eval()
    bn.eval()
    decoder.eval()
    if project != None:
        project.eval()
    gt_list_sp = []
    pr_list_sp = []

    if dataset_name == 'mvtec':
        link_to_normal_sample = '/data/mvtec/' + _class_ + '/train/good/000.png' #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name == 'PACS':
        labels_dict = {
            0: 'dog',
            1: 'elephant',
            2: 'giraffe',
            3: 'guitar',
            4: 'horse',
            5: 'house',
            6: 'person'
        }
        link_to_normal_sample = '/data/PACS/photo/' + labels_dict[_class_] #update the link here
        filenames = [f for f in listdir(link_to_normal_sample)]
        filenames.sort()
        link_to_normal_sample = '/data/PACS/photo/' + labels_dict[_class_] + '/' + filenames[0] #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name != 'mnist':
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)
        ])
    else:
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    normal_image = trans(normal_image)
    normal_image = torch.unsqueeze(normal_image, 0)
    seg_predicted = []
    seg_gt = []
    with torch.no_grad():
        for sample in dataloader:
            img, label = sample[0], sample[1]
            seg_label = label
            label = int(torch.sum(label) != 0)


            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            normal_image = normal_image.to(device)
            img = img.to(device)
            inputs = encoder(img, normal_image, type_of_test, lamda=lamda)
            outputs, _, _, _ = decoder(bn(inputs))
            if project != None:
                outputs = project(outputs)
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            seg_predicted.append(anomaly_map)
            seg_gt.append(seg_label)
            gt_list_sp.append(int(label))
            if math.isnan(np.max(anomaly_map)):
                pr_list_sp.append(0)
            else:
                pr_list_sp.append(np.max(anomaly_map))
        # For pixel-level AUROC
        # auroc_seg = compute_pixelwise_retrieval_metrics(seg_predicted, seg_gt)
        # print('pixel-wise: ', auroc_seg["auroc"])
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_sp


