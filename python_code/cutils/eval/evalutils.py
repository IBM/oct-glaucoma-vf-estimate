import numpy as np
from sklearn.metrics import precision_recall_curve


def dice_coefficient(pred, gt):
    """
    Computes dice coefficients between two masks
    :param pred: predicted masks - [0 ,1]
    :param gt: ground truth  masks - [0 ,1]
    :return: dice coefficient
    """

    # d = ((np.sum(pred[gt == 1]) * 2.0 ) +1)  / ((np.sum(pred) + np.sum(gt))+1)

    d = (2 * np.sum(pred * gt) + 1) / ((np.sum(pred) + np.sum(gt)) + 1)

    return d


def dice_coefficient_batch(pred, gt, eer_thresh=0.5):
    dice_all = []
    n = pred.shape[0]
    for i in range(n):
        seg = pred[i, :, :]
        seg = (seg >= eer_thresh).astype(np.uint8)
        gtd = gt[i, :, :]

        d = dice_coefficient(seg, gtd)
        dice_all.append(d)

    return dice_all


def prec_recall(gt_imgs, pred_imgs):
    """
     Compute precision, recall and threshold values across all images  to plot the precision recall curve
    :param gt_imgs:  pixel values shoudl be either 0 or 1
    :param pred_imgs: the pixel values can be between [0, 1]
    :return:
    """
    count = 1
    gtarr = []
    oparr = []
    count = 1
    N = gt_imgs.shape[0]
    for i in range(N):
        opim = pred_imgs[i, :, :]
        gtim = gt_imgs[i, :, :]

        opvec = np.asarray(opim).astype(np.float32).flatten()
        gtvec = np.asarray(gtim).astype(np.float32).flatten()

        gtarr.extend(gtvec)
        oparr.extend(opvec)
        # print count
    precision, recall, thresholds = precision_recall_curve(gtarr, oparr)
    return precision, recall, thresholds
