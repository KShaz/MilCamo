"""
metrics.py

Evaluation metrics for Camouflaged Object Segmentation:
- S-measure (Sα)
- Adaptive E-measure (Eϕad)
- Weighted F-measure (Fβω)
- Mean Absolute Error (MAE)

References:
Fan et al., Enhanced-alignment Measure for Binary Foreground Map Evaluation.
Margolin et al., How to Evaluate Foreground Maps.
Fan et al., Structure-measure: A New Way to Evaluate Foreground Maps.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


EPS = 1e-8


##########################################################################
# MAE
##########################################################################

def mae(gt, pred):
    """
    Mean Absolute Error
    """
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    gt = gt / (gt.max() + EPS)
    pred = pred / (pred.max() + EPS)

    return np.mean(np.abs(gt - pred))


##########################################################################
# S-MEASURE
##########################################################################

def object_score(pred, gt):
    x = pred[gt == 1]
    if len(x) == 0:
        return 0.0

    mean_val = np.mean(x)
    std_val = np.std(x)

    return (2 * mean_val) / (
        mean_val**2 + 1 + std_val + EPS
    )


def centroid(gt):
    h, w = gt.shape

    if np.sum(gt) == 0:
        return w // 2, h // 2

    x = np.round(np.sum(np.arange(w) * np.sum(gt, axis=0))
                 / np.sum(gt))
    y = np.round(np.sum(np.arange(h) * np.sum(gt, axis=1))
                 / np.sum(gt))

    return int(x), int(y)


def s_measure(pred, gt):
    """
    Structural similarity measure Sα
    """

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    pred /= (pred.max() + EPS)
    gt /= (gt.max() + EPS)

    fg = object_score(pred, gt)
    bg = object_score(1 - pred, 1 - gt)

    alpha = 0.5

    return alpha * fg + (1 - alpha) * bg


##########################################################################
# ADAPTIVE E-MEASURE
##########################################################################

def enhanced_alignment_term(fm, gt):
    mu_fm = np.mean(fm)
    mu_gt = np.mean(gt)

    align = 2 * (fm - mu_fm) * (gt - mu_gt)
    align /= (
        (fm - mu_fm) ** 2 +
        (gt - mu_gt) ** 2 +
        EPS
    )

    enhanced = ((align + 1) ** 2) / 4

    return enhanced


def adaptive_emeasure(pred, gt):
    """
    Adaptive E-measure Eϕad
    """

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    pred /= (pred.max() + EPS)
    gt /= (gt.max() + EPS)

    threshold = min(2 * pred.mean(), 1)

    pred_bin = (pred >= threshold).astype(np.float32)

    enhanced = enhanced_alignment_term(
        pred_bin,
        gt
    )

    return enhanced.mean()


##########################################################################
# WEIGHTED F-MEASURE
##########################################################################

def weighted_fmeasure(pred, gt, beta2=0.3):
    """
    Simplified weighted F-measure implementation.
    """

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    pred /= (pred.max() + EPS)
    gt /= (gt.max() + EPS)

    pred = pred >= 0.5
    gt = gt >= 0.5

    tp = np.sum(pred & gt)
    fp = np.sum(pred & (~gt))
    fn = np.sum((~pred) & gt)

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    return ((1 + beta2) * precision * recall) / (
        beta2 * precision + recall + EPS
    )


##########################################################################
# MASTER EVALUATION
##########################################################################

def evaluate(gt, pred):
    """
    Evaluate a prediction mask against ground truth.
    """

    return {
        "S_alpha": s_measure(pred, gt),
        "E_phi_ad": adaptive_emeasure(pred, gt),
        "F_beta_w": weighted_fmeasure(pred, gt),
        "MAE": mae(gt, pred),
    }