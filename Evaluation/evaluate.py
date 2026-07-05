import os
import cv2
import numpy as np
from tqdm import tqdm

# COD evaluation metrics package
from cos_eval_tf_metrics import (
    SScore,
    ESimilarityMetric,
    WeightedFScoreMetric,
)

import tensorflow as tf


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    return mask


def calculate_mae(gt, pred):
    return np.mean(np.abs(gt - pred))


def evaluate_dataset(gt_dir, pred_dir):

    s_metric = SScore()
    e_metric = ESimilarityMetric()
    wf_metric = WeightedFScoreMetric()

    mae_scores = []

    gt_files = sorted(os.listdir(gt_dir))

    for fname in tqdm(gt_files):

        gt = load_mask(os.path.join(gt_dir, fname))
        pred = load_mask(os.path.join(pred_dir, fname))

        gt_tf = tf.convert_to_tensor(
            gt[np.newaxis, ..., np.newaxis],
            dtype=tf.float32,
        )

        pred_tf = tf.convert_to_tensor(
            pred[np.newaxis, ..., np.newaxis],
            dtype=tf.float32,
        )

        s_metric.update_state(gt_tf, pred_tf)
        e_metric.update_state(gt_tf, pred_tf)
        wf_metric.update_state(gt_tf, pred_tf)

        mae_scores.append(calculate_mae(gt, pred))

    results = {
        "S_alpha": float(s_metric.result()),
        "E_phi_ad": float(e_metric.result()),
        "F_beta_w": float(wf_metric.result()),
        "MAE": float(np.mean(mae_scores)),
    }

    return results


if __name__ == "__main__":

    GT_DIR = "./ground_truth/"
    PRED_DIR = "./predictions/"

    results = evaluate_dataset(
        GT_DIR,
        PRED_DIR,
    )

    print("\nEvaluation Results")
    print("------------------")
    print(f"Sα       : {results['S_alpha']:.4f}")
    print(f"Eφ_ad    : {results['E_phi_ad']:.4f}")
    print(f"Fβω      : {results['F_beta_w']:.4f}")
    print(f"MAE      : {results['MAE']:.4f}")