"""
This code is adapted from PeSTo (Protein Structure Transformer).
Developed by the PeSTo authors, LBM, EPFL.
Original License: CC BY-NC-SA 4.0
Copyright (c) original source: https://github.com/LBM-EPFL/PeSTo/tree/main
"""

import numpy as np
import torch as pt
from typing import Tuple, List
from sklearn.metrics import roc_auc_score, average_precision_score


bc_score_names = ['acc','ppv','npv','tpr','tnr','mcc','auc','std']


@pt.jit.script
def binary_classification_counts(y: pt.Tensor, y_pred: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    true_positives = pt.sum(y_pred * y, dim=0)
    true_negatives = pt.sum((1.0 - y_pred) * (1.0 - y), dim=0)
    false_positives = pt.sum(y_pred * (1.0 - y), dim=0)
    false_negatives = pt.sum((1.0 - y_pred) * y, dim=0)
    
    positives = pt.sum(y, dim=0)
    negatives = pt.sum(1.0 - y, dim=0)
    
    return true_positives, true_negatives, false_positives, false_negatives, positives, negatives


@pt.jit.script
def accuracy(TP: pt.Tensor, TN: pt.Tensor, FP: pt.Tensor, FN: pt.Tensor) -> pt.Tensor:
    return (TP + TN) / (TP + TN + FP + FN)


@pt.jit.script
def precision(TP: pt.Tensor, FP: pt.Tensor, P: pt.Tensor) -> pt.Tensor:
    denominator = TP + FP # Avoid division by zero
    valid_mask = denominator > 0
    result = pt.ones_like(TP) * float('nan')
    result[valid_mask] = TP[valid_mask] / denominator[valid_mask]
    result[~(P > 0)] = float('nan')
    return result


@pt.jit.script
def negative_predictive_value(TN: pt.Tensor, FN: pt.Tensor, N: pt.Tensor) -> pt.Tensor:
    denominator = TN + FN
    valid_mask = denominator > 0
    result = pt.ones_like(TN) * float('nan')
    result[valid_mask] = TN[valid_mask] / denominator[valid_mask]
    result[~(N > 0)] = float('nan')
    return result


@pt.jit.script
def recall(TP: pt.Tensor, FN: pt.Tensor) -> pt.Tensor:
    denominator = TP + FN
    valid_mask = denominator > 0
    result = pt.ones_like(TP) * float('nan')
    result[valid_mask] = TP[valid_mask] / denominator[valid_mask]
    return result


@pt.jit.script
def specificity(TN: pt.Tensor, FP: pt.Tensor) -> pt.Tensor:
    denominator = TN + FP
    valid_mask = denominator > 0
    result = pt.ones_like(TN) * float('nan')
    result[valid_mask] = TN[valid_mask] / denominator[valid_mask]
    return result


@pt.jit.script
def matthews_correlation_coefficient(TP: pt.Tensor, TN: pt.Tensor, FP: pt.Tensor, FN: pt.Tensor) -> pt.Tensor:
    numerator = (TP * TN) - (FP * FN)
    denominator = pt.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    valid_mask = denominator > 0
    result = pt.ones_like(TP) * float('nan')
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    return result


def roc_auc(y: pt.Tensor, y_prob: pt.Tensor, P: pt.Tensor, N: pt.Tensor) -> pt.Tensor:
    valid_mask = (P > 0) & (N > 0)
    result = pt.full_like(P, float('nan'), dtype=pt.float32)
    if pt.any(valid_mask):
        valid_indices = pt.where(valid_mask)[0]
        y_valid = y[:, valid_indices].cpu().numpy()
        y_prob_valid = y_prob[:, valid_indices].cpu().numpy()
    
        try:
            auc_scores = np.array(roc_auc_score(y_valid, y_prob_valid, average=None))
            result[valid_indices] = pt.from_numpy(auc_scores).float().to(y.device)
        except ValueError as e:
            print(f"ROC AUC calculation error: {e}")
            
    return result


def precision_recall_auc(y: pt.Tensor, y_prob: pt.Tensor, P: pt.Tensor, N: pt.Tensor) -> pt.Tensor:
    valid_mask = (P > 0) & (N > 0)
    result = pt.full_like(P, float('nan'), dtype=pt.float32)
    if pt.any(valid_mask):
        valid_indices = pt.where(valid_mask)[0]
        y_valid = y[:, valid_indices].cpu().numpy()
        y_prob_valid = y_prob[:, valid_indices].cpu().numpy()
        
        try:
            pr_auc_scores = np.array(average_precision_score(y_valid, y_prob_valid, average=None))
            result[valid_indices] = pt.from_numpy(pr_auc_scores).float().to(y.device)
        except ValueError as e:
            print(f"PR AUC calculation error: {e}")
            
    return result


@pt.jit.script
def nanmean(x: pt.Tensor) -> pt.Tensor:
    valid_count = pt.sum(~pt.isnan(x), dim=0)
    return pt.nansum(x, dim=0) / pt.clamp(valid_count, min=1)  # Avoid division by zero


def compute_classification_metrics(y_true: pt.Tensor, y_prob: pt.Tensor) -> pt.Tensor:
    # Threshold probabilities to get binary predictions
    y_pred = (y_prob >= 0.5).float()

    # Calculate basic classification counts
    TP, TN, FP, FN, P, N = binary_classification_counts(y_true, y_pred)

    # Compute all metrics
    scores = pt.stack([
        accuracy(TP, TN, FP, FN),                 # Accuracy
        precision(TP, FP, P),                     # Precision (PPV)
        negative_predictive_value(TN, FN, N),     # NPV
        recall(TP, FN),                           # Recall (TPR)
        specificity(TN, FP),                      # Specificity (TNR)
        matthews_correlation_coefficient(TP, TN, FP, FN),  # MCC
        roc_auc(y_true, y_prob, P, N),            # ROC AUC
        pt.std(y_prob, dim=0),                    # Standard deviation of predictions
        precision_recall_auc(y_true, y_prob, P, N) # PR AUC
    ])

    return scores
