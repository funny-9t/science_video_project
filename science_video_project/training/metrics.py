from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, zero_division=0))


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def ranking_accuracy(pos_score: np.ndarray, neg_score: np.ndarray) -> float:
    if pos_score.size == 0:
        return 0.0
    return float((pos_score > neg_score).mean())


def compute_pair_metrics(pos_prob: np.ndarray, neg_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)], axis=0).astype(int)
    y_score = np.concatenate([pos_prob, neg_prob], axis=0)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "auc": auc(y_true, y_score),
        "ranking_accuracy": ranking_accuracy(pos_prob, neg_prob),
    }
