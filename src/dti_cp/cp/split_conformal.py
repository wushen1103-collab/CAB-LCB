from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile with finite-sample correction:
    q = k-th order statistic where k = ceil((n+1)*(1-alpha)).
    Uses absolute residual scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]
    n = scores.shape[0]
    if n == 0:
        return float("nan")

    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = max(1, min(k, n))
    s_sorted = np.sort(scores)
    return float(s_sorted[k - 1])


def split_conformal_interval(y_pred: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lo = y_pred - float(qhat)
    hi = y_pred + float(qhat)
    return lo, hi


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def avg_width(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return float(np.mean(hi - lo))
