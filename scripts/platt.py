"""
Platt scaling (probability calibration) — shared by predict and training scripts.

Fits a logistic regression on (raw_probs → labels) to map overconfident
model outputs to honest probability estimates.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt_scaler(raw_probs: np.ndarray, labels: np.ndarray):
    """
    Fit Platt scaling (logistic regression on raw probs → calibrated probs).
    Returns fitted LogisticRegression model or None if insufficient data.

    Requires ≥30 samples with ≥5 positive and ≥5 negative examples.
    """
    mask = np.isfinite(raw_probs) & np.isfinite(labels)
    probs_clean = raw_probs[mask]
    labels_clean = labels[mask]

    n_pos = int(labels_clean.sum())
    n_neg = len(labels_clean) - n_pos

    if len(labels_clean) < 30 or n_pos < 5 or n_neg < 5:
        return None

    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(probs_clean.reshape(-1, 1), labels_clean)
    return platt


def apply_platt(platt_model, raw_probs: np.ndarray) -> np.ndarray:
    """Apply Platt scaler to raw probabilities. Falls back to raw if model is None."""
    if platt_model is None:
        return raw_probs.copy()
    return platt_model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
