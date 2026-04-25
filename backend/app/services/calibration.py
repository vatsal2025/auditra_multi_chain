"""
Calibration measurement per protected group.

Implements: Expected Calibration Error (ECE) — Guo et al. (2017)
            Calibration gap between groups — Chouldechova (2017)

Chouldechova (2017) proves that when base rates differ across groups,
satisfying both FPR parity AND calibration is mathematically impossible.
This module MEASURES how much each condition holds, so users can see
the tradeoff explicitly rather than discovering it after deployment.

Reference: Chouldechova (2017) "Fair prediction with disparate impact"
           Guo et al. (2017) "On calibration of modern neural networks"
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from app.models.schemas import CalibrationBin, GroupCalibration, CalibrationAudit


def _encode_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


def _ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    """
    Expected Calibration Error with equal-width bins.
    Returns (ece, bins_list).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if i == n_bins - 1:
            in_bin = in_bin | (y_pred_proba == 1.0)

        count = int(in_bin.sum())
        if count == 0:
            continue

        acc = float(y_true[in_bin].mean())
        conf = float(y_pred_proba[in_bin].mean())
        gap = abs(acc - conf)
        ece += (count / n) * gap

        bins.append(CalibrationBin(
            bin_lower=round(lo, 2),
            bin_upper=round(hi, 2),
            confidence=round(conf, 4),
            accuracy=round(acc, 4),
            count=count,
        ))

    return round(float(ece), 4), bins


def compute_calibration_audit(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    positive_outcome: str,
    n_bins: int = 10,
) -> Optional[CalibrationAudit]:
    """
    Train a LightGBM model and compute per-group calibration error.

    Returns CalibrationAudit with ECE per group and cross-group calibration gap.
    A high calibration gap (> 0.05) means the model is better calibrated for
    some groups than others — a form of predictive unfairness.
    """
    if not LGB_AVAILABLE:
        return None
    if protected_attr not in df.columns or outcome_col not in df.columns:
        return None

    subset = df.dropna(subset=[protected_attr, outcome_col]).reset_index(drop=True)
    if len(subset) < 100:
        return None

    feature_cols = [c for c in subset.columns if c != protected_attr and c != outcome_col]
    if not feature_cols:
        return None

    # Binarize outcome
    try:
        numeric_pos = float(positive_outcome)
        y = (pd.to_numeric(subset[outcome_col], errors="coerce") == numeric_pos).astype(int).values
    except (ValueError, TypeError):
        y = (subset[outcome_col].astype(str).str.strip() == str(positive_outcome).strip()).astype(int).values

    if pd.Series(y).nunique() < 2:
        return None

    X = _encode_df(subset[feature_cols])
    protected_col = subset[protected_attr].astype(str)

    model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        verbose=-1, n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        return None

    groups = protected_col.unique().tolist()
    group_calibration: Dict[str, GroupCalibration] = {}

    for gval in groups:
        mask = (protected_col == gval).values
        yt_g = y[mask]
        yp_g = y_proba[mask]
        if len(yt_g) < 20:
            continue
        ece_val, bins_list = _ece(yt_g, yp_g, n_bins=n_bins)
        max_gap = max((abs(b.accuracy - b.confidence) for b in bins_list), default=0.0)
        group_calibration[gval] = GroupCalibration(
            group_value=gval,
            ece=ece_val,
            bins=bins_list,
            max_calibration_gap=round(max_gap, 4),
        )

    if len(group_calibration) < 2:
        return None

    ece_values = [gc.ece for gc in group_calibration.values()]
    calibration_gap = round(max(ece_values) - min(ece_values), 4)
    # Chouldechova threshold: gap < 0.05 considered well-calibrated across groups
    is_calibrated = calibration_gap < 0.05

    return CalibrationAudit(
        protected_attribute=protected_attr,
        outcome_column=outcome_col,
        group_calibration=group_calibration,
        calibration_gap=calibration_gap,
        is_calibrated=is_calibrated,
    )
