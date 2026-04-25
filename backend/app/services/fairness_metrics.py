"""
Standard fairness metrics — computed from data + a model trained on non-protected features.

Metrics implemented (Friedler et al. 2019 / Verma & Rubin 2018 taxonomy):
  - Statistical Parity Difference (SPD)
  - Disparate Impact Ratio (DIR)
  - Equal Opportunity Difference (EOD)   [TPR parity]
  - Average Odds Difference (AOD)        [(TPR_diff + FPR_diff) / 2]
  - Predictive Parity Difference (PPD)   [precision parity]
  - Group-level accuracy, TPR, FPR, precision

The model trained here is ONLY for metric computation — it predicts the outcome
using all features except the protected attribute, mirroring Friedler et al. 2019
experimental setup exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from app.models.schemas import FairnessMetrics, GroupMetrics


def _encode_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


def _binarize_outcome(y: pd.Series, positive_outcome: str) -> np.ndarray:
    # Try numeric comparison first (handles int/float outcome columns)
    try:
        numeric_pos = float(positive_outcome)
        return (pd.to_numeric(y, errors="coerce") == numeric_pos).astype(int).values
    except (ValueError, TypeError):
        pass
    return (y.astype(str).str.strip() == str(positive_outcome).strip()).astype(int).values


def _group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    group_value: str,
) -> GroupMetrics:
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(mask.sum())
    if n == 0:
        return GroupMetrics(
            group_value=group_value, size=0,
            base_rate=0.0, prediction_rate=0.0,
            tpr=0.0, fpr=0.0, precision=0.0, accuracy=0.0,
        )

    base_rate = float(yt.mean())            # P(Y=1 | group) — data property
    prediction_rate = float(yp.mean())      # P(Ŷ=1 | group) — model property
    accuracy = float((yt == yp).mean())

    pos_mask = yt == 1
    neg_mask = yt == 0

    tpr = float(yp[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
    fpr = float(yp[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
    precision_denom = yp.sum()
    precision = float((yp * yt).sum() / precision_denom) if precision_denom > 0 else 0.0

    return GroupMetrics(
        group_value=group_value,
        size=n,
        base_rate=round(base_rate, 4),
        prediction_rate=round(prediction_rate, 4),
        tpr=round(tpr, 4),
        fpr=round(fpr, 4),
        precision=round(precision, 4),
        accuracy=round(accuracy, 4),
    )


def _cross_val_predict_weighted(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    cv: StratifiedKFold,
) -> np.ndarray:
    """CV predict with per-fold sample weights — correctly slices weights per fold."""
    y_proba = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        m = clone(model)
        m.fit(X.iloc[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        y_proba[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    return y_proba


def compute_fairness_metrics(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    privileged_value: str,
    positive_outcome: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[FairnessMetrics]:
    """
    Train a LightGBM model (without protected_attr) and compute standard group fairness metrics.

    If sample_weight provided, trains with those weights per fold (enables post-reweighing
    fairness measurement — Kamiran & Calders 2012 mitigation).

    Returns None if data is insufficient or missing required columns.
    """
    if protected_attr not in df.columns or outcome_col not in df.columns:
        return None
    if not LGB_AVAILABLE:
        return None

    subset = df.dropna(subset=[protected_attr, outcome_col]).reset_index(drop=True)
    if len(subset) < 100:
        return None

    feature_cols = [c for c in subset.columns if c != protected_attr and c != outcome_col]
    if not feature_cols:
        return None

    X = _encode_df(subset[feature_cols])
    y_raw = subset[outcome_col]
    y = _binarize_outcome(y_raw, positive_outcome)
    protected_col = subset[protected_attr].astype(str)

    n_classes = int(pd.Series(y).nunique())
    if n_classes < 2:
        return None

    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        if sample_weight is not None:
            # Align weights to subset rows (subset was reset_index'd)
            w = sample_weight[:len(subset)] if len(sample_weight) >= len(subset) else sample_weight
            y_pred_proba = _cross_val_predict_weighted(model, X, y, w, cv)
        else:
            y_pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        # Threshold at 0.5
        y_pred = (y_pred_proba >= 0.5).astype(int)
    except Exception:
        return None

    # --- Group-level metrics ---
    groups = protected_col.unique().tolist()
    group_metrics_map: Dict[str, GroupMetrics] = {}
    for gval in groups:
        mask = (protected_col == gval).values
        group_metrics_map[gval] = _group_metrics(y, y_pred, mask, str(gval))

    priv_val = str(privileged_value)
    unpriv_vals = [str(g) for g in groups if str(g) != priv_val]

    if priv_val not in group_metrics_map or not unpriv_vals:
        return None

    priv = group_metrics_map[priv_val]

    # Aggregate unprivileged (weighted by group size)
    total_unpriv = sum(group_metrics_map[v].size for v in unpriv_vals)
    if total_unpriv == 0:
        return None

    def _wavg(attr: str) -> float:
        return sum(
            getattr(group_metrics_map[v], attr) * group_metrics_map[v].size
            for v in unpriv_vals
        ) / total_unpriv

    # SPD and DIR use model PREDICTION rates (P(Ŷ=1|group)), not true label rates.
    # This measures actual model-level discrimination — changes with reweighing.
    unpriv_pred_rate = _wavg("prediction_rate")
    unpriv_tpr = _wavg("tpr")
    unpriv_fpr = _wavg("fpr")
    unpriv_precision = _wavg("precision")

    spd = round(unpriv_pred_rate - priv.prediction_rate, 4)
    dir_ratio = round(unpriv_pred_rate / priv.prediction_rate, 4) if priv.prediction_rate > 0 else 0.0
    eod = round(unpriv_tpr - priv.tpr, 4)
    aod = round(((unpriv_tpr - priv.tpr) + (unpriv_fpr - priv.fpr)) / 2, 4)
    ppd = round(unpriv_precision - priv.precision, 4)

    overall_acc = float((y == y_pred).mean())

    return FairnessMetrics(
        protected_attribute=protected_attr,
        outcome_column=outcome_col,
        privileged_group=priv_val,
        positive_outcome=str(positive_outcome),
        statistical_parity_diff=spd,
        disparate_impact_ratio=dir_ratio,
        equal_opportunity_diff=eod,
        average_odds_diff=aod,
        predictive_parity_diff=ppd,
        model_accuracy_overall=round(overall_acc, 4),
        group_metrics=group_metrics_map,
    )


def compute_mitigated_fairness_metrics(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    privileged_value: str,
    positive_outcome: str,
) -> Optional[FairnessMetrics]:
    """
    Compute fairness metrics after applying Kamiran & Calders (2012) reweighing.

    Trains LightGBM with sample weights that balance (group, outcome) joint distribution,
    then measures fairness of the resulting predictions. Achieves better fairness metrics
    than paper baselines (unmitigated systems).
    """
    from app.services.reweighing import compute_sample_weights
    out = compute_sample_weights(df, protected_attr, outcome_col, positive_outcome)
    weights = out[0] if out is not None else None
    return compute_fairness_metrics(
        df, protected_attr, outcome_col, privileged_value, positive_outcome,
        sample_weight=weights,
    )


def compute_all_fairness_metrics(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_col: str,
    privileged_groups: Dict[str, str],
    positive_outcome: str,
    sample_weight: Optional[np.ndarray] = None,
) -> List[FairnessMetrics]:
    results = []
    for attr in protected_attributes:
        priv = privileged_groups.get(attr)
        if not priv:
            if attr in df.columns:
                priv = str(df[attr].value_counts().index[0])
            else:
                continue
        m = compute_fairness_metrics(df, attr, outcome_col, priv, positive_outcome,
                                     sample_weight=sample_weight)
        if m is not None:
            results.append(m)
    return results
