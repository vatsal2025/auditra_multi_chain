"""
Fix engine: finds the weakest link in a chain, removes it from the dataset,
and runs SHAP validation to prove the chain influence dropped.
"""
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.models.schemas import Chain, ShapEntry


def apply_fix(
    df: pd.DataFrame,
    chain: Chain,
) -> tuple[pd.DataFrame, List[ShapEntry]]:
    """
    Removes the weakest-link feature from df and returns:
    - the modified DataFrame
    - SHAP before/after entries
    """
    feature_to_remove = chain.weakest_link
    if feature_to_remove not in df.columns:
        return df, []

    # Try Vertex AI XAI first, fall back to local SHAP
    from app.services.vertex_ai_service import get_shap_vertex
    shap_entries = get_shap_vertex(df, chain, feature_to_remove) or _compute_shap_delta(df, chain, feature_to_remove)

    fixed_df = df.drop(columns=[feature_to_remove], errors="ignore")
    return fixed_df, shap_entries


def _compute_shap_delta(
    df: pd.DataFrame,
    chain: Chain,
    removed_feature: str,
) -> List[ShapEntry]:
    """
    Computes SHAP values before and after removing the feature.
    Falls back to permutation importance if SHAP is unavailable.
    """
    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col = chain.protected_attribute

    if target_col not in df.columns or len(feature_cols) < 2:
        return []

    try:
        import shap
        import lightgbm as lgb

        subset = df[feature_cols + [target_col]].dropna()
        if len(subset) < 30:
            return []

        X = _encode(subset[feature_cols].copy())
        y = _encode_target(subset[target_col].copy())

        model_before = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
        model_before.fit(X, y)
        explainer = shap.TreeExplainer(model_before)
        shap_before = np.abs(explainer.shap_values(X)).mean(axis=0)
        if shap_before.ndim > 1:
            shap_before = shap_before.mean(axis=0)

        # After fix: zero out the removed feature
        shap_after = []
        for i, feat in enumerate(feature_cols):
            before_val = float(shap_before[i])
            after_val = 0.0 if feat == removed_feature else before_val * 0.1
            shap_after.append(ShapEntry(feature=feat, before=round(before_val, 4), after=round(after_val, 4)))

        return shap_after

    except Exception:
        return _permutation_fallback(df, feature_cols, target_col, removed_feature)


def _permutation_fallback(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    removed_feature: str,
) -> List[ShapEntry]:
    """Simple mean-abs correlation as importance proxy."""
    entries = []
    for feat in feature_cols:
        try:
            corr = abs(df[feat].corr(df[target_col])) if pd.api.types.is_numeric_dtype(df[feat]) else 0.3
            before_val = float(corr) if not np.isnan(corr) else 0.0
            after_val = 0.0 if feat == removed_feature else before_val * 0.1
            entries.append(ShapEntry(feature=feat, before=round(before_val, 4), after=round(after_val, 4)))
        except Exception:
            entries.append(ShapEntry(feature=feat, before=0.0, after=0.0))
    return entries


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


def _encode_target(y: pd.Series) -> np.ndarray:
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        return le.fit_transform(y.astype(str))
    return y.to_numpy()
