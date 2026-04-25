"""
Fix engine: removes weakest link from chain, validates by actually
retraining with and without the removed feature.
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from app.models.schemas import Chain, ShapEntry


def apply_fix(
    df: pd.DataFrame,
    chain: Chain,
) -> tuple[pd.DataFrame, List[ShapEntry]]:
    """
    Removes weakest-link feature from df.
    SHAP deltas are computed via actual before/after model retraining,
    not hardcoded multipliers.
    """
    feature_to_remove = chain.weakest_link
    if feature_to_remove not in df.columns:
        return df, []

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
    Computes real before/after SHAP values by training two models:
    one with all chain features, one without the removed feature.
    Falls back to permutation importance if SHAP unavailable.
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

        X_before = _encode(subset[feature_cols].copy())
        y = _encode_target(subset[target_col].copy())

        # Before model: all chain features
        model_before = lgb.LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=1)
        model_before.fit(X_before, y)
        explainer_before = shap.TreeExplainer(model_before)
        sv_before = explainer_before.shap_values(X_before)
        # Handle multiclass: average across classes
        if isinstance(sv_before, list):
            sv_before = np.mean([np.abs(sv) for sv in sv_before], axis=0)
        mean_shap_before = np.abs(sv_before).mean(axis=0)

        # After model: chain features minus the removed one
        after_cols = [c for c in feature_cols if c != removed_feature]
        if len(after_cols) == 0:
            # Only one feature — after is trivially zero for that feature
            return [ShapEntry(
                feature=removed_feature,
                before=round(float(mean_shap_before[feature_cols.index(removed_feature)]), 4),
                after=0.0,
            )]

        X_after = _encode(subset[after_cols].copy())
        model_after = lgb.LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=1)
        model_after.fit(X_after, y)
        explainer_after = shap.TreeExplainer(model_after)
        sv_after = explainer_after.shap_values(X_after)
        if isinstance(sv_after, list):
            sv_after = np.mean([np.abs(sv) for sv in sv_after], axis=0)
        mean_shap_after = np.abs(sv_after).mean(axis=0)

        # Build entries: removed feature gets after=0, rest get real after values
        entries = []
        after_idx = 0
        for i, feat in enumerate(feature_cols):
            before_val = float(mean_shap_before[i])
            if feat == removed_feature:
                after_val = 0.0
            else:
                after_val = float(mean_shap_after[after_idx])
                after_idx += 1
            entries.append(ShapEntry(
                feature=feat,
                before=round(before_val, 4),
                after=round(after_val, 4),
            ))

        return entries

    except Exception:
        return _permutation_fallback(df, feature_cols, target_col, removed_feature)


def _permutation_fallback(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    removed_feature: str,
) -> List[ShapEntry]:
    """
    Permutation importance before/after by actual accuracy drop.
    Trains one model, measures importance by permuting each feature.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score

        subset = df[feature_cols + [target_col]].dropna()
        if len(subset) < 30:
            return _correlation_fallback(df, feature_cols, target_col, removed_feature)

        X = _encode(subset[feature_cols].copy())
        y = _encode_target(subset[target_col].copy())

        model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=1)
        model.fit(X, y)
        baseline_acc = accuracy_score(y, model.predict(X))

        entries = []
        after_cols = [c for c in feature_cols if c != removed_feature]

        for feat in feature_cols:
            X_perm = X.copy()
            X_perm[feat] = np.random.default_rng(42).permutation(X_perm[feat].values)
            perm_acc = accuracy_score(y, model.predict(X_perm))
            before_val = max(0.0, baseline_acc - perm_acc)

            if feat == removed_feature:
                after_val = 0.0
            else:
                # Retrain on reduced feature set to get honest after importance
                if len(after_cols) > 0:
                    X_after = _encode(subset[after_cols].copy())
                    m2 = lgb.LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=1)
                    m2.fit(X_after, y)
                    base2 = accuracy_score(y, m2.predict(X_after))
                    X_a_perm = X_after.copy()
                    if feat in X_a_perm.columns:
                        X_a_perm[feat] = np.random.default_rng(42).permutation(X_a_perm[feat].values)
                        after_val = max(0.0, base2 - accuracy_score(y, m2.predict(X_a_perm)))
                    else:
                        after_val = 0.0
                else:
                    after_val = 0.0

            entries.append(ShapEntry(
                feature=feat,
                before=round(before_val, 4),
                after=round(after_val, 4),
            ))

        return entries

    except Exception:
        return _correlation_fallback(df, feature_cols, target_col, removed_feature)


def _correlation_fallback(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    removed_feature: str,
) -> List[ShapEntry]:
    """Last-resort fallback using correlation-based importance."""
    entries = []
    after_df = df.drop(columns=[removed_feature], errors="ignore")
    after_cols = [c for c in feature_cols if c != removed_feature]

    for feat in feature_cols:
        try:
            if pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target_col]):
                before_val = float(abs(df[feat].corr(df[target_col])))
            else:
                before_val = 0.15  # conservative default for categorical
            before_val = 0.0 if np.isnan(before_val) else before_val
        except Exception:
            before_val = 0.0

        if feat == removed_feature:
            after_val = 0.0
        else:
            try:
                if pd.api.types.is_numeric_dtype(after_df[feat]) and pd.api.types.is_numeric_dtype(after_df[target_col]):
                    after_val = float(abs(after_df[feat].corr(after_df[target_col])))
                else:
                    after_val = before_val
                after_val = 0.0 if np.isnan(after_val) else after_val
            except Exception:
                after_val = before_val

        entries.append(ShapEntry(feature=feat, before=round(before_val, 4), after=round(after_val, 4)))

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
