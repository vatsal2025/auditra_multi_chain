"""
Chain risk scoring via baseline-adjusted reconstructive accuracy.

Trains a model to predict the protected attribute from chain features.
Score = skill score: how much better than majority-class baseline.
  skill = (model_accuracy - baseline) / (1 - baseline)
  skill = 0   → chain adds no predictive power beyond base rates
  skill = 1   → chain perfectly reconstructs protected attribute

This eliminates false positives caused by imbalanced protected attributes.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from app.core.config import settings
from app.models.schemas import Chain
from app.services.graph_engine import _risk_label


def score_chain(df: pd.DataFrame, chain: Chain) -> float:
    """
    Returns skill score [0, 1] for the chain.
    Tries Vertex AI AutoML first, falls back to local LightGBM.
    """
    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col = chain.protected_attribute

    if target_col not in df.columns or not feature_cols:
        return chain.risk_score

    from app.services.vertex_ai_service import score_chain_vertex
    vertex_score = score_chain_vertex(df, chain)
    if vertex_score is not None:
        return vertex_score

    return _score_via_lgbm(df, feature_cols, target_col)


def _score_via_lgbm(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> float:
    if not LGB_AVAILABLE:
        return 0.0

    subset = df[feature_cols + [target_col]].dropna()
    if len(subset) < 50:
        return 0.0

    X = subset[feature_cols].copy()
    y = subset[target_col].copy()

    # Encode categoricals
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))

    n_classes = int(y.nunique())
    if n_classes < 2:
        return 0.0

    objective = "multiclass" if n_classes > 2 else "binary"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = lgb.LGBMClassifier(
        objective=objective,
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
        verbose=-1,
        n_jobs=1,
    )

    try:
        model_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        raw_accuracy = float(np.mean(model_scores))

        # Baseline: majority-class dummy classifier
        dummy = DummyClassifier(strategy="most_frequent")
        baseline_scores = cross_val_score(dummy, X, y, cv=cv, scoring="accuracy")
        baseline = float(np.mean(baseline_scores))

        # Skill score: how much better than baseline, normalized to [0,1]
        max_possible = 1.0 - baseline
        if max_possible <= 1e-6:
            return 0.0
        skill = max(0.0, (raw_accuracy - baseline) / max_possible)
        return round(skill, 4)

    except Exception:
        return 0.0


def score_all_chains(df: pd.DataFrame, chains: List[Chain]) -> List[Chain]:
    scored = []
    for chain in chains:
        skill = score_chain(df, chain)
        scored.append(
            chain.model_copy(
                update={"risk_score": round(skill, 4), "risk_label": _risk_label(skill)}
            )
        )
    return sorted(scored, key=lambda c: c.risk_score, reverse=True)
