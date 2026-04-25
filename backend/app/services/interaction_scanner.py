"""
Conjunctive proxy detection (Zliobaite 2015 Type 2).

A conjunctive proxy occurs when features A and B individually have low predictive
power over the protected attribute, but together they have substantially higher
predictive power. This is invisible to pairwise correlation graphs.

Method:
  1. Compute individual skill scores for each non-protected feature.
  2. For candidate pairs (where at least one feature has skill > min_individual_skill):
     - Compute joint skill score using both features.
     - Interaction gain = joint_skill - max(skill_A, skill_B).
  3. Report pairs where interaction_gain >= min_gain threshold.

Pair selection is O(k^2) where k = features with individual skill > threshold.
Full grid is gated so large datasets don't blow up.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from app.models.schemas import ConjunctiveProxy
from app.services.graph_engine import _risk_label


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


def _encode_target(y: pd.Series) -> np.ndarray:
    if y.dtype == object or str(y.dtype) == "category":
        return LabelEncoder().fit_transform(y.astype(str))
    return y.to_numpy()


def _skill_score(df: pd.DataFrame, feature_cols: List[str], target_col: str, cv: int = 3) -> float:
    """Baseline-adjusted skill score for a set of features predicting target."""
    if not LGB_AVAILABLE:
        return 0.0
    subset = df[feature_cols + [target_col]].dropna()
    if len(subset) < 50:
        return 0.0

    X = _encode(subset[feature_cols])
    y = pd.Series(_encode_target(subset[target_col]))
    if y.nunique() < 2:
        return 0.0

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1, n_jobs=1)
    dummy = DummyClassifier(strategy="most_frequent")

    try:
        model_acc = float(np.mean(cross_val_score(model, X, y, cv=kf, scoring="accuracy")))
        baseline = float(np.mean(cross_val_score(dummy, X, y, cv=kf, scoring="accuracy")))
        max_possible = 1.0 - baseline
        if max_possible <= 1e-6:
            return 0.0
        return round(max(0.0, (model_acc - baseline) / max_possible), 4)
    except Exception:
        return 0.0


def find_conjunctive_proxies(
    df: pd.DataFrame,
    protected_attributes: List[str],
    min_individual_skill: float = 0.02,
    min_interaction_gain: float = 0.05,
    max_pairs: int = 200,
) -> List[ConjunctiveProxy]:
    """
    Find conjunctive proxies for each protected attribute.

    Only pairs where at least one feature exceeds min_individual_skill are evaluated.
    Pairs are sorted by interaction_gain descending.
    """
    if not LGB_AVAILABLE:
        return []

    protected_set = set(protected_attributes)
    non_protected = [c for c in df.columns if c not in protected_set]
    results: List[ConjunctiveProxy] = []

    for protected in protected_attributes:
        if protected not in df.columns:
            continue

        # Step 1: individual skill scores (fast, 3-fold)
        individual_skills: Dict[str, float] = {}
        for feat in non_protected:
            individual_skills[feat] = _skill_score(df, [feat], protected, cv=3)

        # Step 2: candidate pairs — at least one feature with skill > threshold
        candidates = [f for f, s in individual_skills.items() if s >= min_individual_skill]
        # Also add features with moderate skill for conjunctive detection
        moderate = [f for f, s in individual_skills.items()
                    if s >= min_individual_skill * 0.5 and f not in candidates]
        pool = candidates + moderate

        if len(pool) < 2:
            continue

        pairs = list(itertools.combinations(pool, 2))
        # Limit total pairs evaluated
        if len(pairs) > max_pairs:
            # Prioritize: pairs where both features have some individual skill
            pairs_scored = sorted(
                pairs,
                key=lambda p: individual_skills.get(p[0], 0) + individual_skills.get(p[1], 0),
                reverse=True,
            )
            pairs = pairs_scored[:max_pairs]

        # Step 3: evaluate joint skill and interaction gain
        for feat_a, feat_b in pairs:
            joint_skill = _skill_score(df, [feat_a, feat_b], protected, cv=3)
            skill_a = individual_skills.get(feat_a, 0.0)
            skill_b = individual_skills.get(feat_b, 0.0)
            best_individual = max(skill_a, skill_b)
            interaction_gain = joint_skill - best_individual

            if interaction_gain >= min_interaction_gain:
                results.append(ConjunctiveProxy(
                    feature_a=feat_a,
                    feature_b=feat_b,
                    joint_skill=joint_skill,
                    skill_a=skill_a,
                    skill_b=skill_b,
                    interaction_gain=round(interaction_gain, 4),
                    protected_attribute=protected,
                    risk_label=_risk_label(joint_skill),
                ))

    # Deduplicate and sort by interaction gain
    seen = set()
    unique = []
    for r in sorted(results, key=lambda x: x.interaction_gain, reverse=True):
        key = (frozenset([r.feature_a, r.feature_b]), r.protected_attribute)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique
