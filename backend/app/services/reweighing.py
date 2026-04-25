"""
Reweighing — Kamiran & Calders (2012).

Computes sample weights W_i such that the reweighted distribution has
discrimination score D ≈ 0:

    W_i = P(S=s_i) * P(Y=y_i) / P_obs(S=s_i, Y=y_i)

where:
  P(S=s)       = marginal proportion of group s
  P(Y=y)       = marginal proportion of outcome y
  P_obs(S=s,Y=y) = joint proportion of (group s, outcome y)

This produces a uniform expected outcome rate across all groups while
preserving the overall outcome marginal — achieving disc → 0 by construction.

Reference: Kamiran & Calders (2012) "Data preprocessing techniques for
           classification without discrimination"
           Table 1 result: disc score 0.1965 -> ~0 after reweighing on Adult
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.models.schemas import ReweighResult


def compute_sample_weights(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    positive_outcome: str,
) -> Optional[Tuple[np.ndarray, ReweighResult]]:
    """
    Compute Kamiran & Calders reweighing weights for a dataframe.

    Returns (weights_array, ReweighResult) or None if data insufficient.
    weights_array has one weight per row in df (rows without protected_attr
    or outcome_col are assigned weight 1.0).
    """
    if protected_attr not in df.columns or outcome_col not in df.columns:
        return None

    subset = df.dropna(subset=[protected_attr, outcome_col]).copy()
    if len(subset) < 50:
        return None

    # Binarize outcome
    try:
        numeric_pos = float(positive_outcome)
        y = (pd.to_numeric(subset[outcome_col], errors="coerce") == numeric_pos).astype(int)
    except (ValueError, TypeError):
        y = (subset[outcome_col].astype(str).str.strip() == str(positive_outcome).strip()).astype(int)

    if y.nunique() < 2:
        return None

    s = subset[protected_attr].astype(str)
    n = len(subset)

    # Marginal probabilities
    p_y1 = float(y.mean())
    p_y0 = 1.0 - p_y1

    # Discrimination before reweighing
    group_rates = {}
    for gval in s.unique():
        mask = (s == gval).values
        group_rates[gval] = float(y[mask].mean())

    disc_before = max(group_rates.values()) - min(group_rates.values())

    # Compute weights: W_i = P(S) * P(Y) / P_obs(S, Y)
    weights = np.ones(n, dtype=float)
    for gval in s.unique():
        p_s = float((s == gval).mean())
        for yval, p_y in [(1, p_y1), (0, p_y0)]:
            mask = ((s == gval) & (y == yval)).values
            p_obs = float(mask.mean())
            if p_obs > 0:
                w = (p_s * p_y) / p_obs
                weights[mask] = w

    # Discrimination after reweighing (weighted outcome rate per group)
    weighted_rates = {}
    for gval in s.unique():
        mask = (s == gval).values
        w_g = weights[mask]
        y_g = y.values[mask]
        weighted_rates[gval] = float((w_g * y_g).sum() / w_g.sum()) if w_g.sum() > 0 else 0.0

    disc_after = max(weighted_rates.values()) - min(weighted_rates.values())

    # Map weights back to full df index
    full_weights = np.ones(len(df), dtype=float)
    subset_idx = subset.index
    for i, orig_idx in enumerate(subset_idx):
        loc = df.index.get_loc(orig_idx)
        full_weights[loc] = weights[i]

    result = ReweighResult(
        protected_attribute=protected_attr,
        outcome_column=outcome_col,
        disc_before=round(disc_before, 4),
        disc_after=round(disc_after, 4),
        n_samples=n,
    )
    return full_weights, result


def reweigh_dataframe(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    positive_outcome: str,
    weight_col: str = "_sample_weight",
) -> Tuple[pd.DataFrame, Optional[ReweighResult]]:
    """
    Add a sample weight column to df. Returns (weighted_df, ReweighResult).
    If reweighing fails, returns (df, None) with uniform weights.
    """
    out = compute_sample_weights(df, protected_attr, outcome_col, positive_outcome)
    if out is None:
        df2 = df.copy()
        df2[weight_col] = 1.0
        return df2, None

    weights, result = out
    df2 = df.copy()
    df2[weight_col] = weights
    return df2, result
