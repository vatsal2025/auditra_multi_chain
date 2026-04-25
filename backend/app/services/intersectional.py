"""
Intersectional fairness scanner — Kearns et al. (2018).

Detects "fairness gerrymandering": a model can appear fair on each
protected attribute individually while being deeply unfair on subgroups
formed by combinations of attributes.

Method:
  For every pair of protected attributes (A, B), enumerate all (val_a, val_b)
  subgroups. Compute base rate P(Y=1 | A=val_a, B=val_b) for each.
  SPD = subgroup_rate - privileged_combo_rate.
  Subgroups with |SPD| > 0.1 are flagged.

Reference: Kearns et al. (2018) "Preventing Fairness Gerrymandering:
           Auditing and Learning for Subgroup Fairness"
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from app.models.schemas import IntersectionalAudit, IntersectionalGroup

_SPD_FLAG_THRESHOLD = 0.10   # Kearns: violations above 10% disparate impact


def _binarize(y: pd.Series, positive_outcome: str) -> pd.Series:
    try:
        numeric_pos = float(positive_outcome)
        return (pd.to_numeric(y, errors="coerce") == numeric_pos).astype(int)
    except (ValueError, TypeError):
        return (y.astype(str).str.strip() == str(positive_outcome).strip()).astype(int)


def compute_intersectional_audit(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_col: str,
    positive_outcome: str,
    min_group_size: int = 30,
) -> Optional[IntersectionalAudit]:
    """
    Scan all pairwise intersections of protected attributes for subgroup SPD.

    Returns IntersectionalAudit if ≥2 protected attributes exist and data
    is sufficient, else None.
    """
    valid_attrs = [a for a in protected_attributes if a in df.columns]
    if len(valid_attrs) < 2 or outcome_col not in df.columns:
        return None

    subset = df.dropna(subset=valid_attrs + [outcome_col]).reset_index(drop=True)
    if len(subset) < 100:
        return None

    y = _binarize(subset[outcome_col], positive_outcome)
    if y.nunique() < 2:
        return None

    # Take first pair for primary audit (most common case: 2 protected attrs)
    # For >2 attrs, enumerate all pairs and take the worst-case pair
    best_audit: Optional[IntersectionalAudit] = None
    worst_max_gap = -1.0

    for attr_a, attr_b in combinations(valid_attrs, 2):
        s_a = subset[attr_a].astype(str)
        s_b = subset[attr_b].astype(str)

        combos: Dict[Tuple[str, str], List[int]] = {}
        for i in range(len(subset)):
            key = (s_a.iloc[i], s_b.iloc[i])
            combos.setdefault(key, []).append(i)

        # Filter to groups with enough samples
        combos = {k: v for k, v in combos.items() if len(v) >= min_group_size}
        if len(combos) < 2:
            continue

        # Compute base rates
        rates: Dict[Tuple[str, str], float] = {}
        for (va, vb), idxs in combos.items():
            rates[(va, vb)] = float(y.iloc[idxs].mean())

        # Privileged combo = highest base rate
        priv_combo = max(rates, key=rates.__getitem__)
        priv_rate = rates[priv_combo]
        priv_key = f"{attr_a}={priv_combo[0]},{attr_b}={priv_combo[1]}"

        groups: List[IntersectionalGroup] = []
        for (va, vb), idxs in combos.items():
            rate = rates[(va, vb)]
            spd = round(rate - priv_rate, 4)
            group_key = f"{attr_a}={va},{attr_b}={vb}"
            groups.append(IntersectionalGroup(
                group_key=group_key,
                size=len(idxs),
                base_rate=round(rate, 4),
                spd_vs_privileged=spd,
            ))

        groups.sort(key=lambda g: g.spd_vs_privileged)

        spd_vals = [abs(g.spd_vs_privileged) for g in groups]
        max_gap = round(max(spd_vals), 4)
        flagged = [g.group_key for g in groups if abs(g.spd_vs_privileged) > _SPD_FLAG_THRESHOLD]

        audit = IntersectionalAudit(
            protected_attributes=[attr_a, attr_b],
            outcome_column=outcome_col,
            privileged_combo=priv_key,
            privileged_base_rate=round(priv_rate, 4),
            groups=groups,
            max_spd_gap=max_gap,
            flagged_groups=flagged,
        )

        if max_gap > worst_max_gap:
            worst_max_gap = max_gap
            best_audit = audit

    return best_audit
