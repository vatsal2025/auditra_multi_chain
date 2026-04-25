"""
Feature correlation graph + multi-hop discrimination chain detector.

Key design principles:
- Protected attributes are SINK nodes: no outgoing edges, only incoming.
  This prevents chains from flowing backward through protected attrs.
- Pairwise strengths are filtered by statistical significance (Bonferroni-corrected).
- Direction is non-causal but semantically constrained: chains always flow
  FROM non-protected features TOWARD protected attributes.
- DFS enforces protected attrs as terminal-only nodes.
"""
import itertools
import uuid
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pearsonr
from sklearn.preprocessing import LabelEncoder

from app.core.config import settings
from app.models.schemas import Chain, ChainHop, GraphEdge, GraphNode


# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------

# Heuristic patterns for ID-like columns that should be excluded from correlation
_ID_PATTERNS = frozenset(["id", "_id", "uuid", "guid", "key", "index", "idx",
                           "zip", "postal", "fips", "code", "oid", "ssn"])


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify each column as 'numeric' or 'categorical'.
    Low-cardinality numeric cols and ID-like columns are categorical.
    High-cardinality string columns that look like IDs are excluded.
    """
    types: Dict[str, str] = {}
    n_rows = len(df)

    for col in df.columns:
        col_lower = col.lower()

        # Numeric dtype with high cardinality AND not ID-like
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            is_id_like = any(pat in col_lower for pat in _ID_PATTERNS)
            # Float columns are almost never identifiers
            is_float = pd.api.types.is_float_dtype(df[col])
            # For large datasets only: near-unique integer = likely an ID
            # (don't apply to small n to avoid misclassifying salary/score columns)
            is_id_by_cardinality = (
                not is_float
                and n_rows >= 500
                and n_unique / n_rows > 0.90
            )
            if n_unique > 10 and not is_id_like and not is_id_by_cardinality:
                types[col] = "numeric"
            else:
                types[col] = "categorical"
        else:
            types[col] = "categorical"

    return types


def get_excluded_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns that should be excluded from graph (near-unique IDs, etc.)."""
    excluded = []
    n_rows = len(df)
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique / max(n_rows, 1) > 0.95:
            excluded.append(col)
    return excluded


# ---------------------------------------------------------------------------
# Correlation helpers with p-value computation
# ---------------------------------------------------------------------------

def _cramers_v_with_p(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Bias-corrected Cramér's V with chi-squared p-value."""
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(confusion)
    n = confusion.values.sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0, p
    return float(np.sqrt(phi2_corr / denom)), float(p)


def _eta_squared_with_p(numeric: pd.Series, categorical: pd.Series) -> Tuple[float, float]:
    """Eta-squared with ANOVA F-test p-value."""
    cats = categorical.unique()
    groups = [numeric[categorical == cat].dropna().values for cat in cats if (categorical == cat).sum() > 0]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return 0.0, 1.0

    grand_mean = numeric.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((numeric - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0, 1.0

    eta2 = float(ss_between / ss_total)

    # F-test p-value: if all groups constant and groups differ in mean → p ≈ 0
    groups_for_f = [g for g in groups if len(g) >= 2]
    if len(groups_for_f) < 2:
        # Can't compute F; use eta2 magnitude to infer p
        p = 0.0 if eta2 > 0.5 else 1.0
    elif all(g.std() == 0 for g in groups_for_f):
        # Perfect group separation with zero within-group variance → F = ∞ → p = 0
        p = 0.0 if eta2 > 0.0 else 1.0
    else:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = f_oneway(*groups_for_f)
            p = float(p) if not np.isnan(p) else 1.0
        except Exception:
            p = 1.0

    return eta2, p


def _pearson_with_p(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Absolute Pearson correlation with two-tailed p-value."""
    if len(x) < 4:
        return 0.0, 1.0
    try:
        corr, p = pearsonr(x, y)
        return float(abs(corr)) if not np.isnan(corr) else 0.0, float(p)
    except Exception:
        return 0.0, 1.0


def _pairwise_strength(
    df: pd.DataFrame,
    col_types: Dict[str, str],
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise predictive strength [0,1] with Bonferroni-corrected
    significance filtering. Non-significant pairs get strength 0.
    """
    cols = list(col_types.keys())
    n_tests = max(len(cols) * (len(cols) - 1) // 2, 1)
    alpha_corrected = alpha / n_tests

    strengths: Dict[Tuple[str, str], float] = {}

    for a, b in itertools.combinations(cols, 2):
        ta, tb = col_types[a], col_types[b]
        try:
            if ta == "numeric" and tb == "numeric":
                s, p = _pearson_with_p(df[a].dropna(), df[b].dropna())
            elif ta == "categorical" and tb == "categorical":
                valid = df[[a, b]].dropna()
                s, p = _cramers_v_with_p(valid[a].astype(str), valid[b].astype(str))
            elif ta == "numeric" and tb == "categorical":
                valid = df[[a, b]].dropna()
                s, p = _eta_squared_with_p(valid[a], valid[b].astype(str))
            else:
                valid = df[[a, b]].dropna()
                s, p = _eta_squared_with_p(valid[b], valid[a].astype(str))
        except Exception:
            s, p = 0.0, 1.0

        # Zero out statistically insignificant relationships
        if p > alpha_corrected:
            s = 0.0

        strengths[(a, b)] = s
        strengths[(b, a)] = s

    return strengths


# ---------------------------------------------------------------------------
# Graph construction — protected attributes are SINK nodes
# ---------------------------------------------------------------------------

def build_graph(
    df: pd.DataFrame,
    col_types: Dict[str, str],
    threshold: float,
    protected_attributes: Optional[List[str]] = None,
) -> Tuple[nx.DiGraph, Dict[Tuple[str, str], float]]:
    """
    Build a directed feature correlation graph.

    Protected attributes are sink nodes: edges are only added INTO them,
    never out of them. This enforces the semantic that protected attrs
    are prediction targets, not proxy sources.
    """
    if protected_attributes is None:
        protected_attributes = []
    protected_set = set(protected_attributes)

    strengths = _pairwise_strength(df, col_types)
    G = nx.DiGraph()
    G.add_nodes_from(df.columns)

    for (a, b), w in strengths.items():
        if a == b or w < threshold:
            continue

        # Never add outgoing edges FROM protected attributes.
        # Edge a -> b is allowed only if 'a' is not protected OR 'b' is not also protected
        # (we allow protected->protected edges to be skipped entirely)
        if a in protected_set:
            continue  # protected attrs are sinks, no outgoing edges

        G.add_edge(a, b, weight=w)

    return G, strengths


# ---------------------------------------------------------------------------
# Chain detection (DFS) — protected attrs as terminal-only nodes
# ---------------------------------------------------------------------------

def _dfs_chains(
    G: nx.DiGraph,
    target: str,
    max_depth: int,
    current_path: List[str],
    all_chains: List[List[str]],
    protected_set: frozenset,
) -> None:
    """
    DFS from current_path[-1] toward target.
    Protected attributes may only appear as the final (target) node.
    """
    current = current_path[-1]

    if len(current_path) > max_depth + 1:
        return

    if len(current_path) > 1 and current == target:
        all_chains.append(list(current_path))
        return

    for neighbor in G.successors(current):
        if neighbor in current_path:
            continue
        # Protected attrs that are NOT our target are blocked as intermediates
        if neighbor in protected_set and neighbor != target:
            continue
        _dfs_chains(G, target, max_depth, current_path + [neighbor], all_chains, protected_set)


def find_chains(
    G: nx.DiGraph,
    strengths: Dict[Tuple[str, str], float],
    protected_attributes: List[str],
    max_depth: int,
    col_types: Dict[str, str],
) -> List[Chain]:
    protected_set = frozenset(protected_attributes)
    non_protected = [n for n in G.nodes if n not in protected_set]
    chains: List[Chain] = []

    for protected in protected_attributes:
        if protected not in G.nodes:
            continue
        for start in non_protected:
            if start == protected:
                continue
            raw_paths: List[List[str]] = []
            _dfs_chains(G, protected, max_depth, [start], raw_paths, protected_set)

            for path in raw_paths:
                if len(path) < 2:
                    continue
                hops = [
                    ChainHop(
                        source=path[i],
                        target=path[i + 1],
                        weight=round(strengths.get((path[i], path[i + 1]), 0.0), 4),
                    )
                    for i in range(len(path) - 1)
                ]
                weights = [h.weight for h in hops]
                # Geometric mean of hop weights as initial risk proxy
                risk_score = float(np.prod(weights) ** (1.0 / max(len(weights), 1)))
                risk_label = _risk_label(risk_score)
                weakest = min(hops, key=lambda h: h.weight)

                chains.append(
                    Chain(
                        id=str(uuid.uuid4()),
                        path=path,
                        hops=hops,
                        risk_score=round(risk_score, 4),
                        risk_label=risk_label,
                        protected_attribute=protected,
                        weakest_link=weakest.source,
                    )
                )

    # Deduplicate identical paths, keep highest initial score
    seen: dict = {}
    for c in sorted(chains, key=lambda x: x.risk_score, reverse=True):
        key = tuple(c.path)
        if key not in seen:
            seen[key] = c

    return sorted(seen.values(), key=lambda x: x.risk_score, reverse=True)


def _risk_label(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Graph schema for frontend
# ---------------------------------------------------------------------------

def build_graph_schema(
    G: nx.DiGraph,
    chains: List[Chain],
    protected_attributes: List[str],
    col_types: Dict[str, str],
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    node_risk: Dict[str, str] = {n: "none" for n in G.nodes}
    for chain in chains:
        for node in chain.path:
            if node not in node_risk:
                continue
            current = _risk_level_value(node_risk[node])
            new = _risk_level_value(chain.risk_label.lower())
            if new > current:
                node_risk[node] = chain.risk_label.lower()

    nodes = [
        GraphNode(
            id=n,
            label=n,
            dtype=col_types.get(n, "categorical"),
            is_protected=n in protected_attributes,
            risk_level=node_risk.get(n, "none"),
        )
        for n in G.nodes
    ]

    edges = [
        GraphEdge(source=u, target=v, weight=round(d["weight"], 4))
        for u, v, d in G.edges(data=True)
    ]

    return nodes, edges


def _risk_level_value(label: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}.get(label.lower(), 0)
