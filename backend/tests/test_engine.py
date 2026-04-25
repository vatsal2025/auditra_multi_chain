"""
Phase 1.10 - Integration test on COMPAS dataset.
Run: cd backend && python -m pytest tests/test_engine.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.graph_engine import build_graph, detect_column_types, find_chains
from app.services.chain_scorer import score_all_chains

COMPAS_COLS = [
    "age", "c_charge_degree", "race", "age_cat", "score_text",
    "sex", "priors_count", "days_b_screening_arrest", "decile_score",
    "is_recid", "two_year_recid", "juv_fel_count", "juv_misd_count", "juv_other_count"
]
COMPAS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "compas.csv")


@pytest.fixture(scope="module")
def compas_df():
    if not os.path.exists(COMPAS_PATH):
        pytest.skip("COMPAS CSV not found.")
    df_raw = pd.read_csv(COMPAS_PATH)
    keep = [c for c in COMPAS_COLS if c in df_raw.columns]
    return df_raw[keep].dropna(subset=["race", "sex"]).reset_index(drop=True)


def fmt_chain(c):
    return " -> ".join(c.path) + " | risk=" + f"{c.risk_score:.2%}"


def test_column_type_detection(compas_df):
    types = detect_column_types(compas_df)
    assert "race" in types
    assert types["race"] == "categorical"
    assert "age" in types


def test_graph_builds(compas_df):
    col_types = detect_column_types(compas_df)
    G, strengths = build_graph(compas_df, col_types, threshold=0.10, protected_attributes=["race", "sex"])
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    assert G.number_of_nodes() >= 5
    assert G.number_of_edges() > 0


def test_chains_found_to_race(compas_df):
    col_types = detect_column_types(compas_df)
    G, strengths = build_graph(compas_df, col_types, threshold=0.10, protected_attributes=["race"])
    chains = find_chains(G, strengths, ["race"], max_depth=4, col_types=col_types)

    print(f"\nChains to 'race': {len(chains)}")
    for c in chains[:5]:
        print(" ", fmt_chain(c))

    assert len(chains) > 0, "Expected at least 1 chain leading to 'race'"


def test_chains_found_to_sex(compas_df):
    """Sex has weaker correlations in COMPAS; use a lower threshold."""
    col_types = detect_column_types(compas_df)
    G, strengths = build_graph(compas_df, col_types, threshold=0.05, protected_attributes=["sex"])
    chains = find_chains(G, strengths, ["sex"], max_depth=4, col_types=col_types)
    print(f"\nChains to 'sex' (threshold=0.05): {len(chains)}")
    for c in chains[:3]:
        print(" ", fmt_chain(c))
    assert len(chains) > 0, "Expected at least 1 chain leading to 'sex'"


def test_multi_hop_chain_exists(compas_df):
    """The core novelty claim: at least one chain has depth >= 2."""
    col_types = detect_column_types(compas_df)
    G, strengths = build_graph(compas_df, col_types, threshold=0.10, protected_attributes=["race", "sex"])
    chains = find_chains(G, strengths, ["race", "sex"], max_depth=4, col_types=col_types)

    multi_hop = [c for c in chains if len(c.hops) >= 2]
    print(f"\nMulti-hop chains (depth >= 2): {len(multi_hop)}")
    if multi_hop:
        top = multi_hop[0]
        print("  Top:", fmt_chain(top))

    assert len(multi_hop) > 0, "No multi-hop chains found. Lower threshold or increase depth."


def test_risk_scoring(compas_df):
    col_types = detect_column_types(compas_df)
    G, strengths = build_graph(compas_df, col_types, threshold=0.10, protected_attributes=["race"])
    chains = find_chains(G, strengths, ["race"], max_depth=3, col_types=col_types)

    scored = score_all_chains(compas_df, chains[:5])
    print("\nTop scored chains:")
    for c in scored:
        print(f"  {' -> '.join(c.path)} | {c.risk_label} ({c.risk_score:.0%})")

    assert all(0.0 <= c.risk_score <= 1.0 for c in scored)


def test_synthetic_chain_detection():
    """Confirms engine finds a planted chain in synthetic data - runs without COMPAS."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 500

    zip_code = rng.integers(10000, 99999, n).astype(str)
    income = rng.choice(["low", "medium", "high"], n, p=[0.4, 0.35, 0.25])
    credit = np.where(income == "low", rng.choice(["bad", "ok"], n, p=[0.7, 0.3]),
             np.where(income == "medium", rng.choice(["bad", "ok"], n, p=[0.4, 0.6]),
             rng.choice(["bad", "ok"], n, p=[0.2, 0.8])))
    race = np.where(income == "low", rng.choice(["A", "B"], n, p=[0.6, 0.4]),
           np.where(income == "medium", rng.choice(["A", "B"], n, p=[0.5, 0.5]),
           rng.choice(["A", "B"], n, p=[0.3, 0.7])))

    df = pd.DataFrame({"zip_code": zip_code, "income": income, "credit": credit, "race": race})
    col_types = detect_column_types(df)
    G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
    chains = find_chains(G, strengths, ["race"], max_depth=4, col_types=col_types)

    print(f"\nSynthetic chains found: {len(chains)}")
    for c in chains:
        print(" ", " -> ".join(c.path))

    assert len(chains) > 0
    paths = [tuple(c.path) for c in chains]
    assert any("income" in p and "race" in p for p in paths), \
        "income->race chain should be detected"
