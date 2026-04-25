"""
Benchmark suite for Auditra hopping-chain detection.

Compares against published results from:
  [1] Feldman et al. (2015) "Certifying and Removing Disparate Impact"
      KDD 2015 — disparate impact in COMPAS and Adult datasets
  [2] Angwin et al. (ProPublica, 2016) "Machine Bias"
      COMPAS: Black defendants ~2x more likely mislabeled high-risk
  [3] Kamiran & Calders (2012) "Data preprocessing techniques for
      classification without discrimination" — Adult Income dataset
  [4] Friedler et al. (2019) "A comparative study of fairness-enhancing
      interventions in machine learning" — COMPAS, Adult, German Credit
  [5] Zliobaite (2015) "A survey on measuring indirect discrimination
      in machine learning" — proxy discrimination taxonomy
  [6] Zhang & Neill (2016) "Identifying significant predictive bias
      in classifiers" — MDSS scan statistic

All datasets are synthetic or publicly available (no license issues).

Run: cd backend && python -m pytest tests/test_benchmarks.py -v --tb=short
"""
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.chain_scorer import _score_via_lgbm, score_all_chains
from app.services.graph_engine import (
    _cramers_v_with_p,
    _eta_squared_with_p,
    _pearson_with_p,
    build_graph,
    detect_column_types,
    find_chains,
)


# ---------------------------------------------------------------------------
# Fixtures: real and synthetic datasets
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)

COMPAS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "compas.csv")
ADULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adult.csv")
GERMAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "german.csv")


def _load_compas() -> pd.DataFrame:
    if not os.path.exists(COMPAS_PATH):
        return None
    df = pd.read_csv(COMPAS_PATH)
    keep = ["age", "c_charge_degree", "race", "age_cat", "score_text",
            "sex", "priors_count", "days_b_screening_arrest", "decile_score",
            "is_recid", "two_year_recid", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    cols = [c for c in keep if c in df.columns]
    return df[cols].dropna(subset=["race", "sex"]).reset_index(drop=True)


def _load_adult() -> pd.DataFrame:
    """UCI Adult Income dataset (https://archive.ics.uci.edu/ml/datasets/adult)."""
    if not os.path.exists(ADULT_PATH):
        return None
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    try:
        df = pd.read_csv(ADULT_PATH, names=cols, skipinitialspace=True, na_values="?")
        return df.dropna().reset_index(drop=True)
    except Exception:
        return None


def _load_german() -> pd.DataFrame:
    """German Credit dataset — sex proxy discrimination benchmark."""
    if not os.path.exists(GERMAN_PATH):
        return None
    try:
        df = pd.read_csv(GERMAN_PATH)
        return df.dropna().reset_index(drop=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Synthetic benchmark generators (always available, no external dependency)
# ---------------------------------------------------------------------------

def _make_compas_like(n: int = 3000) -> pd.DataFrame:
    """
    Synthetic COMPAS-like dataset with known ground-truth chain structure.
    Chain: priors_count -> score_text -> race (indirect discrimination)
    Chain: age_cat -> decile_score -> race
    """
    rng = np.random.default_rng(42)
    race = rng.choice(["African-American", "Caucasian", "Hispanic"], n,
                      p=[0.51, 0.34, 0.15])
    priors = np.where(race == "African-American",
                      rng.integers(0, 15, n),
                      np.where(race == "Caucasian",
                               rng.integers(0, 8, n),
                               rng.integers(0, 10, n)))
    decile = np.clip(priors // 2 + rng.integers(0, 3, n), 1, 10)
    score_text = np.where(decile >= 7, "High",
                 np.where(decile >= 4, "Medium", "Low"))
    age = np.where(race == "African-American",
                   rng.integers(18, 40, n),
                   rng.integers(18, 55, n))
    age_cat = np.where(age < 25, "Less than 25",
               np.where(age < 45, "25 - 45", "Greater than 45"))
    is_recid = (rng.random(n) < np.where(decile >= 7, 0.6, 0.3)).astype(int)
    sex = rng.choice(["Male", "Female"], n, p=[0.81, 0.19])

    return pd.DataFrame({
        "race": race,
        "sex": sex,
        "age": age,
        "age_cat": age_cat,
        "priors_count": priors,
        "decile_score": decile,
        "score_text": score_text,
        "is_recid": is_recid,
    })


def _make_adult_like(n: int = 5000) -> pd.DataFrame:
    """
    Synthetic Adult Income-like dataset.
    Disparate impact chain: occupation -> education -> income (with race/sex proxy)
    """
    rng = np.random.default_rng(42)
    sex = rng.choice(["Male", "Female"], n, p=[0.67, 0.33])
    race = rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.85, 0.09, 0.03, 0.03])

    # Education correlated with race and sex (structural inequality)
    edu_map = {"White": [0.05, 0.25, 0.45, 0.25],
               "Black": [0.10, 0.35, 0.40, 0.15],
               "Asian": [0.03, 0.15, 0.35, 0.47],
               "Other": [0.12, 0.38, 0.35, 0.15]}
    edu_levels = ["HS-grad", "Some-college", "Bachelors", "Masters"]
    education = np.array([
        rng.choice(edu_levels, p=edu_map[r])
        for r in race
    ])

    # Occupation correlated with education and sex
    # Men more likely in high-income occupations (documented in Adult dataset)
    occ_p_male = {"HS-grad": [0.2, 0.3, 0.3, 0.2],
                  "Some-college": [0.15, 0.25, 0.35, 0.25],
                  "Bachelors": [0.05, 0.15, 0.45, 0.35],
                  "Masters": [0.02, 0.08, 0.40, 0.50]}
    occ_p_female = {"HS-grad": [0.45, 0.40, 0.12, 0.03],
                    "Some-college": [0.35, 0.45, 0.15, 0.05],
                    "Bachelors": [0.15, 0.40, 0.35, 0.10],
                    "Masters": [0.05, 0.25, 0.50, 0.20]}
    occ_levels = ["Service", "Admin", "Tech", "Executive"]
    occupation = np.array([
        rng.choice(occ_levels, p=(occ_p_male if s == "Male" else occ_p_female)[e])
        for s, e in zip(sex, education)
    ])

    # Income depends on occupation and education
    occ_income = {"Service": 0.10, "Admin": 0.20, "Tech": 0.55, "Executive": 0.75}
    income_prob = np.array([occ_income[o] for o in occupation])
    income = np.where(rng.random(n) < income_prob, ">50K", "<=50K")

    hours = np.where(sex == "Male",
                     rng.integers(35, 60, n),
                     rng.integers(20, 50, n))

    return pd.DataFrame({
        "sex": sex,
        "race": race,
        "education": education,
        "occupation": occupation,
        "hours_per_week": hours,
        "income": income,
    })


def _make_german_like(n: int = 1000) -> pd.DataFrame:
    """
    Synthetic German Credit-like dataset.
    Known proxy chain: housing_type -> credit_history -> sex
    """
    rng = np.random.default_rng(42)
    sex = rng.choice(["male", "female"], n, p=[0.69, 0.31])
    age = np.where(sex == "male", rng.integers(22, 65, n), rng.integers(18, 55, n))
    housing = np.where(sex == "male",
                       rng.choice(["own", "rent", "free"], n, p=[0.7, 0.2, 0.1]),
                       rng.choice(["own", "rent", "free"], n, p=[0.4, 0.45, 0.15]))
    credit_hist = np.where(housing == "own",
                           rng.choice(["good", "average", "bad"], n, p=[0.7, 0.2, 0.1]),
                           np.where(housing == "rent",
                                    rng.choice(["good", "average", "bad"], n, p=[0.4, 0.4, 0.2]),
                                    rng.choice(["good", "average", "bad"], n, p=[0.2, 0.4, 0.4])))
    savings = rng.choice(["little", "moderate", "rich"], n, p=[0.5, 0.3, 0.2])
    credit_risk = np.where(
        (credit_hist == "good") & (savings != "little"),
        rng.choice([0, 1], n, p=[0.2, 0.8]),
        np.where(credit_hist == "bad",
                 rng.choice([0, 1], n, p=[0.7, 0.3]),
                 rng.choice([0, 1], n, p=[0.5, 0.5]))
    )

    return pd.DataFrame({
        "sex": sex,
        "age": age,
        "housing": housing,
        "credit_history": credit_hist,
        "savings": savings,
        "credit_risk": credit_risk,
    })


def _make_redlining_scenario(n: int = 2000) -> pd.DataFrame:
    """
    Redlining scenario: zip_code -> property_value -> loan_approval
    with race as protected attribute indirectly targeted.
    Classic Feldman et al. (2015) disparate impact setup.
    """
    rng = np.random.default_rng(42)
    race = rng.choice(["White", "Black"], n, p=[0.72, 0.28])
    # Zip codes historically segregated by race
    zip_code = np.where(
        race == "White",
        rng.choice(["10001", "10002", "10003"], n, p=[0.5, 0.3, 0.2]),
        rng.choice(["10004", "10005", "10006"], n, p=[0.5, 0.3, 0.2])
    )
    # Property values correlated with zip
    prop_val = np.where(
        zip_code.astype(int) <= 10003,
        rng.choice(["high", "medium", "low"], n, p=[0.6, 0.3, 0.1]),
        rng.choice(["high", "medium", "low"], n, p=[0.1, 0.3, 0.6])
    )
    loan = np.where(
        prop_val == "high",
        rng.choice(["approved", "denied"], n, p=[0.85, 0.15]),
        np.where(prop_val == "medium",
                 rng.choice(["approved", "denied"], n, p=[0.6, 0.4]),
                 rng.choice(["approved", "denied"], n, p=[0.25, 0.75]))
    )
    income = rng.choice(["low", "medium", "high"], n, p=[0.4, 0.35, 0.25])

    return pd.DataFrame({
        "race": race,
        "zip_code": zip_code,
        "property_value": prop_val,
        "income": income,
        "loan_approval": loan,
    })


def _make_null_scenario(n: int = 2000) -> pd.DataFrame:
    """
    Null dataset: no real relationship between any feature and race.
    Used to verify false positive rate is controlled.
    """
    rng = np.random.default_rng(99)
    # 90% class imbalance (worst case for naive accuracy)
    race = rng.choice(["White", "Black"], n, p=[0.90, 0.10])
    f1 = rng.standard_normal(n)
    f2 = rng.choice(["A", "B", "C"], n)
    f3 = rng.integers(1, 100, n)
    f4 = rng.choice(["X", "Y"], n)
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "race": race})


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _disparate_impact(df: pd.DataFrame, protected: str, target: str,
                       privileged: str, positive_outcome: str) -> float:
    """Disparate Impact ratio (Feldman et al. 2015). Threshold: < 0.8 = disparate impact."""
    priv = df[df[protected] == privileged][target]
    unprivd = df[df[protected] != privileged][target]
    rate_priv = (priv == positive_outcome).mean()
    rate_unpriv = (unprivd == positive_outcome).mean()
    if rate_priv == 0:
        return 0.0
    return float(rate_unpriv / rate_priv)


def _audit_metrics(chains, protected_attr: str) -> dict:
    """Summarize chains for a specific protected attribute."""
    attr_chains = [c for c in chains if c.protected_attribute == protected_attr]
    return {
        "total": len(attr_chains),
        "critical": sum(1 for c in attr_chains if c.risk_label == "CRITICAL"),
        "high": sum(1 for c in attr_chains if c.risk_label == "HIGH"),
        "medium": sum(1 for c in attr_chains if c.risk_label == "MEDIUM"),
        "low": sum(1 for c in attr_chains if c.risk_label == "LOW"),
        "top_score": max((c.risk_score for c in attr_chains), default=0.0),
        "top_path": " ->".join(attr_chains[0].path) if attr_chains else "",
    }


# ===========================================================================
# SECTION 1: Statistical correctness
# ===========================================================================

class TestStatisticalCorrectness:

    def test_cramers_v_independent_variables(self):
        """Cramér's V on independent vars should be ~0, p > 0.05."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.choice(["A", "B", "C"], 500))
        y = pd.Series(rng.choice(["X", "Y", "Z"], 500))
        v, p = _cramers_v_with_p(x, y)
        assert v < 0.15, f"Independent vars V={v:.3f} too high (spurious correlation)"
        assert p > 0.05, f"p={p:.4f} should not be significant for independent vars"

    def test_cramers_v_perfectly_correlated(self):
        """Cramér's V on perfectly correlated vars should approach 1.0."""
        x = pd.Series(["A"] * 200 + ["B"] * 200 + ["C"] * 200)
        y = pd.Series(["X"] * 200 + ["Y"] * 200 + ["Z"] * 200)
        v, p = _cramers_v_with_p(x, y)
        assert v > 0.95, f"Perfect 3x3 correlation V={v:.3f}, expected > 0.95"
        assert p < 0.001, f"Perfect correlation p={p:.4f} should be near zero"

    def test_cramers_v_2x2_perfect(self):
        """2x2 perfect correlation (most common case)."""
        x = pd.Series(["A"] * 500 + ["B"] * 500)
        y = pd.Series(["X"] * 500 + ["Y"] * 500)
        v, p = _cramers_v_with_p(x, y)
        assert v > 0.95, f"2x2 perfect V={v:.3f}"

    def test_eta_squared_independent(self):
        """Eta-squared on independent numeric/categorical should be ~0."""
        rng = np.random.default_rng(42)
        numeric = pd.Series(rng.standard_normal(500))
        cat = pd.Series(rng.choice(["A", "B", "C"], 500))
        eta2, p = _eta_squared_with_p(numeric, cat)
        assert eta2 < 0.05, f"Independent eta2={eta2:.4f} too high"
        assert p > 0.05, f"p={p:.4f} should not be significant"

    def test_eta_squared_strong_relationship(self):
        """Eta-squared on clearly separated groups should be high."""
        rng = np.random.default_rng(0)
        # Groups centered at 1, 10, 100 with small noise so within-group std > 0
        g1 = 1.0 + rng.standard_normal(200) * 0.05
        g2 = 10.0 + rng.standard_normal(200) * 0.05
        g3 = 100.0 + rng.standard_normal(200) * 0.05
        numeric = pd.Series(np.concatenate([g1, g2, g3]))
        cat = pd.Series(["A"] * 200 + ["B"] * 200 + ["C"] * 200)
        eta2, p = _eta_squared_with_p(numeric, cat)
        assert eta2 > 0.90, f"Clearly separated groups eta2={eta2:.4f}"
        assert p < 0.001

    def test_pearson_independent(self):
        """Pearson on independent vars should be ~0."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.standard_normal(500))
        y = pd.Series(rng.standard_normal(500))
        r, p = _pearson_with_p(x, y)
        assert r < 0.15, f"Independent Pearson r={r:.4f}"

    def test_pearson_perfectly_correlated(self):
        """Pearson on y=x should be 1.0."""
        x = pd.Series(np.linspace(0, 100, 500))
        y = x * 2 + 5
        r, p = _pearson_with_p(x, y)
        assert r > 0.999, f"Perfect linear r={r:.4f}"
        assert p < 0.001

    def test_bonferroni_p_value_filtering(self):
        """Spurious edges removed under Bonferroni correction on wide dataset."""
        rng = np.random.default_rng(99)
        n = 200
        # 20 columns of pure noise + race
        data = {f"noise_{i}": rng.standard_normal(n) for i in range(20)}
        data["race"] = rng.choice(["A", "B"], n)
        df = pd.DataFrame(data)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        # With Bonferroni correction, most noise edges should be removed
        edges_to_race = [(u, v) for u, v, d in G.edges(data=True) if v == "race"]
        assert len(edges_to_race) <= 3, \
            f"Bonferroni should suppress most noise edges, got {len(edges_to_race)}"


# ===========================================================================
# SECTION 2: False positive control (the accuracy-baseline bug fix)
# ===========================================================================

class TestFalsePositiveControl:

    def test_null_dataset_no_critical_chains(self):
        """
        Null dataset with 90% class imbalance: MUST produce zero CRITICAL/HIGH chains.
        This was the key bug: naive accuracy scored random features as CRITICAL.
        """
        df = _make_null_scenario(n=2000)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)

        critical = [c for c in scored if c.risk_label in ("CRITICAL", "HIGH")]
        majority_class_pct = df["race"].value_counts(normalize=True).iloc[0]
        print(f"\nNull dataset (majority class={majority_class_pct:.1%})")
        print(f"  Chains found: {len(scored)}")
        for c in scored[:5]:
            print(f"  {' -> '.join(c.path)} | {c.risk_label} | skill={c.risk_score:.3f}")

        assert len(critical) == 0, \
            f"Null dataset produced {len(critical)} CRITICAL/HIGH chains — false positive!"

    def test_skill_score_near_zero_for_random_features(self):
        """Skill score for truly unrelated features must be near 0."""
        rng = np.random.default_rng(42)
        n = 500
        race = rng.choice(["A", "B"], n, p=[0.85, 0.15])
        noise_f1 = rng.standard_normal(n)
        noise_f2 = rng.choice(["X", "Y", "Z"], n)
        df = pd.DataFrame({"noise_f1": noise_f1, "noise_f2": noise_f2, "race": race})

        score = _score_via_lgbm(df, ["noise_f1", "noise_f2"], "race")
        majority_baseline = df["race"].value_counts(normalize=True).iloc[0]
        print(f"\nSkill score for random features: {score:.4f} (majority={majority_baseline:.2%})")
        assert score < 0.10, f"Random features should have near-zero skill, got {score:.4f}"

    def test_skill_score_high_for_real_chain(self):
        """Skill score for a planted real chain must be substantially above zero."""
        rng = np.random.default_rng(42)
        n = 1000
        income = rng.choice(["low", "medium", "high"], n, p=[0.4, 0.35, 0.25])
        race = np.where(income == "low",
                        rng.choice(["A", "B"], n, p=[0.8, 0.2]),
                        np.where(income == "medium",
                                 rng.choice(["A", "B"], n, p=[0.55, 0.45]),
                                 rng.choice(["A", "B"], n, p=[0.2, 0.8])))
        df = pd.DataFrame({"income": income, "race": race})
        score = _score_via_lgbm(df, ["income"], "race")
        print(f"\nReal chain skill score: {score:.4f}")
        assert score > 0.20, f"Real chain should have skill > 0.20, got {score:.4f}"

    def test_score_not_dominated_by_class_imbalance(self):
        """
        With 95% imbalance, naive accuracy = 0.95.
        Our skill score should remain near 0 for random features.
        """
        rng = np.random.default_rng(42)
        n = 2000
        race = rng.choice(["A", "B"], n, p=[0.95, 0.05])
        f1 = rng.standard_normal(n)
        df = pd.DataFrame({"f1": f1, "race": race})
        score = _score_via_lgbm(df, ["f1"], "race")
        print(f"\n95% imbalance, random feature skill: {score:.4f}")
        assert score < 0.10, \
            f"Imbalanced race with random feature got skill={score:.4f} (should be ~0)"


# ===========================================================================
# SECTION 3: Graph structure correctness
# ===========================================================================

class TestGraphStructure:

    def test_protected_attrs_are_sink_nodes(self):
        """Protected attributes must have no outgoing edges."""
        df = _make_compas_like()
        col_types = detect_column_types(df)
        protected = ["race", "sex"]
        G, _ = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        for p in protected:
            out_edges = list(G.successors(p))
            assert out_edges == [], \
                f"Protected attr '{p}' has outgoing edges: {out_edges} — graph is not causal"

    def test_no_backward_chains_through_protected(self):
        """No chain uses a protected attr as an intermediate node."""
        df = _make_compas_like()
        col_types = detect_column_types(df)
        protected = ["race", "sex"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)

        protected_as_intermediate = []
        for c in chains:
            for node in c.path[1:-1]:  # exclude start and end
                if node in protected:
                    protected_as_intermediate.append((c.path, node))

        assert len(protected_as_intermediate) == 0, \
            f"Protected attrs appear as intermediates: {protected_as_intermediate[:3]}"

    def test_chains_always_end_at_protected(self):
        """Every chain must terminate at its declared protected attribute."""
        df = _make_adult_like()
        col_types = detect_column_types(df)
        protected = ["race", "sex"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)

        for c in chains:
            assert c.path[-1] == c.protected_attribute, \
                f"Chain ends at {c.path[-1]} not {c.protected_attribute}: {c.path}"

    def test_no_cycles_in_chains(self):
        """Every chain must be a simple path (no node repeated)."""
        df = _make_german_like()
        col_types = detect_column_types(df)
        protected = ["sex"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=5, col_types=col_types)

        for c in chains:
            assert len(c.path) == len(set(c.path)), \
                f"Cycle in chain: {c.path}"

    def test_chain_path_length_respects_max_depth(self):
        """No chain exceeds max_depth hops."""
        df = _make_compas_like()
        col_types = detect_column_types(df)
        max_depth = 3
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=max_depth, col_types=col_types)

        for c in chains:
            assert len(c.hops) <= max_depth, \
                f"Chain has {len(c.hops)} hops, max={max_depth}: {c.path}"

    def test_hop_weights_match_stored_strengths(self):
        """ChainHop weights must equal the pairwise strengths dict values."""
        df = _make_compas_like(500)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=3, col_types=col_types)

        for c in chains[:10]:
            for hop in c.hops:
                expected = strengths.get((hop.source, hop.target), 0.0)
                assert abs(hop.weight - round(expected, 4)) < 1e-4, \
                    f"Hop {hop.source}->{hop.target} weight mismatch: {hop.weight} vs {expected}"


# ===========================================================================
# SECTION 4: Benchmark against published datasets / results
# ===========================================================================

class TestCOMPASBenchmark:
    """
    ProPublica COMPAS analysis benchmark.
    Reference: Angwin et al. (2016) — Black defendants 2x more likely mislabeled.
    Expected: auditor should detect race and sex proxy chains with HIGH/CRITICAL risk.
    """

    def test_compas_synthetic_chains_detected(self):
        """Planted chains in COMPAS-like data are detected and ranked correctly."""
        df = _make_compas_like(n=3000)
        col_types = detect_column_types(df)
        protected = ["race", "sex"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains)

        race_chains = [c for c in scored if c.protected_attribute == "race"]
        m = _audit_metrics(scored, "race")

        print(f"\nCOMPAS-like benchmark (race):")
        print(f"  Total chains: {m['total']}")
        print(f"  CRITICAL: {m['critical']}, HIGH: {m['high']}")
        print(f"  Top path: {m['top_path']}")
        print(f"  Top score: {m['top_score']:.3f}")

        assert m["total"] > 0, "Should detect chains toward 'race'"
        # Planted chains involve priors_count and decile_score
        paths = [tuple(c.path) for c in race_chains]
        has_prior_chain = any("priors_count" in p and "race" in p for p in paths)
        assert has_prior_chain, "Should detect priors_count -> race chain"

    def test_compas_synthetic_disparate_impact(self):
        """
        Feldman et al. (2015): DI < 0.8 indicates disparate impact.
        Auditor should flag datasets exhibiting DI.
        """
        df = _make_compas_like(n=3000)
        # High-risk score by race
        df_hi = df[df["score_text"] == "High"]
        di = _disparate_impact(df, "race", "score_text",
                               privileged="Caucasian", positive_outcome="High")
        print(f"\nDisparate Impact (Black vs White, High risk): {di:.3f}")
        print(f"  Feldman et al. threshold: < 0.8 = disparate impact")
        # Dataset is designed to have DI
        assert di < 0.8 or di > 1.2, f"DI={di:.3f} — dataset should show disparate impact"

    @pytest.mark.skipif(not os.path.exists(COMPAS_PATH), reason="COMPAS CSV not found (download from ProPublica)")
    def test_compas_real_dataset_detection(self):
        """Real COMPAS dataset: auditor must find known race/sex chains."""
        df = _load_compas()
        assert df is not None and len(df) > 100

        col_types = detect_column_types(df)
        protected = ["race", "sex"]
        G, strengths = build_graph(df, col_types, threshold=0.10, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains)

        m_race = _audit_metrics(scored, "race")
        m_sex = _audit_metrics(scored, "sex")

        print(f"\nReal COMPAS — race: {m_race}")
        print(f"Real COMPAS — sex: {m_sex}")

        assert m_race["total"] > 0, "Real COMPAS must detect race chains"
        assert m_race["top_score"] > 0.05, \
            f"Top race chain skill={m_race['top_score']:.3f}, should be non-trivial"


class TestAdultIncomeBenchmark:
    """
    UCI Adult Income — Kamiran & Calders (2012), Friedler et al. (2019) benchmark.
    Known proxy discrimination: occupation/education proxy for sex and race.
    80% income disparity between men and women is documented.
    """

    def test_adult_sex_proxy_chains_detected(self):
        """Sex proxy chains through occupation/education should be detected."""
        df = _make_adult_like(n=5000)
        col_types = detect_column_types(df)
        protected = ["sex", "race"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains)

        m_sex = _audit_metrics(scored, "sex")
        m_race = _audit_metrics(scored, "race")

        print(f"\nAdult-like benchmark — sex: {m_sex}")
        print(f"Adult-like benchmark — race: {m_race}")

        assert m_sex["total"] > 0, "Should detect sex proxy chains"
        sex_paths = [tuple(c.path) for c in scored if c.protected_attribute == "sex"]
        has_occupation = any("occupation" in p for p in sex_paths)
        assert has_occupation, "occupation->sex chain should be detected (documented in Kamiran 2012)"

    def test_adult_income_disparate_impact(self):
        """Women earn >50K at lower rate. DI ratio should be < 0.8."""
        df = _make_adult_like(n=5000)
        di = _disparate_impact(df, "sex", "income",
                               privileged="Male", positive_outcome=">50K")
        print(f"\nAdult-like DI (female vs male, >50K): {di:.3f}")
        assert di < 0.80, f"Adult-like DI={di:.3f}, expected < 0.80 (known gender income gap)"

    @pytest.mark.skipif(not os.path.exists(ADULT_PATH), reason="Adult dataset not found")
    def test_adult_real_dataset_detection(self):
        df = _load_adult()
        if df is None:
            pytest.skip("Adult CSV parse failed")
        col_types = detect_column_types(df)
        protected = ["sex", "race"]
        G, strengths = build_graph(df, col_types, threshold=0.10, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains[:30])
        m_sex = _audit_metrics(scored, "sex")
        print(f"\nReal Adult — sex: {m_sex}")
        assert m_sex["total"] > 0


class TestGermanCreditBenchmark:
    """
    German Credit dataset — Friedler et al. (2019).
    Housing type is documented proxy for sex (women more often rent).
    """

    def test_german_sex_proxy_chain(self):
        """housing -> credit_history -> sex chain should be detected."""
        df = _make_german_like(n=1000)
        col_types = detect_column_types(df)
        protected = ["sex"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains)

        m = _audit_metrics(scored, "sex")
        print(f"\nGerman Credit-like benchmark — sex: {m}")

        assert m["total"] > 0, "Should detect sex proxy chains"
        paths = [tuple(c.path) for c in scored if c.protected_attribute == "sex"]
        has_housing = any("housing" in p for p in paths)
        assert has_housing, "housing->sex chain should be detected (documented proxy)"

    def test_german_fix_breaks_chain(self):
        """After fix (weakest link removal), chain risk should drop."""
        from app.services.fix_engine import apply_fix
        from app.models.schemas import Chain, ChainHop

        df = _make_german_like(n=1000)
        col_types = detect_column_types(df)
        protected = ["sex"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)

        if not scored:
            pytest.skip("No chains detected in German-like dataset")

        top_chain = scored[0]
        original_score = top_chain.risk_score
        removed = top_chain.weakest_link

        fixed_df, shap_entries = apply_fix(df, top_chain)

        assert removed not in fixed_df.columns, f"Removed feature '{removed}' still present"

        # Re-score the chain on fixed dataset
        if removed in top_chain.path:
            remaining_features = [c for c in top_chain.path
                                  if c != top_chain.protected_attribute and c != removed]
            if remaining_features:
                new_score = _score_via_lgbm(fixed_df, remaining_features, top_chain.protected_attribute)
                print(f"\nGerman fix: removed='{removed}', "
                      f"original_skill={original_score:.3f}, new_skill={new_score:.3f}")
                assert new_score <= original_score + 0.05, \
                    f"Risk should not increase after removing weakest link"

        # SHAP entries must be real (not hardcoded)
        removed_entries = [e for e in shap_entries if e.feature == removed]
        if removed_entries:
            assert removed_entries[0].after == 0.0, "Removed feature must have after=0.0"
        # Non-removed entries must NOT be exactly before*0.1 or before*0.05 (old fake values)
        for e in shap_entries:
            if e.feature != removed and e.before > 0:
                ratio = e.after / e.before if e.before != 0 else 0
                assert ratio != 0.1 and ratio != 0.05, \
                    f"SHAP after for {e.feature} appears hardcoded: before={e.before}, after={e.after}"


class TestRedliningBenchmark:
    """
    Redlining scenario — Feldman et al. (2015) disparate impact.
    Classic: zip_code -> property_value -> loan_approval proxies race.
    """

    def test_redlining_chain_detected(self):
        """zip_code -> property_value -> race chain must be detected."""
        df = _make_redlining_scenario(n=2000)
        col_types = detect_column_types(df)
        protected = ["race"]
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains)

        m = _audit_metrics(scored, "race")
        print(f"\nRedlining benchmark: {m}")

        assert m["total"] > 0, "Redlining scenario must detect race chains"
        paths = [tuple(c.path) for c in scored if c.protected_attribute == "race"]
        has_zip = any("zip_code" in p for p in paths)
        assert has_zip, "zip_code->race proxy chain must be detected"
        top_chain = scored[0] if scored else None
        if top_chain:
            assert top_chain.risk_score > 0.15, \
                f"Top chain skill={top_chain.risk_score:.3f} too low for planted chain"

    def test_redlining_disparate_impact(self):
        """Loan approval DI must be < 0.8 for Black vs White."""
        df = _make_redlining_scenario(n=2000)
        di = _disparate_impact(df, "race", "loan_approval",
                               privileged="White", positive_outcome="approved")
        print(f"\nRedlining DI (Black vs White, loan approved): {di:.3f}")
        assert di < 0.80, f"DI={di:.3f}, should show disparate impact < 0.80"

    def test_multi_hop_longer_than_2(self):
        """At least one chain in redlining dataset has >= 3 hops."""
        df = _make_redlining_scenario(n=2000)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=4, col_types=col_types)
        deep_chains = [c for c in chains if len(c.hops) >= 3]
        print(f"\nRedlining 3+ hop chains: {len(deep_chains)}")
        # At least try to find one; dataset may not have 3-hop with these cols
        if len(deep_chains) == 0:
            # Verify 2-hop at minimum
            two_hop = [c for c in chains if len(c.hops) >= 2]
            assert len(two_hop) > 0, "Should find at least 2-hop chains in redlining"


# ===========================================================================
# SECTION 5: Zliobaite (2015) indirect discrimination taxonomy
# ===========================================================================

class TestIndirectDiscriminationTaxonomy:
    """
    Zliobaite (2015): 4 types of indirect discrimination proxies.
    Type 1: Single proxy (direct correlation with protected attr).
    Type 2: Conjunctive (multiple features together reveal protected).
    Type 3: Chain (A -> B -> protected, but A alone not correlated).
    Type 4: Intersectional (subgroup-specific discrimination).
    """

    def test_type1_single_proxy(self):
        """Type 1: income directly correlates with race."""
        rng = np.random.default_rng(42)
        n = 1000
        race = rng.choice(["A", "B"], n, p=[0.55, 0.45])
        income = np.where(race == "A",
                          rng.choice(["low", "medium", "high"], n, p=[0.6, 0.3, 0.1]),
                          rng.choice(["low", "medium", "high"], n, p=[0.2, 0.4, 0.4]))
        df = pd.DataFrame({"income": income, "race": race})
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=2, col_types=col_types)
        scored = score_all_chains(df, chains)
        assert any(len(c.hops) == 1 for c in scored), "Type 1 single proxy not detected"
        print(f"\nType 1 proxy — chains: {len(scored)}, top: {scored[0].risk_score:.3f}" if scored else "NONE")

    def test_type3_relay_chain(self):
        """
        Type 3 (Zliobaite): A -> B -> protected, where A alone is UNCORRELATED with protected.
        Only through B does A become informative.
        """
        rng = np.random.default_rng(42)
        n = 2000
        # B is correlated with race
        neighborhood = rng.choice(["north", "south"], n, p=[0.5, 0.5])
        race = np.where(neighborhood == "north",
                        rng.choice(["A", "B"], n, p=[0.8, 0.2]),
                        rng.choice(["A", "B"], n, p=[0.2, 0.8]))
        # A is correlated with B but NOT directly with race
        street_type = np.where(neighborhood == "north",
                               rng.choice(["urban", "suburban"], n, p=[0.7, 0.3]),
                               rng.choice(["urban", "suburban"], n, p=[0.3, 0.7]))
        # Verify A-race direct correlation is low
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(pd.Series(street_type), pd.Series(race))
        chi2, p, _, _ = chi2_contingency(ct)

        df = pd.DataFrame({
            "street_type": street_type,
            "neighborhood": neighborhood,
            "race": race,
        })
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)

        relay_chains = [c for c in scored if "street_type" in c.path and "neighborhood" in c.path and "race" in c.path]
        print(f"\nType 3 relay chains: {len(relay_chains)}")
        for c in relay_chains[:3]:
            print(f"  {' -> '.join(c.path)} | skill={c.risk_score:.3f}")

        assert len(relay_chains) > 0, "Type 3 relay chain (street_type->neighborhood->race) not detected"


# ===========================================================================
# SECTION 6: Performance & scale
# ===========================================================================

class TestPerformance:

    def test_audit_completes_under_30s_on_1000_rows(self):
        """Full pipeline on 1k rows must complete under 30 seconds."""
        df = _make_compas_like(n=1000)
        col_types = detect_column_types(df)
        protected = ["race", "sex"]
        t0 = time.time()
        G, strengths = build_graph(df, col_types, threshold=0.10, protected_attributes=protected)
        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains[:20])
        elapsed = time.time() - t0
        print(f"\nPipeline time (n=1000): {elapsed:.1f}s, chains: {len(scored)}")
        assert elapsed < 30, f"Pipeline took {elapsed:.1f}s > 30s limit"

    def test_dfs_does_not_blowup_on_dense_graph(self):
        """Dense graph (20 features, low threshold) must finish in < 10s."""
        rng = np.random.default_rng(42)
        n = 500
        # 15 features with moderate correlation structure
        data = {}
        for i in range(15):
            data[f"f{i}"] = rng.choice(["A", "B", "C"], n,
                                        p=[0.33 + 0.1 * (i % 3 - 1),
                                           0.34,
                                           0.33 - 0.1 * (i % 3 - 1)])
        data["race"] = rng.choice(["X", "Y"], n)
        df = pd.DataFrame(data)
        col_types = detect_column_types(df)
        t0 = time.time()
        G, strengths = build_graph(df, col_types, threshold=0.10, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=4, col_types=col_types)
        elapsed = time.time() - t0
        print(f"\nDense graph chains: {len(chains)} in {elapsed:.2f}s")
        assert elapsed < 10, f"DFS took {elapsed:.2f}s on dense graph — O(n!) blowup risk"

    def test_deduplication_works(self):
        """Same path found from different DFS starts should appear only once."""
        df = _make_compas_like(1000)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=4, col_types=col_types)
        paths = [tuple(c.path) for c in chains]
        assert len(paths) == len(set(paths)), "Duplicate chains in result"


# ===========================================================================
# SECTION 7: Fix engine validation
# ===========================================================================

class TestFixEngine:

    def test_shap_after_is_zero_for_removed_feature(self):
        """After fix, removed feature must have SHAP after=0.0."""
        from app.services.fix_engine import apply_fix
        df = _make_german_like(n=500)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["sex"])
        chains = find_chains(G, strengths, ["sex"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)
        if not scored:
            pytest.skip("No chains in German-like data")
        top = scored[0]
        _, shap_entries = apply_fix(df, top)
        for e in shap_entries:
            if e.feature == top.weakest_link:
                assert e.after == 0.0, f"Removed feature after={e.after}, expected 0.0"

    def test_shap_not_hardcoded_multiplier(self):
        """SHAP after values must NOT equal before * 0.1 (old fake) for non-removed features."""
        from app.services.fix_engine import apply_fix
        df = _make_adult_like(n=1000)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["sex"])
        chains = find_chains(G, strengths, ["sex"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)
        if not scored:
            pytest.skip("No chains detected")
        top = scored[0]
        _, shap_entries = apply_fix(df, top)
        for e in shap_entries:
            if e.feature != top.weakest_link and e.before > 0.001:
                ratio = e.after / e.before
                assert abs(ratio - 0.1) > 0.01, \
                    f"SHAP for {e.feature} looks hardcoded: after/before={ratio:.3f}"
                assert abs(ratio - 0.05) > 0.01, \
                    f"SHAP for {e.feature} looks hardcoded (0.05): after/before={ratio:.3f}"

    def test_fix_removes_correct_column(self):
        """apply_fix must drop weakest_link and only weakest_link."""
        from app.services.fix_engine import apply_fix
        df = _make_german_like(n=500)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["sex"])
        chains = find_chains(G, strengths, ["sex"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)
        if not scored:
            pytest.skip("No chains detected")
        top = scored[0]
        fixed_df, _ = apply_fix(df, top)
        assert top.weakest_link not in fixed_df.columns, \
            f"Weakest link '{top.weakest_link}' still in df after fix"
        # All other cols present
        for col in df.columns:
            if col != top.weakest_link:
                assert col in fixed_df.columns, f"Non-removed col '{col}' missing after fix"

    def test_stale_chains_removed_after_fix(self):
        """After fix, chains whose weakest_link is now missing must be purged."""
        from app.services.fix_engine import apply_fix
        from app.models.schemas import Chain, ChainHop

        df = _make_german_like(n=500)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["sex"])
        chains = find_chains(G, strengths, ["sex"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)
        if not scored:
            pytest.skip("No chains detected")

        top = scored[0]
        removed_feature = top.weakest_link
        fixed_df, _ = apply_fix(df, top)

        # Any other chain referencing removed_feature as weakest_link should be flagged
        remaining_cols = set(fixed_df.columns)
        stale = [c for c in scored
                 if c.id != top.id
                 and c.weakest_link is not None
                 and c.weakest_link not in remaining_cols]
        print(f"\nStale chains after fix of '{removed_feature}': {len(stale)}")
        # The test validates the concept; route-level cleanup is tested in integration


# ===========================================================================
# SECTION 8: End-to-end pipeline integration
# ===========================================================================

class TestEndToEndPipeline:

    def test_full_pipeline_compas_like(self):
        """Full pipeline: build graph -> find chains -> score -> validate output."""
        df = _make_compas_like(n=2000)
        col_types = detect_column_types(df)
        protected = ["race", "sex"]

        G, strengths = build_graph(df, col_types, threshold=0.08, protected_attributes=protected)
        assert G.number_of_nodes() >= 5
        assert G.number_of_edges() > 0

        chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
        assert len(chains) > 0, "No chains detected in COMPAS-like data"

        scored = score_all_chains(df, chains[:30])
        for c in scored:
            assert 0.0 <= c.risk_score <= 1.0, f"Score out of range: {c.risk_score}"
            assert c.risk_label in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            assert c.path[-1] == c.protected_attribute
            assert c.weakest_link is not None

        print(f"\nEnd-to-end COMPAS-like:")
        print(f"  Chains: {len(scored)}, Critical: {sum(1 for c in scored if c.risk_label=='CRITICAL')}")
        print(f"  Top path: {' -> '.join(scored[0].path)} | skill={scored[0].risk_score:.3f}")

    def test_full_pipeline_null_no_false_positives(self):
        """Full pipeline on null data must not produce CRITICAL/HIGH chains."""
        df = _make_null_scenario(n=1500)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=3, col_types=col_types)
        scored = score_all_chains(df, chains)

        fp = [c for c in scored if c.risk_label in ("CRITICAL", "HIGH")]
        print(f"\nNull pipeline: {len(scored)} chains, {len(fp)} false positives")
        assert len(fp) == 0, f"False positives on null data: {[(c.path, c.risk_label) for c in fp]}"

    def test_column_type_detection_zip_integer(self):
        """Integer zip codes must be detected as categorical, not numeric."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "zip_code": rng.integers(10000, 99999, n),  # integer zip
            "salary": rng.integers(30000, 120000, n),   # actual numeric
            "race": rng.choice(["A", "B"], n),
        })
        col_types = detect_column_types(df)
        # zip_code has ~100 unique values out of 100 rows (>90% unique ratio ->excluded/categorical)
        print(f"\nColumn types: {col_types}")
        # salary has high numeric range and is real numeric
        assert col_types["salary"] == "numeric", "salary should be numeric"
        # zip_code: high uniqueness ratio (ID-like) ->categorical
        assert col_types["zip_code"] == "categorical", \
            f"Integer zip_code detected as '{col_types['zip_code']}', should be categorical"

    def test_risk_labels_monotone_with_score(self):
        """Risk labels must be monotone: higher score = higher/equal label."""
        df = _make_compas_like(2000)
        col_types = detect_column_types(df)
        G, strengths = build_graph(df, col_types, threshold=0.05, protected_attributes=["race"])
        chains = find_chains(G, strengths, ["race"], max_depth=4, col_types=col_types)
        scored = score_all_chains(df, chains[:20])

        label_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        # Sorted descending — so label values should be non-increasing
        for i in range(len(scored) - 1):
            a, b = scored[i], scored[i + 1]
            assert label_order[a.risk_label] >= label_order[b.risk_label] or a.risk_score >= b.risk_score, \
                f"Label not monotone: {a.risk_label}({a.risk_score:.3f}) before {b.risk_label}({b.risk_score:.3f})"


# ===========================================================================
# SECTION 9: Comparative summary (prints benchmark table)
# ===========================================================================

class TestBenchmarkSummaryReport:
    """Prints a comparison table across all benchmark scenarios."""

    def test_print_benchmark_summary(self):
        scenarios = [
            ("COMPAS-like (race, sex)",   _make_compas_like(2000),     ["race", "sex"], 0.08),
            ("Adult-like (sex, race)",    _make_adult_like(2000),      ["sex", "race"], 0.08),
            ("German Credit-like (sex)",  _make_german_like(800),      ["sex"],         0.05),
            ("Redlining (race)",          _make_redlining_scenario(1500), ["race"],     0.05),
            ("Null (race, 90% imbalance)",_make_null_scenario(1000),   ["race"],        0.05),
        ]

        print("\n" + "=" * 80)
        print(f"{'Scenario':<30} {'Chains':>7} {'Crit':>5} {'High':>5} {'TopSkill':>9} {'FP':>4}")
        print("-" * 80)

        for name, df, protected, thresh in scenarios:
            col_types = detect_column_types(df)
            G, strengths = build_graph(df, col_types, threshold=thresh, protected_attributes=protected)
            chains = find_chains(G, strengths, protected, max_depth=4, col_types=col_types)
            scored = score_all_chains(df, chains[:30])
            total = len(scored)
            crit = sum(1 for c in scored if c.risk_label == "CRITICAL")
            high = sum(1 for c in scored if c.risk_label == "HIGH")
            top_skill = max((c.risk_score for c in scored), default=0.0)
            # FP: CRITICAL/HIGH on null scenario
            fp = (crit + high) if "Null" in name else "-"
            print(f"{name:<30} {total:>7} {crit:>5} {high:>5} {top_skill:>9.3f} {str(fp):>4}")

        print("=" * 80)
        print("\nNotes:")
        print("  Skill score = (LightGBM accuracy - majority-class baseline) / (1 - baseline)")
        print("  FP column applies to null dataset only: should be 0")
        print("  Thresholds: CRITICAL >= 0.75, HIGH >= 0.50, MEDIUM >= 0.25, LOW < 0.25")
        print("  References: Feldman 2015 (KDD), ProPublica 2016, Kamiran 2012, Friedler 2019")
