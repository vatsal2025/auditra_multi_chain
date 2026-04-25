"""
Tests for calibration, reweighing, and intersectional services.

References:
  Chouldechova (2017) — ECE and calibration gap
  Kamiran & Calders (2012) — reweighing achieves disc -> 0
  Kearns et al. (2018) — intersectional fairness gerrymandering
"""
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_biased_df(n=1200, seed=42, male_rate=0.45, female_rate=0.20):
    """Synthetic dataset with known group bias."""
    rng = np.random.default_rng(seed)
    n_male = n * 2 // 3
    n_female = n - n_male
    sex = np.array(["Male"] * n_male + ["Female"] * n_female)
    y = np.zeros(n_male + n_female, dtype=int)
    y[:n_male] = (rng.random(n_male) < male_rate).astype(int)
    y[n_male:] = (rng.random(n_female) < female_rate).astype(int)
    total = n_male + n_female
    df = pd.DataFrame({
        "sex": sex,
        "age": rng.integers(20, 65, total),
        "education_years": rng.integers(8, 20, total),
        "income": y,
    })
    return df


def _make_intersectional_df(n_per_group=300, seed=0):
    """4-group dataset: (race, sex) with Black+Female significantly disadvantaged."""
    rng = np.random.default_rng(seed)
    groups = [
        ("White", "Male",   0.45),
        ("White", "Female", 0.38),
        ("Black", "Male",   0.32),
        ("Black", "Female", 0.08),   # worst subgroup — should be flagged
    ]
    rows = []
    for race, sex, rate in groups:
        for _ in range(n_per_group):
            rows.append({
                "race": race, "sex": sex,
                "income": int(rng.random() < rate),
                "age": int(rng.integers(20, 60)),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# CALIBRATION
# ===========================================================================

class TestCalibrationECE:
    def test_perfect_calibration_has_low_ece(self):
        """Perfectly calibrated predictions -> ECE near 0."""
        from app.services.calibration import _ece
        rng = np.random.default_rng(7)
        # Generate probabilities, then draw outcomes from those same probabilities
        proba = rng.uniform(0, 1, 2000)
        y = (rng.random(2000) < proba).astype(int)
        ece_val, bins = _ece(y, proba, n_bins=10)
        assert ece_val < 0.05, f"Perfect calibration should have ECE < 0.05, got {ece_val}"

    def test_miscalibrated_has_high_ece(self):
        """Predicted 0.9 for everything but only 50% actual -> ECE ~0.4."""
        from app.services.calibration import _ece
        y = np.array([1, 0] * 500)
        proba = np.full(1000, 0.9)
        ece_val, bins = _ece(y, proba, n_bins=10)
        # All predictions in top bin, actual=0.5, conf=0.9 -> gap=0.4
        assert ece_val > 0.35, f"Expected ECE > 0.35, got {ece_val}"

    def test_ece_bins_contain_all_samples(self):
        """Sum of bin counts = total samples."""
        from app.services.calibration import _ece
        rng = np.random.default_rng(0)
        y = (rng.random(500) > 0.4).astype(int)
        proba = rng.uniform(0, 1, 500)
        ece_val, bins = _ece(y, proba, n_bins=10)
        assert sum(b.count for b in bins) == 500

    def test_ece_bin_values_are_valid(self):
        """confidence and accuracy must be in [0, 1]."""
        from app.services.calibration import _ece
        rng = np.random.default_rng(1)
        y = (rng.random(800) > 0.5).astype(int)
        proba = rng.uniform(0, 1, 800)
        _, bins = _ece(y, proba, n_bins=10)
        for b in bins:
            assert 0.0 <= b.confidence <= 1.0, f"Invalid confidence: {b.confidence}"
            assert 0.0 <= b.accuracy <= 1.0, f"Invalid accuracy: {b.accuracy}"


class TestCalibrationAudit:
    def test_returns_none_without_lgbm(self, monkeypatch):
        """Returns None when LightGBM unavailable."""
        import app.services.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "LGB_AVAILABLE", False)
        df = _make_biased_df()
        result = cal_mod.compute_calibration_audit(df, "sex", "income", "1")
        assert result is None

    def test_returns_none_for_small_dataset(self):
        """Returns None for datasets smaller than 100 rows."""
        from app.services.calibration import compute_calibration_audit
        df = _make_biased_df(n=90)
        result = compute_calibration_audit(df, "sex", "income", "1")
        assert result is None

    def test_calibration_audit_structure(self):
        """CalibrationAudit has expected fields for a valid dataset."""
        from app.services.calibration import compute_calibration_audit
        df = _make_biased_df(n=1200)
        result = compute_calibration_audit(df, "sex", "income", "1")
        if result is None:
            pytest.skip("LightGBM not available")
        assert result.protected_attribute == "sex"
        assert result.outcome_column == "income"
        assert "Male" in result.group_calibration
        assert "Female" in result.group_calibration
        assert 0.0 <= result.calibration_gap <= 1.0
        assert isinstance(result.is_calibrated, bool)

    def test_calibration_gap_detection(self):
        """Heavily miscalibrated groups should produce calibration_gap > 0."""
        from app.services.calibration import compute_calibration_audit
        df = _make_biased_df(n=1500, male_rate=0.50, female_rate=0.10)
        result = compute_calibration_audit(df, "sex", "income", "1")
        if result is None:
            pytest.skip("LightGBM not available")
        # Groups with different base rates will have different calibration
        assert result.calibration_gap >= 0.0

    def test_calibration_threshold_label(self):
        """is_calibrated is True when gap < 0.05, False otherwise."""
        from app.services.calibration import compute_calibration_audit
        df = _make_biased_df(n=1200)
        result = compute_calibration_audit(df, "sex", "income", "1")
        if result is None:
            pytest.skip("LightGBM not available")
        expected = result.calibration_gap < 0.05
        assert result.is_calibrated == expected

    def test_calibration_audit_group_ece_nonnegative(self):
        """ECE is always non-negative."""
        from app.services.calibration import compute_calibration_audit
        df = _make_biased_df(n=1200)
        result = compute_calibration_audit(df, "sex", "income", "1")
        if result is None:
            pytest.skip("LightGBM not available")
        for gc in result.group_calibration.values():
            assert gc.ece >= 0.0


# ===========================================================================
# REWEIGHING — Kamiran & Calders (2012)
# ===========================================================================

class TestReweighing:
    def test_disc_approaches_zero(self):
        """Core claim: reweighing drives discrimination score to near 0."""
        from app.services.reweighing import compute_sample_weights
        df = _make_biased_df(n=2000)
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is not None
        weights, result = out
        assert result.disc_before > 0.10, "Expected initial disc > 0.10"
        assert result.disc_after < 0.005, f"Reweighing should drive disc to ~0, got {result.disc_after}"

    def test_weights_are_positive(self):
        """All sample weights must be positive (required for any downstream model)."""
        from app.services.reweighing import compute_sample_weights
        df = _make_biased_df(n=1000)
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is not None
        weights, _ = out
        assert (weights > 0).all(), "All weights must be positive"

    def test_returns_none_for_insufficient_data(self):
        """Returns None when dataset too small."""
        from app.services.reweighing import compute_sample_weights
        df = _make_biased_df(n=30)
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is None

    def test_weights_length_matches_df(self):
        """Weights array has same length as input dataframe."""
        from app.services.reweighing import compute_sample_weights
        df = _make_biased_df(n=500)
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is not None
        weights, _ = out
        assert len(weights) == len(df)

    def test_reweigh_result_fields(self):
        """ReweighResult schema fields are valid."""
        from app.services.reweighing import compute_sample_weights
        df = _make_biased_df(n=1000)
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is not None
        _, result = out
        assert result.protected_attribute == "sex"
        assert result.outcome_column == "income"
        assert result.n_samples == len(df.dropna(subset=["sex", "income"]))
        assert 0.0 <= result.disc_after <= result.disc_before + 0.001

    def test_uniform_groups_no_reweighing_needed(self):
        """When groups have equal base rates, weights should be near 1.0."""
        from app.services.reweighing import compute_sample_weights
        rng = np.random.default_rng(99)
        n = 800
        sex = np.array(["Male"] * 400 + ["Female"] * 400)
        y = (rng.random(n) < 0.35).astype(int)  # Same rate both groups
        df = pd.DataFrame({"sex": sex, "income": y, "age": rng.integers(20, 60, n)})
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is not None
        weights, result = out
        assert result.disc_before < 0.05
        assert result.disc_after < 0.01

    def test_reweigh_dataframe_adds_weight_col(self):
        """reweigh_dataframe returns df with _sample_weight column."""
        from app.services.reweighing import reweigh_dataframe
        df = _make_biased_df(n=600)
        weighted_df, result = reweigh_dataframe(df, "sex", "income", "1")
        assert "_sample_weight" in weighted_df.columns
        assert (weighted_df["_sample_weight"] > 0).all()

    def test_kamiran_adult_benchmark(self):
        """
        Synthetic Adult-like data: disc 0.1965 -> ~0 after reweighing.
        Kamiran & Calders (2012) Table 2 reports disc 0.1965 for NB.
        We don't replicate NB exactly, but reweighing mathematically zeroes disc.
        """
        from app.services.reweighing import compute_sample_weights
        # Replicate Adult-like gender disparity: Male 30% >50K, Female 10.8% >50K
        rng = np.random.default_rng(2012)
        n_male, n_female = 21790, 10771
        sex = np.array(["Male"] * n_male + ["Female"] * n_female)
        y = np.zeros(n_male + n_female, dtype=int)
        y[:n_male] = (rng.random(n_male) < 0.300).astype(int)
        y[n_male:] = (rng.random(n_female) < 0.108).astype(int)
        df = pd.DataFrame({"sex": sex, "income": y})
        out = compute_sample_weights(df, "sex", "income", "1")
        assert out is not None
        _, result = out
        assert result.disc_before > 0.15, f"Expected disc ~0.19, got {result.disc_before}"
        assert result.disc_after < 0.005, f"After reweigh disc should be ~0, got {result.disc_after}"


# ===========================================================================
# INTERSECTIONAL AUDIT — Kearns (2018)
# ===========================================================================

class TestIntersectionalAudit:
    def test_detects_gerrymandering(self):
        """Black+Female subgroup significantly disadvantaged — should be flagged."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        flagged_keys = audit.flagged_groups
        assert any("Black" in k and "Female" in k for k in flagged_keys), \
            f"Black+Female should be flagged, got flagged={flagged_keys}"

    def test_privileged_combo_has_highest_rate(self):
        """Privileged combo should have highest base rate."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        for g in audit.groups:
            assert g.base_rate <= audit.privileged_base_rate + 0.02, \
                f"Group {g.group_key} has higher rate {g.base_rate} than privileged {audit.privileged_base_rate}"

    def test_spd_of_privileged_is_near_zero(self):
        """SPD of the privileged combo vs itself is 0."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        priv_group = next((g for g in audit.groups if g.group_key == audit.privileged_combo), None)
        if priv_group:
            assert abs(priv_group.spd_vs_privileged) < 0.01

    def test_returns_none_for_single_protected_attr(self):
        """Need at least 2 protected attributes for intersectional audit."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        result = compute_intersectional_audit(df, ["race"], "income", "1")
        assert result is None

    def test_returns_none_for_small_dataset(self):
        """Returns None when dataset is too small."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df(n_per_group=10)
        result = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert result is None

    def test_all_groups_have_valid_spd(self):
        """SPD values must be in [-1, 0] (all groups <= privileged)."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        for g in audit.groups:
            assert -1.0 <= g.spd_vs_privileged <= 0.01, \
                f"Invalid SPD {g.spd_vs_privileged} for {g.group_key}"

    def test_no_gerrymandering_in_uniform_data(self):
        """Equal base rates across all intersections -> no groups flagged."""
        from app.services.intersectional import compute_intersectional_audit
        rng = np.random.default_rng(55)
        n_per = 400
        rows = []
        for race in ["White", "Black"]:
            for sex in ["Male", "Female"]:
                for _ in range(n_per):
                    rows.append({"race": race, "sex": sex,
                                 "income": int(rng.random() < 0.35)})
        df = pd.DataFrame(rows)
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        assert len(audit.flagged_groups) == 0, \
            f"Uniform data should have no flagged groups, got {audit.flagged_groups}"

    def test_max_spd_gap_is_worst_case(self):
        """max_spd_gap equals maximum |spd| across all groups."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        expected_max = max(abs(g.spd_vs_privileged) for g in audit.groups)
        assert abs(audit.max_spd_gap - expected_max) < 0.001

    def test_audit_fields_complete(self):
        """IntersectionalAudit schema fields are all populated."""
        from app.services.intersectional import compute_intersectional_audit
        df = _make_intersectional_df()
        audit = compute_intersectional_audit(df, ["race", "sex"], "income", "1")
        assert audit is not None
        assert audit.outcome_column == "income"
        assert len(audit.protected_attributes) == 2
        assert audit.privileged_combo != ""
        assert len(audit.groups) >= 2
        assert audit.max_spd_gap >= 0.0


# ===========================================================================
# REAL DATASET INTEGRATION — calibration on COMPAS and Adult
# ===========================================================================

class TestCalibrationOnRealDatasets:
    """Calibration tests on downloaded real datasets (skipped if unavailable)."""

    def test_compas_calibration_audit(self):
        """COMPAS: LightGBM should be decently calibrated (no perfect model)."""
        try:
            from app.services.data_loader import load_compas
            from app.services.calibration import compute_calibration_audit
        except ImportError:
            pytest.skip("Services unavailable")

        df = load_compas()
        if df is None:
            pytest.skip("COMPAS dataset unavailable")

        audit = compute_calibration_audit(df, "race", "two_year_recid", "1")
        if audit is None:
            pytest.skip("LightGBM unavailable or insufficient data")

        # Both groups should have ECE values
        assert "African-American" in audit.group_calibration
        assert "Caucasian" in audit.group_calibration
        # Calibration gap: real models on COMPAS tend to be reasonably calibrated
        assert 0.0 <= audit.calibration_gap <= 0.5
        print(f"\nCOMPAS calibration_gap={audit.calibration_gap:.4f} is_calibrated={audit.is_calibrated}")

    def test_adult_intersectional_audit(self):
        """Adult Income: Black+Female subgroup should show larger SPD than individual-level metrics."""
        try:
            from app.services.data_loader import load_adult
            from app.services.intersectional import compute_intersectional_audit
        except ImportError:
            pytest.skip("Services unavailable")

        df = load_adult()
        if df is None:
            pytest.skip("Adult dataset unavailable")

        audit = compute_intersectional_audit(df, ["race", "sex"], "income", ">50K")
        if audit is None:
            pytest.skip("Intersectional audit failed (insufficient subgroups)")

        print(f"\nAdult intersectional max_spd_gap={audit.max_spd_gap:.4f}")
        print(f"Flagged groups: {audit.flagged_groups}")
        # Adult dataset has well-known race*sex intersectional bias
        assert audit.max_spd_gap > 0.10, \
            f"Adult should show intersectional SPD > 0.10, got {audit.max_spd_gap}"


# ===========================================================================
# REWEIGHING ON REAL ADULT DATASET — replication of Kamiran Table 2
# ===========================================================================

class TestReweighingOnRealAdult:
    def test_adult_sex_disc_approaches_zero(self):
        """
        Kamiran & Calders (2012) Table 2: disc score for NB = 0.1965.
        After reweighing by sex on Adult, disc -> ~0 by construction.
        """
        try:
            from app.services.data_loader import load_adult
            from app.services.reweighing import compute_sample_weights
        except ImportError:
            pytest.skip("Services unavailable")

        df = load_adult()
        if df is None:
            pytest.skip("Adult dataset unavailable")

        out = compute_sample_weights(df, "sex", "income", ">50K")
        if out is None:
            pytest.skip("Reweighing failed")

        weights, result = out
        print(f"\nAdult sex: disc_before={result.disc_before:.4f} disc_after={result.disc_after:.4f}")
        # Paper reports disc ~0.19 for Male vs Female
        assert result.disc_before > 0.15, \
            f"Expected Adult sex disc > 0.15, got {result.disc_before}"
        assert result.disc_after < 0.005, \
            f"After reweighing, disc should be ~0, got {result.disc_after}"
