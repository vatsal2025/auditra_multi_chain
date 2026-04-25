"""
Real dataset downloader and preprocessor for benchmark comparisons.

Datasets:
  - COMPAS: ProPublica broward county criminal justice scores
  - Adult Income: UCI ML Repository (Dua & Graff 2019)
  - German Credit: UCI ML Repository (Hofmann 1994)

All preprocessing matches the paper-standard approaches:
  - COMPAS: Angwin et al. (2016) / Friedler et al. (2019) filtering
  - Adult: Kamiran & Calders (2012) / Feldman et al. (2015) standard split
  - German: Friedler et al. (2019) encoding
"""
from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _download(url: str, local_path: str) -> Optional[bytes]:
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        _ensure_dir()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return resp.content
    except Exception as e:
        print(f"[data_loader] Download failed {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# COMPAS
# Paper filtering: Angwin et al. (2016), replicated in Friedler et al. (2019)
# ---------------------------------------------------------------------------

COMPAS_LOCAL = os.path.join(DATA_DIR, "compas.csv")

COMPAS_FEATURES = [
    "age", "c_charge_degree", "race", "age_cat", "score_text",
    "sex", "priors_count", "days_b_screening_arrest", "decile_score",
    "is_recid", "two_year_recid", "juv_fel_count", "juv_misd_count", "juv_other_count",
]

def load_compas() -> Optional[pd.DataFrame]:
    """
    Load COMPAS dataset with ProPublica-standard filtering.

    Filters applied (Angwin et al. 2016 methodology):
      - days_b_screening_arrest in [-30, 30]
      - is_recid != -1
      - c_charge_degree != 'O' (ordinary traffic violations)
      - score_text != 'N/A'
      - race in ['African-American', 'Caucasian'] (paper focuses on these)
    """
    raw = _download(COMPAS_URL, COMPAS_LOCAL)
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw))

    # Paper-standard ProPublica filters
    df = df[df["days_b_screening_arrest"] <= 30]
    df = df[df["days_b_screening_arrest"] >= -30]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]
    df = df[df["race"].isin(["African-American", "Caucasian"])]

    cols = [c for c in COMPAS_FEATURES if c in df.columns]
    df = df[cols].dropna(subset=["race", "sex", "two_year_recid"]).reset_index(drop=True)

    # Binary prediction variable: COMPAS score >= 5 = "High risk"
    df["high_risk_pred"] = (df["decile_score"] >= 5).astype(int)

    return df


# ---------------------------------------------------------------------------
# Adult Income
# Paper standard: Kamiran & Calders (2012), Feldman et al. (2015)
# ---------------------------------------------------------------------------

ADULT_LOCAL = os.path.join(DATA_DIR, "adult.csv")

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

def load_adult() -> Optional[pd.DataFrame]:
    """
    Load UCI Adult Income dataset (train + test combined).
    Standard preprocessing: Kamiran & Calders (2012).
      - Remove rows with missing values (coded as ' ?')
      - Binarize income: '>50K' = 1, '<=50K' = 0
      - Race: keep all (Feldman uses White vs Others)
      - Sex: Male (privileged) vs Female (unprivileged)
    """
    train_raw = _download(ADULT_TRAIN_URL, os.path.join(DATA_DIR, "adult_train.csv"))
    test_raw = _download(ADULT_TEST_URL, os.path.join(DATA_DIR, "adult_test.csv"))

    dfs = []
    if train_raw:
        df_train = pd.read_csv(io.BytesIO(train_raw), names=ADULT_COLS,
                               skipinitialspace=True, na_values="?")
        dfs.append(df_train)
    if test_raw:
        df_test = pd.read_csv(io.BytesIO(test_raw), names=ADULT_COLS,
                              skipinitialspace=True, na_values="?", skiprows=1)
        dfs.append(df_test)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True).dropna()
    # Normalize income labels (test set has trailing '.')
    df["income"] = df["income"].str.strip().str.rstrip(".").str.strip()
    # Remove fnlwgt (sample weight, not a feature)
    df = df.drop(columns=["fnlwgt"], errors="ignore")
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# German Credit
# Paper standard: Friedler et al. (2019), Kamiran & Calders (2012)
# ---------------------------------------------------------------------------

GERMAN_LOCAL = os.path.join(DATA_DIR, "german.csv")

# German numeric format column names
GERMAN_COLS = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment",
    "housing", "existing_credits", "job", "liable_people", "telephone",
    "foreign_worker", "credit_risk",
]

def load_german() -> Optional[pd.DataFrame]:
    """
    Load German Credit dataset.
    Sex encoding from personal_status_sex attribute (Friedler et al. 2019 standard):
      A91 = male divorced/separated
      A92 = female divorced/separated/married
      A93 = male single
      A94 = male married/widowed
      A95 = female single
    Credit risk: 1 = good credit, 2 = bad credit
    Standard for fairness: female = unprivileged, good credit = positive outcome.
    """
    raw = _download(GERMAN_URL, GERMAN_LOCAL)
    if raw is None:
        return None

    df = pd.read_csv(io.BytesIO(raw), sep=" ", header=None, names=GERMAN_COLS)

    # Extract sex from personal_status_sex
    female_codes = {"A92", "A95"}
    df["sex"] = df["personal_status_sex"].apply(
        lambda x: "female" if x in female_codes else "male"
    )
    df = df.drop(columns=["personal_status_sex"])

    # Binarize credit risk: 1 = good (positive), 2 = bad
    df["credit_risk_binary"] = (df["credit_risk"] == 1).astype(int)
    # Drop original credit_risk to prevent leakage (credit_risk_binary is derived from it)
    df = df.drop(columns=["credit_risk"])

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Dataset metadata for benchmark
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "compas": {
        "loader": load_compas,
        "protected": ["race", "sex"],
        "outcome": "two_year_recid",
        "positive_outcome": "0",     # 0 = not recidivate = positive (good outcome)
        "privileged": {"race": "Caucasian", "sex": "Male"},
        "prediction_col": "high_risk_pred",  # COMPAS score >= 5
        "paper_metrics": {
            # ProPublica (2016) Table
            "race": {
                "fpr_ratio": 1.91,    # Black FPR / White FPR
                "fpr_black": 0.449,
                "fpr_white": 0.235,
                "fnr_black": 0.280,
                "fnr_white": 0.477,
            },
            # Friedler et al. (2019) Table 1 — LR baseline on COMPAS
            "friedler_spd": -0.200,   # approx stat parity diff (Black vs White)
        },
    },
    "adult": {
        "loader": load_adult,
        "protected": ["sex", "race"],
        "outcome": "income",
        "positive_outcome": ">50K",
        "privileged": {"sex": "Male", "race": "White"},
        "paper_metrics": {
            # Kamiran & Calders (2012) Table 2 — NB baseline
            "sex": {
                "disc_score": 0.1965,   # P(>50K|Male) - P(>50K|Female)
                "di_ratio": 0.36,       # Feldman et al. (2015) Table 1
            },
            # Friedler et al. (2019) approximate
            "race": {
                "spd_approx": -0.12,
            },
        },
    },
    "german": {
        "loader": load_german,
        "protected": ["sex"],
        "outcome": "credit_risk_binary",
        "positive_outcome": "1",
        "privileged": {"sex": "male"},
        "paper_metrics": {
            # Friedler et al. (2019) — Logistic Regression baseline
            "sex": {
                "spd_approx": -0.09,    # female vs male stat parity diff
            },
        },
    },
}
