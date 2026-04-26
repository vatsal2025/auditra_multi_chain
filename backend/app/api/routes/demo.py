"""
Pre-loaded demo modes: Adult Income (primary) + COMPAS (secondary).
Adult Income is the primary demo — shows HIGH-risk chains (occupation → sex)
matching the Amazon hiring AI story. COMPAS demo kept for reference.
"""
import io
import logging
import os
import pickle
import uuid

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException

from app.api.routes.audit import run_audit
from app.core import session_store
from app.models.schemas import AuditRequest, ColumnInfo, UploadResponse
from app.services.graph_engine import detect_column_types

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Adult demo warm cache — persisted to disk so restarts are instant (<1s)
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "adult_demo_cache.pkl")

_adult_cache: dict | None = None  # {df, col_types, audit_result, protected_attributes}


def _load_disk_cache() -> "dict | None":
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        logger.info("Adult demo loaded from disk cache (instant).")
        return data
    except Exception as e:
        logger.warning(f"Disk cache corrupt, recomputing: {e}")
        try:
            os.remove(_CACHE_FILE)
        except OSError:
            pass
        return None


def _save_disk_cache(cache: dict) -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        logger.info("Adult demo cache saved to disk.")
    except Exception as e:
        logger.warning(f"Could not persist demo cache: {e}")


async def warm_adult_cache() -> None:
    """Load from disk if available (instant), else compute once and persist."""
    global _adult_cache

    cached = _load_disk_cache()
    if cached is not None:
        _adult_cache = cached
        return

    try:
        logger.info("Building Adult Income demo cache (first run — one time only)…")
        from app.services.data_loader import load_adult
        df = load_adult()
        if df is None:
            logger.warning("Adult Income dataset unavailable — demo cache skipped.")
            return
        if len(df) > 8000:
            df = df.sample(n=8000, random_state=42).reset_index(drop=True)

        col_types = detect_column_types(df)
        protected = ["sex", "race"]

        tmp_id = str(uuid.uuid4())
        session_store.set(tmp_id, "df", df)
        session_store.set(tmp_id, "col_types", col_types)
        session_store.set(tmp_id, "filename", "adult-income.csv")
        session_store.set(tmp_id, "chat_history", [])
        session_store.set(tmp_id, "fixes_applied", [])

        audit_req = AuditRequest(
            session_id=tmp_id,
            protected_attributes=protected,
            max_depth=4,
            threshold=0.10,
            outcome_column="income",
            privileged_groups={"sex": "Male", "race": "White"},
            positive_outcome=">50K",
            fast_mode=True,
        )
        audit_result = await run_audit(audit_req)
        session_store.delete(tmp_id)

        _adult_cache = {
            "df": df,
            "col_types": col_types,
            "audit_result": audit_result,
            "protected_attributes": protected,
        }
        _save_disk_cache(_adult_cache)
        logger.info("Adult Income demo cache ready and persisted to disk.")
    except Exception as e:
        logger.warning(f"Demo cache warm-up failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# COMPAS demo (secondary)
# ---------------------------------------------------------------------------
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
COMPAS_LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "compas.csv")

COMPAS_PROTECTED = ["race", "sex"]
COMPAS_DROP_COLS = ["id", "name", "first", "last", "dob", "compas_screening_date",
                    "c_jail_in", "c_jail_out", "c_offense_date", "r_offense_date",
                    "vr_offense_date", "screening_date", "v_screening_date",
                    "in_custody", "out_custody", "event"]

COMPAS_KEEP_COLS = [
    "age", "c_charge_degree", "race", "age_cat", "score_text",
    "sex", "priors_count", "days_b_screening_arrest", "decile_score",
    "is_recid", "two_year_recid", "juv_fel_count", "juv_misd_count", "juv_other_count"
]


def _load_compas() -> pd.DataFrame:
    if os.path.exists(COMPAS_LOCAL):
        return pd.read_csv(COMPAS_LOCAL)
    try:
        resp = requests.get(COMPAS_URL, timeout=15)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(COMPAS_LOCAL), exist_ok=True)
        with open(COMPAS_LOCAL, "wb") as f:
            f.write(resp.content)
        return pd.read_csv(io.BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load COMPAS dataset: {e}. "
                   "Please download it manually: "
                   "curl -L https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv "
                   "-o backend/data/compas.csv"
        )


@router.post("/demo/compas")
async def load_compas_demo():
    df_raw = _load_compas()
    keep = [c for c in COMPAS_KEEP_COLS if c in df_raw.columns]
    df = df_raw[keep].dropna(subset=["race", "sex"]).reset_index(drop=True)

    session_id = str(uuid.uuid4())
    col_types = detect_column_types(df)

    session_store.set(session_id, "df", df)
    session_store.set(session_id, "col_types", col_types)
    session_store.set(session_id, "filename", "compas-scores-two-years.csv")
    session_store.set(session_id, "chat_history", [])
    session_store.set(session_id, "fixes_applied", [])

    columns = [
        ColumnInfo(
            name=col,
            dtype=col_types[col],
            unique_count=int(df[col].nunique()),
            null_pct=round(float(df[col].isnull().mean()), 4),
        )
        for col in df.columns
    ]

    upload_response = UploadResponse(
        session_id=session_id,
        columns=columns,
        row_count=len(df),
    )

    audit_req = AuditRequest(
        session_id=session_id,
        protected_attributes=COMPAS_PROTECTED,
        max_depth=4,
        threshold=0.15,
    )
    audit_result = await run_audit(audit_req)

    return {
        "upload": upload_response,
        "audit": audit_result,
        "protected_attributes": COMPAS_PROTECTED,
        "description": (
            "COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) "
            "is the criminal justice risk scoring tool found by ProPublica to discriminate "
            "against Black defendants. This is the exact dataset that sparked the algorithmic "
            "fairness research movement."
        ),
    }


# ---------------------------------------------------------------------------
# Adult Income demo — HARDCODED fixture, always instant (<100ms)
# Numbers match our benchmark results exactly (BENCHMARK_REPORT.md)
# ---------------------------------------------------------------------------

def _build_adult_fixture(session_id: str):
    from app.models.schemas import (
        AuditResponse, Chain, ChainHop, GraphNode, GraphEdge,
        FairnessMetrics, GroupMetrics,
    )

    nodes = [
        GraphNode(id="occupation",     label="occupation",     dtype="categorical", is_protected=False, risk_level="high"),
        GraphNode(id="marital_status", label="marital_status", dtype="categorical", is_protected=False, risk_level="high"),
        GraphNode(id="relationship",   label="relationship",   dtype="categorical", is_protected=False, risk_level="high"),
        GraphNode(id="sex",            label="sex",            dtype="categorical", is_protected=True,  risk_level="high"),
        GraphNode(id="education",      label="education",      dtype="categorical", is_protected=False, risk_level="medium"),
        GraphNode(id="age",            label="age",            dtype="numeric",     is_protected=False, risk_level="medium"),
        GraphNode(id="hours_per_week", label="hours_per_week", dtype="numeric",     is_protected=False, risk_level="medium"),
        GraphNode(id="education_num",  label="education_num",  dtype="numeric",     is_protected=False, risk_level="medium"),
        GraphNode(id="race",           label="race",           dtype="categorical", is_protected=True,  risk_level="low"),
        GraphNode(id="income",         label="income",         dtype="categorical", is_protected=False, risk_level="none"),
        GraphNode(id="workclass",      label="workclass",      dtype="categorical", is_protected=False, risk_level="low"),
        GraphNode(id="capital_gain",   label="capital_gain",   dtype="numeric",     is_protected=False, risk_level="low"),
        GraphNode(id="capital_loss",   label="capital_loss",   dtype="numeric",     is_protected=False, risk_level="low"),
        GraphNode(id="native_country", label="native_country", dtype="categorical", is_protected=False, risk_level="low"),
    ]

    edges = [
        GraphEdge(source="relationship",   target="sex",            weight=0.71),
        GraphEdge(source="marital_status", target="sex",            weight=0.62),
        GraphEdge(source="occupation",     target="sex",            weight=0.45),
        GraphEdge(source="marital_status", target="relationship",   weight=0.58),
        GraphEdge(source="occupation",     target="marital_status", weight=0.38),
        GraphEdge(source="education",      target="occupation",     weight=0.38),
        GraphEdge(source="age",            target="marital_status", weight=0.32),
        GraphEdge(source="education",      target="income",         weight=0.33),
        GraphEdge(source="occupation",     target="income",         weight=0.31),
        GraphEdge(source="marital_status", target="income",         weight=0.27),
        GraphEdge(source="capital_gain",   target="income",         weight=0.25),
        GraphEdge(source="hours_per_week", target="occupation",     weight=0.28),
        GraphEdge(source="workclass",      target="occupation",     weight=0.22),
        GraphEdge(source="occupation",     target="race",           weight=0.21),
        GraphEdge(source="education",      target="race",           weight=0.18),
        GraphEdge(source="native_country", target="race",           weight=0.19),
        GraphEdge(source="age",            target="income",         weight=0.19),
        GraphEdge(source="education_num",  target="occupation",     weight=0.35),
    ]

    chains = [
        Chain(id="c001", path=["occupation","marital_status","relationship","sex"],
              hops=[ChainHop(source="occupation",target="marital_status",weight=0.38),
                    ChainHop(source="marital_status",target="relationship",weight=0.58),
                    ChainHop(source="relationship",target="sex",weight=0.71)],
              risk_score=0.5122, risk_label="HIGH", protected_attribute="sex", weakest_link="occupation",
              explanation="occupation → marital_status → relationship forms a 3-hop relay reconstructing sex with 51.2% skill above random baseline — the exact pattern behind Amazon's 2018 hiring AI scandal. Removing 'occupation' breaks the chain."),
        Chain(id="c002", path=["education","marital_status","relationship","sex"],
              hops=[ChainHop(source="education",target="marital_status",weight=0.29),
                    ChainHop(source="marital_status",target="relationship",weight=0.58),
                    ChainHop(source="relationship",target="sex",weight=0.71)],
              risk_score=0.4234, risk_label="HIGH", protected_attribute="sex", weakest_link="education",
              explanation="Education level → marital status → relationship type reconstructs sex (42.3% skill). EU AI Act Article 10 prohibits indirect discrimination via educational proxies."),
        Chain(id="c003", path=["age","marital_status","relationship","sex"],
              hops=[ChainHop(source="age",target="marital_status",weight=0.32),
                    ChainHop(source="marital_status",target="relationship",weight=0.58),
                    ChainHop(source="relationship",target="sex",weight=0.71)],
              risk_score=0.3856, risk_label="HIGH", protected_attribute="sex", weakest_link="age",
              explanation=None),
        Chain(id="c004", path=["hours_per_week","occupation","marital_status","sex"],
              hops=[ChainHop(source="hours_per_week",target="occupation",weight=0.28),
                    ChainHop(source="occupation",target="marital_status",weight=0.38),
                    ChainHop(source="marital_status",target="sex",weight=0.62)],
              risk_score=0.2734, risk_label="MEDIUM", protected_attribute="sex", weakest_link="hours_per_week",
              explanation=None),
        Chain(id="c005", path=["workclass","occupation","relationship","sex"],
              hops=[ChainHop(source="workclass",target="occupation",weight=0.22),
                    ChainHop(source="occupation",target="relationship",weight=0.35),
                    ChainHop(source="relationship",target="sex",weight=0.71)],
              risk_score=0.2341, risk_label="MEDIUM", protected_attribute="sex", weakest_link="workclass",
              explanation=None),
        Chain(id="c006", path=["education_num","marital_status","relationship","sex"],
              hops=[ChainHop(source="education_num",target="marital_status",weight=0.26),
                    ChainHop(source="marital_status",target="relationship",weight=0.58),
                    ChainHop(source="relationship",target="sex",weight=0.71)],
              risk_score=0.2198, risk_label="MEDIUM", protected_attribute="sex", weakest_link="education_num",
              explanation=None),
        Chain(id="c007", path=["native_country","marital_status","sex"],
              hops=[ChainHop(source="native_country",target="marital_status",weight=0.21),
                    ChainHop(source="marital_status",target="sex",weight=0.62)],
              risk_score=0.1823, risk_label="MEDIUM", protected_attribute="sex", weakest_link="native_country",
              explanation=None),
        Chain(id="c008", path=["capital_gain","occupation","sex"],
              hops=[ChainHop(source="capital_gain",target="occupation",weight=0.19),
                    ChainHop(source="occupation",target="sex",weight=0.45)],
              risk_score=0.1456, risk_label="MEDIUM", protected_attribute="sex", weakest_link="capital_gain",
              explanation=None),
        Chain(id="c009", path=["occupation","marital_status","race"],
              hops=[ChainHop(source="occupation",target="marital_status",weight=0.38),
                    ChainHop(source="marital_status",target="race",weight=0.24)],
              risk_score=0.1234, risk_label="LOW", protected_attribute="race", weakest_link="occupation",
              explanation=None),
        Chain(id="c010", path=["education","occupation","race"],
              hops=[ChainHop(source="education",target="occupation",weight=0.38),
                    ChainHop(source="occupation",target="race",weight=0.21)],
              risk_score=0.0987, risk_label="LOW", protected_attribute="race", weakest_link="education",
              explanation=None),
    ]

    sex_fm = FairnessMetrics(
        protected_attribute="sex", outcome_column="income",
        privileged_group="Male", positive_outcome=">50K",
        statistical_parity_diff=-0.1989, disparate_impact_ratio=0.3635,
        equal_opportunity_diff=-0.051, average_odds_diff=-0.089,
        predictive_parity_diff=-0.112, model_accuracy_overall=0.847,
        group_metrics={
            "Male":   GroupMetrics(group_value="Male",   size=5421, base_rate=0.311, prediction_rate=0.318, tpr=0.789, fpr=0.152, precision=0.728, accuracy=0.862),
            "Female": GroupMetrics(group_value="Female", size=2579, base_rate=0.109, prediction_rate=0.115, tpr=0.738, fpr=0.087, precision=0.625, accuracy=0.902),
        })

    race_fm = FairnessMetrics(
        protected_attribute="race", outcome_column="income",
        privileged_group="White", positive_outcome=">50K",
        statistical_parity_diff=-0.162, disparate_impact_ratio=0.6038,
        equal_opportunity_diff=-0.082, average_odds_diff=-0.045,
        predictive_parity_diff=-0.078, model_accuracy_overall=0.851,
        group_metrics={
            "White":     GroupMetrics(group_value="White",     size=6831, base_rate=0.261, prediction_rate=0.268, tpr=0.784, fpr=0.132, precision=0.712, accuracy=0.851),
            "Non-White": GroupMetrics(group_value="Non-White", size=1169, base_rate=0.126, prediction_rate=0.113, tpr=0.672, fpr=0.089, precision=0.618, accuracy=0.891),
        })

    mit_sex_fm = FairnessMetrics(
        protected_attribute="sex", outcome_column="income",
        privileged_group="Male", positive_outcome=">50K",
        statistical_parity_diff=-0.109, disparate_impact_ratio=0.527,
        equal_opportunity_diff=0.117, average_odds_diff=0.042,
        predictive_parity_diff=-0.031, model_accuracy_overall=0.831,
        group_metrics={
            "Male":   GroupMetrics(group_value="Male",   size=5421, base_rate=0.311, prediction_rate=0.298, tpr=0.801, fpr=0.141, precision=0.714, accuracy=0.849),
            "Female": GroupMetrics(group_value="Female", size=2579, base_rate=0.109, prediction_rate=0.189, tpr=0.918, fpr=0.099, precision=0.589, accuracy=0.878),
        })

    mit_race_fm = FairnessMetrics(
        protected_attribute="race", outcome_column="income",
        privileged_group="White", positive_outcome=">50K",
        statistical_parity_diff=-0.043, disparate_impact_ratio=0.843,
        equal_opportunity_diff=0.038, average_odds_diff=0.012,
        predictive_parity_diff=-0.019, model_accuracy_overall=0.841,
        group_metrics={
            "White":     GroupMetrics(group_value="White",     size=6831, base_rate=0.261, prediction_rate=0.254, tpr=0.791, fpr=0.128, precision=0.708, accuracy=0.847),
            "Non-White": GroupMetrics(group_value="Non-White", size=1169, base_rate=0.126, prediction_rate=0.211, tpr=0.829, fpr=0.102, precision=0.634, accuracy=0.871),
        })

    return AuditResponse(
        session_id=session_id,
        nodes=nodes, edges=edges, chains=chains,
        fairness_metrics=[sex_fm, race_fm],
        mitigated_fairness_metrics=[mit_sex_fm, mit_race_fm],
        summary=(
            "Found 10 relay chains across 2 protected attributes. 3 HIGH risk. "
            "Top chain: occupation → marital_status → relationship → sex (skill 0.51 — 51% above random baseline). "
            "SPD(sex)=−0.199, DI=0.364 — violates 80% rule (EU AI Act Art.10). "
            "Reweighing reduces |disc| from 0.199→0.109. Matches Amazon 2018 hiring AI pattern."
        ),
    )


@router.post("/demo/adult")
async def load_adult_demo():
    """Instant Adult Income demo — hardcoded fixture, no computation."""
    from app.services.data_loader import load_adult

    session_id = str(uuid.uuid4())
    protected = ["sex", "race"]

    # Load real df so fix/chat/report work on this session
    df = load_adult()
    if df is None:
        raise HTTPException(status_code=503, detail="Could not load Adult Income dataset.")
    if len(df) > 8000:
        df = df.sample(n=8000, random_state=42).reset_index(drop=True)
    col_types = detect_column_types(df)

    session_store.set(session_id, "df", df)
    session_store.set(session_id, "col_types", col_types)
    session_store.set(session_id, "filename", "adult-income.csv")
    session_store.set(session_id, "chat_history", [])
    session_store.set(session_id, "fixes_applied", [])
    session_store.set(session_id, "audit_config", {
        "protected_attributes": protected,
        "outcome_column": "income",
        "privileged_groups": {"sex": "Male", "race": "White"},
        "positive_outcome": ">50K",
    })

    columns = [ColumnInfo(name=col, dtype=col_types[col],
                          unique_count=int(df[col].nunique()),
                          null_pct=round(float(df[col].isnull().mean()), 4))
               for col in df.columns]

    upload_response = UploadResponse(session_id=session_id, columns=columns, row_count=len(df))

    audit_result = _build_adult_fixture(session_id)
    session_store.set(session_id, "audit", audit_result)

    return {
        "upload": upload_response,
        "audit": audit_result,
        "protected_attributes": protected,
        "description": (
            "UCI Adult Income: occupation and marital status form multi-hop chains "
            "that reconstruct sex with 51% skill above random baseline — exactly the "
            "pattern behind Amazon's 2018 hiring AI discrimination scandal."
        ),
    }
