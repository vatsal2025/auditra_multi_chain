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
# Adult Income demo (primary — instant from cache after first run)
# ---------------------------------------------------------------------------

@router.post("/demo/adult")
async def load_adult_demo():
    """
    UCI Adult Income dataset. Returns instantly from disk cache after first run.
    Shows HIGH-risk relay chains: occupation → sex (skill 0.51).
    """
    session_id = str(uuid.uuid4())
    protected = ["sex", "race"]

    if _adult_cache is not None:
        df: pd.DataFrame = _adult_cache["df"].copy()
        col_types = _adult_cache["col_types"]
        audit_result = _adult_cache["audit_result"].model_copy(
            update={"session_id": session_id}
        )
    else:
        # Cold path (cache not ready yet — rare, only within first ~60s of first-ever boot)
        from app.services.data_loader import load_adult
        df = load_adult()
        if df is None:
            raise HTTPException(
                status_code=503,
                detail="Could not load Adult Income dataset from UCI repository."
            )
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

    if _adult_cache is None:
        audit_req = AuditRequest(
            session_id=session_id,
            protected_attributes=protected,
            max_depth=4,
            threshold=0.10,
            outcome_column="income",
            privileged_groups={"sex": "Male", "race": "White"},
            positive_outcome=">50K",
            fast_mode=True,
        )
        audit_result = await run_audit(audit_req)

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
