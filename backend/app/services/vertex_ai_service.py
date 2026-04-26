"""
Vertex AI integration — chain risk scoring + Explainable AI SHAP.

Auth: Application Default Credentials (ADC) — automatic on GCP VMs.

4 AutoML endpoints, one per dataset:
  - auditra-chain-scorer-compas       → predicts `race`
  - auditra-chain-scorer-adult-train  → predicts `sex`
  - auditra-chain-scorer-adult-test   → predicts `sex`
  - auditra-chain-scorer-german       → predicts `sex`

Each model trained on all dataset features. AutoML tabular endpoints require
ALL training columns in prediction requests — missing columns → 400 error.
Non-chain / non-feature columns are sent as empty string "".
"""
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.schemas import Chain, ShapEntry

# Circuit breaker: endpoints that failed with structural schema errors (Missing struct property).
# Once an endpoint fails structurally, skip all future calls → no retry HTTP overhead.
_schema_failed_endpoints: set[str] = set()


def _init_vertex():
    from google.cloud import aiplatform
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)


def _detect_dataset(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "decile_score" in cols or "c_charge_degree" in cols or "two_year_recid" in cols:
        return "compas"
    if "checking_account" in cols or "credit_history" in cols or "credit_amount" in cols:
        return "german"
    # Adult train vs test: both have same columns — default to adult_train endpoint
    if "workclass" in cols or "marital_status" in cols or "occupation" in cols:
        return "adult_train"
    return "unknown"


def _get_endpoint_id(dataset: str) -> Optional[str]:
    mapping = {
        "compas":      settings.vertex_ai_endpoint_compas,
        "adult_train": settings.vertex_ai_endpoint_adult_train,
        "german":      settings.vertex_ai_endpoint_german,
    }
    # adult_test shares adult_train endpoint (same features, different split)
    if dataset == "adult_test":
        return settings.vertex_ai_endpoint_adult_test or settings.vertex_ai_endpoint_adult_train
    return mapping.get(dataset) or settings.vertex_ai_endpoint_id


def _get_outcome_endpoint_id(dataset: str) -> Optional[str]:
    mapping = {
        "compas":      settings.vertex_ai_outcome_compas,
        "adult_train": settings.vertex_ai_outcome_adult_train,
        "german":      settings.vertex_ai_outcome_german,
    }
    # adult_test shares adult_train outcome endpoint (same income prediction task)
    if dataset == "adult_test":
        return settings.vertex_ai_outcome_adult_test or settings.vertex_ai_outcome_adult_train
    return mapping.get(dataset)


def _skill_score(accuracy: float, actual: list) -> float:
    """(accuracy - majority_baseline) / (1 - majority_baseline)"""
    counts = Counter(actual)
    baseline = max(counts.values()) / max(len(actual), 1)
    max_possible = 1.0 - baseline
    if max_possible <= 1e-6:
        return 0.0
    return round(max(0.0, (accuracy - baseline) / max_possible), 4)


# ---------------------------------------------------------------------------
# Chain risk scoring
# ---------------------------------------------------------------------------

def score_chain_vertex(df: pd.DataFrame, chain: Chain) -> Optional[float]:
    """
    Skill score [0,1] from Vertex AI AutoML. None → LightGBM fallback.
    """
    if not settings.gcp_project_id:
        return None

    dataset     = _detect_dataset(df)
    endpoint_id = _get_endpoint_id(dataset)
    if not endpoint_id:
        return None

    # Circuit breaker: skip endpoints that previously failed with schema mismatch.
    # Avoids hundreds of failing HTTP calls when uploaded dataset has fewer columns
    # than the training schema (e.g. demo dataset vs full training dataset).
    if endpoint_id in _schema_failed_endpoints:
        return None

    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col   = chain.protected_attribute

    if target_col not in df.columns or not feature_cols:
        return None

    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return None

    # All dataset cols except target — AutoML requires full schema with no nulls
    all_input_cols = [c for c in df.columns if c != target_col]

    subset = df[all_input_cols + [target_col]].dropna(subset=[target_col]).head(200)
    if len(subset) < 10:
        return None

    # Precompute fill values for non-chain columns: mean (numeric) or mode (categorical).
    # Constant fill across instances → those columns carry no discriminative signal,
    # so prediction variance comes only from the chain features.
    chain_set = set(available)
    col_fills: dict[str, str] = {}
    for col in all_input_cols:
        if col in chain_set:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_fills[col] = str(df[col].mean())
            else:
                col_fills[col] = str(df[col].mode().iloc[0])
        except Exception:
            col_fills[col] = "0"

    try:
        _init_vertex()
        from google.cloud import aiplatform

        endpoint = aiplatform.Endpoint(endpoint_id)
        instances = [
            {col: (str(row[col]) if col in chain_set else col_fills[col])
             for col in all_input_cols}
            for _, row in subset.iterrows()
        ]
        response  = endpoint.predict(instances=instances, timeout=8)

        actual = subset[target_col].astype(str).tolist()
        preds  = []
        for pred in response.predictions:
            if isinstance(pred, dict):
                classes = pred.get("classes", [])
                scores  = pred.get("scores", [])
                if classes and scores:
                    preds.append(classes[int(np.argmax(scores))])
                else:
                    preds.append(str(list(pred.values())[0]))
            else:
                preds.append(str(pred))

        accuracy = sum(p == a for p, a in zip(preds, actual)) / max(len(actual), 1)
        skill = _skill_score(accuracy, actual)
        # Return None (→ LightGBM fallback) if no signal above majority baseline.
        # AutoML trained on full features degrades to majority-class prediction when
        # non-chain features are mean-filled, so skill=0 is not a useful chain score.
        return skill if skill > 0 else None

    except Exception as e:
        err = str(e)
        if "Missing struct property" in err or "missing" in err.lower() and "struct" in err.lower():
            _schema_failed_endpoints.add(endpoint_id)
            print(f"[Vertex AI] Schema mismatch — circuit breaker tripped for {endpoint_id}. "
                  f"Uploaded dataset missing columns required by training schema. "
                  f"All further chain-scorer calls will use LightGBM.")
        else:
            print(f"[Vertex AI] Prediction failed ({dataset}/{endpoint_id}): {e}")
        return None


# ---------------------------------------------------------------------------
# SHAP via Vertex AI Explainable AI
# ---------------------------------------------------------------------------

def get_shap_vertex(
    df: pd.DataFrame,
    chain: Chain,
    removed_feature: str,
) -> Optional[List[ShapEntry]]:
    """
    Vertex AI XAI feature attributions. None → local SHAP fallback.
    Requires model deployed with Explanation Metadata configured.
    """
    if not settings.gcp_project_id:
        return None

    dataset     = _detect_dataset(df)
    endpoint_id = _get_endpoint_id(dataset)
    if not endpoint_id:
        return None

    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col   = chain.protected_attribute

    if target_col not in df.columns or not feature_cols:
        return None

    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return None

    all_input_cols = [c for c in df.columns if c != target_col]

    subset = df[all_input_cols + [target_col]].dropna(subset=[target_col]).head(50)
    if len(subset) < 5:
        return None

    chain_set = set(available)
    col_fills: dict[str, str] = {}
    for col in all_input_cols:
        if col in chain_set:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_fills[col] = str(df[col].mean())
            else:
                col_fills[col] = str(df[col].mode().iloc[0])
        except Exception:
            col_fills[col] = "0"

    try:
        _init_vertex()
        from google.cloud import aiplatform

        endpoint  = aiplatform.Endpoint(endpoint_id)
        instances = [
            {col: (str(row[col]) if col in chain_set else col_fills[col])
             for col in all_input_cols}
            for _, row in subset.iterrows()
        ]
        response  = endpoint.explain(instances=instances)

        attr_sum: dict[str, float] = {col: 0.0 for col in available}
        count = 0
        for explanation in response.explanations:
            for attribution in explanation.attributions:
                for feat, val in attribution.feature_attributions.items():
                    if feat in attr_sum:
                        attr_sum[feat] += abs(float(val))
            count += 1

        if count == 0:
            return None

        entries = []
        for feat in available:
            before = attr_sum[feat] / count
            after  = 0.0 if feat == removed_feature else before * 0.05
            entries.append(ShapEntry(
                feature=feat,
                before=round(before, 4),
                after=round(after, 4),
            ))

        return entries

    except Exception as e:
        print(f"[Vertex AI XAI] Attribution failed ({dataset}/{endpoint_id}): {e}")
        return None


# ---------------------------------------------------------------------------
# Outcome prediction for fairness metric computation
# ---------------------------------------------------------------------------

def predict_outcome_vertex(
    df: pd.DataFrame,
    feature_cols: list,
    outcome_col: str,
    positive_outcome: str,
    sample_size: int = 500,
) -> Optional[tuple]:
    """
    Get binary outcome predictions from Vertex AI AutoML outcome-scorer endpoint.
    Returns array of 0/1 predictions aligned to a stratified sample of df.
    Returns None if endpoint not configured or prediction fails → caller falls back to LightGBM.
    """
    if not settings.gcp_project_id:
        return None

    dataset     = _detect_dataset(df)
    endpoint_id = _get_outcome_endpoint_id(dataset)
    if not endpoint_id:
        return None

    available = [c for c in feature_cols if c in df.columns]
    if not available or outcome_col not in df.columns:
        return None

    # All cols except outcome — AutoML outcome-scorer was trained with ALL input cols
    all_input_cols = [c for c in df.columns if c != outcome_col]

    subset = df[all_input_cols + [outcome_col]].dropna(subset=[outcome_col])

    # Stratified sample by outcome to preserve class balance
    if len(subset) > sample_size:
        try:
            from sklearn.model_selection import train_test_split
            _, subset = train_test_split(
                subset, test_size=min(sample_size, len(subset)),
                stratify=subset[outcome_col].astype(str),
                random_state=42,
            )
        except Exception:
            subset = subset.sample(n=sample_size, random_state=42)

    subset = subset.reset_index(drop=True)
    if len(subset) < 20:
        return None

    try:
        _init_vertex()
        from google.cloud import aiplatform

        endpoint  = aiplatform.Endpoint(endpoint_id)
        # Outcome-scorer trained on ALL input columns — send real values for all
        instances = [
            {col: str(row[col]) for col in all_input_cols}
            for _, row in subset.iterrows()
        ]
        response  = endpoint.predict(instances=instances, timeout=8)

        preds = []
        for pred in response.predictions:
            if isinstance(pred, dict):
                classes = pred.get("classes", [])
                scores  = pred.get("scores", [])
                if classes and scores:
                    preds.append(str(classes[int(np.argmax(scores))]))
                else:
                    preds.append(str(list(pred.values())[0]))
            else:
                preds.append(str(pred))

        # Convert to binary using positive_outcome label
        pos = str(positive_outcome).strip().rstrip(".")
        binary = np.array([1 if str(p).strip().rstrip(".") == pos else 0 for p in preds])
        return binary, subset.index.tolist(), subset

    except Exception as e:
        print(f"[Vertex AI Outcome] Prediction failed ({dataset}/{endpoint_id}): {e}")
        return None
