"""
Vertex AI integration — chain risk scoring + Explainable AI SHAP.

Auth: Application Default Credentials (ADC) — automatic on GCP VMs.

4 AutoML endpoints, one per dataset:
  - auditra-chain-scorer-compas       → predicts `race`
  - auditra-chain-scorer-adult-train  → predicts `sex`
  - auditra-chain-scorer-adult-test   → predicts `sex`
  - auditra-chain-scorer-german       → predicts `sex`

Each model trained on all dataset features. At chain scoring time, only
chain-path features are sent; Vertex AI AutoML treats missing cols as null.
"""
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.schemas import Chain, ShapEntry


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
        "adult_test":  settings.vertex_ai_endpoint_adult_test,
        "german":      settings.vertex_ai_endpoint_german,
    }
    return mapping.get(dataset) or settings.vertex_ai_endpoint_id


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

    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col   = chain.protected_attribute

    if target_col not in df.columns or not feature_cols:
        return None

    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return None

    subset = df[available + [target_col]].dropna(subset=[target_col]).head(200)
    if len(subset) < 10:
        return None

    try:
        _init_vertex()
        from google.cloud import aiplatform

        endpoint  = aiplatform.Endpoint(endpoint_id)
        instances = subset[available].astype(str).to_dict(orient="records")
        response  = endpoint.predict(instances=instances)

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
        return _skill_score(accuracy, actual)

    except Exception as e:
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

    subset = df[available + [target_col]].dropna(subset=[target_col]).head(50)
    if len(subset) < 5:
        return None

    try:
        _init_vertex()
        from google.cloud import aiplatform

        endpoint  = aiplatform.Endpoint(endpoint_id)
        instances = subset[available].astype(str).to_dict(orient="records")
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
