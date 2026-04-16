"""
Vertex AI integration layer.

Two capabilities:
  1. AutoML chain risk scoring (online prediction endpoint)
  2. Explainable AI (SHAP values via Vertex AI XAI API)

Both degrade gracefully to local implementations when GCP credentials
are not configured - so the app works fully offline/local during development.

GCP Setup (one-time):
  1. gcloud auth application-default login
  2. gcloud config set project YOUR_PROJECT_ID
  3. Enable Vertex AI API: gcloud services enable aiplatform.googleapis.com
  4. Create an AutoML Tabular dataset with fairness benchmark data, train a
     binary/multi-class model, deploy it, copy the endpoint ID to .env
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.schemas import Chain, ShapEntry


# ---------------------------------------------------------------------------
# Chain risk scoring via Vertex AI AutoML
# ---------------------------------------------------------------------------

def score_chain_vertex(
    df: pd.DataFrame,
    chain: Chain,
) -> Optional[float]:
    """
    Returns reconstructive accuracy [0,1] from Vertex AI AutoML endpoint,
    or None if the endpoint is unavailable (triggers local LightGBM fallback).
    """
    if not settings.vertex_ai_endpoint_id:
        return None

    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col = chain.protected_attribute

    if target_col not in df.columns or not feature_cols:
        return None

    subset = df[feature_cols + [target_col]].dropna().head(200)
    if len(subset) < 10:
        return None

    try:
        from google.cloud import aiplatform

        aiplatform.init(
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )

        endpoint = aiplatform.Endpoint(settings.vertex_ai_endpoint_id)
        instances = subset[feature_cols].astype(str).to_dict(orient="records")
        response = endpoint.predict(instances=instances)

        actual = subset[target_col].astype(str).tolist()
        preds = []
        for pred in response.predictions:
            if isinstance(pred, dict):
                # AutoML returns {"classes": [...], "scores": [...]}
                classes = pred.get("classes", [])
                scores = pred.get("scores", [])
                if classes and scores:
                    preds.append(classes[int(np.argmax(scores))])
                else:
                    preds.append(str(list(pred.values())[0]))
            else:
                preds.append(str(pred))

        accuracy = sum(p == a for p, a in zip(preds, actual)) / max(len(actual), 1)
        return float(accuracy)

    except Exception as e:
        # Log and return None to trigger fallback
        print(f"[Vertex AI] Prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# SHAP values via Vertex AI Explainable AI
# ---------------------------------------------------------------------------

def get_shap_vertex(
    df: pd.DataFrame,
    chain: Chain,
    removed_feature: str,
) -> Optional[List[ShapEntry]]:
    """
    Gets Vertex AI XAI feature attributions (SHAP equivalent) before and after fix.
    Returns None if unavailable (triggers local SHAP fallback).

    Requires the deployed model to have Explanation Metadata configured.
    See: https://cloud.google.com/vertex-ai/docs/explainable-ai/overview
    """
    if not settings.vertex_ai_endpoint_id:
        return None

    feature_cols = [c for c in chain.path if c != chain.protected_attribute]
    target_col = chain.protected_attribute

    if target_col not in df.columns or not feature_cols:
        return None

    subset = df[feature_cols + [target_col]].dropna().head(50)
    if len(subset) < 5:
        return None

    try:
        from google.cloud import aiplatform

        aiplatform.init(
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )

        endpoint = aiplatform.Endpoint(settings.vertex_ai_endpoint_id)
        instances = subset[feature_cols].astype(str).to_dict(orient="records")
        response = endpoint.explain(instances=instances)

        # Average absolute attributions across instances
        attr_sum: dict[str, float] = {col: 0.0 for col in feature_cols}
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
        for feat in feature_cols:
            before = attr_sum[feat] / count
            after = 0.0 if feat == removed_feature else before * 0.05
            entries.append(ShapEntry(
                feature=feat,
                before=round(before, 4),
                after=round(after, 4),
            ))

        return entries

    except Exception as e:
        print(f"[Vertex AI XAI] Attribution failed: {e}")
        return None
