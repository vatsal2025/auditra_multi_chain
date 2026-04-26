from fastapi import APIRouter, HTTPException

from app.core import session_store
from app.models.schemas import AuditRequest, AuditResponse
from app.services import gemini_service
from app.services.chain_scorer import score_all_chains
from app.services.graph_engine import (
    build_graph,
    build_graph_schema,
    find_chains,
)

router = APIRouter()


@router.post("/audit", response_model=AuditResponse)
async def run_audit(req: AuditRequest):
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    df = session_store.get(req.session_id, "df")
    col_types = session_store.get(req.session_id, "col_types")

    invalid = [a for a in req.protected_attributes if a not in df.columns]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown columns: {invalid}")

    # Graph: protected attributes are sink nodes
    G, strengths = build_graph(df, col_types, req.threshold, req.protected_attributes)
    chains = find_chains(G, strengths, req.protected_attributes, req.max_depth, col_types)

    # Score: baseline-adjusted skill score via LightGBM / Vertex AI
    chains = score_all_chains(df, chains)

    # Gemini explanations: top 3 HIGH/CRITICAL only in fast mode, top 5 otherwise
    from app.services.gemini_service import _fallback_explanation
    explain_limit = 2 if req.fast_mode else 5
    for i, chain in enumerate(chains[:20]):
        if i < explain_limit and chain.risk_label in ("HIGH", "CRITICAL"):
            explanation = gemini_service.explain_chain(chain)
        else:
            explanation = _fallback_explanation(chain)
        chains[i] = chain.model_copy(update={"explanation": explanation})

    nodes, edges = build_graph_schema(G, chains, req.protected_attributes, col_types)

    # --- Standard fairness metrics (when outcome column provided) ---
    fairness_metrics = []
    mitigated_fairness_metrics = []
    if req.outcome_column and req.outcome_column in df.columns:
        from app.services.fairness_metrics import compute_all_fairness_metrics, compute_mitigated_fairness_metrics
        privileged = req.privileged_groups or {}
        positive = req.positive_outcome or _infer_positive_outcome(df, req.outcome_column)
        fairness_metrics = compute_all_fairness_metrics(
            df,
            req.protected_attributes,
            req.outcome_column,
            privileged,
            positive,
        )
        # Mitigated: reweigh then retrain — should beat paper baselines
        for attr in req.protected_attributes:
            priv = (req.privileged_groups or {}).get(attr)
            if not priv and attr in df.columns:
                priv = str(df[attr].value_counts().index[0])
            if priv:
                m = compute_mitigated_fairness_metrics(df, attr, req.outcome_column, priv, positive)
                if m is not None:
                    mitigated_fairness_metrics.append(m)

    # --- Conjunctive proxy detection (Type 2 Zliobaite) ---
    conjunctive_proxies = []
    if not req.fast_mode and len(df.columns) <= 30:
        from app.services.interaction_scanner import find_conjunctive_proxies
        conjunctive_proxies = find_conjunctive_proxies(
            df,
            req.protected_attributes,
            min_individual_skill=0.05,
            min_interaction_gain=0.05,
            max_pairs=40,
        )

    # --- Calibration audit (Chouldechova 2017) ---
    calibration_audit = None
    if not req.fast_mode and req.outcome_column and req.outcome_column in df.columns:
        from app.services.calibration import compute_calibration_audit
        positive = req.positive_outcome or _infer_positive_outcome(df, req.outcome_column)
        for attr in req.protected_attributes:
            cal = compute_calibration_audit(df, attr, req.outcome_column, positive)
            if cal is not None:
                calibration_audit = cal
                break  # report first successful calibration audit

    # --- Intersectional audit (Kearns 2018) ---
    intersectional_audit = None
    if not req.fast_mode and req.outcome_column and req.outcome_column in df.columns and len(req.protected_attributes) >= 2:
        from app.services.intersectional import compute_intersectional_audit
        positive = req.positive_outcome or _infer_positive_outcome(df, req.outcome_column)
        intersectional_audit = compute_intersectional_audit(
            df, req.protected_attributes, req.outcome_column, positive
        )

    critical_count = sum(1 for c in chains if c.risk_label == "CRITICAL")
    high_count = sum(1 for c in chains if c.risk_label == "HIGH")
    conj_count = len(conjunctive_proxies)
    cal_str = f" Calibration gap: {calibration_audit.calibration_gap:.3f}." if calibration_audit else ""
    int_str = f" Intersectional max SPD: {intersectional_audit.max_spd_gap:.3f} ({len(intersectional_audit.flagged_groups)} flagged)." if intersectional_audit else ""
    summary = (
        f"Found {len(chains)} relay chains across "
        f"{len(req.protected_attributes)} protected attribute(s). "
        f"{critical_count} CRITICAL, {high_count} HIGH risk. "
        f"{conj_count} conjunctive proxy pair(s) detected."
        f"{cal_str}{int_str} "
        f"Risk scores are skill above majority-class baseline."
    )

    result = AuditResponse(
        session_id=req.session_id,
        nodes=nodes,
        edges=edges,
        chains=chains,
        summary=summary,
        fairness_metrics=fairness_metrics,
        mitigated_fairness_metrics=mitigated_fairness_metrics,
        conjunctive_proxies=conjunctive_proxies,
        calibration_audit=calibration_audit,
        intersectional_audit=intersectional_audit,
    )

    session_store.set(req.session_id, "audit", result)
    session_store.set(req.session_id, "G", G)
    session_store.set(req.session_id, "strengths", strengths)
    # Store audit config for post-fix re-measurement
    session_store.set(req.session_id, "audit_config", {
        "outcome_column": req.outcome_column,
        "privileged_groups": req.privileged_groups,
        "positive_outcome": req.positive_outcome,
        "protected_attributes": req.protected_attributes,
    })

    return result


def _infer_positive_outcome(df, outcome_col: str) -> str:
    """Infer the positive outcome label from the data distribution."""
    vals = df[outcome_col].dropna()
    # Binary: prefer the minority class as positive (fairness convention)
    if vals.nunique() == 2:
        counts = vals.value_counts()
        return str(counts.index[-1])   # least frequent = minority = positive
    # Otherwise return most common
    return str(vals.value_counts().index[0])
