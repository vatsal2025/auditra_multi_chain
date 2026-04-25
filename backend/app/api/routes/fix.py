import threading
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.core import session_store
from app.models.schemas import FixRequest, FixResponse, FixMetricsComparison, MetricDelta, ReweighResult
from app.services.fix_engine import apply_fix

router = APIRouter()

_session_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _compute_metrics_comparison(
    df_before,
    df_after,
    removed_feature: str,
    audit_config: dict,
) -> Optional[FixMetricsComparison]:
    """Re-compute fairness metrics before and after fix, return delta."""
    if not audit_config:
        return None
    outcome_col = audit_config.get("outcome_column")
    if not outcome_col or outcome_col not in df_before.columns:
        return None

    from app.services.fairness_metrics import compute_all_fairness_metrics
    protected = audit_config.get("protected_attributes", [])
    privileged = audit_config.get("privileged_groups") or {}
    positive = audit_config.get("positive_outcome") or ""

    if not protected or not positive:
        return None

    before_metrics = compute_all_fairness_metrics(df_before, protected, outcome_col, privileged, positive)
    after_metrics = compute_all_fairness_metrics(df_after, protected, outcome_col, privileged, positive)

    if not before_metrics or not after_metrics:
        return None

    # Build delta for first protected attribute (primary)
    bm = before_metrics[0]
    am = after_metrics[0]

    metric_pairs = [
        ("statistical_parity_diff",   bm.statistical_parity_diff,   am.statistical_parity_diff),
        ("disparate_impact_ratio",     bm.disparate_impact_ratio,     am.disparate_impact_ratio),
        ("equal_opportunity_diff",     bm.equal_opportunity_diff,     am.equal_opportunity_diff),
        ("average_odds_diff",          bm.average_odds_diff,          am.average_odds_diff),
        ("predictive_parity_diff",     bm.predictive_parity_diff,     am.predictive_parity_diff),
        ("model_accuracy_overall",     bm.model_accuracy_overall,     am.model_accuracy_overall),
    ]

    deltas: List[MetricDelta] = []
    for name, before_val, after_val in metric_pairs:
        delta = round(after_val - before_val, 4)
        # For disparity metrics: closer to 0 = better (improved if |after| < |before|)
        # For accuracy: higher = better
        if name == "model_accuracy_overall":
            improved = after_val >= before_val - 0.005
        else:
            improved = abs(after_val) <= abs(before_val)
        deltas.append(MetricDelta(
            metric=name,
            before=round(before_val, 4),
            after=round(after_val, 4),
            delta=delta,
            improved=improved,
        ))

    return FixMetricsComparison(removed_feature=removed_feature, deltas=deltas)


@router.post("/fix", response_model=FixResponse)
async def apply_chain_fix(req: FixRequest):
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    audit = session_store.get(req.session_id, "audit")
    if not audit:
        raise HTTPException(status_code=400, detail="Run /audit first.")

    chain = next((c for c in audit.chains if c.id == req.chain_id), None)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found.")

    session_lock = _get_session_lock(req.session_id)

    with session_lock:
        # Re-fetch under lock
        audit = session_store.get(req.session_id, "audit")
        chain = next((c for c in audit.chains if c.id == req.chain_id), None)
        if not chain:
            raise HTTPException(status_code=404, detail="Chain already removed by concurrent request.")

        df_before = session_store.get(req.session_id, "df")
        audit_config = session_store.get(req.session_id, "audit_config") or {}

        reweigh_result: Optional[ReweighResult] = None

        if req.fix_strategy == "reweigh":
            # Reweighing: add sample weights, do NOT drop any feature
            from app.services.reweighing import reweigh_dataframe
            outcome_col = audit_config.get("outcome_column") or ""
            positive = audit_config.get("positive_outcome") or ""
            protected = chain.protected_attribute
            if outcome_col and positive and protected:
                fixed_df, reweigh_result = reweigh_dataframe(
                    df_before, protected, outcome_col, positive
                )
            else:
                fixed_df = df_before.copy()
            removed_feature = ""
            shap_entries = []
            msg = f"Applied reweighing on '{protected}'. Sample weights added. disc_after={reweigh_result.disc_after if reweigh_result else 'N/A'}."
        else:
            # Default: drop weakest link
            fixed_df, shap_entries = apply_fix(df_before, chain)
            removed_feature = chain.weakest_link or ""
            msg = f"Removed '{removed_feature}' from dataset. Chain broken."

        # Post-fix: re-compute fairness metrics and report delta
        metrics_comparison = _compute_metrics_comparison(
            df_before, fixed_df, removed_feature, audit_config
        )

        # Purge stale chains whose weakest_link no longer exists
        remaining_cols = set(fixed_df.columns)
        remaining_chains = [
            c for c in audit.chains
            if c.id != req.chain_id
            and (c.weakest_link is None or c.weakest_link in remaining_cols)
        ]

        # Also purge conjunctive proxies that reference the removed feature
        remaining_conj = [
            cp for cp in (audit.conjunctive_proxies or [])
            if (not removed_feature)
            or (cp.feature_a != removed_feature and cp.feature_b != removed_feature)
        ]

        updated_audit = audit.model_copy(update={
            "chains": remaining_chains,
            "conjunctive_proxies": remaining_conj,
        })

        session_store.set(req.session_id, "df", fixed_df)
        session_store.set(req.session_id, "audit", updated_audit)

        fixes = session_store.get(req.session_id, "fixes_applied") or []
        if removed_feature:
            fixes.append(removed_feature)
        session_store.set(req.session_id, "fixes_applied", fixes)

    return FixResponse(
        session_id=req.session_id,
        chain_id=req.chain_id,
        removed_feature=removed_feature,
        shap_values=shap_entries,
        success=True,
        message=msg,
        metrics_comparison=metrics_comparison,
        reweigh_result=reweigh_result,
    )
