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

    G, strengths = build_graph(df, col_types, req.threshold)
    chains = find_chains(G, strengths, req.protected_attributes, req.max_depth, col_types)

    # Score chains with LightGBM / Vertex AI
    chains = score_all_chains(df, chains)

    # Generate Gemini explanations for top chains
    for i, chain in enumerate(chains[:20]):  # limit to avoid rate limits
        explanation = gemini_service.explain_chain(chain)
        chains[i] = chain.model_copy(update={"explanation": explanation})

    nodes, edges = build_graph_schema(G, chains, req.protected_attributes, col_types)

    critical_count = sum(1 for c in chains if c.risk_label == "CRITICAL")
    summary = (
        f"Found {len(chains)} discrimination chains across "
        f"{len(req.protected_attributes)} protected attribute(s). "
        f"{critical_count} chain(s) rated CRITICAL."
    )

    # Persist for later use
    session_store.set(req.session_id, "audit", AuditResponse(
        session_id=req.session_id,
        nodes=nodes,
        edges=edges,
        chains=chains,
        summary=summary,
    ))
    session_store.set(req.session_id, "G", G)
    session_store.set(req.session_id, "strengths", strengths)

    return AuditResponse(
        session_id=req.session_id,
        nodes=nodes,
        edges=edges,
        chains=chains,
        summary=summary,
    )
