"""
Gemini integration via Vertex AI — all AI calls use GCP credits (ADC auth).

No AI Studio API key used. On GCP VM, Application Default Credentials handle auth
automatically. On local dev: run `gcloud auth application-default login`.
"""
from typing import List, Optional

from app.core.config import settings
from app.models.schemas import Chain

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

# Cache keyed on (path_tuple, protected_attribute) — same chain = same explanation
_explanation_cache: dict[tuple, str] = {}

_VERTEX_INIT = False


def _init_vertex():
    global _VERTEX_INIT
    if not _VERTEX_INIT:
        vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
        _VERTEX_INIT = True


def _vertex_model(model_name: str) -> GenerativeModel:
    _init_vertex()
    return GenerativeModel(model_name)


# ---------------------------------------------------------------------------
# Chain explanation
# ---------------------------------------------------------------------------

CHAIN_EXPLANATION_PROMPT = """You are a fairness auditor explaining a discrimination risk to a non-technical stakeholder.

A multi-hop proxy discrimination chain was found in a dataset:

Chain path: {path}
Protected attribute: {protected}
Skill score: {risk_score} ({risk_label}) — skill above random-chance baseline
Hop details:
{hop_details}

Write a 3-4 sentence plain English explanation that:
1. Describes how this chain allows indirect discrimination against people with attribute "{protected}"
2. Mentions the historical or social reason this chain is problematic (e.g., redlining, systemic bias)
3. States which regulation this likely violates (EU AI Act Article 10, US ECOA, etc.)
4. Is direct and professional - no jargon, no hedging

Do NOT use bullet points. Write in paragraph form."""


def explain_chain(chain: Chain) -> str:
    cache_key = (tuple(chain.path), chain.protected_attribute)
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    if not settings.gcp_project_id:
        return _fallback_explanation(chain)

    hop_details = "\n".join(
        f"  {h.source} → {h.target} (predictive strength: {h.weight:.2%})"
        for h in chain.hops
    )
    prompt = CHAIN_EXPLANATION_PROMPT.format(
        path=" → ".join(chain.path),
        protected=chain.protected_attribute,
        risk_score=f"{chain.risk_score:.0%}",
        risk_label=chain.risk_label,
        hop_details=hop_details,
    )

    try:
        model = _vertex_model("gemini-1.5-flash-8b")
        response = model.generate_content(prompt)
        result = response.text.strip()
        _explanation_cache[cache_key] = result
        return result
    except Exception:
        return _fallback_explanation(chain)


def _fallback_explanation(chain: Chain) -> str:
    path_str = " → ".join(chain.path)
    return (
        f"This {len(chain.hops)}-hop chain ({path_str}) allows your model to "
        f"indirectly reconstruct '{chain.protected_attribute}' with {chain.risk_score:.0%} "
        f"skill above the majority-class baseline. Each hop individually appears neutral, "
        f"but together they form a discrimination pathway. "
        f"This chain likely violates EU AI Act Article 10 data governance requirements. "
        f"Removing '{chain.weakest_link}' will break the chain."
    )


# ---------------------------------------------------------------------------
# Audit chat assistant
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are FairLens, an AI fairness auditing assistant. You help users understand bias risks in their ML training datasets.

Audit context:
{audit_context}

RESPONSE FORMAT RULES — follow strictly:
- Lead with a 2-3 sentence plain-English summary.
- Then use bullet points (- item) for key findings, actions, or explanations.
- Use a markdown table when comparing metrics, chains, or groups (| Col | Col |).
- Never write long prose paragraphs. Max 4 lines of prose total.
- Be direct and specific — name exact features, exact metrics, exact regulations.
- Regulations to cite when relevant: EU AI Act Article 10, US ECOA, GDPR Article 22."""


def chat(
    user_message: str,
    chains: List[Chain],
    history: List[dict],
    dataset_name: Optional[str] = None,
) -> str:
    if not settings.gcp_project_id:
        return (
            "Vertex AI not configured. Set GCP_PROJECT_ID in .env and ensure "
            "Application Default Credentials are active (run `gcloud auth application-default login`)."
        )

    audit_context = _build_audit_context(chains, dataset_name)
    system_content = SYSTEM_PROMPT.format(audit_context=audit_context)
    full_prompt = f"{system_content}\n\n---\nUser: {user_message}"

    try:
        model = _vertex_model("gemini-1.5-flash")
        vertex_history = [
            Content(role=turn["role"], parts=[Part.from_text(turn["content"])])
            for turn in history
        ]
        chat_session = model.start_chat(history=vertex_history)
        response = chat_session.send_message(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Vertex AI Gemini error: {str(e)}"


def _build_audit_context(chains: List[Chain], dataset_name: Optional[str]) -> str:
    lines = []
    if dataset_name:
        lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Total chains found: {len(chains)}")

    for i, c in enumerate(chains[:10], 1):
        lines.append(
            f"Chain {i}: {' → '.join(c.path)} | "
            f"Protected: {c.protected_attribute} | "
            f"Risk: {c.risk_label} ({c.risk_score:.0%} skill) | "
            f"Weakest link: {c.weakest_link}"
        )

    return "\n".join(lines)
