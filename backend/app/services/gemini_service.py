"""
Gemini integration via Vertex AI (preferred — uses GCP credits) or AI Studio API key.

Priority:
  1. Vertex AI — if GCP_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS configured
  2. AI Studio — if GEMINI_API_KEY configured
  3. Fallback text — no credentials at all
"""
from typing import List, Optional

from app.core.config import settings
from app.models.schemas import Chain

# Cache keyed on (path_tuple, protected_attribute) — same chain = same explanation
_explanation_cache: dict[tuple, str] = {}


def _use_vertex() -> bool:
    # On GCP VM, ADC handles auth automatically — only project ID needed
    return bool(settings.gcp_project_id)


def _use_aistudio() -> bool:
    return bool(settings.gemini_api_key)


def _get_vertex_model(model_name: str):
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
    return GenerativeModel(model_name)


def _get_aistudio_model(model_name: str):
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(model_name)


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

    # Use lighter model for explanations — deterministic text, no reasoning needed
    explanation_model = "gemini-1.5-flash-8b"

    try:
        if _use_vertex():
            model = _get_vertex_model(explanation_model)
            response = model.generate_content(prompt)
        elif _use_aistudio():
            model = _get_aistudio_model(explanation_model)
            response = model.generate_content(prompt)
        else:
            return _fallback_explanation(chain)

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

SYSTEM_PROMPT = """You are FairLens, an AI fairness auditing assistant. You help users understand
bias risks in their ML training datasets. You have access to the audit results below.

Audit context:
{audit_context}

Answer questions clearly, concisely, and practically. When recommending fixes, be specific about
which features to remove or transform. Always connect findings to real regulatory requirements
(EU AI Act Article 10, US ECOA, GDPR Article 22) when relevant."""


def chat(
    user_message: str,
    chains: List[Chain],
    history: List[dict],
    dataset_name: Optional[str] = None,
) -> str:
    audit_context = _build_audit_context(chains, dataset_name)
    system_content = SYSTEM_PROMPT.format(audit_context=audit_context)
    full_prompt = f"{system_content}\n\n---\nUser: {user_message}"

    try:
        if _use_vertex():
            return _chat_vertex(full_prompt, history)
        elif _use_aistudio():
            return _chat_aistudio(full_prompt, history)
        else:
            return "No AI credentials configured. Set GCP_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS (Vertex AI) or GEMINI_API_KEY in your .env file."
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"


def _chat_vertex(full_prompt: str, history: List[dict]) -> str:
    from vertexai.generative_models import Content, Part
    model = _get_vertex_model("gemini-2.5-flash")

    vertex_history = [
        Content(role=turn["role"], parts=[Part.from_text(turn["content"])])
        for turn in history
    ]
    chat_session = model.start_chat(history=vertex_history)
    response = chat_session.send_message(full_prompt)
    return response.text.strip()


def _chat_aistudio(full_prompt: str, history: List[dict]) -> str:
    model = _get_aistudio_model("gemini-2.5-flash")
    gemini_history = [
        {"role": turn["role"], "parts": [turn["content"]]}
        for turn in history
    ]
    chat_session = model.start_chat(history=gemini_history)
    response = chat_session.send_message(full_prompt)
    return response.text.strip()


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
