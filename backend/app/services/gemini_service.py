"""
Gemini integration:
  1. Chain explanation generator
  2. Conversational audit assistant
"""
import json
from typing import List, Optional

from app.core.config import settings
from app.models.schemas import Chain


def _get_model():
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Chain explanation
# ---------------------------------------------------------------------------

CHAIN_EXPLANATION_PROMPT = """You are a fairness auditor explaining a discrimination risk to a non-technical stakeholder.

A multi-hop proxy discrimination chain was found in a dataset:

Chain path: {path}
Protected attribute: {protected}
Risk score: {risk_score} ({risk_label})
Hop details:
{hop_details}

Write a 3-4 sentence plain English explanation that:
1. Describes how this chain allows indirect discrimination against people with attribute "{protected}"
2. Mentions the historical or social reason this chain is problematic (e.g., redlining, systemic bias)
3. States which regulation this likely violates (EU AI Act Article 10, US ECOA, etc.)
4. Is direct and professional - no jargon, no hedging

Do NOT use bullet points. Write in paragraph form."""


def explain_chain(chain: Chain) -> str:
    if not settings.gemini_api_key:
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
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return _fallback_explanation(chain)


def _fallback_explanation(chain: Chain) -> str:
    path_str = " → ".join(chain.path)
    return (
        f"This {len(chain.hops)}-hop chain ({path_str}) allows your model to "
        f"indirectly reconstruct '{chain.protected_attribute}' with approximately "
        f"{chain.risk_score:.0%} accuracy. Each hop individually appears neutral, "
        f"but together they form a discrimination pathway. "
        f"This chain likely violates EU AI Act Article 10 data governance requirements. "
        f"Removing the weakest link ('{chain.weakest_link}') will break the chain."
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
    if not settings.gemini_api_key:
        return "Gemini API key not configured. Please set GEMINI_API_KEY in your .env file."

    audit_context = _build_audit_context(chains, dataset_name)
    system = SYSTEM_PROMPT.format(audit_context=audit_context)

    try:
        model = _get_model()
        # Build conversation history
        gemini_history = []
        for turn in history:
            gemini_history.append({"role": turn["role"], "parts": [turn["content"]]})

        chat_session = model.start_chat(history=gemini_history)
        full_prompt = f"{system}\n\nUser: {user_message}" if not gemini_history else user_message
        response = chat_session.send_message(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"


def _build_audit_context(chains: List[Chain], dataset_name: Optional[str]) -> str:
    lines = []
    if dataset_name:
        lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Total chains found: {len(chains)}")

    for i, c in enumerate(chains[:10], 1):
        lines.append(
            f"Chain {i}: {' → '.join(c.path)} | "
            f"Protected: {c.protected_attribute} | "
            f"Risk: {c.risk_label} ({c.risk_score:.0%}) | "
            f"Weakest link: {c.weakest_link}"
        )

    return "\n".join(lines)
