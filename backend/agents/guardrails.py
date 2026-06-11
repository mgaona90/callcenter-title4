"""
Legal guardrails layer.

Responsibilities:
  1. Block Tier 3 queries from reaching the LLM and return the correct escalation text
  2. Append appropriate disclaimers to Tier 2 answers
  3. Append low-confidence disclaimers when retrieval confidence is below threshold
  4. Audit log every guardrail action
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents.prompts.safety_templates import (
    LOW_CONFIDENCE_DISCLAIMER,
    REGULATORY_DISCLAIMER,
    get_escalation_response,
)
from rag.pipeline import RAGResult
from rag.query_classifier import QueryTier

logger = logging.getLogger(__name__)


@dataclass
class GuardrailDecision:
    blocked: bool
    final_response: str | None   # set when blocked=True
    append_disclaimer: str | None  # set when blocked=False and a disclaimer should follow


def apply_guardrails(rag_result: RAGResult, llm_response: str | None = None) -> GuardrailDecision:
    """
    Apply legal guardrails to a RAG result.

    Call this BEFORE the LLM for Tier 3 (pass llm_response=None).
    Call this AFTER the LLM for Tier 1/2 (pass llm_response=<generated text>).
    """
    # ── Tier 3: hard block ──────────────────────────────────────────────────
    if rag_result.escalate:
        escalation_text = get_escalation_response(rag_result.escalation_reason)
        logger.info(
            "GUARDRAIL ESCALATION | tier=3 | reason=%s", rag_result.escalation_reason
        )
        return GuardrailDecision(blocked=True, final_response=escalation_text, append_disclaimer=None)

    # ── Tier 2: answer allowed but disclaimer required ──────────────────────
    if rag_result.classification.tier == QueryTier.COMPLEX_REGULATORY:
        disclaimer = REGULATORY_DISCLAIMER
        logger.info(
            "GUARDRAIL DISCLAIMER | tier=2 | confidence=%.2f", rag_result.confidence
        )
        return GuardrailDecision(blocked=False, final_response=None, append_disclaimer=disclaimer)

    # ── Tier 1 with low confidence: soft disclaimer ─────────────────────────
    if rag_result.needs_disclaimer:
        logger.info(
            "GUARDRAIL LOW_CONFIDENCE | confidence=%.2f", rag_result.confidence
        )
        return GuardrailDecision(
            blocked=False, final_response=None, append_disclaimer=LOW_CONFIDENCE_DISCLAIMER
        )

    # ── Tier 1, high confidence: clean pass-through ─────────────────────────
    return GuardrailDecision(blocked=False, final_response=None, append_disclaimer=None)


def build_final_response(llm_text: str, decision: GuardrailDecision) -> str:
    """Combine LLM output with any disclaimer into the final spoken response."""
    if decision.blocked and decision.final_response:
        return decision.final_response
    if decision.append_disclaimer:
        return f"{llm_text.rstrip()} {decision.append_disclaimer}"
    return llm_text
