"""
Main financial aid voice agent.

Flow per turn:
  1. run_rag_pipeline(query)         → RAGResult
  2. apply_guardrails(rag_result)    → GuardrailDecision
  3. [if blocked] return escalation text immediately
  4. [otherwise]  call LLM (streaming), then append disclaimer if needed

Model routing:
  Tier 1 → claude-haiku-4-5-20251001   (low latency, simple questions)
  Tier 2 → claude-opus-4-8             (deep reasoning, regulatory complexity)
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import anthropic

from agents.guardrails import GuardrailDecision, apply_guardrails, build_final_response
from agents.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_CONTEXT
from observability.langfuse_client import trace_agent_call
from rag.pipeline import RAGResult, run_rag_pipeline
from rag.query_classifier import QueryTier

MODEL_TIER1 = os.getenv("LLM_MODEL_TIER1", "claude-haiku-4-5-20251001")
MODEL_TIER2 = os.getenv("LLM_MODEL_TIER2", "claude-opus-4-8")

_client = anthropic.AsyncAnthropic()


def _pick_model(tier: QueryTier) -> str:
    return MODEL_TIER2 if tier == QueryTier.COMPLEX_REGULATORY else MODEL_TIER1


def _build_system_prompt(rag_result: RAGResult) -> str:
    if rag_result.context_text:
        return SYSTEM_PROMPT_WITH_CONTEXT.format(
            system_prompt=SYSTEM_PROMPT,
            context=rag_result.context_text,
        )
    return SYSTEM_PROMPT


async def generate_response(
    query: str,
    conversation_history: list[dict],
    session_id: str,
    rag_result: RAGResult | None = None,
) -> str:
    """
    Generate a full (non-streaming) response for the given query.
    Pass rag_result if already computed to avoid running the pipeline twice.
    """
    if rag_result is None:
        rag_result = await run_rag_pipeline(query)

    # Pre-LLM guardrail check (catches Tier 3)
    pre_decision = apply_guardrails(rag_result)
    if pre_decision.blocked:
        await trace_agent_call(
            session_id=session_id,
            query=query,
            response=pre_decision.final_response,
            rag_result=rag_result,
            model="ESCALATED",
            escalated=True,
        )
        return pre_decision.final_response

    model = _pick_model(rag_result.classification.tier)
    system = _build_system_prompt(rag_result)

    messages = [*conversation_history, {"role": "user", "content": query}]

    # Opus uses adaptive thinking; Haiku does not support it
    create_kwargs: dict = dict(
        model=model,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    if model == MODEL_TIER2:
        create_kwargs["thinking"] = {"type": "adaptive"}

    message = await _client.messages.create(**create_kwargs)
    llm_text = message.content[-1].text  # last block is always the text block

    final = build_final_response(llm_text, apply_guardrails(rag_result, llm_text))

    await trace_agent_call(
        session_id=session_id,
        query=query,
        response=final,
        rag_result=rag_result,
        model=model,
        escalated=False,
    )

    return final


async def stream_response(
    query: str,
    conversation_history: list[dict],
    session_id: str,
) -> AsyncIterator[str]:
    """
    Streaming version — yields text chunks as they arrive.
    Disclaimer is appended as a final chunk after streaming ends.
    Guardrail escalations are yielded as a single chunk immediately.
    """
    rag_result = await run_rag_pipeline(query)

    pre_decision = apply_guardrails(rag_result)
    if pre_decision.blocked:
        await trace_agent_call(
            session_id=session_id,
            query=query,
            response=pre_decision.final_response,
            rag_result=rag_result,
            model="ESCALATED",
            escalated=True,
        )
        yield pre_decision.final_response
        return

    model = _pick_model(rag_result.classification.tier)
    system = _build_system_prompt(rag_result)
    messages = [*conversation_history, {"role": "user", "content": query}]

    create_kwargs: dict = dict(
        model=model,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    if model == MODEL_TIER2:
        create_kwargs["thinking"] = {"type": "adaptive"}

    full_text = ""
    async with _client.messages.stream(**create_kwargs) as stream:
        async for chunk in stream.text_stream:
            full_text += chunk
            yield chunk

    # Append disclaimer as trailing text (pause-friendly for TTS)
    post_decision = apply_guardrails(rag_result, full_text)
    if post_decision.append_disclaimer:
        yield f" {post_decision.append_disclaimer}"
        full_text += f" {post_decision.append_disclaimer}"

    await trace_agent_call(
        session_id=session_id,
        query=query,
        response=full_text,
        rag_result=rag_result,
        model=model,
        escalated=False,
    )
