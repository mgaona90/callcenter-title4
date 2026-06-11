"""
Full RAG pipeline:

  classify → [tier 3: escalate] → retrieve → rerank → confidence score → answer

Tier routing:
  Tier 1 (Simple FAQ)        → Haiku + top 3 docs
  Tier 2 (Complex Regulatory) → Opus + top 5 docs + mandatory disclaimer
  Tier 3 (Personal Advice)   → hard escalation, zero LLM calls after classifier
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rag.query_classifier import ClassificationResult, QueryTier, classify_query
from rag.reranker import RankedDocument, compute_confidence, rerank
from rag.retriever import RetrievedDocument, hybrid_retrieve

CONFIDENCE_DISCLAIMER_THRESHOLD = float(
    os.getenv("CONFIDENCE_DISCLAIMER_THRESHOLD", "0.72")
)


@dataclass
class RAGResult:
    classification: ClassificationResult
    ranked_docs: list[RankedDocument] = field(default_factory=list)
    confidence: float = 0.0
    context_text: str = ""
    needs_disclaimer: bool = False
    escalate: bool = False
    escalation_reason: str | None = None


async def run_rag_pipeline(query: str) -> RAGResult:
    """
    Full pipeline from raw query to RAG context ready for the LLM.
    The agent calls this, then decides what to say based on the result.
    """
    # Step 1: Classify
    classification = await classify_query(query)

    # Tier 3 — stop immediately, no retrieval
    if classification.tier == QueryTier.PERSONAL_ADVICE:
        return RAGResult(
            classification=classification,
            escalate=True,
            escalation_reason=classification.escalation_reason
            or "This question requires a personalized review by a financial aid advisor.",
        )

    # Steps 2 & 3: Retrieve + rerank
    top_k_retrieve = 12 if classification.tier == QueryTier.COMPLEX_REGULATORY else 8
    top_n_rerank = 5 if classification.tier == QueryTier.COMPLEX_REGULATORY else 3

    raw_docs = await hybrid_retrieve(query, top_k=top_k_retrieve)
    ranked_docs = await rerank(query, raw_docs, top_n=top_n_rerank)

    # Step 4: Confidence
    confidence = compute_confidence(ranked_docs)

    # Step 5: Build context block for the LLM
    context_parts = []
    for i, ranked in enumerate(ranked_docs, 1):
        doc = ranked.document
        context_parts.append(
            f"[Source {i}] {doc.source} (relevance: {ranked.rerank_score:.2f})\n{doc.content}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    needs_disclaimer = (
        classification.tier == QueryTier.COMPLEX_REGULATORY
        or confidence < CONFIDENCE_DISCLAIMER_THRESHOLD
    )

    return RAGResult(
        classification=classification,
        ranked_docs=ranked_docs,
        confidence=confidence,
        context_text=context_text,
        needs_disclaimer=needs_disclaimer,
        escalate=False,
    )
