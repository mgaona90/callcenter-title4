"""
Langfuse observability wrapper.

Every agent call is traced with:
  - query tier + confidence score
  - model used
  - whether escalation was triggered
  - retrieved doc sources
  - final response text (for compliance audit)

Traces are fire-and-forget — a Langfuse failure never blocks a user response.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if secret and public:
            from langfuse import Langfuse
            _langfuse = Langfuse(secret_key=secret, public_key=public, host=host)
        else:
            logger.warning("Langfuse keys not set — observability disabled")
    return _langfuse


async def trace_agent_call(
    session_id: str,
    query: str,
    response: str,
    rag_result,  # RAGResult — avoid circular import with type hint
    model: str,
    escalated: bool,
) -> None:
    """Record a complete agent turn in Langfuse. Non-blocking."""
    lf = _get_langfuse()
    if lf is None:
        return

    try:
        doc_sources = []
        if hasattr(rag_result, "ranked_docs"):
            doc_sources = [
                {
                    "source": rd.document.source,
                    "doc_type": rd.document.doc_type,
                    "score": rd.rerank_score,
                }
                for rd in rag_result.ranked_docs
            ]

        trace = lf.trace(
            name="financial-aid-agent",
            session_id=session_id,
            input=query,
            output=response,
            metadata={
                "tier": int(rag_result.classification.tier),
                "tier_confidence": rag_result.classification.confidence,
                "retrieval_confidence": getattr(rag_result, "confidence", None),
                "model": model,
                "escalated": escalated,
                "needs_disclaimer": getattr(rag_result, "needs_disclaimer", False),
                "doc_sources": doc_sources,
            },
            tags=[
                f"tier-{int(rag_result.classification.tier)}",
                "escalated" if escalated else "answered",
                model,
            ],
        )
        lf.flush()
    except Exception as exc:
        logger.warning("Langfuse trace failed (non-fatal): %s", exc)


async def trace_ingestion(doc_count: int, collection: str) -> None:
    """Record a knowledge base ingestion event."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.event(
            name="knowledge-base-ingestion",
            metadata={"doc_count": doc_count, "collection": collection},
        )
        lf.flush()
    except Exception as exc:
        logger.warning("Langfuse ingestion trace failed: %s", exc)
