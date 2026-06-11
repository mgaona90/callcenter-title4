"""
Cohere Rerank v3 — second-pass scoring of retrieved documents.

If COHERE_API_KEY is not set, falls back to using the raw RRF scores
from hybrid retrieval — slightly less precise but fully functional.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

_client = None
_cohere_available: bool | None = None  # None = not yet checked


def _get_client():
    global _client, _cohere_available
    if _cohere_available is None:
        api_key = os.getenv("COHERE_API_KEY", "").strip()
        # Treat placeholder values as unset
        valid = api_key and api_key not in ("...", "your_key_here", "")
        if valid:
            try:
                import cohere
                _client = cohere.AsyncClientV2(api_key=api_key)
                _cohere_available = True
                logger.info("Cohere reranker enabled")
            except ImportError:
                logger.warning("cohere package not installed — reranker disabled")
                _cohere_available = False
        else:
            logger.info("COHERE_API_KEY not configured — using raw retrieval scores")
            _cohere_available = False
    return _client if _cohere_available else None


@dataclass
class RankedDocument:
    document: RetrievedDocument
    rerank_score: float


async def rerank(
    query: str,
    documents: list[RetrievedDocument],
    top_n: int = 5,
    model: str = "rerank-v3.5",
) -> list[RankedDocument]:
    """
    Rerank documents using Cohere Rerank v3.
    Falls back to RRF scores if Cohere is not configured.
    """
    if not documents:
        return []

    client = _get_client()

    if client is None:
        # Fallback: use existing RRF scores, just trim to top_n
        ranked = [RankedDocument(document=d, rerank_score=d.score) for d in documents]
        return ranked[:top_n]

    try:
        response = await client.rerank(
            model=model,
            query=query,
            documents=[d.content for d in documents],
            top_n=top_n,
        )
        ranked = []
        for result in response.results:
            doc = documents[result.index]
            ranked.append(RankedDocument(document=doc, rerank_score=result.relevance_score))
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked
    except Exception as exc:
        logger.warning("Cohere rerank failed (%s) — falling back to retrieval scores", exc)
        return [RankedDocument(document=d, rerank_score=d.score) for d in documents][:top_n]


def compute_confidence(ranked_docs: list[RankedDocument]) -> float:
    """
    Confidence score in [0, 1].
    With Cohere: uses relevance score directly.
    Without Cohere: normalizes the RRF score (typically 0.01–0.05) to 0–1 range.
    """
    if not ranked_docs:
        return 0.0
    top_score = ranked_docs[0].rerank_score
    # RRF scores are tiny (~0.016–0.033); scale them up for display
    if top_score < 0.1:
        top_score = min(top_score * 20, 1.0)
    return min(top_score, 1.0)
