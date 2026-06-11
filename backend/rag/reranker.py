"""
Cohere Rerank v3 — second-pass scoring of retrieved documents.

Reranking is applied after hybrid retrieval to sharpen precision.
The reranker score is also used as a proxy for retrieval confidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cohere

from rag.retriever import RetrievedDocument

_client: cohere.AsyncClientV2 | None = None


def _get_client() -> cohere.AsyncClientV2:
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is not set")
        _client = cohere.AsyncClientV2(api_key=api_key)
    return _client


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
    Returns top_n documents sorted by rerank_score descending.
    """
    if not documents:
        return []

    client = _get_client()
    response = await client.rerank(
        model=model,
        query=query,
        documents=[d.content for d in documents],
        top_n=top_n,
        return_documents=False,
    )

    ranked = []
    for result in response.results:
        doc = documents[result.index]
        ranked.append(RankedDocument(document=doc, rerank_score=result.relevance_score))

    ranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return ranked


def compute_confidence(ranked_docs: list[RankedDocument]) -> float:
    """
    Confidence score in [0, 1] derived from the top reranked document's score.
    If no documents, returns 0.
    """
    if not ranked_docs:
        return 0.0
    top_score = ranked_docs[0].rerank_score
    # Cohere rerank scores are in [0, 1]; treat > 0.8 as high confidence
    return min(top_score, 1.0)
