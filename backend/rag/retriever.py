"""
Hybrid retriever using Qdrant dense + sparse vectors with RRF fusion.

Dense model : BAAI/bge-small-en-v1.5  (384 dims, via fastembed)
Sparse model: Qdrant/bm42-all-minilm-l6-v2-attentions (via fastembed)

Both models run locally — no external embedding API needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    NamedSparseVector,
    NamedVector,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "title4_knowledge")

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_DIM = 384

# Module-level singletons — loaded once on first use
_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_model


@dataclass
class RetrievedDocument:
    id: str
    content: str
    source: str
    doc_type: str  # "fsa_handbook" | "university_policy" | "faq"
    score: float
    metadata: dict = field(default_factory=dict)


def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)


def _rrf_merge(
    dense_hits: list,
    sparse_hits: list,
    top_k: int,
    k: int = 60,
) -> list[tuple[str, float, dict]]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        doc_id = str(hit.id)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = hit.payload or {}

    for rank, hit in enumerate(sparse_hits):
        doc_id = str(hit.id)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        if doc_id not in payloads:
            payloads[doc_id] = hit.payload or {}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score, payloads[doc_id]) for doc_id, score in ranked[:top_k]]


async def hybrid_retrieve(
    query: str,
    top_k: int = 8,
    filter_doc_type: Optional[str] = None,
) -> list[RetrievedDocument]:
    """
    Hybrid dense + sparse retrieval with RRF merge.
    Returns top_k documents ranked by combined relevance.
    """
    client = get_qdrant_client()

    # Compute embeddings (CPU, no API call)
    dense_vec = list(_get_dense_model().embed([query]))[0].tolist()
    sparse_obj = list(_get_sparse_model().embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_obj.indices.tolist(),
        values=sparse_obj.values.tolist(),
    )

    search_limit = top_k * 3

    # Run both searches concurrently
    import asyncio

    dense_task = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=NamedVector(name="dense", vector=dense_vec),
        limit=search_limit,
        with_payload=True,
    )
    sparse_task = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=NamedSparseVector(name="sparse", vector=sparse_vec),
        limit=search_limit,
        with_payload=True,
    )

    dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

    merged = _rrf_merge(dense_hits, sparse_hits, top_k=top_k)

    results = []
    for doc_id, rrf_score, payload in merged:
        if filter_doc_type and payload.get("doc_type") != filter_doc_type:
            continue
        results.append(
            RetrievedDocument(
                id=doc_id,
                content=payload.get("content", ""),
                source=payload.get("source", ""),
                doc_type=payload.get("doc_type", "unknown"),
                score=rrf_score,
                metadata=payload.get("metadata", {}),
            )
        )

    return results[:top_k]


async def ensure_collection_exists() -> None:
    """Create the Qdrant collection with dense + sparse vectors if it doesn't exist."""
    client = get_qdrant_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]

    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        # Payload index for filtered search
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="doc_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )


async def upsert_documents(documents: list[dict]) -> None:
    """
    Upsert documents into Qdrant.

    Each document dict must have:
      id       : str  (unique, e.g. SHA-256 of content)
      content  : str
      source   : str
      doc_type : str
      metadata : dict (optional)
    """
    await ensure_collection_exists()
    client = get_qdrant_client()

    contents = [d["content"] for d in documents]
    dense_vecs = list(_get_dense_model().embed(contents))
    sparse_vecs = list(_get_sparse_model().embed(contents))

    points = []
    for doc, dv, sv in zip(documents, dense_vecs, sparse_vecs):
        points.append(
            PointStruct(
                id=doc["id"],
                vector={
                    "dense": dv.tolist(),
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    ),
                },
                payload={
                    "content": doc["content"],
                    "source": doc["source"],
                    "doc_type": doc["doc_type"],
                    "metadata": doc.get("metadata", {}),
                },
            )
        )

    await client.upsert(collection_name=COLLECTION_NAME, points=points)
