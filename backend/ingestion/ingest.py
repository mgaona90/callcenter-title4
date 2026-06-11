"""
Knowledge base ingestion pipeline.

Usage:
  python -m ingestion.ingest --data-dir /app/data/knowledge_base
  python -m ingestion.ingest --data-dir /app/data/knowledge_base --clear

The pipeline:
  1. Load documents from disk (PDF, DOCX, TXT, FAQ JSON)
  2. Chunk into ~800-token passages with 150-token overlap
  3. Compute dense + sparse embeddings (local, no API)
  4. Upsert into Qdrant
  5. Log to Langfuse
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ingestion.document_loaders import load_directory
from observability.langfuse_client import trace_ingestion
from rag.retriever import COLLECTION_NAME, ensure_collection_exists, get_qdrant_client, upsert_documents

logger = logging.getLogger(__name__)

BATCH_SIZE = 64  # Qdrant upsert batch size


async def ingest(data_dir: str, clear: bool = False) -> int:
    """
    Ingest all documents in data_dir into Qdrant.
    Returns total number of chunks ingested.
    """
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    await ensure_collection_exists()

    if clear:
        client = get_qdrant_client()
        await client.delete_collection(COLLECTION_NAME)
        await ensure_collection_exists()
        logger.info("Collection cleared and recreated: %s", COLLECTION_NAME)

    logger.info("Loading documents from %s …", data_dir)
    docs = load_directory(directory)
    logger.info("Loaded %d chunks from disk", len(docs))

    if not docs:
        logger.warning("No documents found — add files to %s", data_dir)
        return 0

    # Upsert in batches
    total = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        await upsert_documents(batch)
        total += len(batch)
        logger.info("Upserted %d / %d chunks …", total, len(docs))

    await trace_ingestion(doc_count=total, collection=COLLECTION_NAME)
    logger.info("Ingestion complete — %d chunks in collection '%s'", total, COLLECTION_NAME)
    return total


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Ingest knowledge base into Qdrant")
    parser.add_argument("--data-dir", default="/app/data/knowledge_base", help="Path to knowledge base directory")
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingestion")
    args = parser.parse_args()

    asyncio.run(ingest(args.data_dir, clear=args.clear))
