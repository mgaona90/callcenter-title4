#!/usr/bin/env python
"""
Standalone ingestion script — run from project root.

Usage:
  python scripts/ingest_documents.py
  python scripts/ingest_documents.py --clear
  python scripts/ingest_documents.py --data-dir ./data/knowledge_base
  python scripts/ingest_documents.py --verify-only
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path so we can import the ingestion module
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main(data_dir: str, clear: bool, verify_only: bool) -> None:
    from ingestion.ingest import ingest
    from rag.retriever import COLLECTION_NAME, get_qdrant_client

    if verify_only:
        client = get_qdrant_client()
        info = await client.get_collection(COLLECTION_NAME)
        logger.info(
            "Collection '%s': %d vectors",
            COLLECTION_NAME,
            info.points_count,
        )
        return

    total = await ingest(data_dir, clear=clear)
    logger.info("Done — %d chunks indexed.", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/knowledge_base")
    parser.add_argument("--clear", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--verify-only", action="store_true", help="Just print collection stats")
    args = parser.parse_args()

    asyncio.run(main(args.data_dir, args.clear, args.verify_only))
