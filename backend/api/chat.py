"""
Chat API endpoint — used by the Streamlit prototype and any REST client.

POST /api/chat          → full response (JSON)
POST /api/chat/stream   → SSE streaming response
GET  /api/session/{id}  → retrieve conversation history
DELETE /api/session/{id} → clear conversation
"""

from __future__ import annotations

import json
import os
import uuid
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.financial_aid_agent import generate_response, stream_response
from rag.pipeline import run_rag_pipeline

router = APIRouter(tags=["chat"])

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_CONTEXT_TURNS = int(os.getenv("MAX_CONTEXT_TURNS", "10"))


# ── Redis dependency ─────────────────────────────────────────────────────────

_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


# ── Session helpers ───────────────────────────────────────────────────────────

async def load_history(session_id: str, redis: aioredis.Redis) -> list[dict]:
    raw = await redis.get(f"session:{session_id}:history")
    if raw:
        return json.loads(raw)
    return []


async def save_history(session_id: str, history: list[dict], redis: aioredis.Redis) -> None:
    # Keep only the last N turns to stay within context window
    trimmed = history[-(MAX_CONTEXT_TURNS * 2):]
    await redis.setex(
        f"session:{session_id}:history",
        3600 * 4,  # 4-hour TTL per call session
        json.dumps(trimmed),
    )


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tier: int
    tier_label: str
    confidence: float
    escalated: bool
    doc_sources: list[dict]


_TIER_LABELS = {1: "Simple FAQ", 2: "Complex Regulatory", 3: "Personal Advice (escalated)"}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, redis: aioredis.Redis = Depends(get_redis)):
    session_id = req.session_id or str(uuid.uuid4())
    history = await load_history(session_id, redis)

    # Run RAG ahead so we can return metadata to the UI
    rag_result = await run_rag_pipeline(req.query)

    response_text = await generate_response(
        query=req.query,
        conversation_history=history,
        session_id=session_id,
    )

    # Persist turn
    history.extend([
        {"role": "user", "content": req.query},
        {"role": "assistant", "content": response_text},
    ])
    await save_history(session_id, history, redis)

    doc_sources = [
        {
            "source": rd.document.source,
            "doc_type": rd.document.doc_type,
            "content_snippet": rd.document.content[:200],
            "rerank_score": round(rd.rerank_score, 3),
        }
        for rd in rag_result.ranked_docs
    ]

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        tier=int(rag_result.classification.tier),
        tier_label=_TIER_LABELS[int(rag_result.classification.tier)],
        confidence=round(rag_result.confidence, 3),
        escalated=rag_result.escalate,
        doc_sources=doc_sources,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, redis: aioredis.Redis = Depends(get_redis)):
    session_id = req.session_id or str(uuid.uuid4())
    history = await load_history(session_id, redis)

    async def sse_generator() -> AsyncIterator[str]:
        full_response = ""
        async for chunk in stream_response(
            query=req.query,
            conversation_history=history,
            session_id=session_id,
        ):
            full_response += chunk
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

        # Save history after streaming completes
        updated_history = history + [
            {"role": "user", "content": req.query},
            {"role": "assistant", "content": full_response},
        ]
        await save_history(session_id, updated_history, redis)
        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id},
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str, redis: aioredis.Redis = Depends(get_redis)):
    history = await load_history(session_id, redis)
    return {"session_id": session_id, "turns": len(history) // 2, "history": history}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str, redis: aioredis.Redis = Depends(get_redis)):
    await redis.delete(f"session:{session_id}:history")
    return {"session_id": session_id, "cleared": True}
