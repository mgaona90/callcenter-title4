"""
VAPI webhook handler.

VAPI orchestrates the full voice pipeline:
  caller → Deepgram Nova-3 (STT) → this webhook → Cartesia Sonic-3 (TTS) → caller

VAPI sends the transcribed text to POST /vapi/webhook and expects a JSON response
with the assistant's text reply. All session memory lives in Redis keyed by call ID.

Webhook event types handled:
  assistant-request   → generate a response to the caller's utterance
  end-of-call-report  → log call summary (Langfuse)
  status-update       → informational, acknowledged only

Signature verification: VAPI signs requests with HMAC-SHA256 using VAPI_WEBHOOK_SECRET.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import ORJSONResponse

from agents.financial_aid_agent import generate_response

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vapi"])

WEBHOOK_SECRET = os.getenv("VAPI_WEBHOOK_SECRET", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_CONTEXT_TURNS = int(os.getenv("MAX_CONTEXT_TURNS", "10"))

_redis_pool: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


def _verify_signature(body: bytes, signature: str) -> bool:
    """Verify VAPI HMAC-SHA256 signature. Skip if secret is not configured."""
    if not WEBHOOK_SECRET:
        return True
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _load_call_history(call_id: str, redis: aioredis.Redis) -> list[dict]:
    raw = await redis.get(f"vapi:call:{call_id}:history")
    return json.loads(raw) if raw else []


async def _save_call_history(call_id: str, history: list[dict], redis: aioredis.Redis) -> None:
    trimmed = history[-(MAX_CONTEXT_TURNS * 2):]
    await redis.setex(
        f"vapi:call:{call_id}:history",
        3600 * 2,  # 2-hour TTL
        json.dumps(trimmed),
    )


@router.post("/webhook")
async def vapi_webhook(request: Request):
    body = await request.body()

    # Signature check
    sig = request.headers.get("x-vapi-signature", "")
    if WEBHOOK_SECRET and not _verify_signature(body, sig):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = payload.get("message", {})
    event_type = message.get("type", "")

    # ── assistant-request: main call path ────────────────────────────────────
    if event_type == "assistant-request":
        call_info = message.get("call", {})
        call_id = call_info.get("id", str(uuid.uuid4()))

        # Latest transcript from VAPI
        transcript = message.get("transcript", "").strip()
        if not transcript:
            return ORJSONResponse(
                {"message": {"role": "assistant", "content": "I'm sorry, I didn't catch that. Could you repeat your question?"}}
            )

        redis = await _get_redis()
        history = await _load_call_history(call_id, redis)

        try:
            response_text = await generate_response(
                query=transcript,
                conversation_history=history,
                session_id=call_id,
            )
        except Exception as exc:
            logger.exception("Agent error for call %s: %s", call_id, exc)
            response_text = (
                "I apologize, I'm experiencing a technical issue. "
                "Please hold for a moment or try calling back. "
                "You can also visit StudentAid.gov for assistance."
            )

        # Persist
        history.extend([
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": response_text},
        ])
        await _save_call_history(call_id, history, redis)

        return ORJSONResponse(
            {"message": {"role": "assistant", "content": response_text}}
        )

    # ── end-of-call-report ───────────────────────────────────────────────────
    if event_type == "end-of-call-report":
        call_id = message.get("call", {}).get("id", "unknown")
        duration = message.get("durationSeconds", 0)
        logger.info("Call ended | id=%s | duration=%ds", call_id, duration)
        return ORJSONResponse({"received": True})

    # ── all other events: acknowledge ────────────────────────────────────────
    return ORJSONResponse({"received": True})
