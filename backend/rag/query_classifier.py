"""
Query safety classifier for Title IV financial aid.

Three-tier system:
  Tier 1 - Simple FAQ        → fast answer with Haiku
  Tier 2 - Complex Regulatory → deep reasoning with Opus + disclaimer
  Tier 3 - Personal Advice   → hard escalation, no LLM answer generated
"""

import json
import os
from dataclasses import dataclass
from enum import IntEnum

from anthropic import AsyncAnthropic

_client = AsyncAnthropic()
_MODEL = os.getenv("LLM_MODEL_CLASSIFIER", "claude-haiku-4-5-20251001")

_PROMPT = """\
You are a safety classifier for a US federal financial aid (Title IV) helpline.
Classify the user query into exactly one tier.

TIER 1 — Simple FAQ  (general information, no personal details required)
Examples:
  "What is FAFSA?"
  "What is a Pell Grant?"
  "What documents do I need to apply for financial aid?"
  "When is the federal FAFSA deadline?"
  "What is the difference between subsidized and unsubsidized loans?"

TIER 2 — Complex Regulatory  (program rules, multi-step processes, calculations)
Examples:
  "How is the Expected Family Contribution calculated?"
  "What are the Satisfactory Academic Progress requirements?"
  "How does dependency status affect aid?"
  "What happens during verification?"
  "What are the income thresholds for automatic zero EFC?"

TIER 3 — Personal Advice  ← MUST escalate, NEVER answer directly
Triggers (any one is sufficient):
  • "Am I eligible for…" / "Do I qualify for…" (specific to the caller)
  • Specific dollar amount advice for a named individual
  • Appeal recommendation for a specific situation
  • Immigration status + personal aid eligibility
  • Loan forgiveness advice for a named individual
  • Anything requiring review of a specific student's documents

Respond ONLY with valid JSON — no markdown, no prose:
{
  "tier": 1 | 2 | 3,
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one sentence>",
  "escalation_reason": "<null or short reason if tier 3>"
}

User query: {query}"""


class QueryTier(IntEnum):
    SIMPLE_FAQ = 1
    COMPLEX_REGULATORY = 2
    PERSONAL_ADVICE = 3


@dataclass
class ClassificationResult:
    tier: QueryTier
    confidence: float
    reasoning: str
    escalation_reason: str | None = None


async def classify_query(query: str) -> ClassificationResult:
    message = await _client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
    )
    raw = message.content[0].text.strip()
    data = json.loads(raw)
    return ClassificationResult(
        tier=QueryTier(int(data["tier"])),
        confidence=float(data["confidence"]),
        reasoning=data["reasoning"],
        escalation_reason=data.get("escalation_reason"),
    )
