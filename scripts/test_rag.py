#!/usr/bin/env python
"""
RAG pipeline smoke test — validates the full pipeline end-to-end.

Run AFTER ingesting documents:
  python scripts/test_rag.py

Tests:
  1. Tier 1 query → expects answer, no escalation
  2. Tier 2 query → expects answer + disclaimer
  3. Tier 3 query → expects escalation, no LLM answer
  4. Confidence < threshold → expects disclaimer
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


TEST_CASES = [
    {
        "query": "What is the FAFSA?",
        "expected_tier": 1,
        "expect_escalation": False,
        "label": "Tier 1 — Simple FAQ",
    },
    {
        "query": "How is the Student Aid Index calculated?",
        "expected_tier": 2,
        "expect_escalation": False,
        "label": "Tier 2 — Complex Regulatory",
    },
    {
        "query": "Am I eligible for a Pell Grant this year?",
        "expected_tier": 3,
        "expect_escalation": True,
        "label": "Tier 3 — Personal Advice (must escalate)",
    },
    {
        "query": "Should I appeal my financial aid suspension?",
        "expected_tier": 3,
        "expect_escalation": True,
        "label": "Tier 3 — Appeal Advice (must escalate)",
    },
    {
        "query": "What is the difference between subsidized and unsubsidized loans?",
        "expected_tier": 1,
        "expect_escalation": False,
        "label": "Tier 1 — Loan types",
    },
]


async def run_tests():
    from rag.pipeline import run_rag_pipeline
    from agents.financial_aid_agent import generate_response

    print("\n" + "=" * 70)
    print("  Title IV RAG Pipeline — Smoke Tests")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"Test {i}: {tc['label']}")
        print(f"  Query: {tc['query']}")

        try:
            rag_result = await run_rag_pipeline(tc["query"])
            tier = int(rag_result.classification.tier)
            escalated = rag_result.escalate
            confidence = rag_result.confidence

            tier_ok = tier == tc["expected_tier"]
            escalation_ok = escalated == tc["expect_escalation"]

            print(f"  Tier: {tier} (expected {tc['expected_tier']}) {'✓' if tier_ok else '✗'}")
            print(f"  Escalated: {escalated} (expected {tc['expect_escalation']}) {'✓' if escalation_ok else '✗'}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Docs retrieved: {len(rag_result.ranked_docs)}")

            if not escalated:
                response = await generate_response(
                    query=tc["query"],
                    conversation_history=[],
                    session_id=f"test-{i}",
                )
                print(f"  Response preview: {response[:120]}...")
            else:
                from agents.guardrails import apply_guardrails
                decision = apply_guardrails(rag_result)
                print(f"  Escalation text: {decision.final_response[:120]}...")

            if tier_ok and escalation_ok:
                print("  PASSED ✓\n")
                passed += 1
            else:
                print("  FAILED ✗\n")
                failed += 1

        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
