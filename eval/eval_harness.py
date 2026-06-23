"""
eval/eval_harness.py

Evaluation harness for SpinWheel Card Intelligence RAG pipeline.
Runs 30 questions through the full RAG pipeline and scores each answer on:
  - Groundedness: does the answer stay within the retrieved chunks?
  - Relevance: are the retrieved chunks relevant to the question?

Both scores are 0.0-1.0 per question, averaged across all 30 questions.
Run this BEFORE and AFTER prompt/retrieval tuning to capture improvement delta.

Usage:
    python3 eval/eval_harness.py
    python3 eval/eval_harness.py --output eval/results_baseline.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from google import genai

# ── Add project root to path so we can import pipeline/rag.py ─────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.rag import rag

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JUDGE_MODEL = "gemini-2.5-flash"

# ── 30 Evaluation Questions ───────────────────────────────────────────────────
# Covers: PSA grading criteria, card valuation, player/set facts,
# comparison questions, and edge cases where corpus may not have answers.

EVAL_QUESTIONS = [
    # PSA Grading Criteria (1-10)
    {
        "id": "eval_001",
        "question": "What are the criteria for a PSA 10 Gem Mint grade?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_002",
        "question": "What is the centering requirement for a PSA 9 card?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_003",
        "question": "What distinguishes a PSA 7 from a PSA 8?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_004",
        "question": "What defects would cause a card to receive a PSA 5 grade?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_005",
        "question": "How does PSA define a Mint condition card?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_006",
        "question": "What is the lowest PSA grade and what condition does it represent?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_007",
        "question": "Can a PSA 10 card have any printing defects?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_008",
        "question": "What does PSA look for when grading card corners?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_009",
        "question": "How does PSA grade card surface condition?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    {
        "id": "eval_010",
        "question": "What is a half-point grade in PSA grading?",
        "category": "psa_grading",
        "expected_grounded": True
    },
    # Card Valuation (11-18)
    {
        "id": "eval_011",
        "question": "What makes a rookie card more valuable than other cards?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_012",
        "question": "How does card condition affect its market value?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_013",
        "question": "What is a population report and why does it matter for card value?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_014",
        "question": "Why are low population PSA 10 cards especially valuable?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_015",
        "question": "What role does rarity play in sports card valuation?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_016",
        "question": "How does player performance affect the value of their cards?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_017",
        "question": "What is the most expensive baseball card ever sold?",
        "category": "valuation",
        "expected_grounded": True
    },
    {
        "id": "eval_018",
        "question": "What factors make a 1952 Topps Mickey Mantle card valuable?",
        "category": "valuation",
        "expected_grounded": True
    },
    # Card History and Sets (19-24)
    {
        "id": "eval_019",
        "question": "What is a rookie card?",
        "category": "card_history",
        "expected_grounded": True
    },
    {
        "id": "eval_020",
        "question": "When did Topps start producing baseball cards?",
        "category": "card_history",
        "expected_grounded": True
    },
    {
        "id": "eval_021",
        "question": "What is the history of PSA as a grading company?",
        "category": "card_history",
        "expected_grounded": True
    },
    {
        "id": "eval_022",
        "question": "What are the most collected baseball card sets?",
        "category": "card_history",
        "expected_grounded": True
    },
    {
        "id": "eval_023",
        "question": "What makes a card a short print?",
        "category": "card_history",
        "expected_grounded": True
    },
    {
        "id": "eval_024",
        "question": "How has the sports card market changed over the decades?",
        "category": "card_history",
        "expected_grounded": True
    },
    # Comparison Questions (25-27)
    {
        "id": "eval_025",
        "question": "What is the difference between a PSA 9 and a PSA 10 in terms of value?",
        "category": "comparison",
        "expected_grounded": True
    },
    {
        "id": "eval_026",
        "question": "How does a graded card differ from a raw ungraded card in terms of value?",
        "category": "comparison",
        "expected_grounded": True
    },
    {
        "id": "eval_027",
        "question": "What is the difference between a first print and a reprint card?",
        "category": "comparison",
        "expected_grounded": True
    },
    # Edge Cases — corpus may not have full answers (28-30)
    {
        "id": "eval_028",
        "question": "What is the current market price for a 2023 Shohei Ohtani rookie card?",
        "category": "edge_case",
        "expected_grounded": False  # corpus unlikely to have current prices
    },
    {
        "id": "eval_029",
        "question": "How do I submit a card to PSA for grading?",
        "category": "edge_case",
        "expected_grounded": True
    },
    {
        "id": "eval_030",
        "question": "What is the PSA grading scale for autographed cards?",
        "category": "edge_case",
        "expected_grounded": True
    },
]

# ── LLM Judge Prompts ─────────────────────────────────────────────────────────

GROUNDEDNESS_JUDGE_PROMPT = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Your task is to evaluate whether an AI answer is grounded in the provided context chunks.

A grounded answer:
- Only makes claims that are directly supported by the context chunks
- Does not introduce facts, numbers, or details not present in the chunks
- Correctly cites the source chunks

An ungrounded answer:
- Introduces facts not present in the chunks (hallucination)
- Makes claims that contradict the chunks
- Ignores the chunks and answers from general knowledge

Rate the groundedness of the answer on a scale of 0.0 to 1.0:
- 1.0: Fully grounded, every claim is supported by the chunks
- 0.75: Mostly grounded, minor unsupported claims
- 0.5: Partially grounded, some hallucination present
- 0.25: Mostly ungrounded, significant hallucination
- 0.0: Completely ungrounded, ignores the chunks entirely

Question: {question}

Context Chunks:
{context}

Answer to evaluate:
{answer}

Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "reason": "brief explanation"}}"""

RELEVANCE_JUDGE_PROMPT = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Your task is to evaluate whether the retrieved context chunks are relevant to the question asked.

Rate the relevance of the chunks on a scale of 0.0 to 1.0:
- 1.0: All chunks are highly relevant to answering the question
- 0.75: Most chunks are relevant, one or two are off-topic
- 0.5: Some chunks are relevant, some are not
- 0.25: Few chunks are relevant
- 0.0: No chunks are relevant to the question

Question: {question}

Retrieved Chunks:
{context}

Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "reason": "brief explanation"}}"""

# ── Judge Function ────────────────────────────────────────────────────────────

def judge(prompt: str, retries: int = 3) -> dict:
    """Call Gemini as a judge and parse the JSON response."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt
            )
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"score": 0.0, "reason": f"Judge failed: {str(e)}"}


def score_question(eval_item: dict) -> dict:
    """Run a single eval question through RAG and score it."""
    question = eval_item["question"]
    print(f"  [{eval_item['id']}] {question[:60]}...")

    # Run RAG pipeline
    try:
        result = rag(question)
    except Exception as e:
        print(f"    ERROR: {e}")
        return {
            **eval_item,
            "answer": "",
            "chunks_used": [],
            "groundedness_score": 0.0,
            "groundedness_reason": f"RAG failed: {str(e)}",
            "relevance_score": 0.0,
            "relevance_reason": f"RAG failed: {str(e)}",
        }

    answer = result["answer"]
    chunks = result["chunks_used"]

    # Build context string for judge
    context = "\n\n".join([
        f"[{c['chunk_id']}]: {c['text'][:500]}"
        for c in chunks
    ])

    # Score groundedness
    g_prompt = GROUNDEDNESS_JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )
    g_result = judge(g_prompt)

    # Small delay to avoid rate limiting
    time.sleep(1)

    # Score relevance
    r_prompt = RELEVANCE_JUDGE_PROMPT.format(
        question=question,
        context=context
    )
    r_result = judge(r_prompt)

    print(f"    Groundedness: {g_result['score']:.2f} | Relevance: {r_result['score']:.2f}")

    return {
        **eval_item,
        "answer": answer,
        "chunks_used": [c["chunk_id"] for c in chunks],
        "groundedness_score": g_result["score"],
        "groundedness_reason": g_result["reason"],
        "relevance_score": r_result["score"],
        "relevance_reason": r_result["reason"],
    }


# ── Main Eval Runner ──────────────────────────────────────────────────────────

def run_eval(output_path: str = None) -> dict:
    """Run full eval harness across all 30 questions."""
    print("=" * 60)
    print("SpinWheel — RAG Eval Harness")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Questions: {len(EVAL_QUESTIONS)}")
    print("=" * 60)

    results = []

    for i, eval_item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Category: {eval_item['category']}")
        result = score_question(eval_item)
        results.append(result)
        # Delay between questions to respect rate limits
        time.sleep(2)

    # ── Compute aggregate scores ───────────────────────────────────────────────
    groundedness_scores = [r["groundedness_score"] for r in results]
    relevance_scores = [r["relevance_score"] for r in results]

    avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
    avg_relevance = sum(relevance_scores) / len(relevance_scores)

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"groundedness": [], "relevance": []}
        categories[cat]["groundedness"].append(r["groundedness_score"])
        categories[cat]["relevance"].append(r["relevance_score"])

    category_summary = {}
    for cat, scores in categories.items():
        category_summary[cat] = {
            "avg_groundedness": sum(scores["groundedness"]) / len(scores["groundedness"]),
            "avg_relevance": sum(scores["relevance"]) / len(scores["relevance"]),
            "count": len(scores["groundedness"])
        }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(EVAL_QUESTIONS),
        "avg_groundedness": round(avg_groundedness, 4),
        "avg_relevance": round(avg_relevance, 4),
        "category_breakdown": category_summary,
        "results": results
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Average Groundedness Score: {avg_groundedness:.4f} ({avg_groundedness*100:.1f}%)")
    print(f"Average Relevance Score:    {avg_relevance:.4f} ({avg_relevance*100:.1f}%)")
    print("\nPer-Category Breakdown:")
    for cat, scores in category_summary.items():
        print(f"  {cat:20s} | G: {scores['avg_groundedness']:.2f} | R: {scores['avg_relevance']:.2f} | N: {scores['count']}")

    # ── Save results ───────────────────────────────────────────────────────────
    if output_path is None:
        output_path = Path(__file__).parent / "results_baseline.json"

    Path(output_path).write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {output_path}")
    print("=" * 60)

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpinWheel RAG Eval Harness")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: eval/results_baseline.json)"
    )
    args = parser.parse_args()
    run_eval(output_path=args.output)
