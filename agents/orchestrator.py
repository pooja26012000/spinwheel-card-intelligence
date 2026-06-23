"""
agents/orchestrator.py

Orchestrator agent for SpinWheel Card Intelligence.
Classifies every incoming query as either:
  - "single"     → routes to retrieval_agent (one card, one retrieval call)
  - "comparison" → routes to comparison_agent (two cards, two retrieval calls)

Uses Gemini 2.5 Flash for classification — not regex or keyword matching —
because natural language intent is ambiguous and needs a model to decide.
"""

import json
import os
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
CLASSIFIER_MODEL = "gemini-2.5-flash"

# ── Classification prompt ─────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a query router for a sports card grading and valuation assistant.

Your job is to classify the user's question into exactly one of two categories:

1. "single" — the question is about ONE card, ONE player, ONE grade, or ONE concept.
   Examples:
   - "What is a PSA 10?"
   - "What makes a Mickey Mantle rookie card valuable?"
   - "How does centering affect a card's grade?"
   - "What is the population of PSA 10 1952 Topps Mantle cards?"

2. "comparison" — the question explicitly or implicitly compares TWO cards, TWO grades,
   TWO players, or asks about the difference between two things.
   Examples:
   - "What is the difference between a PSA 9 and PSA 10?"
   - "Is a Mantle rookie worth more than a Ruth rookie?"
   - "How does a graded card compare to a raw card?"
   - "Which is more valuable, a PSA 8 or PSA 9 Honus Wagner?"

Respond with ONLY a JSON object in this exact format:
{{"intent": "single", "reason": "brief explanation"}}
or
{{"intent": "comparison", "reason": "brief explanation"}}

User question: {query}"""

# ── Orchestrator ──────────────────────────────────────────────────────────────

def classify(query: str) -> dict:
    """
    Classify a query as 'single' or 'comparison' using Gemini.
    Returns dict with 'intent' and 'reason' keys.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = CLASSIFIER_PROMPT.format(query=query)

    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=prompt
        )
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        result = json.loads(text.strip())

        # Validate intent value
        if result.get("intent") not in ("single", "comparison"):
            result["intent"] = "single"
            result["reason"] = "defaulted to single — unrecognized intent"

        return result

    except Exception as e:
        # Default to single on failure — safer than crashing
        return {
            "intent": "single",
            "reason": f"classification failed, defaulted to single: {str(e)}"
        }


def route(query: str) -> dict:
    """
    Main orchestrator entry point.
    Classifies query and routes to the appropriate agent.
    Returns the full result dict from whichever agent handled it.
    """
    # Import agents here to avoid circular imports
    from agents.retrieval_agent import handle as retrieval_handle
    from agents.comparison_agent import handle as comparison_handle

    classification = classify(query)
    intent = classification["intent"]

    if intent == "comparison":
        result = comparison_handle(query)
    else:
        result = retrieval_handle(query)

    # Attach routing metadata to result
    result["routing"] = {
        "intent": intent,
        "reason": classification["reason"]
    }

    return result


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is a PSA 10?",
        "What is the difference between a PSA 9 and PSA 10?",
        "What makes a Mickey Mantle rookie card valuable?",
        "Is a graded card worth more than a raw card?",
        "How does centering affect a card grade?",
        "Which is better, a PSA 8 Honus Wagner or a PSA 9 Babe Ruth rookie?",
    ]

    print("=" * 60)
    print("SpinWheel — Orchestrator Classification Test")
    print("=" * 60)

    client = genai.Client(api_key=GEMINI_API_KEY)

    for query in test_queries:
        result = classify(query)
        print(f"\nQuery:  {query}")
        print(f"Intent: {result['intent']}")
        print(f"Reason: {result['reason']}")
