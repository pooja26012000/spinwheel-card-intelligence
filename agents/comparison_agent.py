"""
agents/comparison_agent.py

Comparison agent for SpinWheel Card Intelligence.
Handles questions that compare two cards, two grades, or two concepts.

Flow:
  1. Receives comparison query from orchestrator
  2. Decomposes query into two sub-queries (one per subject)
  3. Makes two separate retrieval calls to Vertex AI Vector Search
  4. Merges and deduplicates the retrieved chunks
  5. Generates a single grounded answer that reasons across both subjects
  6. Returns structured result with answer, chunks, and agent metadata

Why two retrieval calls?
  A single retrieval call for "PSA 9 vs PSA 10" may return chunks biased
  toward whichever term appears more in the corpus. Two separate calls
  guarantee representation for both subjects before generation.
"""

import json
import os
import sys
from pathlib import Path

from google import genai

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.rag import embed_query, retrieve, build_context

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GENERATION_MODEL = "gemini-2.5-flash"

COMPARISON_SYSTEM_PROMPT = """You are SpinWheel Card Intelligence, an expert assistant for sports card grading and valuation.

You have been given context chunks about TWO subjects for comparison.
Your job is to write a clear, grounded comparison answer.

Rules:
- Answer ONLY from the provided context chunks
- Clearly address both subjects being compared
- Structure your answer to highlight the key differences or similarities
- Be specific — use numbers, grades, and facts from the chunks
- If the chunks don't have enough information about one subject, say so explicitly
- Do not hallucinate facts not present in the chunks
- End your answer with: Sources: [chunk_id_1, chunk_id_2, ...]
"""

DECOMPOSE_PROMPT = """You are a query decomposer for a sports card question answering system.

Given a comparison question, extract the TWO subjects being compared.
Return ONLY a JSON object with two sub-queries optimized for vector search retrieval.

Examples:
Question: "What is the difference between a PSA 9 and PSA 10?"
{{"query_a": "PSA 10 Gem Mint grading criteria", "query_b": "PSA 9 Mint grading criteria"}}

Question: "Is a graded card worth more than a raw ungraded card?"
{{"query_a": "graded card value benefits", "query_b": "raw ungraded card value"}}

Question: "Which is better, a PSA 8 Honus Wagner or a PSA 9 Babe Ruth rookie?"
{{"query_a": "Honus Wagner card PSA 8 value", "query_b": "Babe Ruth rookie card PSA 9 value"}}

Now decompose this question:
Question: "{query}"

Respond with ONLY the JSON object, no other text."""

# ── Query decomposition ───────────────────────────────────────────────────────

def decompose(query: str) -> tuple[str, str]:
    """
    Decompose a comparison query into two sub-queries.
    Returns (query_a, query_b) tuple.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = DECOMPOSE_PROMPT.format(query=query)

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        result = json.loads(text.strip())
        return result["query_a"], result["query_b"]

    except Exception as e:
        # Fallback: use original query for both (still better than one call)
        print(f"  [decompose fallback] {e}")
        return query, query


# ── Chunk merging ─────────────────────────────────────────────────────────────

def merge_chunks(chunks_a: list[dict], chunks_b: list[dict]) -> list[dict]:
    """
    Merge two lists of retrieved chunks, deduplicate by chunk_id,
    and keep the best distance score for duplicates.
    Returns up to 8 unique chunks sorted by distance descending.
    """
    seen = {}
    for chunk in chunks_a + chunks_b:
        cid = chunk["chunk_id"]
        if cid not in seen or chunk["distance"] > seen[cid]["distance"]:
            seen[cid] = chunk

    # Sort by distance descending (higher = more relevant for DOT_PRODUCT)
    merged = sorted(seen.values(), key=lambda x: x["distance"], reverse=True)
    return merged[:8]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_comparison(query: str, chunks: list[dict]) -> str:
    """
    Generate a grounded comparison answer using merged chunks from both subjects.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    context = build_context(chunks)
    full_prompt = (
        f"{COMPARISON_SYSTEM_PROMPT}\n\n"
        f"Context chunks (covering both subjects):\n\n{context}\n\n"
        f"Comparison question: {query}"
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=full_prompt
    )
    return response.text


# ── Agent handler ─────────────────────────────────────────────────────────────

def handle(query: str) -> dict:
    """
    Handle a comparison query with two retrieval calls.

    Args:
        query: The user's comparison question

    Returns:
        dict with keys: query, answer, chunks_used, sub_queries, agent
    """
    # Step 1: Decompose into two sub-queries
    query_a, query_b = decompose(query)
    print(f"  [comparison] Sub-query A: {query_a}")
    print(f"  [comparison] Sub-query B: {query_b}")

    # Step 2: Two separate retrieval calls
    embedding_a = embed_query(query_a)
    chunks_a = retrieve(embedding_a, top_k=5)

    embedding_b = embed_query(query_b)
    chunks_b = retrieve(embedding_b, top_k=5)

    # Step 3: Merge and deduplicate chunks
    merged = merge_chunks(chunks_a, chunks_b)

    # Step 4: Generate grounded comparison answer
    answer = generate_comparison(query, merged)

    return {
        "query": query,
        "answer": answer,
        "chunks_used": merged,
        "sub_queries": {"query_a": query_a, "query_b": query_b},
        "agent": "comparison_agent",
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is the difference between a PSA 9 and PSA 10?",
        "Is a graded card worth more than a raw ungraded card?",
    ]

    print("=" * 60)
    print("SpinWheel — Comparison Agent Smoke Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = handle(query)
        print(f"Agent: {result['agent']}")
        print(f"Sub-queries: {result['sub_queries']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nChunks used ({len(result['chunks_used'])} total):")
        for c in result["chunks_used"]:
            print(f"  [{c['chunk_id']}] {c['label']} (dist: {c['distance']:.4f})")
        print()
