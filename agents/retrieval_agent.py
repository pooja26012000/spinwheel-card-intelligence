"""
agents/retrieval_agent.py

Retrieval agent for SpinWheel Card Intelligence.
Handles single-card and single-concept questions.

Flow:
  1. Receives query from orchestrator
  2. Calls rag() pipeline — embed → retrieve top-5 → generate grounded answer
  3. Returns structured result with answer, chunks, and agent metadata
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.rag import rag

# ── Agent handler ─────────────────────────────────────────────────────────────

def handle(query: str, top_k: int = 5) -> dict:
    """
    Handle a single-card or single-concept query.
    Calls the full RAG pipeline and returns structured result.

    Args:
        query:  The user's question
        top_k:  Number of chunks to retrieve (default 5)

    Returns:
        dict with keys: query, answer, chunks_used, agent
    """
    result = rag(query, top_k=top_k)

    return {
        "query": result["query"],
        "answer": result["answer"],
        "chunks_used": result["chunks_used"],
        "agent": "retrieval_agent",
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    test_queries = [
        "What is a PSA 10?",
        "What makes a rookie card valuable?",
    ]

    print("=" * 60)
    print("SpinWheel — Retrieval Agent Smoke Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = handle(query)
        print(f"Agent: {result['agent']}")
        print(f"Answer:\n{result['answer']}")
        print(f"\nChunks used:")
        for c in result["chunks_used"]:
            print(f"  [{c['chunk_id']}] {c['label']} (dist: {c['distance']:.4f})")
        print()
