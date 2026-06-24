"""
api/main.py

FastAPI backend for SpinWheel Card Intelligence.
Exposes the full agentic RAG pipeline via HTTP endpoints.

Endpoints:
  GET  /health          — health check for Cloud Run
  POST /query           — routes through orchestrator (auto single vs comparison)
  POST /compare         — forces comparison agent explicitly

Usage (local):
  uvicorn api.main:app --reload --port 8080

Usage (production):
  Deployed on Cloud Run via Dockerfile
"""

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.orchestrator import route
from agents.comparison_agent import handle as comparison_handle

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SpinWheel Card Intelligence API",
    description="Agentic RAG system for sports card grading and valuation",
    version="1.0.0"
)

# CORS — allow Next.js frontend on Vercel to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class ChunkUsed(BaseModel):
    chunk_id: str
    label: str
    distance: float
    text: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    agent: str
    routing_intent: str
    chunks_used: list[dict]
    latency_ms: float

class CompareRequest(BaseModel):
    question: str

class CompareResponse(BaseModel):
    question: str
    answer: str
    agent: str
    sub_queries: dict
    chunks_used: list[dict]
    latency_ms: float

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint for Cloud Run."""
    return {"status": "ok", "service": "spinwheel-card-intelligence"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main query endpoint. Routes through orchestrator automatically.
    Single-card questions → retrieval agent.
    Comparison questions → comparison agent.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()

    try:
        result = route(request.question.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        question=result["query"],
        answer=result["answer"],
        agent=result.get("agent", "unknown"),
        routing_intent=result.get("routing", {}).get("intent", "unknown"),
        chunks_used=result["chunks_used"],
        latency_ms=round(latency_ms, 2)
    )


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    """
    Explicit comparison endpoint. Always uses comparison agent.
    Use this when the frontend has a dedicated comparison mode.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()

    try:
        result = comparison_handle(request.question.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison pipeline error: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    return CompareResponse(
        question=result["query"],
        answer=result["answer"],
        agent=result.get("agent", "comparison_agent"),
        sub_queries=result.get("sub_queries", {}),
        chunks_used=result["chunks_used"],
        latency_ms=round(latency_ms, 2)
    )
