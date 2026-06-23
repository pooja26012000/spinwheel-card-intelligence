"""
pipeline/rag.py

RAG pipeline: embed query → retrieve top-5 chunks → generate grounded answer with Gemini 2.5 Flash.
This is the core of SpinWheel Card Intelligence — all agents and API endpoints call into this.

Generation uses Google AI (google-genai) client with gemini-2.5-flash.
Retrieval uses Vertex AI Vector Search via REST (unchanged from test_retrieval.py).
"""

import json
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import google.auth
import google.auth.transport.requests
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────

METADATA_FILE = Path(__file__).parent / "index_metadata.json"
metadata = json.loads(METADATA_FILE.read_text())

PROJECT_ID        = metadata["project_id"]
REGION            = metadata["region"]
ENDPOINT_ID       = metadata["endpoint_id"]
DEPLOYED_INDEX_ID = metadata["deployed_index_id"]
EMBEDDING_MODEL   = metadata["embedding_model"]
TOP_K             = metadata["top_k"]

PUBLIC_DOMAIN = "1086430222.us-central1-119676722067.vdb.vertexai.goog"

CHUNKS_FILE = Path(__file__).parent.parent / "corpus" / "chunks" / "chunks.jsonl"
chunks_by_id: dict = {}
for _line in CHUNKS_FILE.read_text().splitlines():
    if _line:
        _chunk = json.loads(_line)
        chunks_by_id[_chunk["chunk_id"]] = _chunk

GENERATION_MODEL = "gemini-2.5-flash"

# Load API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are SpinWheel Card Intelligence, an expert assistant for sports card grading and valuation.

Answer the user's question using ONLY the context chunks provided below.
- Be specific and cite evidence from the chunks.
- If the chunks do not contain enough information to answer, say so explicitly — do not hallucinate.
- Keep answers concise but complete (3-6 sentences unless the question demands more).
- At the end of your answer, list the chunk IDs you drew from, like: Sources: [chunk_id_1, chunk_id_2]
"""

# ── Step 1: Embed query ───────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Embed a query string using text-embedding-004 with RETRIEVAL_QUERY task type."""
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    result = model.get_embeddings([TextEmbeddingInput(query, "RETRIEVAL_QUERY")])
    return result[0].values


# ── Step 2: Retrieve top-k chunks ─────────────────────────────────────────────

def retrieve(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """
    Query Vertex AI Vector Search via REST API with public endpoint.
    Returns full chunk dicts (not truncated) for generation context.
    """
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    endpoint_url = (
        f"https://{PUBLIC_DOMAIN}/v1/projects/{PROJECT_ID}/locations/{REGION}/"
        f"indexEndpoints/{ENDPOINT_ID}:findNeighbors"
    )

    body = json.dumps({
        "deployed_index_id": DEPLOYED_INDEX_ID,
        "queries": [{
            "datapoint": {"feature_vector": query_embedding},
            "neighbor_count": top_k
        }]
    }).encode("utf-8")

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    req = urllib.request.Request(
        endpoint_url,
        data=body,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(req, context=ssl_context) as response:
        result = json.loads(response.read())

    neighbors = result["nearestNeighbors"][0]["neighbors"]

    chunks = []
    for neighbor in neighbors:
        chunk_id = neighbor["datapoint"]["datapointId"]
        distance = neighbor["distance"]
        chunk = chunks_by_id.get(chunk_id, {})
        chunks.append({
            "chunk_id": chunk_id,
            "distance": distance,
            "label": chunk.get("label", "unknown"),
            "source_file": chunk.get("source_file", ""),
            "text": chunk.get("text", ""),  # full text, not truncated
        })

    return chunks


# ── Step 3: Build context string for Gemini ───────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | ID: {chunk['chunk_id']} | Source: {chunk['label']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Step 4: Generate grounded answer with Gemini 2.5 Flash ───────────────────

def generate(query: str, chunks: list[dict]) -> str:
    """Send retrieved chunks + query to Gemini 2.5 Flash and return grounded answer."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    context = build_context(chunks)
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context chunks:\n\n{context}\n\n"
        f"Question: {query}"
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=full_prompt
    )
    return response.text


# ── Step 5: Full RAG pipeline (entry point for agents + API) ──────────────────

def rag(query: str, top_k: int = TOP_K) -> dict:
    """
    Full pipeline: embed → retrieve → generate.
    Returns structured output consumed by agents, eval harness, and API.
    """
    query_embedding = embed_query(query)
    chunks = retrieve(query_embedding, top_k=top_k)
    answer = generate(query, chunks)

    return {
        "query": query,
        "answer": answer,
        "chunks_used": [
            {
                "chunk_id": c["chunk_id"],
                "label": c["label"],
                "distance": c["distance"],
                "text": c["text"]
            }
            for c in chunks
        ]
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is the difference between a PSA 9 and PSA 10?",
        "What makes a sports card valuable?",
    ]

    print("=" * 60)
    print("SpinWheel — RAG Pipeline Smoke Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = rag(query)
        print(f"Answer:\n{result['answer']}")
        print(f"\nChunks used:")
        for c in result["chunks_used"]:
            print(f"  [{c['chunk_id']}] {c['label']} (dist: {c['distance']:.4f})")
        print()
