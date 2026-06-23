"""
pipeline/test_retrieval.py
Runs a test query against the deployed Vertex AI Vector Search index.
Proves end-to-end retrieval is working before building the RAG pipeline.
"""

import json
from pathlib import Path

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.cloud import aiplatform_v1

import ssl
import certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Load metadata
METADATA_FILE = Path(__file__).parent / "index_metadata.json"
metadata = json.loads(METADATA_FILE.read_text())

PROJECT_ID = metadata["project_id"]
REGION = metadata["region"]
ENDPOINT_ID = metadata["endpoint_id"]
DEPLOYED_INDEX_ID = metadata["deployed_index_id"]
EMBEDDING_MODEL = metadata["embedding_model"]
TOP_K = metadata["top_k"]

# Load chunks for lookup
CHUNKS_FILE = Path(__file__).parent.parent / "corpus" / "chunks" / "chunks.jsonl"
chunks_by_id = {}
for line in CHUNKS_FILE.read_text().splitlines():
    if line:
        chunk = json.loads(line)
        chunks_by_id[chunk["chunk_id"]] = chunk


def embed_query(query: str) -> list[float]:
    """Embed a query string using the same model used for indexing."""
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    result = model.get_embeddings([TextEmbeddingInput(query, "RETRIEVAL_QUERY")])
    return result[0].values

def retrieve(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """Query Vertex AI Vector Search public endpoint via REST."""
    import ssl
    import certifi
    import google.auth
    import google.auth.transport.requests
    import urllib.request

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    PUBLIC_DOMAIN = "1086430222.us-central1-119676722067.vdb.vertexai.goog"
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
    results = []
    for neighbor in neighbors:
        chunk_id = neighbor["datapoint"]["datapointId"]
        distance = neighbor["distance"]
        chunk = chunks_by_id.get(chunk_id, {})
        results.append({
            "chunk_id": chunk_id,
            "distance": distance,
            "label": chunk.get("label", "unknown"),
            "text": chunk.get("text", "")[:300],
        })

    return results


def run():
    test_queries = [
        "What is the difference between a PSA 9 and PSA 10?",
        "What makes a sports card valuable?",
        "What is a rookie card?",
    ]

    print("=" * 60)
    print("SpinWheel Card Intelligence — Retrieval Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        query_embedding = embed_query(query)
        results = retrieve(query_embedding)

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result['chunk_id']}")
            print(f"      Source: {result['label']}")
            print(f"      Distance: {result['distance']:.4f}")
            print(f"      Preview: {result['text'][:150]}...")
            print()


if __name__ == "__main__":
    run()
