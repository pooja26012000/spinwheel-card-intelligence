"""
pipeline/embed_and_index.py (v2)

Adds embedding cache so reruns skip re-embedding.
Step 1: Embed all chunks (skipped if cache exists)
Step 2: Upload embeddings to GCS
Step 3: Create Vertex AI Vector Search index
Step 4: Deploy index to endpoint
"""

import json
import os
import time
from pathlib import Path

from google.cloud import aiplatform, storage

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "spinwheel-card-intel-2630")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET_NAME = f"{PROJECT_ID}-vectors"
GCS_PREFIX = "embeddings"
INDEX_DISPLAY_NAME = "spinwheel-card-index"
ENDPOINT_DISPLAY_NAME = "spinwheel-card-endpoint"

CHUNKS_FILE = Path(__file__).parent.parent / "corpus" / "chunks" / "chunks.jsonl"
CACHE_FILE = Path(__file__).parent / "embeddings_cache.jsonl"
INDEX_METADATA_FILE = Path(__file__).parent / "index_metadata.json"

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSIONS = 768
TOP_K = 5


def embed_chunks(chunks: list[dict]) -> list[dict]:
    # Return from cache if it exists
    if CACHE_FILE.exists():
        print(f"\nLoading embeddings from cache: {CACHE_FILE}")
        cached = [json.loads(l) for l in CACHE_FILE.read_text().splitlines() if l]
        print(f"  Loaded {len(cached)} cached embeddings.")
        return cached

    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    import vertexai
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    print(f"\nEmbedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embedded = []
    batch_size = 20

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [TextEmbeddingInput(c["text"], "RETRIEVAL_DOCUMENT") for c in batch]
        result = model.get_embeddings(texts)
        for chunk, r in zip(batch, result):
            embedded.append({"id": chunk["chunk_id"], "embedding": r.values})
        print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(0.5)

    # Save cache immediately
    CACHE_FILE.write_text("\n".join(json.dumps(e) for e in embedded))
    print(f"  Embeddings cached → {CACHE_FILE}")
    return embedded


def upload_to_gcs(embedded: list[dict]) -> str:
    storage_client = storage.Client(project=PROJECT_ID)

    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"\nUsing existing GCS bucket: gs://{BUCKET_NAME}")
    except Exception:
        bucket = storage_client.create_bucket(BUCKET_NAME, location=REGION)
        print(f"\nCreated GCS bucket: gs://{BUCKET_NAME}")

    local_path = Path(__file__).parent / "embeddings_upload.jsonl"
    with open(local_path, "w") as f:
        for item in embedded:
            f.write(json.dumps({"id": item["id"],
                                "embedding": item["embedding"]}) + "\n")

    blob = bucket.blob(f"{GCS_PREFIX}/embeddings.json")
    blob.upload_from_filename(str(local_path))
    local_path.unlink()

    gcs_uri = f"gs://{BUCKET_NAME}/{GCS_PREFIX}/"
    print(f"Uploaded {len(embedded)} embeddings → {gcs_uri}")
    return gcs_uri


def create_index(gcs_uri: str) -> str:
    from google.cloud import aiplatform_v1
    from google.cloud.aiplatform_v1.types import Index

    client = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )

    parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    index = Index(
        display_name=INDEX_DISPLAY_NAME,
        description="SpinWheel Card Intelligence RAG index",
        metadata_schema_uri="gs://google-cloud-aiplatform/schema/matchingengine/metadata/nearest_neighbor_search_1.0.0.yaml",
        metadata={
            "config": {
                "dimensions": EMBEDDING_DIMENSIONS,
                "approximateNeighborsCount": TOP_K,
                "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
                "algorithm_config": {
                    "treeAhConfig": {
                        "leafNodeEmbeddingCount": 500,
                        "leafNodesToSearchPercent": 7,
                    }
                },
                "contentsDeltaUri": gcs_uri,
            }
        },
    )

    print(f"\nCreating Vertex AI Vector Search index (10-20 mins)...")
    operation = client.create_index(parent=parent, index=index)
    print("Waiting for index creation to complete...")
    result = operation.result(timeout=1800)
    print(f"Index created: {result.name}")
    return result.name


def deploy_index(index_resource_name: str) -> tuple[str, str]:
    print(f"\nCreating endpoint and deploying index (10-15 mins)...")

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
    )

    deployed_index_id = "spinwheel_card_deployed"
    endpoint.deploy_index(
        index=aiplatform.MatchingEngineIndex(index_resource_name),
        deployed_index_id=deployed_index_id,
    )

    print(f"Deployed. Endpoint: {endpoint.resource_name}")
    return endpoint.resource_name, deployed_index_id


def save_metadata(index_resource_name, endpoint_resource_name, deployed_index_id):
    metadata = {
        "project_id": PROJECT_ID,
        "region": REGION,
        "index_resource_name": index_resource_name,
        "endpoint_resource_name": endpoint_resource_name,
        "deployed_index_id": deployed_index_id,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "top_k": TOP_K,
        "bucket_name": BUCKET_NAME,
    }
    INDEX_METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata saved → {INDEX_METADATA_FILE}")


def run():
    print("=" * 60)
    print("SpinWheel — Embed & Index Pipeline v2 (with cache)")
    print("=" * 60)

    chunks = [json.loads(l) for l in CHUNKS_FILE.read_text().splitlines() if l]
    print(f"Loaded {len(chunks)} chunks.")

    embedded = embed_chunks(chunks)
    gcs_uri = upload_to_gcs(embedded)
    index_resource_name = create_index(gcs_uri)
    endpoint_resource_name, deployed_index_id = deploy_index(index_resource_name)
    save_metadata(index_resource_name, endpoint_resource_name, deployed_index_id)

    print("\n" + "=" * 60)
    print("Pipeline complete. Index is live and queryable.")
    print("=" * 60)


if __name__ == "__main__":
    run()
