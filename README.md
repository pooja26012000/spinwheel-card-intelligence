# SpinWheel Card Intelligence

An agentic RAG (Retrieval-Augmented Generation) system for sports card grading and valuation. Built on GCP with a live API and Next.js frontend.

**Live demo:** [spinwheel-card-intelligence.vercel.app](https://spinwheel-card-intelligence.vercel.app)
**API:** [spinwheel-api-119676722067.us-central1.run.app](https://spinwheel-api-119676722067.us-central1.run.app)

---

## What It Does

Users can ask natural language questions about sports card grading and valuation. The system retrieves relevant chunks from a curated corpus using vector similarity search, then generates grounded answers using Gemini 2.5 Flash.

It supports two query modes:
- **Single queries** — e.g. "What is a PSA 10?"
- **Comparison queries** — e.g. "Compare PSA 9 and PSA 10 grading standards" (dual retrieval via comparison agent)

An orchestrator agent classifies each incoming query and routes it to the appropriate agent automatically.

---

## Architecture

```
User Query
    ↓
Next.js Frontend (Vercel)
    ↓
FastAPI Backend (Cloud Run)
    ↓
Orchestrator Agent (Gemini 2.5 Flash)
    ↓ classifies: single vs comparison
    ↓
Retrieval Agent          Comparison Agent
    ↓                         ↓
Vertex AI Vector Search   2x Vector Search
    ↓                         ↓
Top-5 chunks            Merged chunks
    ↓                         ↓
        Gemini 2.5 Flash (generation)
                ↓
            Answer + Sources
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Google `text-embedding-004` (768 dimensions) |
| Vector Search | Vertex AI Vector Search |
| Generation | Gemini 2.5 Flash via `google-genai` |
| Backend | FastAPI on Cloud Run |
| Frontend | Next.js on Vercel |
| CI/CD | GitHub Actions |

---

## Eval Results

Evaluated on 30 questions across grading standards, valuation, and card history categories.

| Metric | Baseline (RAG only) | After Agents | Delta |
|---|---|---|---|
| Groundedness | 40.0% | 39.2% | -0.8% |
| Relevance | 24.2% | 29.2% | **+5.0%** |

**Key finding:** Relevance improved 5% from dual retrieval on comparison queries. Groundedness remained flat — the eval correctly diagnosed this as a corpus coverage gap in the valuation and card history categories, not an agent problem.

---

## Project Structure

```
spinwheel-card-intelligence/
├── corpus/
│   ├── scraper.py          # Web scraping
│   ├── chunker.py          # Heading → paragraph → sentence chunking
│   ├── merge_psa.py        # PSA document merging
│   ├── raw/                # 9 source documents
│   └── chunks/
│       └── chunks.jsonl    # 159 chunks (512 tokens, 50 overlap)
├── pipeline/
│   ├── embed_and_index.py  # Embeds chunks, uploads to Vertex AI
│   ├── test_retrieval.py   # Retrieval smoke tests
│   └── rag.py              # Core RAG: embed → retrieve → generate
├── agents/
│   ├── orchestrator.py     # Classifies single vs comparison queries
│   ├── retrieval_agent.py  # Single card questions
│   └── comparison_agent.py # Dual retrieval + merge
├── api/
│   └── main.py             # FastAPI: /health, /query, /compare
├── eval/
│   ├── eval_harness_baseline.py    # 30-question baseline eval
│   ├── eval_harness_agents.py      # Same eval via orchestrator
│   ├── results_baseline.json       # G=40%, R=24.2%
│   └── results_after_agents.json   # G=39.2%, R=29.2%
├── frontend/
│   └── src/app/
│       ├── layout.tsx
│       ├── page.tsx        # Chat UI with source chips
│       └── globals.css
├── .github/workflows/
│   └── deploy.yml          # Auto-deploy Cloud Run on push to main
└── Dockerfile
```

---

## GCP Infrastructure

| Resource | Value |
|---|---|
| Project | `spinwheel-card-intel-2630` |
| Region | `us-central1` |
| Vector Search Index | `4575804555966545920` |
| Vector Search Endpoint | `1807027569040556032` |
| Cloud Run Service | `spinwheel-api` |
| GCS Bucket | `spinwheel-card-intel-2630-vectors` |

---

## Running Locally

**Prerequisites:** Python 3.11+, Node.js 18+, GCP service account key

**Backend:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/spinwheel-key.json
export GEMINI_API_KEY=your_key_here
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8080
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## API Endpoints

**`POST /query`**
```json
{ "question": "What is a PSA 10?" }
```

**`POST /compare`**
```json
{ "question": "Compare PSA 9 and PSA 10 grading standards" }
```

**`GET /health`**
```json
{ "status": "ok" }
```

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow that:
1. Authenticates with GCP using a service account key stored as a GitHub secret
2. Builds a container image via Cloud Build
3. Deploys to Cloud Run

The service account requires `roles/run.admin`, `roles/artifactregistry.admin`, `roles/cloudbuild.builds.editor`, and `roles/iam.serviceAccountUser` on the compute service account.

---

## Authentication Notes

Two separate auth systems are in use:

- **Vector Search** — `google.auth.default()` with `GOOGLE_APPLICATION_CREDENTIALS`
- **Gemini generation** — `google-genai` client with `GEMINI_API_KEY` (Vertex AI publisher models are blocked on free tier)