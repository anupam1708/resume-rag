# Resume RAG — Hybrid Retrieval over a Resume Corpus

A production-pattern RAG system that answers natural-language questions over a corpus of resumes ("who has Java experience with 10+ years and fintech background?"). Built as a reference implementation demonstrating hybrid retrieval, structured extraction, query routing, and continuous evaluation.

## Architecture

```
┌─────────────┐      ┌──────────────────────────────────────────┐
│   Resumes   │ ───► │  INGESTION                               │
│   (PDFs)    │      │  1. Parse text                           │
└─────────────┘      │  2. LLM structured extraction (Claude)   │
                     │  3. Embed chunks (sentence-transformers) │
                     └──────────┬───────────────────────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │  POSTGRES                         │
                │   • candidates (structured JSON)  │
                │   • chunks  (text + section)      │
                │   • chunk_embeddings (pgvector)   │
                │   • Full-text index (tsvector)    │
                └───────────────────┬───────────────┘
                                    ▼
┌──────────┐    ┌─────────────────────────────────────────────────┐
│  Query   │ ─► │  QUERY ROUTER (LLM classifier)                  │
└──────────┘    │   ├─ Filter query ─► SQL on candidates          │
                │   ├─ Semantic    ─► pgvector cosine             │
                │   └─ Hybrid      ─► BM25 + vector + RRF fusion  │
                └─────────────────────┬───────────────────────────┘
                                      ▼
                        ┌─────────────────────────┐
                        │ GENERATION (Claude)     │
                        │  Answer with citations  │
                        └─────────────────────────┘
```

## Why this design

This is **not** pure vector RAG — and that's the point.

- Pure vector RAG fails on filter queries: "who knows Java" should return Java engineers ranked by depth, not whoever's resume happens to phrase things in vocab close to the query.
- LLM extraction at ingest pays back 100x in query quality — the hard work happens once per resume, every query is cheap.
- Hybrid (BM25 + vector + RRF) catches both exact-match terms ("LangGraph") and semantic equivalents ("agentic systems" matches "multi-agent orchestration").

## Setup (~10 min)

```bash
# 1. Start Postgres + pgvector
docker compose up -d

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY and LANGSMITH_API_KEY

# 4. Schema
psql postgresql://postgres:postgres@localhost:5432/resumes -f schema.sql

# 5. Get data
python scripts/download_data.py  # downloads Kaggle resume dataset, trims to 100

# 6. Ingest
python -m src.ingestion

# 7. Run API
uvicorn src.api:app --reload

# 8. Query
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Who has Java experience with fintech background?"}'
```

## Running evals

```bash
python evals/run_evals.py
```

Outputs recall@10 and precision@10 for three experiments:
- `vector-only` — naive RAG baseline
- `bm25-only` — keyword-only baseline
- `hybrid-rrf` — production config

LangSmith dashboard shows side-by-side comparison.

## Demo script (5 min for interview)

1. **Show the architecture diagram** (this README) — 30 sec
2. **Show a structured-filter query** — `curl` for "Java engineers with 10+ years" → returns ranked SQL filter result. Talk through why this isn't a vector search.
3. **Show a semantic query** — "candidates with experience in event-driven microservices at scale" → hybrid retrieval returns matches even when exact phrasing differs.
4. **Show LangSmith experiment** — vector-only vs hybrid recall@10. Numbers matter.
5. **Show a trace** — click into a single eval failure, walk through retrieved chunks, what the LLM saw, why it answered as it did.

## Tradeoffs and what I'd add with more time

- **Reranking** with a cross-encoder (BGE-reranker) on top of RRF — typically +5-10% recall@10
- **Streaming generation** with citations rendered as they resolve
- **Online evals** on production traces (5% sample) to catch drift
- **Entity resolution** when the same person appears across multiple resume versions
- **Per-tenant isolation** for multi-recruiter SaaS deployment
