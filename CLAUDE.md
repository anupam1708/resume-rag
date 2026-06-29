# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

**Resume RAG** is a production-pattern Retrieval-Augmented Generation system that
answers natural-language questions over a corpus of resumes (e.g. *"Who has Java
experience with 10+ years and a fintech background?"*). It is a reference
implementation demonstrating **hybrid retrieval** (BM25 + vector + RRF),
**LLM structured extraction at ingest**, **query routing**, and **continuous
evaluation** with LangSmith.

The central design thesis: this is deliberately **not** pure vector RAG. Filter
queries ("who knows Java") are answered by SQL over LLM-extracted structured
fields, semantic queries by pgvector, and mixed queries by rank fusion. Read
`DESIGN.md` for the full rationale — it is the source of truth for architecture
decisions.

## Tech stack

- **Python 3.10+** (uses `list[dict]` / `float | None` syntax)
- **Postgres 16 + pgvector** (`pgvector/pgvector:pg16` via Docker Compose) — stores
  structured candidate JSONB, text chunks, vector embeddings, and FTS `tsvector`
- **Anthropic Claude** — extraction & generation (`claude-sonnet-4-6`), query
  routing (`claude-haiku-4-5-20251001`)
- **sentence-transformers** — `all-MiniLM-L6-v2`, 384-dim embeddings (local, no API)
- **FastAPI + uvicorn** — serving layer
- **LangSmith** — eval tracking and tracing
- **pandas, pydantic, tqdm, kagglehub** — supporting libs

There is **no test suite, linter config, or CI** in this repo. "Validation" is the
eval harness (`evals/run_evals.py`). Match the existing code style when editing.

## Repository layout

```
src/
  config.py       Central config, loads .env. Model IDs, embed model, dims live here.
  db.py           Postgres connection contextmanager (get_conn) with pgvector registration.
  extraction.py   LLM structured extraction + section-aware chunk building. THE core IP.
  ingestion.py    Pipeline: CSV -> extract -> embed -> Postgres. Run: python -m src.ingestion
  retrieval.py    Query router + 3 retrievers (filter/vector/bm25) + RRF fusion + retrieve().
  graph_build.py  Graph RAG: materializes graph_nodes/graph_edges from candidates. Run: python -m src.graph_build
  graph_retrieval.py  Graph RAG retrievers: graph_search / graph_rank / similar_candidates.
  generation.py   Final cited-answer generation from retrieved candidates.
  api.py          FastAPI app. POST /query, GET /health, GET /similar/{candidate_id}.
scripts/
  download_data.py  Pulls Kaggle resume dataset, stratified-samples to 100 rows.
evals/
  eval_set.json     5 labeled queries with expected categories/skills.
  run_evals.py      LangSmith runner comparing vector-only / bm25-only / hybrid-rrf.
data/
  resumes.csv       Ingestion input. Columns: candidate_id, category, resume_text.
schema.sql          Postgres DDL (run once against the DB).
graph.sql           Graph RAG DDL: graph_nodes/graph_edges + undirected view (run after schema.sql + ingest).
docker-compose.yml  Postgres + pgvector service.
requirements.txt    Python deps.
.env.example        Required env vars (copy to .env).
README.md           Quickstart + demo script.
DESIGN.md           Full system design doc (layered L0–L6). Authoritative.
AWS_DEPLOYMENT.md   Production deployment guide (ECS/RDS/ElastiCache). Aspirational, not deployed.
```

## Data model (schema.sql)

- **`candidates`** — one row per resume. `skills`/`roles`/`education` are JSONB,
  `total_years_experience` numeric, `raw_text` the full resume. GIN index on `skills`
  drives filter queries.
- **`chunks`** — one row per resume *section* (`summary` | `experience` | `skills` |
  `education`). Holds `text`, `embedding vector(384)`, and a generated `fts tsvector`.
  HNSW index for cosine, GIN index for FTS.

Embedding dimension (384) is tied to `all-MiniLM-L6-v2`. **If you change `EMBED_MODEL`
in `config.py`, you must also change `EMBED_DIM` and the `vector(384)` column in
`schema.sql`, then re-ingest.**

## Request flow

1. `POST /query` → `api.py` validates the `Query` model (`query`, `mode`, `top_k`).
2. `retrieve()` in `retrieval.py`:
   - If `mode == "auto"`, the LLM **router** classifies the query into
     `filter` / `semantic` / `hybrid` and extracts filterable entities (skills,
     min_years, domains).
   - `filter` → `filter_search()` SQL over JSONB, scored by summed years (rewards depth).
   - `semantic` → `vector_search()` pgvector cosine, best-chunk-per-candidate.
   - `hybrid` → `vector_search()` + `bm25_search()` fused via `rrf_fuse()` (RRF, k=60).
   - `graph` → `graph_rank()` spreading-activation walk over the knowledge graph
     (multi-hop/relational queries); `graph_hybrid` fuses it with vector via RRF.
   - Results hydrated with candidate metadata.
3. `generate()` in `generation.py` produces the final answer with `[candidate_id]`
   citations, instructed to never fabricate candidates.

## Development workflow

### One-time setup
```bash
docker compose up -d                                   # Postgres + pgvector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # add ANTHROPIC_API_KEY, LANGSMITH_API_KEY
psql "$DATABASE_URL" -f schema.sql                     # create tables/indexes
python scripts/download_data.py                        # fetch + sample data (needs Kaggle creds)
```

### Run
```bash
python -m src.ingestion                                # ingest data/resumes.csv (idempotent)
psql "$DATABASE_URL" -f graph.sql                      # Graph RAG: create graph tables (one-time)
python -m src.graph_build                              # Graph RAG: materialize graph from candidates (idempotent)
uvicorn src.api:app --reload                           # serve API on :8000
python evals/run_evals.py                              # run LangSmith evals
```

> **Graph RAG** is an optional add-on for learning — see `GRAPH_RAG.md`. It derives a
> knowledge graph from the already-extracted candidate fields (no new LLM calls) and
> adds `graph` / `graph_hybrid` retrieval modes plus `GET /similar/{candidate_id}`.
> It requires `graph.sql` + `python -m src.graph_build` after ingestion; the core
> filter/vector/hybrid pipeline works without it.

### Example query
```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Who has Java experience with fintech background?"}'
```

## Conventions & gotchas

- **Imports are absolute from `src.`** (e.g. `from src.config import ...`). Run modules
  as packages (`python -m src.ingestion`), not as scripts, so imports resolve.
- **All config flows through `src/config.py`.** Don't read env vars or hard-code model
  IDs elsewhere — add them here. `DATABASE_URL` and `ANTHROPIC_API_KEY` are required
  (will `KeyError` if missing); `LANGSMITH_API_KEY` is optional.
- **DB access goes through `get_conn()`** (contextmanager in `db.py`) which commits on
  success, rolls back on exception, and registers pgvector. Don't open raw connections.
- **Ingestion is idempotent** — it skips candidates already present by `candidate_id`.
  To re-ingest a changed candidate, delete its row first (cascades to chunks).
- **Model choice is intentional**: Sonnet for extraction/generation (quality matters),
  Haiku for routing (cheap, high-volume classification). Preserve this split.
- **LLM JSON parsing is defensive** — extraction strips code fences and has a
  self-correction retry; the router strips fences too. Keep this robustness if you
  touch prompt/parse code.
- **Use the latest Claude models** when adding LLM calls. Current IDs live in
  `config.py`. Don't downgrade to older model IDs.
- **Eval ground truth is category-based and approximate** (uses the resume CSV's
  `category` column), good enough to show hybrid beats the baselines. It is not
  hand-labeled relevance — see the note at the top of `run_evals.py`.

## Where to make changes

- New retrieval strategy → `retrieval.py` (and wire into `retrieve()` dispatch + `mode` options in `api.py`).
- Graph RAG (nodes/edges, traversal) → `graph.sql`, `graph_build.py`, `graph_retrieval.py`; see `GRAPH_RAG.md`.
- Change extracted fields → update `EXTRACTION_PROMPT` and `build_chunks()` in
  `extraction.py`, the `candidates` columns/insert in `ingestion.py`, and `schema.sql`.
- New API surface → `api.py`.
- Tune embeddings/models → `config.py` (mind the dimension coupling above).
- New eval queries/metrics → `evals/eval_set.json` and `evals/run_evals.py`.

## Git workflow

- Active development branch for this work: `claude/claude-md-docs-7l18b4`.
- Commit with clear messages; push with `git push -u origin <branch>`.
- Do **not** open a pull request unless explicitly asked.
- `.env`, `.venv/`, `__pycache__/`, and `.claude/` are gitignored — never commit secrets.

## Further reading

- `DESIGN.md` — authoritative layered architecture (L0 data → L6 evaluation),
  design decisions, scaling path, and known limitations.
- `AWS_DEPLOYMENT.md` — production deployment blueprint (ECS Fargate, RDS, ElastiCache,
  CI/CD, IaC, cost). Reference design, not currently deployed.
- `GRAPH_RAG.md` — Graph RAG add-on (knowledge-graph retriever): design, setup, and
  extension ideas. Optional, for learning.
- `README.md` — quickstart and the interview demo script.
