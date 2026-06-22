# Resume RAG — System Design Document

> Detailed architecture for a hybrid-retrieval RAG system that answers natural-language
> questions over a resume corpus. This document covers every layer end to end: data,
> ingestion, storage, retrieval, generation, serving, and evaluation — plus the design
> rationale, failure modes, and scaling path for each.
>
> For setup/quickstart see [README.md](README.md). This document is the *why* and the *how it fits together*.

---

## 1. System overview

The system turns a pile of unstructured resumes into a queryable knowledge base and
answers recruiter questions like *"Who has Java experience with 10+ years and a fintech
background?"* with cited, grounded answers.

The core thesis: **pure vector RAG is the wrong default for this domain.** Recruiter
queries split into two shapes — hard *filters* ("knows Java", "5+ years") and soft
*semantic* intent ("built large-scale distributed systems"). A single embedding search
serves neither well. So the system does expensive structured extraction **once** at
ingest time, then routes each query to the cheapest retrieval strategy that fits it.

### Layered view

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L0  DATA SOURCE          Kaggle resume CSV  →  data/resumes.csv (100 rows)     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1  INGESTION            CSV → LLM extraction → section chunks → embeddings    │
│                          src/ingestion.py · src/extraction.py                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2  STORAGE              Postgres 16 + pgvector                                │
│                          candidates (JSONB)  ·  chunks (vector + tsvector)     │
│                          schema.sql · src/db.py                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  RETRIEVAL            Router → {filter | semantic | hybrid} → RRF fuse      │
│                          src/retrieval.py                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ L4  GENERATION           Claude answer with per-candidate citations           │
│                          src/generation.py                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L5  SERVING              FastAPI  POST /query · GET /health                    │
│                          src/api.py                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ L6  EVALUATION           LangSmith: vector-only vs bm25-only vs hybrid-rrf     │
│                          evals/run_evals.py · evals/eval_set.json              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### End-to-end request path

```
client ──POST /query──► api.py
                         │
                         ├─► retrieve(query, mode="auto")            [retrieval.py]
                         │     ├─ route_query()      Haiku classifier → {mode, skills, min_years, domains}
                         │     ├─ filter_search()    SQL over candidates.skills JSONB
                         │     ├─ vector_search()    pgvector cosine over chunks
                         │     ├─ bm25_search()      Postgres FTS over chunks
                         │     ├─ rrf_fuse()         reciprocal-rank fusion (hybrid only)
                         │     └─ hydrate            join candidate metadata
                         │
                         └─► generate(query, candidates)            [generation.py]
                               └─ Sonnet → grounded answer with [c_0042] citations
                         ◄── {query, mode, route, candidates[], answer}
```

### Component / model map

| Concern            | Choice                               | Where                |
|--------------------|--------------------------------------|----------------------|
| Extraction LLM     | `claude-sonnet-4-6`                  | `config.EXTRACT_MODEL` |
| Generation LLM     | `claude-sonnet-4-6`                  | `config.GEN_MODEL`     |
| Query router LLM   | `claude-haiku-4-5` (cheap classify)  | `config.ROUTER_MODEL`  |
| Embeddings         | `all-MiniLM-L6-v2`, 384-dim          | `config.EMBED_MODEL`   |
| Vector index       | pgvector HNSW, cosine                | `schema.sql`           |
| Keyword index      | Postgres FTS (`tsvector` + GIN)      | `schema.sql`           |
| Structured filter  | JSONB + GIN                          | `schema.sql`           |
| Serving            | FastAPI + Uvicorn                    | `src/api.py`           |
| Eval               | LangSmith experiments                | `evals/`               |

---

## L0 · Data layer

**Source.** Kaggle "Resume Dataset" (`snehaanbhawal/resume-dataset`), 2,484 resumes across
24 occupational categories (`ID, Resume_str, Resume_html, Category`).

**Preparation** ([scripts/download_data.py](scripts/download_data.py)). Downloads via
`kagglehub`, then **stratified-samples 5 resumes per category** (deterministic,
`random_state=42`) and trims to 100 rows. Stratification is deliberate: it keeps the eval
set's relevance judgments interesting — a random sample could over-represent one category
and make retrieval look artificially easy.

**Canonical output** — `data/resumes.csv`:

| column         | meaning                                  |
|----------------|------------------------------------------|
| `candidate_id` | synthetic stable id, `c_0000`…`c_0099`   |
| `category`     | occupational label (used as eval ground truth) |
| `resume_text`  | raw plain-text resume                    |

**Design note.** The `category` column is *only* used as approximate ground truth for
evals — it is never fed to retrieval or generation. This keeps the eval honest: the system
never sees the label it's being graded against.

---

## L1 · Ingestion layer

`CSV row → structured JSON (LLM) → section chunks → embeddings → Postgres`
Entry point: [src/ingestion.py](src/ingestion.py) · `python -m src.ingestion`

### 1.1 Structured extraction — the core IP

[src/extraction.py](src/extraction.py) sends each resume to Claude Sonnet with a strict
JSON-schema prompt that pulls out:

- `name`, `total_years_experience`, `summary`
- `skills[]` — each with `name`, `years`, `last_used_year`, `proficiency`
- `roles[]` — `title`, `company`, `start_year`, `end_year`, `domain`
- `education[]` — `degree`, `institution`, `year`

Two prompt-engineering decisions carry most of the system's query quality:

1. **Canonical-name normalization.** The prompt forces `"JavaScript" not "JS"`,
   `"Amazon Web Services" → "AWS"`, `"PostgreSQL" not "postgres"`. This is what makes the
   downstream JSONB filter (`WHERE LOWER(s->>'name') = ANY(...)`) reliable — exact-match
   filters only work if the corpus speaks one vocabulary. The normalization burden is paid
   once at ingest, not pushed onto every query.
2. **Constrained domain taxonomy.** Roles are classified into a fixed set
   (`fintech, banking, healthcare, …, other`) so "fintech experience" queries match
   consistently instead of fighting free-text drift.

**Self-correction loop.** Extraction must return valid JSON or the row is unusable. The
flow is: strip stray markdown fences → `json.loads` → on `JSONDecodeError`, send the
broken text back to the model with "fix this JSON" → parse again. This is a pragmatic
reliability hedge against the one-in-N malformed generation.

> Hardening note: the second parse is not itself wrapped, so a twice-broken response would
> raise. In ingestion that's caught per-row (the resume is skipped with a log line); for a
> production pipeline you'd add a dead-letter record. See §8.

### 1.2 Section-aware chunking

`build_chunks()` does **not** do naive fixed-window splitting. It emits one chunk per
*semantic unit*, tagged with a `section`:

| section      | content                                                          |
|--------------|------------------------------------------------------------------|
| `summary`    | the 2–3 sentence professional summary                            |
| `experience` | one chunk **per role**: `"<title> at <company> (2019-present). Domain: fintech."` |
| `skills`     | one aggregated line: `"Skills: Java (6y), Spring Boot (4y)…"`     |
| `education`  | one chunk per degree                                             |

Why this matters: embeddings of clean, role-scoped text retrieve far better than
embeddings of an arbitrary 512-token slice that straddles two jobs. The chunk text is
*synthesized from the extracted structure*, not cut from the raw resume — so the embedding
model sees normalized, context-complete sentences.

### 1.3 Embedding + write

- Model: `all-MiniLM-L6-v2` (384-dim) — small, fast, CPU-friendly, strong for short text.
  Loaded once (lazy singleton in retrieval; explicit in ingestion).
- All chunk texts for a resume are encoded in one batch.
- Writes are wrapped in a per-candidate transaction; **idempotent** via a
  `SELECT 1 FROM candidates WHERE candidate_id = %s` guard, so re-running ingestion resumes
  rather than duplicates.

### Ingestion data flow

```
resume_text
   │ extract()              [Sonnet + self-correct]
   ▼
{name, years, skills[], roles[], education[], summary}
   │                         │ build_chunks()
   ▼ INSERT candidates       ▼
candidates row          [{section, text}, …]
                             │ embedder.encode(batch)
                             ▼
                        INSERT chunks (text, section, embedding)
```

---

## L2 · Storage layer

Single Postgres 16 instance with the `pgvector` extension
([docker-compose.yml](docker-compose.yml) uses `pgvector/pgvector:pg16`). One store backs
all three retrieval modes — no separate vector DB, no separate search engine. That's the
key operational simplification: **one system to run, back up, and reason about.**

Connection management ([src/db.py](src/db.py)): a `get_conn()` context manager opens a
`psycopg` connection, registers the pgvector type adapter, and commits-or-rolls-back
around the `with` block.

### Schema ([schema.sql](schema.sql))

**`candidates`** — one row per person; the structured/filter surface.

| column                   | type        | role                                   |
|--------------------------|-------------|----------------------------------------|
| `candidate_id`           | TEXT PK     | stable id                              |
| `name`                   | TEXT        |                                        |
| `total_years_experience` | NUMERIC     | filter predicate                       |
| `skills`                 | JSONB       | `[{name, years, …}]` — filter + scoring |
| `roles` / `education`    | JSONB       | hydration / display                    |
| `raw_text`               | TEXT        | provenance / future re-extraction      |

Indexes: **GIN on `skills`** (containment / array-element queries),
B-tree on `total_years_experience` (range filters).

**`chunks`** — one row per resume section; the retrieval surface.

| column         | type           | role                                      |
|----------------|----------------|-------------------------------------------|
| `chunk_id`     | BIGSERIAL PK   |                                           |
| `candidate_id` | TEXT FK        | `ON DELETE CASCADE` → cleanup is automatic |
| `section`      | TEXT           | `summary|experience|skills|education`     |
| `text`         | TEXT           | chunk text                                |
| `embedding`    | `vector(384)`  | semantic search                           |
| `fts`          | `tsvector`     | **`GENERATED ALWAYS … STORED`** from `text` |

Indexes: **HNSW** on `embedding` (`vector_cosine_ops`) for ANN cosine search;
**GIN** on `fts` for keyword search; B-tree on `candidate_id` for cheap candidate→chunks
joins.

### Storage design decisions

- **`fts` is a generated column** — it can never drift from `text`; there's no application
  code that can forget to update the search index. The database owns that invariant.
- **HNSW over IVFFlat** — better recall/latency at this scale and no `train`/`lists`
  tuning step (pgvector ≥ 0.5).
- **Cosine distance** (`<=>` with `vector_cosine_ops`) matches MiniLM's normalized output.
- **JSONB, not normalized skill tables** — skills are read as a unit, the corpus is small,
  and a GIN index makes containment queries fast. Normalizing would add joins for no win
  at this scale (revisit at millions of candidates — see §7).

---

## L3 · Retrieval layer

The heart of the system ([src/retrieval.py](src/retrieval.py)). A query is classified,
dispatched to one of three strategies, and the result is hydrated with candidate metadata.

### 3.1 Query router

`route_query()` calls **Haiku** (cheap, fast classification) to label the query and pull
out structured entities:

```json
{"mode": "filter|semantic|hybrid", "skills": [...], "min_years": 5, "domains": [...]}
```

| mode       | when                                              | example                                            |
|------------|---------------------------------------------------|----------------------------------------------------|
| `filter`   | concrete skill / years / role constraint          | "who knows Python", "Java engineers, 5+ years"     |
| `semantic` | abstract capability, paraphrase-heavy             | "built large-scale distributed systems"            |
| `hybrid`   | both a hard constraint **and** abstract intent    | "senior Java engineers with fintech + team lead"   |

Using a small model here is a deliberate cost decision: routing runs on every query, so it
must be cheap; the heavy models run only where their quality pays off (extraction once,
generation once per query).

### 3.2 The three retrievers

**Filter** — `filter_search()`. Pure SQL over `candidates`. Scores each candidate by the
**sum of `years` across matching skills**, so depth wins ("10 years of Java" outranks
"touched Java once"). `min_years` becomes a `WHERE total_years_experience >= …` predicate.
This is the path pure-vector RAG gets wrong — it returns ranked, explainable matches
instead of vocabulary-proximity guesses.

**Semantic** — `vector_search()`. Encodes the query, runs pgvector cosine over `chunks`,
and aggregates to the candidate level with **`MAX(1 - (embedding <=> q))`** — a candidate's
score is their single best-matching chunk. (Best-chunk, not mean, so one strongly relevant
role isn't diluted by unrelated sections.)

**BM25-like** — `bm25_search()`. Postgres FTS: `ts_rank_cd(fts, plainto_tsquery(...))`,
again aggregated by best chunk per candidate. Catches exact tokens an embedding may smooth
over (specific tool/framework names like "LangGraph").

> Note on terminology: this is Postgres `ts_rank_cd`, not literal Okapi BM25. It's the
> keyword-precision half of the hybrid; the eval label "bm25-only" names the role, not the
> exact formula.

### 3.3 Reciprocal Rank Fusion

`rrf_fuse()` combines the vector and keyword rankings:

```
score(d) = Σ_rankings  1 / (k_const + rank_in_ranking(d))      # k_const = 60
```

RRF is **score-agnostic — only ranks matter.** That's exactly right here: cosine
similarity (~0–1) and `ts_rank_cd` (unbounded) live on incomparable scales, so you can't
just add them. RRF sidesteps normalization entirely and is robust to one retriever
producing wild score magnitudes. A document ranked highly by *both* retrievers floats to
the top; agreement is rewarded structurally.

For hybrid, each retriever is run with `k = top_k * 2` to give fusion a deeper pool to
reconcile before truncating to `top_k`.

### 3.4 Dispatch + hydration — `retrieve()`

```
retrieve(query, mode="auto", top_k=10)
  ├─ mode=="auto" → route_query()  (else trust caller-supplied mode)
  ├─ filter   → filter_search(skills, min_years, domains)
  ├─ semantic → vector_search(query)
  └─ hybrid   → rrf_fuse([vector_search(·, 2k), bm25_search(·, 2k)])
  └─ hydrate: for each (id, score) → SELECT name, years, skills, roles
  return {mode, route, candidates[]}
```

The retriever returns lightweight `(candidate_id, score)` tuples; hydration happens once at
the end so the ranking math stays cheap and the heavy JSONB is fetched only for the final
top-k.

> Efficiency note: hydration issues one `SELECT` per candidate in a loop. Fine for k≤10;
> for larger k batch it into a single `WHERE candidate_id = ANY(...)`. See §8.

---

## L4 · Generation layer

[src/generation.py](src/generation.py) takes the hydrated candidates and asks **Sonnet** to
write the answer. The prompt enforces the properties that make a RAG answer trustworthy:

- **Grounding** — "use ONLY the retrieved candidate profiles."
- **Mandatory citations** — every claim cites a candidate id in brackets, `[c_0042]`, so
  any statement is traceable to a source row.
- **Refusal over hallucination** — "if the retrieved candidates don't answer the question,
  say so plainly — do NOT make up candidates."
- **Fixed answer shape** — lead with a direct answer, then 2–5 bulleted candidates with
  one-line justifications.

Candidate context is serialized compactly (id, name, years, and JSON-truncated skills/roles
capped at 500 chars each) to keep the prompt bounded regardless of how rich a profile is.

The trust boundary is explicit: the model only ever sees retrieved, structured profiles —
never the corpus at large — and must cite within it. Retrieval quality is therefore the
ceiling on answer quality, which is why the eval layer measures retrieval directly.

---

## L5 · Serving layer

[src/api.py](src/api.py) — a thin FastAPI surface. Deliberately thin: it orchestrates
`retrieve()` then `generate()` and shapes the response. No business logic leaks into the
transport layer.

**`POST /query`**

```jsonc
// request
{ "query": "Who has Java experience with fintech background?",
  "mode": "auto",        // auto | filter | semantic | hybrid
  "top_k": 10 }

// response
{ "query":  "...",
  "mode":   "hybrid",      // mode actually executed
  "route":  { ... },       // router's classification + extracted entities (observability)
  "candidates": [ {"id","name","score","years"}, ... ],
  "answer": "Three candidates stand out… [c_0042] …" }
```

Returning `mode` and `route` alongside the answer makes the system **inspectable from the
outside** — a caller (or a demo) can see *why* a strategy was chosen, not just the result.
`mode` can be pinned by the caller to bypass the router (useful for evals and debugging).

**`GET /health`** — liveness probe for orchestration.

---

## L6 · Evaluation layer

[evals/run_evals.py](evals/run_evals.py) is the layer that turns "hybrid feels better" into
a number. It runs **three LangSmith experiments over the same dataset** — `vector-only`,
`bm25-only`, `hybrid-rrf` — so the production config is always measured against both naive
baselines.

**Dataset.** [evals/eval_set.json](evals/eval_set.json) — 5 queries, each with
`expected_categories`, `expected_skills`, and `min_relevant`. Uploaded once to LangSmith
(`resume-rag-eval-v1`); creation is idempotent.

**Ground truth.** A candidate is "relevant" if its resume `category` is in the query's
`expected_categories`. This is *approximate* — category-level, not hand-labeled id sets —
and the code says so explicitly. It's strong enough to rank the three strategies against
each other, which is the actual question; it is not a production relevance benchmark.

**Metrics** (each computed over the top-10):

| metric         | definition                                            | answers                              |
|----------------|-------------------------------------------------------|--------------------------------------|
| `recall@10`    | relevant retrieved ÷ total relevant in corpus         | did we find the people who exist?    |
| `precision@10` | relevant retrieved ÷ retrieved                        | how much of the result is signal?    |
| `hit@10`       | 1 if relevant-in-top-10 ≥ `min_relevant`, else 0      | is the result *usable* for this query? |

**Why it's structured this way.** The three target functions call the retrievers
*directly* (`vector_search`, `bm25_search`, `rrf_fuse`) rather than going through the
router — so the experiment isolates the retrieval strategy as the only variable. Every run
streams to LangSmith, where you can open a single failing query, inspect what each strategy
retrieved, and see exactly where vector-only or bm25-only lost to hybrid.

---

## 7. Cross-cutting concerns & design rationale

**Why hybrid at all (the central bet).** Recruiter queries are bimodal: exact constraints
and fuzzy intent. Vector search alone misranks exact constraints (it returns
vocabulary-proximity, not skill-depth); keyword search alone misses paraphrase ("agentic
systems" ≠ "multi-agent orchestration"). Routing + RRF gives each query the strategy it
needs, and fusion hedges when the query is genuinely mixed.

**Why pay for extraction at ingest.** Extraction is the expensive step (one Sonnet call per
resume) but it runs **once per resume, not once per query**. The payoff compounds: clean
canonical skills make filters exact, normalized chunks make embeddings sharper, and the
structured fields make generation citable. Front-loading cost where it amortizes is the
defining architectural choice.

**Cost shaping by model tier.** Haiku for routing (every query, must be cheap), Sonnet for
extraction and generation (quality-sensitive, runs once). The router is the only
per-query LLM cost besides generation, and it's the cheapest model.

**One datastore.** Filters, vectors, and keyword search all live in Postgres. No
cross-store consistency problem, one backup story, one thing to operate. This is the
biggest operational simplification in the design and is viable well past this corpus size.

**Determinism where it counts.** Data sampling is seeded; ingestion is idempotent; eval
targets bypass the router. Each removes a source of run-to-run variance so results are
reproducible.

### Scaling path

| dimension        | today                     | next step                                                |
|------------------|---------------------------|----------------------------------------------------------|
| corpus size      | 100 resumes, one PG node  | HNSW scales to ~1–10M chunks on one node; then shard / pgvector replicas |
| skills storage   | JSONB + GIN               | normalized `skills` table once filters get multi-dimensional |
| ingestion        | serial, in-process        | queue + worker pool; batch embed; dead-letter on extract failure |
| hydration        | per-candidate SELECT loop | single `WHERE candidate_id = ANY(...)` batch fetch        |
| embedding model  | MiniLM-384 (CPU)          | larger model / hosted embeddings if recall plateaus      |
| ranking          | RRF of 2 retrievers       | cross-encoder rerank on top of RRF (README cites +5–10% recall@10) |

---

## 8. Known limitations & hardening backlog

Honest accounting of what's intentionally simple in this reference build:

- **Extraction self-correction isn't doubly-guarded.** A twice-malformed JSON response
  raises; in ingestion it's caught per-row and skipped, but there's no dead-letter capture.
- **Hydration N+1.** One `SELECT` per candidate. Negligible at k=10, worth batching for
  larger result sets.
- **No connection pooling.** `get_conn()` opens a fresh connection per call. Add a pool
  (e.g. `psycopg_pool`) before any real concurrency.
- **No auth / rate limiting / tenancy** on the API — it's a single-tenant reference surface.
- **Eval ground truth is category-level**, not hand-labeled id sets — good for ranking
  strategies, not for absolute quality claims.
- **Router has no fallback** — if Haiku returns non-JSON, `route_query()` raises rather than
  degrading to a default mode (e.g. hybrid).
- **Embedding model is fixed at 384-dim.** Changing it is a schema migration
  (`vector(384)`) plus a full re-embed.

These are deliberate scope choices for a reference implementation, not oversights — each
has a clear upgrade path in §7.

---

## Appendix · File map

| file                       | layer | responsibility                                  |
|----------------------------|-------|-------------------------------------------------|
| `scripts/download_data.py` | L0    | fetch + stratified-sample the corpus            |
| `src/extraction.py`        | L1    | LLM structured extraction + section chunking    |
| `src/ingestion.py`         | L1    | orchestrate extract → embed → write             |
| `schema.sql`               | L2    | tables + HNSW / GIN / FTS indexes               |
| `src/db.py`                | L2    | connection context manager + pgvector adapter   |
| `src/config.py`            | all   | env, model ids, embedding dims                  |
| `src/retrieval.py`         | L3    | router + 3 retrievers + RRF + hydration         |
| `src/generation.py`        | L4    | grounded, cited answer synthesis                |
| `src/api.py`               | L5    | FastAPI `/query` + `/health`                    |
| `evals/run_evals.py`       | L6    | 3 LangSmith experiments + metrics               |
| `evals/eval_set.json`      | L6    | query set + approximate ground truth            |
| `docker-compose.yml`       | infra | Postgres 16 + pgvector                          |
