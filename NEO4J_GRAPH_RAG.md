# Neo4j Graph RAG — Design & Learning Notes

A **Neo4j**-backed knowledge-graph retriever added to Resume RAG **for learning**.
It sits alongside the existing filter / vector / BM25 retrievers and answers
**relational, multi-hop** questions the others can't — using a real graph database
and Cypher instead of recursive SQL.

> Why a separate graph store? The rest of the system retrieves candidates by
> *attributes* (does this resume match these terms / filters / embeddings?).
> Graph RAG retrieves by *connections* (which candidates are linked, through
> shared skills, companies, and domains, to the entities in the query?). Neo4j is
> purpose-built for that traversal — relationships are first-class, and Cypher
> expresses multi-hop patterns far more naturally than SQL joins or CTEs.

---

## The key insight: the graph is free

Graph RAG normally has two hard parts — extracting entities and the relationships
between them. **Resume RAG already does both** in `extraction.py`, which produces
per-candidate `skills`, `roles` (with `company` + `domain`), and `education`. So
building the graph needs **no new LLM calls** — `neo4j_graph_build.py` reads the
Postgres `candidates` table (populated by the normal ingestion) and projects it
into Neo4j.

This is the architecture: **Postgres stays the system of record** (structured
fields, chunks, embeddings, FTS); **Neo4j is a derived projection** used only for
graph traversal. You rebuild it any time from Postgres.

## The graph model

```
(:Candidate {id, name, years})
(:Skill {name})  (:Company {name})  (:Domain {name})  (:Institution {name})

(Candidate)-[:HAS_SKILL {years}]->(Skill)
(Candidate)-[:WORKED_AT {tenure}]->(Company)
(Company)-[:IN_DOMAIN]->(Domain)
(Candidate)-[:STUDIED_AT]->(Institution)
(Skill)-[:CO_OCCURS {count}]-(Skill)     // undirected; weight = corpus co-occurrence
```

`CO_OCCURS` is the one *derived* edge: skills that appear together on the same
candidate get linked, weighted by how often across the corpus (e.g. React +
TypeScript). It's computed in one Cypher pass after the per-candidate nodes exist.

## Files

| File | Role |
|------|------|
| `docker-compose.yml` | adds a `neo4j:5` service (Browser :7474, Bolt :7687) |
| `src/config.py` | `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` |
| `src/neo4j_db.py` | shared driver + `get_session()` contextmanager |
| `src/neo4j_graph_build.py` | reads Postgres `candidates`, builds the Neo4j graph. Run: `python -m src.neo4j_graph_build` |
| `src/neo4j_retrieval.py` | `graph_search()` / `graph_rank()` / `similar_candidates()` (Cypher) |
| `src/retrieval.py` | `retrieve()` dispatch gains `graph` + `graph_hybrid`; router can emit `graph` |
| `src/api.py` | `mode` accepts `graph` / `graph_hybrid`; new `GET /similar/{candidate_id}` |
| `evals/run_evals.py` | adds a `graph-neo4j` experiment via `target_graph` |
| `scripts/neo4j_graph_demo.py` | print / `--run` demo helper |

## How retrieval works: weighted spreading activation in Cypher

Graph retrieval returns the **same shape** as every other retriever —
`list[tuple[candidate_id, score]]` — so it drops straight into `rrf_fuse()` and
the existing dispatch.

1. **Seed.** Reuse the LLM router (`route_query`) to pull `skills` and `domains`
   from the query; those names match `Skill` / `Domain` nodes.
2. **Walk.** A variable-length undirected pattern from the seeds to `Candidate`
   nodes:
   ```cypher
   MATCH (seed)
   WHERE (seed:Skill AND seed.name IN $skills) OR (seed:Domain AND seed.name IN $domains)
   MATCH path = (seed)-[*1..2]-(c:Candidate)
   RETURN c.id AS candidate_id, sum($decay ^ length(path)) AS score
   ORDER BY score DESC LIMIT $k
   ```
3. **Score.** Each candidate accumulates `decay ** path_length` over every path
   that reaches it — closer and more densely connected candidates rank higher.

> Cypher doesn't allow a **parameter** in the variable-length bound (`*1..$hops`),
> so `hops` is inlined as a validated integer in `neo4j_retrieval.py`; all entity
> values stay parameterized.

### `similar_candidates(candidate_id)`
A direct shared-neighbor query — clean to express in Cypher:
```cypher
MATCH (c:Candidate {id: $id})--(n)--(other:Candidate)
WHERE other <> c
RETURN other.id AS candidate_id, count(DISTINCT n) AS score
ORDER BY score DESC LIMIT $k
```
Ranks peers by how many skills/companies/domains/institutions they share —
different from embedding similarity (structure vs. text). Exposed at
`GET /similar/{candidate_id}`.

## Setup & run

```bash
docker compose up -d                      # postgres + neo4j
cp .env.example .env                      # set ANTHROPIC_API_KEY (+ NEO4J_* if changed)
pip install -r requirements.txt           # now includes the neo4j driver

# Normal Postgres pipeline first (the graph is derived from it):
psql "$DATABASE_URL" -f schema.sql
python -m src.ingestion

# Build + serve the graph:
python -m src.neo4j_graph_build           # project Postgres -> Neo4j (idempotent)
uvicorn src.api:app --reload
```

Neo4j Browser is at <http://localhost:7474> (user `neo4j`, password `password`) —
handy for eyeballing the graph with `MATCH (n) RETURN n LIMIT 100`.

Query it:
```bash
# Explicit graph mode
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Engineers who know technologies commonly paired with Java", "mode": "graph"}'

# Graph fused with semantic similarity
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "fintech engineers similar to our backend team", "mode": "graph_hybrid"}'

# Similarity by shared graph neighbors
curl localhost:8000/similar/c_0042
```

`mode: "auto"` also works — the router was taught to send relational / multi-hop
phrasings to `graph`.

### Demo helper

```bash
python -m scripts.neo4j_graph_demo          # print example queries + Cypher paths (no DB/API needed)
python -m scripts.neo4j_graph_demo --run    # execute them against the built graph + print graph stats
```

## Evaluating

```bash
python -m src.neo4j_graph_build      # build the graph first
python evals/run_evals.py            # now logs a 4th experiment: "graph-neo4j"
```

The eval set gains one relational query
(*"Engineers who know technologies commonly paired with Java"*). As with the rest
of the harness, ground truth is **category-based and approximate** (see the note
atop `run_evals.py`) — it shows directional behavior, not hand-labeled relevance.
Expect graph to shine on relational queries and underperform plain vector/BM25 on
simple semantic ones; that contrast is the point.

## Postgres-CTE vs. Neo4j — what this teaches

The same Graph RAG idea can be built two ways. This branch is the Neo4j version;
the trade-offs worth internalizing:

| | Postgres recursive CTE | Neo4j + Cypher |
|---|---|---|
| Infra | none (reuses Postgres) | a second database to run/sync |
| Traversal | recursive CTE over an undirected edge **view** | native variable-length patterns |
| Expressiveness | verbose; awkward past 2–3 hops | concise multi-hop patterns |
| Tuning | manual decay in SQL | same, plus GDS library (PageRank, communities) if you go further |
| Best when | you want one store and shallow hops | traversal is central and deep/most varied |

## Knobs & extensions (good next experiments)

- **`hops`** (default 2) / **`decay`** (default 0.5) in `graph_rank`.
- **Relationship-weighted scoring** — factor `HAS_SKILL.years`, `WORKED_AT.tenure`,
  or `CO_OCCURS.count` into the path score instead of decay-by-length only.
- **Neo4j GDS** — run PageRank / Louvain community detection for centrality priors
  or candidate clustering.
- **Full-text seed matching** — use a Neo4j full-text index so seeds match fuzzy
  skill names, not just exact normalized strings.

## Limitations (it's a learning implementation)

- Entity dedup is naive (`lower(name)`): "AWS" and "Amazon Web Services" are
  distinct nodes unless extraction already canonicalized them.
- Neo4j is a derived projection — it can go stale; re-run `neo4j_graph_build`
  after re-ingesting.
- Path scoring is heuristic spreading activation, not a learned graph ranker.
- `graph_rank` enumerates all paths up to `hops`; fine for this corpus size, but
  you'd bound/optimize it (or use GDS) at scale.
