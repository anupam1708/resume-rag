# Graph RAG — Design & Learning Notes

A Postgres-native knowledge-graph retriever added to Resume RAG **for learning**.
It sits alongside the existing filter / vector / BM25 retrievers and answers
**relational, multi-hop** questions the others can't.

> Why this exists: the rest of the system retrieves candidates by *attributes*
> (does this resume match these terms / filters / embeddings?). Graph RAG retrieves
> by *connections* (which candidates are linked, through shared skills, companies,
> and domains, to the entities in the query?). It's the same corpus viewed as a
> network instead of a bag of documents.

---

## The key insight: the graph is free

Graph RAG normally has two hard parts — extracting entities, and extracting the
relationships between them. **Resume RAG already does both** in `extraction.py`,
which produces per-candidate `skills`, `roles` (each with `company` + `domain`),
and `education`. So building the graph needs **no new LLM calls** — it's a pure
transform over the `candidates` table that ingestion already populated.

## The graph

**Nodes** (`graph_nodes`): `candidate`, `skill`, `company`, `domain`, `institution`

**Edges** (`graph_edges`):

| Edge | Direction | Source field | Weight |
|------|-----------|--------------|--------|
| `HAS_SKILL` | candidate → skill | `skills[].name` | `skills[].years` |
| `WORKED_AT` | candidate → company | `roles[].company` | tenure (end − start) |
| `IN_DOMAIN` | company → domain | `roles[].domain` | role count |
| `STUDIED_AT` | candidate → institution | `education[].institution` | 1 |
| `CO_OCCURS` | skill ↔ skill | derived: skill pairs per candidate | corpus co-occurrence count |

`CO_OCCURS` is the one *derived* edge: for each candidate we take every unordered
pair of their skills and increment a global counter. Skills that frequently appear
together (e.g. React + TypeScript) end up strongly linked. It's stored as two
directed rows so traversal can treat it as undirected.

## Files

| File | Role |
|------|------|
| `graph.sql` | DDL: `graph_nodes`, `graph_edges`, indexes, and the `graph_edges_undirected` view |
| `src/graph_build.py` | Reads `candidates`, materializes nodes + edges. Run: `python -m src.graph_build` |
| `src/graph_retrieval.py` | `graph_search()` / `graph_rank()` / `similar_candidates()` retrievers |
| `src/retrieval.py` | `retrieve()` dispatch gains `graph` + `graph_hybrid` modes; router can emit `graph` |
| `src/api.py` | `mode` accepts `graph` / `graph_hybrid`; new `GET /similar/{candidate_id}` |
| `evals/run_evals.py` | adds a `graph` experiment via `target_graph` |

## How retrieval works: weighted spreading activation

Graph retrieval returns the **same shape** as every other retriever —
`list[tuple[candidate_id, score]]` — so it drops straight into `rrf_fuse()` and the
existing dispatch.

1. **Seed.** Reuse the LLM router (`route_query`) to pull `skills` and `domains`
   from the query, and look up their graph node IDs (`_seed_nodes`).
2. **Walk.** A recursive CTE (`_walk_to_candidates`) starts at the seed nodes and
   spreads outward up to `hops` edges over the **undirected** view, multiplying the
   contribution by `decay` (default 0.5) at each hop.
3. **Score.** Every `candidate` node accumulates weight from every path that
   reaches it. Candidates closer to (and more densely connected to) the query's
   entities score higher.

```
"skills paired with Kubernetes":
   Kubernetes ──CO_OCCURS──▶ Docker ──HAS_SKILL──▶ [candidate]   (2 hops, weight 0.25)
   Kubernetes ──CO_OCCURS──▶ Helm   ──HAS_SKILL──▶ [candidate]   (2 hops, weight 0.25)
```

The recursive term joins the working set directly to `graph_edges_undirected`
(not a subquery) because Postgres forbids a recursive self-reference inside a
subquery — that's why the undirected edges are exposed as a flat view.

### `similar_candidates(candidate_id)`
A focused 2-hop walk seeded at one candidate: `candidate → {skills,companies,…} →
candidate`, excluding the seed. Ranks peers by shared neighbors — genuinely
different from embedding similarity, which compares text rather than structure.
Exposed at `GET /similar/{candidate_id}`.

## Setup & run

```bash
# After the normal setup + ingestion (so `candidates` is populated):
psql "$DATABASE_URL" -f graph.sql        # create graph tables + view
python -m src.graph_build                 # materialize the graph (idempotent)

uvicorn src.api:app --reload
```

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

## Evaluating

```bash
python -m src.graph_build      # build the graph first
python evals/run_evals.py      # now logs a 4th experiment: "graph"
```

The eval set gains one relational query
(*"Engineers who know technologies commonly paired with Java"*). As with the rest
of the harness, ground truth is **category-based and approximate** (see the note
atop `run_evals.py`) — it shows directional behavior, not hand-labeled relevance.
Expect graph to shine on relational queries and underperform vanilla vector/BM25
on plain semantic ones; that contrast is the point.

## Knobs & extensions (good next experiments)

- **`hops`** (default 2) — higher reaches further but blurs relevance and costs more.
- **`decay`** (default 0.5) — how fast distant nodes lose influence.
- **Edge-weight in the walk** — the current walk decays by hop only; try multiplying
  by normalized edge weight so years/tenure/co-occurrence strength matter.
- **Path-type weighting** — weight `HAS_SKILL` vs `CO_OCCURS` differently.
- **Centrality** — precompute degree / PageRank as a candidate prior.
- **Cycle guard** — track visited nodes per path if you raise `hops`.

## Limitations (it's a learning implementation)

- Entity dedup is naive (`lower(name)`): "AWS" and "Amazon Web Services" are
  distinct nodes unless extraction already canonicalized them.
- The walk has no cycle guard; it relies on the small `hops` cap to terminate.
- Co-occurrence is corpus-global, not time-aware.
- Scoring is heuristic spreading activation, not a learned/principled graph ranker.
