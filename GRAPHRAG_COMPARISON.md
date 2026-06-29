# Graph RAG: Postgres vs. Neo4j — A Comparison

Resume RAG has two **learning** implementations of the same Graph RAG idea — a
knowledge-graph retriever that answers relational, multi-hop questions over the
resume corpus. They are identical in intent and integration but differ in the
storage engine and how traversal is expressed.

This document compares them. The two implementations live on separate branches
(neither is merged into `main`):

| Approach | Branch | Design doc |
|----------|--------|------------|
| **Postgres-native** (recursive CTE over edge tables) | `claude/claude-md-docs-7l18b4` | `GRAPH_RAG.md` |
| **Neo4j** (Cypher over a property graph) | `claude/graphrag-neo4j` | `NEO4J_GRAPH_RAG.md` |

> Both are deliberately simple, for learning — not production graph ranking. See
> each branch's design doc and the shared *Limitations* section below.

---

## 1. What they share

The two are the **same system** above the storage layer. Everything here is
identical (or all-but-identical) between them:

- **The graph is free.** Neither makes new LLM calls. Both derive the graph from
  the structured fields `extraction.py` already produced (`skills`, `roles` with
  `company`/`domain`, `education`) and stored in the Postgres `candidates` table.
- **The same conceptual model** — nodes: candidate, skill, company, domain,
  institution; edges: `HAS_SKILL` (weighted by years), `WORKED_AT` (tenure),
  `IN_DOMAIN`, `STUDIED_AT`, and a derived `CO_OCCURS` (skill–skill, weighted by
  corpus co-occurrence).
- **The same retrieval contract** — every retriever returns
  `list[tuple[candidate_id, score]]`, so graph results drop straight into the
  existing `rrf_fuse()` and `retrieve()` dispatch. Both add `graph` and
  `graph_hybrid` modes and a `GET /similar/{candidate_id}` endpoint.
- **The same seeding** — both reuse the LLM router (`route_query`) to pull
  `skills`/`domains` from the query as traversal seed entities, and both extend
  the router prompt with a `graph` class for relational/multi-hop phrasing.
- **The same scoring idea** — weighted **spreading activation**: start at the
  seed entities, walk outward, and score candidates by decayed path contribution.
- **The same `similar_candidates`** — rank peers by shared graph neighbors
  (structure-based similarity, distinct from embedding similarity).
- **The same evaluation hook** — a graph experiment added to `run_evals.py` plus
  one relational query in `eval_set.json`, against the existing category-based
  (approximate) ground truth.
- **The same demo helper** — a `scripts/*graph_demo.py` with a no-DB print mode
  and a live `--run` mode.

Because the contract is identical, **the storage engine is a swappable backend**
behind `graph_rank()` / `graph_search()` / `similar_candidates()`.

---

## 2. Side-by-side

| Dimension | Postgres-native | Neo4j |
|-----------|-----------------|-------|
| **Extra infrastructure** | None — reuses the existing Postgres | A second database (`neo4j:5` service) to run, secure, and keep in sync |
| **New dependency** | None (`psycopg` already present) | `neo4j` driver |
| **Source of truth** | Postgres (graph tables live next to the data) | Postgres remains source of truth; Neo4j is a **derived projection** that can go stale |
| **Schema / model** | Generic `graph_nodes` + `graph_edges` tables with a `node_type` / `rel` discriminator column | Native labels (`:Skill`) and relationship types (`[:HAS_SKILL]`) — types are first-class |
| **Build** | `graph_build.py`: `TRUNCATE` + bulk `INSERT`; co-occurrence computed in Python | `neo4j_graph_build.py`: `DETACH DELETE` + `MERGE`; co-occurrence computed in one Cypher pass |
| **Traversal mechanism** | `WITH RECURSIVE` CTE over an `graph_edges_undirected` view | Native variable-length path pattern `(seed)-[*1..2]-(c)` |
| **Query language** | SQL (verbose; awkward beyond 2–3 hops) | Cypher (concise multi-hop patterns) |
| **Undirected edges** | Must be materialized as a `UNION ALL` view — a recursive CTE can't self-reference inside a subquery | Just omit direction in the pattern: `-[]-` |
| **Hop depth** | Parameterized in the CTE (`WHERE depth < %s`) | Must be **inlined** as an integer — Cypher forbids a parameter in the `*1..n` bound |
| **Advanced graph algos** | Hand-rolled in SQL | Neo4j GDS library (PageRank, Louvain communities, etc.) |
| **Transactional consistency w/ source** | Same DB and transaction as the candidate data | Separate store; needs a rebuild/sync step after re-ingest |
| **Ops surface** | One service to back up and monitor | Two services; plus the Neo4j Browser (`:7474`) for visual exploration |

---

## 3. The core difference: how the walk is expressed

Both do the same thing — seed at skill/domain nodes, spread outward, sum a decayed
contribution per candidate. The expression is where they diverge.

### Postgres — recursive CTE over an undirected view

```sql
WITH RECURSIVE walk AS (
    SELECT node_id AS nid, 0 AS depth, 1.0::float AS w
    FROM graph_nodes WHERE node_id = ANY(%(seeds)s)
  UNION ALL
    SELECT u.dst, walk.depth + 1, walk.w * %(decay)s
    FROM walk
    JOIN graph_edges_undirected u ON u.src = walk.nid   -- view, not a subquery
    WHERE walk.depth < %(hops)s
)
SELECT n.key AS candidate_id, SUM(walk.w) AS score
FROM walk JOIN graph_nodes n ON n.node_id = walk.nid
WHERE n.node_type = 'candidate'
GROUP BY n.key ORDER BY score DESC LIMIT %(k)s
```

Why the `graph_edges_undirected` **view** exists: a recursive CTE's self-reference
can't appear inside a subquery, so the "follow edges in either direction" logic is
pre-flattened into a `UNION ALL` view the recursive term can join to directly.
Decay is applied as `walk.w * decay` at each step, so a path of length *d*
contributes `decay^d`.

### Neo4j — a variable-length Cypher pattern

```cypher
MATCH (seed)
WHERE (seed:Skill  AND seed.name IN $skills)
   OR (seed:Domain AND seed.name IN $domains)
MATCH path = (seed)-[*1..2]-(c:Candidate)        -- undirected, multi-hop, native
RETURN c.id AS candidate_id, sum($decay ^ length(path)) AS score
ORDER BY score DESC LIMIT $k
```

Undirectedness is free (`-[*1..2]-`), and the multi-hop walk is a single pattern.
The one wrinkle: the `2` in `*1..2` can't be a query parameter, so `hops` is
inlined as a validated integer in code. Scoring is the explicit
`sum(decay ^ length(path))`.

**Takeaway:** the same algorithm is ~12 lines of recursive SQL plus a helper view,
or ~4 lines of Cypher. That gap widens fast as hop depth and pattern complexity
grow — which is precisely what graph databases are built for.

---

## 4. Building the graph

| | Postgres | Neo4j |
|---|----------|-------|
| Reset | `TRUNCATE graph_nodes, graph_edges RESTART IDENTITY CASCADE` | `MATCH (n) DETACH DELETE n` |
| Upsert | `INSERT ... ON CONFLICT (node_type, key) DO UPDATE` | `MERGE (:Skill {name})` (uniqueness constraints enforce dedup) |
| Co-occurrence | Python: enumerate skill pairs per candidate, count, insert both directions | Cypher: `MATCH (s1)<-[:HAS_SKILL]-(c)-[:HAS_SKILL]->(s2) WHERE id(s1)<id(s2)` then `MERGE`/`SET count` |
| Empty-list edge case | N/A (Python loops) | `UNWIND []` drops the row, so the build is split into guarded per-list statements |
| Idempotency | Full rebuild each run | Full rebuild each run |

Both are idempotent by rebuilding from scratch. The Neo4j build leans on
**uniqueness constraints** for dedup and does co-occurrence **in the database**;
the Postgres build does dedup with an in-memory node cache and co-occurrence **in
Python**.

---

## 5. When to choose which

**Choose Postgres-native when…**
- You want **zero new infrastructure** — one database to run, back up, and secure.
- Traversals are **shallow** (1–3 hops), which covers most resume-search needs.
- You value the graph being **transactionally consistent** with the source data
  (same DB, no sync/staleness).
- The design thesis is "everything in Postgres" (as in this repo's `DESIGN.md`).

**Choose Neo4j when…**
- Traversal is **central and deep/variable** — many hops, varied path shapes,
  pathfinding — where Cypher stays readable and SQL doesn't.
- You want **graph algorithms** off the shelf (centrality, community detection,
  similarity) via GDS.
- A **visual graph explorer** (Neo4j Browser) materially helps development.
- The graph is large enough that a purpose-built traversal engine and graph-native
  indexes pay for the operational cost of a second store.

**Rule of thumb:** for *this* corpus (100 resumes, 2-hop queries), Postgres-native
is the pragmatic winner — no second system to operate. Neo4j earns its keep when
the graph and its traversals become the primary workload rather than an add-on.

---

## 6. Shared limitations

Both are learning implementations and share the same caveats:

- **Naive entity resolution** — dedup is `lower(name)`, so "AWS" and "Amazon Web
  Services" are distinct nodes unless extraction already canonicalized them.
- **Heuristic scoring** — decayed spreading activation, not a learned/principled
  graph ranker.
- **Approximate evaluation** — category-based ground truth (see the note atop
  `run_evals.py`); it shows directional behavior, not hand-labeled relevance.
- **Unbounded path enumeration** — fine at this corpus size; both would need
  bounding/optimization (or GDS, on the Neo4j side) at scale.
- **No cycle guard beyond the hop cap** — both rely on the small `hops` bound to
  terminate.

---

## 7. Further reading

- `GRAPH_RAG.md` (branch `claude/claude-md-docs-7l18b4`) — Postgres-native design,
  setup, and extension ideas.
- `NEO4J_GRAPH_RAG.md` (branch `claude/graphrag-neo4j`) — Neo4j design, setup, and
  a Postgres-vs-Neo4j table from the Neo4j side.
- `DESIGN.md` — the broader Resume RAG architecture these add-ons plug into.
