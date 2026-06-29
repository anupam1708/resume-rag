"""
Graph retrieval (Graph RAG add-on).

A fourth retriever that ranks candidates by their *connectivity* to the entities
in a query, rather than by text similarity or structured filters. It answers
multi-hop / relational questions the other retrievers can't, e.g.:

  - "candidates who know skills commonly paired with Kubernetes"
        skill --CO_OCCURS--> skill --HAS_SKILL--> candidate   (2 hops)
  - "engineers from fintech companies who know Java"
        candidate --WORKED_AT--> company --IN_DOMAIN--> fintech (+ Java skill seed)
  - "candidates similar to c_0042"
        candidate --HAS_SKILL--> skill --HAS_SKILL--> candidate (shared neighbors)

Like every other retriever (vector_search / bm25_search / filter_search), the
public functions return list[tuple[candidate_id, score]] so results drop straight
into rrf_fuse() and the retrieve() dispatch.

Method: weighted spreading activation. We seed the graph at the query's entity
nodes (skills + domains, reused from the LLM router) and walk outward up to
`hops` edges, decaying the contribution by `decay` per hop. Candidate nodes
accumulate score from every path that reaches them — well-connected candidates
score higher.
"""
from src.db import get_conn
from src.retrieval import route_query


# --------------------------------------------------------------------
# Seed lookup: map query entities -> graph node ids
# --------------------------------------------------------------------

def _seed_nodes(cur, skills: list[str], domains: list[str]) -> list[int]:
    keys_skill = [s.strip().lower() for s in (skills or []) if s and s.strip()]
    keys_domain = [d.strip().lower() for d in (domains or []) if d and d.strip()]
    if not keys_skill and not keys_domain:
        return []
    cur.execute(
        """
        SELECT node_id FROM graph_nodes
        WHERE (node_type = 'skill'  AND key = ANY(%s))
           OR (node_type = 'domain' AND key = ANY(%s))
        """,
        (keys_skill or [""], keys_domain or [""]),
    )
    return [r[0] for r in cur.fetchall()]


# --------------------------------------------------------------------
# Spreading-activation walk from seed nodes to candidate nodes
# --------------------------------------------------------------------

def _walk_to_candidates(cur, seed_ids: list[int], hops: int, decay: float,
                        k: int) -> list[tuple[str, float]]:
    if not seed_ids:
        return []
    cur.execute(
        """
        WITH RECURSIVE walk AS (
            SELECT node_id AS nid, 0 AS depth, 1.0::float AS w
            FROM graph_nodes
            WHERE node_id = ANY(%(seeds)s)
          UNION ALL
            SELECT u.dst, walk.depth + 1, walk.w * %(decay)s
            FROM walk
            JOIN graph_edges_undirected u ON u.src = walk.nid
            WHERE walk.depth < %(hops)s
        )
        SELECT n.key AS candidate_id, SUM(walk.w) AS score
        FROM walk
        JOIN graph_nodes n ON n.node_id = walk.nid
        WHERE n.node_type = 'candidate'
        GROUP BY n.key
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        {"seeds": seed_ids, "decay": decay, "hops": hops, "k": k},
    )
    return [(row[0], float(row[1])) for row in cur.fetchall()]


# --------------------------------------------------------------------
# Public retrievers
# --------------------------------------------------------------------

def graph_rank(skills: list[str], domains: list[str], k: int = 10,
               hops: int = 2, decay: float = 0.5) -> list[tuple[str, float]]:
    """
    Rank candidates by graph connectivity to the given seed entities.
    Takes already-extracted skills/domains so callers that already routed the
    query (e.g. retrieve() in auto mode) don't pay for a second router call.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        seeds = _seed_nodes(cur, skills, domains)
        return _walk_to_candidates(cur, seeds, hops=hops, decay=decay, k=k)


def graph_search(query: str, k: int = 10, hops: int = 2,
                 decay: float = 0.5) -> list[tuple[str, float]]:
    """
    Returns [(candidate_id, score), ...] ranked by graph connectivity to the
    query's entities. Reuses the LLM router to pull skills/domains as seeds.
    """
    route = route_query(query)
    return graph_rank(route.get("skills", []), route.get("domains", []),
                      k=k, hops=hops, decay=decay)


def similar_candidates(candidate_id: str, k: int = 10) -> list[tuple[str, float]]:
    """
    Returns [(candidate_id, score), ...] of candidates most similar to the given
    one by shared graph neighbors (skills, companies, domains, institutions) —
    a 2-hop walk that excludes the seed candidate itself.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT node_id FROM graph_nodes WHERE node_type = 'candidate' AND key = %s",
            (candidate_id,),
        )
        row = cur.fetchone()
        if not row:
            return []
        ranked = _walk_to_candidates(cur, [row[0]], hops=2, decay=0.5, k=k + 1)
        return [(cid, score) for cid, score in ranked if cid != candidate_id][:k]
