"""
Neo4j graph retrieval (Graph RAG add-on).

Ranks candidates by *connectivity* to the entities in a query, using Cypher
traversal over the Neo4j knowledge graph. Returns the same shape as the Postgres
retrievers — list[tuple[candidate_id, score]] — so results drop straight into
rrf_fuse() and the retrieve() dispatch.

Method: weighted spreading activation. Seed at the query's skill/domain nodes,
walk undirected variable-length paths to Candidate nodes, and score each
candidate by sum(decay ** path_length) over all reaching paths. Closer / more
densely connected candidates score higher.
"""
from src.neo4j_db import get_session
from src.retrieval import route_query


def _norm_list(xs) -> list[str]:
    return [str(x).strip().lower() for x in (xs or []) if x and str(x).strip()]


def graph_rank(skills: list[str], domains: list[str], k: int = 10,
               hops: int = 2, decay: float = 0.5) -> list[tuple[str, float]]:
    """
    Rank candidates by graph connectivity to the given seed entities. Takes
    already-extracted skills/domains so callers that already routed the query
    don't pay for a second router call.
    """
    skills = _norm_list(skills)
    domains = _norm_list(domains)
    if not skills and not domains:
        return []

    # Neo4j doesn't allow a parameter in the variable-length bound, so `hops` is
    # inlined as a validated integer; entity values stay parameterized.
    hops = max(1, int(hops))
    cypher = f"""
        MATCH (seed)
        WHERE (seed:Skill  AND seed.name IN $skills)
           OR (seed:Domain AND seed.name IN $domains)
        MATCH path = (seed)-[*1..{hops}]-(c:Candidate)
        RETURN c.id AS candidate_id, sum($decay ^ length(path)) AS score
        ORDER BY score DESC
        LIMIT $k
    """
    with get_session() as session:
        result = session.run(
            cypher, skills=skills, domains=domains, decay=decay, k=k
        )
        return [(r["candidate_id"], float(r["score"])) for r in result]


def graph_search(query: str, k: int = 10, hops: int = 2,
                 decay: float = 0.5) -> list[tuple[str, float]]:
    """Route the query, then rank by graph connectivity. Standalone entry point."""
    route = route_query(query)
    return graph_rank(route.get("skills", []), route.get("domains", []),
                      k=k, hops=hops, decay=decay)


def similar_candidates(candidate_id: str, k: int = 10) -> list[tuple[str, float]]:
    """
    Candidates most similar to the given one by shared graph neighbors (skills,
    companies, domains, institutions). Score = number of distinct shared
    neighbors. Genuinely different from embedding similarity (structure vs text).
    """
    cypher = """
        MATCH (c:Candidate {id: $id})--(n)--(other:Candidate)
        WHERE other <> c
        RETURN other.id AS candidate_id, count(DISTINCT n) AS score
        ORDER BY score DESC
        LIMIT $k
    """
    with get_session() as session:
        result = session.run(cypher, id=candidate_id, k=k)
        return [(r["candidate_id"], float(r["score"])) for r in result]
