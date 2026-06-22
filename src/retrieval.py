"""
Hybrid retrieval: vector + BM25 + RRF fusion, with query routing.

Three retrieval modes:
  1. Filter   - SQL on structured JSONB (for "who knows Java")
  2. Semantic - pgvector cosine (for paraphrased queries)
  3. Hybrid   - both, fused with Reciprocal Rank Fusion (RRF)

Query router (LLM) picks the mode.
"""
import json
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from src.config import ANTHROPIC_API_KEY, EMBED_MODEL, ROUTER_MODEL
from src.db import get_conn

client = Anthropic(api_key=ANTHROPIC_API_KEY)
_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# --------------------------------------------------------------------
# Query routing
# --------------------------------------------------------------------

ROUTER_PROMPT = """Classify this resume-search query into ONE of:
- "filter": user wants candidates matching a specific skill, years, or role
   (e.g. "who knows Python", "Java engineers with 5+ years", "data scientists")
- "semantic": user describes an abstract capability or experience
   (e.g. "who has built large-scale distributed systems")
- "hybrid": query has both filter constraints AND abstract capability
   (e.g. "senior Java engineers with fintech experience and team leadership")

Also extract any filterable entities as JSON.

Query: "{query}"

Output JSON only:
{{"mode": "filter|semantic|hybrid",
  "skills": [string],
  "min_years": number | null,
  "domains": [string]}}
"""


def route_query(query: str) -> dict:
    msg = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": ROUTER_PROMPT.replace("{query}", query)}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)


# --------------------------------------------------------------------
# Filter retrieval (structured SQL on JSONB)
# --------------------------------------------------------------------

def filter_search(skills: list[str], min_years: float | None,
                  domains: list[str], k: int = 20) -> list[tuple[str, float]]:
    """
    Returns [(candidate_id, score), ...] ranked by skill match strength.

    Score = sum of years for each matching skill. Encourages depth.
    """
    where = []
    params = []
    if min_years:
        where.append("total_years_experience >= %s")
        params.append(min_years)

    sql = """
        SELECT candidate_id,
               COALESCE((
                 SELECT SUM((s->>'years')::numeric)
                 FROM jsonb_array_elements(skills) s
                 WHERE LOWER(s->>'name') = ANY(%s)
               ), 0) as skill_score
        FROM candidates
        WHERE {where}
        ORDER BY skill_score DESC
        LIMIT %s
    """.format(where=" AND ".join(where) if where else "TRUE")

    skill_lower = [s.lower() for s in skills] if skills else [""]
    params = [skill_lower] + params + [k]

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [(row[0], float(row[1])) for row in cur.fetchall()]


# --------------------------------------------------------------------
# Vector retrieval (pgvector cosine)
# --------------------------------------------------------------------

def vector_search(query: str, k: int = 20) -> list[tuple[str, float]]:
    """
    Returns [(candidate_id, score), ...] from semantic chunk search.
    Aggregates chunks per candidate by taking best-chunk similarity.
    """
    qvec = embedder().encode(query).tolist()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT candidate_id, MAX(1 - (embedding <=> %s::vector)) AS sim
            FROM chunks
            GROUP BY candidate_id
            ORDER BY sim DESC
            LIMIT %s
            """,
            (qvec, k),
        )
        return [(row[0], float(row[1])) for row in cur.fetchall()]


# --------------------------------------------------------------------
# BM25 retrieval (Postgres FTS)
# --------------------------------------------------------------------

def bm25_search(query: str, k: int = 20) -> list[tuple[str, float]]:
    """Returns [(candidate_id, score), ...] from full-text search."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT candidate_id, MAX(ts_rank_cd(fts, plainto_tsquery('english', %s))) AS r
            FROM chunks
            WHERE fts @@ plainto_tsquery('english', %s)
            GROUP BY candidate_id
            ORDER BY r DESC
            LIMIT %s
            """,
            (query, query, k),
        )
        return [(row[0], float(row[1])) for row in cur.fetchall()]


# --------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------

def rrf_fuse(rankings: list[list[tuple[str, float]]], k_const: int = 60,
             top_k: int = 10) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)).
    Score-agnostic; only ranks matter. Robust to scale differences between
    BM25 and cosine similarity.
    """
    scores = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# --------------------------------------------------------------------
# Main retrieve() entry point
# --------------------------------------------------------------------

def retrieve(query: str, mode: str = "auto", top_k: int = 10) -> dict:
    """
    Main retrieval entry point.

    Returns: {"mode": str, "candidates": [{"id", "score", ...}], "route_info": dict}
    """
    if mode == "auto":
        route = route_query(query)
        mode = route["mode"]
    else:
        route = {"mode": mode, "skills": [], "min_years": None, "domains": []}

    if mode == "filter":
        ranked = filter_search(
            route.get("skills", []), route.get("min_years"),
            route.get("domains", []), k=top_k
        )
    elif mode == "semantic":
        ranked = vector_search(query, k=top_k)
    else:  # hybrid
        vec = vector_search(query, k=top_k * 2)
        bm = bm25_search(query, k=top_k * 2)
        ranked = rrf_fuse([vec, bm], top_k=top_k)

    # Hydrate with candidate metadata
    candidates = []
    if ranked:
        with get_conn() as conn:
            cur = conn.cursor()
            for cid, score in ranked:
                cur.execute(
                    "SELECT name, total_years_experience, skills, roles "
                    "FROM candidates WHERE candidate_id = %s",
                    (cid,),
                )
                row = cur.fetchone()
                if row:
                    candidates.append({
                        "id": cid,
                        "score": score,
                        "name": row[0],
                        "years": float(row[1]) if row[1] is not None else None,
                        "skills": row[2],
                        "roles": row[3],
                    })

    return {"mode": mode, "route": route, "candidates": candidates}
