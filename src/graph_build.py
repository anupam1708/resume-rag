"""
Knowledge-graph builder (Graph RAG add-on).

Materializes a graph (graph_nodes + graph_edges) from the structured fields the
LLM already extracted into the `candidates` table. No new LLM calls — this is a
pure SQL/Python transform over data ingestion.py already produced.

Run AFTER ingestion:  python -m src.graph_build

Idempotent: rebuilds the whole graph from scratch each run (TRUNCATE + reinsert),
so it's safe to re-run after a fresh ingest.

Graph shape
-----------
  candidate --HAS_SKILL-->  skill        weight = years on that skill
  candidate --WORKED_AT-->  company      weight = tenure (end_year - start_year)
  company   --IN_DOMAIN-->  domain       weight = number of roles seen in domain
  candidate --STUDIED_AT--> institution
  skill     --CO_OCCURS-->  skill        weight = how often the two skills appear
                                          together across the corpus (undirected,
                                          stored as two directed edges)
"""
import json
from collections import defaultdict
from itertools import combinations

from src.db import get_conn


def _norm(s) -> str:
    """Canonical dedup key: trimmed, lower-cased."""
    return str(s).strip().lower()


def main():
    with get_conn() as conn:
        cur = conn.cursor()

        # Rebuild from scratch so re-runs are deterministic and idempotent.
        cur.execute("TRUNCATE graph_nodes, graph_edges RESTART IDENTITY CASCADE")

        # In-memory node cache: (node_type, key) -> node_id, to avoid round-trips.
        node_ids: dict[tuple[str, str], int] = {}

        def upsert_node(node_type: str, key: str, label: str) -> int:
            cache_key = (node_type, key)
            if cache_key in node_ids:
                return node_ids[cache_key]
            cur.execute(
                """
                INSERT INTO graph_nodes (node_type, key, label)
                VALUES (%s, %s, %s)
                ON CONFLICT (node_type, key) DO UPDATE SET label = EXCLUDED.label
                RETURNING node_id
                """,
                (node_type, key, label),
            )
            nid = cur.fetchone()[0]
            node_ids[cache_key] = nid
            return nid

        # Accumulate edges in Python, then bulk-insert. We sum weights for
        # duplicate (src, dst, rel) triples (e.g. the same skill from two roles).
        edge_weights: dict[tuple[int, int, str], float] = defaultdict(float)

        def add_edge(src: int, dst: int, rel: str, weight: float = 1.0):
            edge_weights[(src, dst, rel)] += weight

        # Corpus-wide skill co-occurrence counts.
        cooccur: dict[tuple[str, str], int] = defaultdict(int)

        cur.execute(
            "SELECT candidate_id, name, skills, roles, education FROM candidates"
        )
        rows = cur.fetchall()
        print(f"Building graph from {len(rows)} candidates...")

        for cid, name, skills, roles, education in rows:
            # psycopg returns JSONB as already-parsed Python objects; be defensive.
            skills = skills if isinstance(skills, list) else json.loads(skills or "[]")
            roles = roles if isinstance(roles, list) else json.loads(roles or "[]")
            education = education if isinstance(education, list) else json.loads(education or "[]")

            cand_node = upsert_node("candidate", cid, name or cid)

            # --- HAS_SKILL ---
            skill_keys = []
            for s in skills:
                sname = s.get("name")
                if not sname:
                    continue
                key = _norm(sname)
                skill_keys.append(key)
                skill_node = upsert_node("skill", key, sname)
                years = s.get("years") or 0
                add_edge(cand_node, skill_node, "HAS_SKILL", float(years))

            # --- skill co-occurrence (unordered pairs, deduped per candidate) ---
            for a, b in combinations(sorted(set(skill_keys)), 2):
                cooccur[(a, b)] += 1

            # --- WORKED_AT + IN_DOMAIN ---
            for r in roles:
                company = r.get("company")
                if company:
                    comp_node = upsert_node("company", _norm(company), company)
                    start, end = r.get("start_year"), r.get("end_year")
                    tenure = float((end or start or 0) - (start or 0)) if start else 1.0
                    add_edge(cand_node, comp_node, "WORKED_AT", max(tenure, 1.0))

                    domain = r.get("domain")
                    if domain:
                        dom_node = upsert_node("domain", _norm(domain), domain)
                        add_edge(comp_node, dom_node, "IN_DOMAIN", 1.0)

            # --- STUDIED_AT ---
            for e in education:
                inst = e.get("institution")
                if inst:
                    inst_node = upsert_node("institution", _norm(inst), inst)
                    add_edge(cand_node, inst_node, "STUDIED_AT", 1.0)

        # --- CO_OCCURS (stored both directions for undirected traversal) ---
        for (a, b), count in cooccur.items():
            na = upsert_node("skill", a, a)
            nb = upsert_node("skill", b, b)
            add_edge(na, nb, "CO_OCCURS", float(count))
            add_edge(nb, na, "CO_OCCURS", float(count))

        # Bulk insert edges.
        for (src, dst, rel), weight in edge_weights.items():
            cur.execute(
                "INSERT INTO graph_edges (src, dst, rel, weight) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (src, dst, rel) DO UPDATE SET weight = EXCLUDED.weight",
                (src, dst, rel, weight),
            )

        # Report.
        cur.execute("SELECT node_type, COUNT(*) FROM graph_nodes GROUP BY node_type ORDER BY 1")
        node_counts = cur.fetchall()
        cur.execute("SELECT rel, COUNT(*) FROM graph_edges GROUP BY rel ORDER BY 1")
        edge_counts = cur.fetchall()

    print("Graph build complete.")
    print("  Nodes:", ", ".join(f"{t}={c}" for t, c in node_counts))
    print("  Edges:", ", ".join(f"{r}={c}" for r, c in edge_counts))


if __name__ == "__main__":
    main()
