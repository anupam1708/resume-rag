"""
Neo4j knowledge-graph builder (Graph RAG add-on).

Reads the structured fields the LLM already extracted into the Postgres
`candidates` table and materializes a property graph in Neo4j. No new LLM calls —
a pure transform over data ingestion.py already produced.

Run AFTER Postgres ingestion:  python -m src.neo4j_graph_build
Idempotent: wipes the graph and rebuilds it each run.

Graph model
-----------
  (:Candidate {id, name, years})
  (:Skill {name})  (:Company {name})  (:Domain {name})  (:Institution {name})

  (Candidate)-[:HAS_SKILL {years}]->(Skill)
  (Candidate)-[:WORKED_AT {tenure}]->(Company)
  (Company)-[:IN_DOMAIN]->(Domain)
  (Candidate)-[:STUDIED_AT]->(Institution)
  (Skill)-[:CO_OCCURS {count}]-(Skill)   # undirected; weight = corpus co-occurrence
"""
import json

from src.db import get_conn
from src.neo4j_db import get_session


CONSTRAINTS = [
    "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT institution_name IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
]

# Built with several small statements rather than one combined query: UNWIND over
# an empty list drops the row, which would skip a candidate's companies/education
# when they have no skills. So each list gets its own guarded statement (only run
# when non-empty), all keyed off the already-MERGEd Candidate. Names are
# normalized (lower-cased) on the way in so nodes dedupe cleanly.
MERGE_CANDIDATE = """
MERGE (c:Candidate {id: $id})
SET c.name = $name, c.years = $years
"""

ADD_SKILLS = """
MATCH (c:Candidate {id: $id})
UNWIND $skills AS sk
  MERGE (s:Skill {name: sk.name})
  MERGE (c)-[r:HAS_SKILL]->(s)
  SET r.years = sk.years
"""

ADD_ROLES = """
MATCH (c:Candidate {id: $id})
UNWIND $roles AS ro
  MERGE (co:Company {name: ro.company})
  MERGE (c)-[w:WORKED_AT]->(co)
  SET w.tenure = ro.tenure
  WITH co, ro
  WHERE ro.domain IS NOT NULL AND ro.domain <> ''
  MERGE (d:Domain {name: ro.domain})
  MERGE (co)-[:IN_DOMAIN]->(d)
"""

ADD_INSTITUTIONS = """
MATCH (c:Candidate {id: $id})
UNWIND $institutions AS inst
  MERGE (i:Institution {name: inst})
  MERGE (c)-[:STUDIED_AT]->(i)
"""

# Derive skill co-occurrence across the whole corpus in one pass. id(s1) < id(s2)
# yields each unordered pair once; count = number of candidates sharing both.
BUILD_COOCCURRENCE = """
MATCH (s1:Skill)<-[:HAS_SKILL]-(c:Candidate)-[:HAS_SKILL]->(s2:Skill)
WHERE id(s1) < id(s2)
WITH s1, s2, count(DISTINCT c) AS cnt
MERGE (s1)-[r:CO_OCCURS]-(s2)
SET r.count = cnt
"""


def _norm(s) -> str:
    return str(s).strip().lower()


def _candidate_payload(cid, name, years, skills, roles, education) -> dict:
    skills = skills if isinstance(skills, list) else json.loads(skills or "[]")
    roles = roles if isinstance(roles, list) else json.loads(roles or "[]")
    education = education if isinstance(education, list) else json.loads(education or "[]")

    skill_params = [
        {"name": _norm(s["name"]), "years": float(s.get("years") or 0)}
        for s in skills if s.get("name")
    ]
    role_params = []
    for r in roles:
        if not r.get("company"):
            continue
        start, end = r.get("start_year"), r.get("end_year")
        tenure = float((end or start or 0) - start) if start else 1.0
        role_params.append({
            "company": _norm(r["company"]),
            "tenure": max(tenure, 1.0),
            "domain": _norm(r["domain"]) if r.get("domain") else None,
        })
    inst_params = [_norm(e["institution"]) for e in education if e.get("institution")]

    return {
        "id": cid,
        "name": name or cid,
        "years": float(years) if years is not None else None,
        "skills": skill_params,
        "roles": role_params,
        "institutions": inst_params,
    }


def main():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT candidate_id, name, total_years_experience, skills, roles, education "
            "FROM candidates"
        )
        rows = cur.fetchall()

    print(f"Building Neo4j graph from {len(rows)} candidates...")

    with get_session() as session:
        # Fresh rebuild for determinism/idempotency.
        session.run("MATCH (n) DETACH DELETE n")
        for stmt in CONSTRAINTS:
            session.run(stmt)

        for row in rows:
            p = _candidate_payload(*row)
            session.run(MERGE_CANDIDATE, id=p["id"], name=p["name"], years=p["years"])
            if p["skills"]:
                session.run(ADD_SKILLS, id=p["id"], skills=p["skills"])
            if p["roles"]:
                session.run(ADD_ROLES, id=p["id"], roles=p["roles"])
            if p["institutions"]:
                session.run(ADD_INSTITUTIONS, id=p["id"], institutions=p["institutions"])

        session.run(BUILD_COOCCURRENCE)

        # Report node/relationship counts.
        node_counts = session.run(
            "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS c ORDER BY l"
        ).data()
        rel_counts = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY rel"
        ).data()

    print("Neo4j graph build complete.")
    print("  Nodes:", ", ".join(f"{r['label']}={r['c']}" for r in node_counts))
    print("  Edges:", ", ".join(f"{r['rel']}={r['c']}" for r in rel_counts))


if __name__ == "__main__":
    main()
