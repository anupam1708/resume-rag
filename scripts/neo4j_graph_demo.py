"""
Neo4j Graph RAG demo helper — print (and optionally run) example graph queries.

Two modes:
    python -m scripts.neo4j_graph_demo          # print examples + traversal paths
                                                # (no DB or API key needed)
    python -m scripts.neo4j_graph_demo --run    # execute against the live Neo4j graph
                                                # (needs Neo4j + a built graph + ANTHROPIC_API_KEY)

Build the graph first for --run:
    docker compose up -d                 # starts postgres + neo4j
    python -m src.ingestion              # populate Postgres candidates
    python -m src.neo4j_graph_build      # materialize the Neo4j graph
"""
import sys

# Each example: the query, what graph relationship it exercises, and the path it walks.
EXAMPLES = [
    {
        "query": "Engineers who know technologies commonly paired with Java",
        "shows": "skill co-occurrence expansion, then back to candidates",
        "path": "(Java)-[:CO_OCCURS]-(Spring)<-[:HAS_SKILL]-(:Candidate)",
        "mode": "graph",
    },
    {
        "query": "Candidates from fintech companies who also know Python",
        "shows": "domain + skill seeds converging on the same candidates",
        "path": "(fintech)<-[:IN_DOMAIN]-(:Company)<-[:WORKED_AT]-(:Candidate)-[:HAS_SKILL]->(Python)",
        "mode": "graph",
    },
    {
        "query": "Who has skills related to cloud and DevOps?",
        "shows": "multi-skill seeding with spreading activation",
        "path": "(AWS|Kubernetes)-[:CO_OCCURS|HAS_SKILL]-(:Candidate)",
        "mode": "graph_hybrid",
    },
]


def print_examples():
    print("=" * 78)
    print("Neo4j Graph RAG — example queries")
    print("=" * 78)
    print(
        "\nGraph retrieval ranks candidates by *connectivity* to the entities in a\n"
        "query (shared skills / companies / domains), not by text similarity. Each\n"
        "example names the relationship it exercises and the Cypher path it walks.\n"
    )
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"[{i}] {ex['query']}")
        print(f"    mode  : {ex['mode']}")
        print(f"    shows : {ex['shows']}")
        print(f"    path  : {ex['path']}")
        print(
            "    curl  : "
            'curl -X POST localhost:8000/query -H "Content-Type: application/json" \\\n'
            f"            -d '{{\"query\": \"{ex['query']}\", \"mode\": \"{ex['mode']}\"}}'"
        )
        print()
    print("Similarity by shared graph neighbors (pick any ingested candidate_id):")
    print("    curl localhost:8000/similar/c_0000\n")
    print("Run these live against the built graph with:  python -m scripts.neo4j_graph_demo --run")


def run_examples():
    # Imported lazily so plain `print` mode needs no DB / API key / deps.
    from src.neo4j_db import get_session
    from src.retrieval import route_query
    from src.neo4j_retrieval import graph_rank, similar_candidates

    # 1. Graph stats — also confirms the graph has actually been built.
    with get_session() as session:
        nodes = session.run(
            "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS c ORDER BY l"
        ).data()
        edges = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY rel"
        ).data()
        first = session.run(
            "MATCH (c:Candidate) RETURN c.id AS id ORDER BY c.id LIMIT 1"
        ).single()

    if not nodes:
        print("Graph is empty. Build it first:")
        print("  docker compose up -d")
        print("  python -m src.ingestion")
        print("  python -m src.neo4j_graph_build")
        sys.exit(1)

    print("Graph stats")
    print("  nodes:", ", ".join(f"{r['label']}={r['c']}" for r in nodes))
    print("  edges:", ", ".join(f"{r['rel']}={r['c']}" for r in edges))
    print()

    # 2. Run each example query through the graph retriever.
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"[{i}] {ex['query']}")
        route = route_query(ex["query"])
        print(f"    router -> mode={route.get('mode')} "
              f"skills={route.get('skills')} domains={route.get('domains')}")
        ranked = graph_rank(route.get("skills", []), route.get("domains", []), k=5)
        if ranked:
            for cid, score in ranked:
                print(f"      {cid}  score={score:.3f}")
        else:
            print("      (no connected candidates — entities may be absent from the graph)")
        print()

    # 3. Similarity-by-shared-neighbors for the first candidate.
    if first:
        cid = first["id"]
        print(f"Candidates similar to {cid} (shared graph neighbors):")
        for sid, score in similar_candidates(cid, k=5):
            print(f"  {sid}  score={score:.3f}")


def main():
    if "--run" in sys.argv[1:]:
        run_examples()
    else:
        print_examples()


if __name__ == "__main__":
    main()
