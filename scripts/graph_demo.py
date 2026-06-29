"""
Graph RAG demo helper — print (and optionally run) example graph queries.

Two modes:
    python -m scripts.graph_demo          # just print the examples + traversal paths
                                          # (no DB or API key needed)
    python -m scripts.graph_demo --run    # execute each against the live graph
                                          # (needs Postgres + a built graph + ANTHROPIC_API_KEY)

Run as a module from the repo root so `from src...` imports resolve
(see CLAUDE.md: run modules as packages, not scripts).

The graph must be built first for --run:
    psql "$DATABASE_URL" -f graph.sql
    python -m src.ingestion
    python -m src.graph_build
"""
import sys

# Each example: the query, what graph relationship it exercises, and the path it walks.
EXAMPLES = [
    {
        "query": "Engineers who know technologies commonly paired with Java",
        "shows": "skill co-occurrence expansion, then back to candidates",
        "path": "Java --CO_OCCURS--> {Spring, Maven, ...} --HAS_SKILL--> [candidate]",
        "mode": "graph",
    },
    {
        "query": "Candidates from fintech companies who also know Python",
        "shows": "domain + skill seeds converging on the same candidates",
        "path": "fintech <--IN_DOMAIN-- company <--WORKED_AT-- [candidate] --HAS_SKILL--> Python",
        "mode": "graph",
    },
    {
        "query": "Who has skills related to cloud and DevOps?",
        "shows": "multi-skill seeding with spreading activation",
        "path": "{AWS, Kubernetes, ...} --CO_OCCURS/HAS_SKILL--> [candidate]",
        "mode": "graph_hybrid",
    },
]


def print_examples():
    print("=" * 78)
    print("Graph RAG — example queries")
    print("=" * 78)
    print(
        "\nGraph retrieval ranks candidates by *connectivity* to the entities in a\n"
        "query (shared skills / companies / domains), not by text similarity. It\n"
        "shines on relational, multi-hop questions. Each example below names the\n"
        "relationship it exercises and the path the walk follows.\n"
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
    print("Run these live against the built graph with:  python -m scripts.graph_demo --run")


def run_examples():
    # Imported lazily so plain `print` mode needs no DB / API key / deps.
    from src.db import get_conn
    from src.retrieval import route_query
    from src.graph_retrieval import graph_rank, similar_candidates

    # 1. Graph stats — also confirms the graph has actually been built.
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT node_type, COUNT(*) FROM graph_nodes GROUP BY node_type ORDER BY 1")
        nodes = cur.fetchall()
        cur.execute("SELECT rel, COUNT(*) FROM graph_edges GROUP BY rel ORDER BY 1")
        edges = cur.fetchall()
        cur.execute("SELECT key FROM graph_nodes WHERE node_type = 'candidate' ORDER BY key LIMIT 1")
        first = cur.fetchone()

    if not nodes:
        print("Graph is empty. Build it first:")
        print('  psql "$DATABASE_URL" -f graph.sql')
        print("  python -m src.ingestion")
        print("  python -m src.graph_build")
        sys.exit(1)

    print("Graph stats")
    print("  nodes:", ", ".join(f"{t}={c}" for t, c in nodes))
    print("  edges:", ", ".join(f"{r}={c}" for r, c in edges))
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
        cid = first[0]
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
