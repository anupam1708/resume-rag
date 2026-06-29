"""
LangSmith eval runner. Compares three retrieval modes on the eval set.

Run: python evals/run_evals.py

Outputs recall@10 and precision@10 per query and overall, logged to LangSmith
as three experiments: vector-only, bm25-only, hybrid-rrf.

Uses the resume CSV's `category` column as ground truth: a candidate is
"relevant" if its category is in the query's expected_categories.

This is approximate ground truth — good enough to show that hybrid beats
vector-only and bm25-only. For production you'd want hand-labeled
candidate ID expected sets, not category-based.
"""
import json
import os
from pathlib import Path
import pandas as pd
from langsmith import Client
from langsmith.evaluation import evaluate

from src.retrieval import vector_search, bm25_search, rrf_fuse, filter_search, route_query
from src.graph_retrieval import graph_search

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESUMES = Path(__file__).parent.parent / "data" / "resumes.csv"
DATASET_NAME = "resume-rag-eval-v1"

client = Client()


def load_ground_truth() -> dict[str, str]:
    """candidate_id -> category"""
    df = pd.read_csv(RESUMES)
    return dict(zip(df["candidate_id"], df["category"]))


def ensure_dataset() -> None:
    """Create dataset in LangSmith if it doesn't exist."""
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' already exists.")
        return
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Resume RAG retrieval eval set",
    )
    with open(EVAL_SET) as f:
        examples = json.load(f)
    client.create_examples(
        inputs=[{"query": e["query"]} for e in examples],
        outputs=[{
            "expected_categories": e["expected_categories"],
            "expected_skills": e["expected_skills"],
            "min_relevant": e["min_relevant"],
        } for e in examples],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples.")


GROUND_TRUTH = load_ground_truth()


def is_relevant(candidate_id: str, expected_categories: list[str]) -> bool:
    return GROUND_TRUTH.get(candidate_id) in expected_categories


def recall_at_10(run, example) -> dict:
    retrieved = [c["id"] for c in run.outputs["candidates"][:10]]
    expected_cats = example.outputs["expected_categories"]
    # Total relevant in corpus
    total_relevant = sum(1 for cat in GROUND_TRUTH.values() if cat in expected_cats)
    hits = sum(1 for cid in retrieved if is_relevant(cid, expected_cats))
    score = hits / total_relevant if total_relevant else 0.0
    return {"key": "recall@10", "score": score}


def precision_at_10(run, example) -> dict:
    retrieved = [c["id"] for c in run.outputs["candidates"][:10]]
    expected_cats = example.outputs["expected_categories"]
    hits = sum(1 for cid in retrieved if is_relevant(cid, expected_cats))
    score = hits / len(retrieved) if retrieved else 0.0
    return {"key": "precision@10", "score": score}


def hit_at_10(run, example) -> dict:
    """Did we get at least min_relevant correct candidates in top 10?"""
    retrieved = [c["id"] for c in run.outputs["candidates"][:10]]
    expected_cats = example.outputs["expected_categories"]
    min_rel = example.outputs["min_relevant"]
    hits = sum(1 for cid in retrieved if is_relevant(cid, expected_cats))
    return {"key": "hit@10", "score": 1.0 if hits >= min_rel else 0.0}


# --------------------------------------------------------------------
# Three target functions (the things we're comparing)
# --------------------------------------------------------------------

def hydrate(ranked):
    return {"candidates": [{"id": cid, "score": s} for cid, s in ranked]}


def target_vector(inputs: dict) -> dict:
    return hydrate(vector_search(inputs["query"], k=10))


def target_bm25(inputs: dict) -> dict:
    return hydrate(bm25_search(inputs["query"], k=10))


def target_hybrid(inputs: dict) -> dict:
    vec = vector_search(inputs["query"], k=20)
    bm = bm25_search(inputs["query"], k=20)
    fused = rrf_fuse([vec, bm], top_k=10)
    return hydrate(fused)


def target_graph(inputs: dict) -> dict:
    # Requires the graph to be built first: python -m src.graph_build
    return hydrate(graph_search(inputs["query"], k=10))


# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------

def main():
    ensure_dataset()
    evaluators = [recall_at_10, precision_at_10, hit_at_10]

    for name, target in [
        ("vector-only", target_vector),
        ("bm25-only", target_bm25),
        ("hybrid-rrf", target_hybrid),
        ("graph", target_graph),
    ]:
        print(f"\n=== Running experiment: {name} ===")
        results = evaluate(
            target,
            data=DATASET_NAME,
            evaluators=evaluators,
            experiment_prefix=name,
        )
        print(f"Done. View at https://smith.langchain.com")


if __name__ == "__main__":
    main()
