"""FastAPI surface. Run: uvicorn src.api:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval import retrieve
from src.generation import generate
from src.neo4j_retrieval import similar_candidates

app = FastAPI(title="Resume RAG")


class Query(BaseModel):
    query: str
    # "auto" | "filter" | "semantic" | "hybrid" | "graph" | "graph_hybrid"
    mode: str = "auto"
    top_k: int = 10


@app.post("/query")
def query_endpoint(q: Query):
    result = retrieve(q.query, mode=q.mode, top_k=q.top_k)
    answer = generate(q.query, result["candidates"])
    return {
        "query": q.query,
        "mode": result["mode"],
        "route": result["route"],
        "candidates": [
            {"id": c["id"], "name": c["name"], "score": c["score"], "years": c["years"]}
            for c in result["candidates"]
        ],
        "answer": answer,
    }


@app.get("/similar/{candidate_id}")
def similar_endpoint(candidate_id: str, top_k: int = 10):
    """Graph RAG: candidates most similar to this one by shared graph neighbors (Neo4j)."""
    ranked = similar_candidates(candidate_id, k=top_k)
    return {
        "candidate_id": candidate_id,
        "similar": [{"id": cid, "score": score} for cid, score in ranked],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
