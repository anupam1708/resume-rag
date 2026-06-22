"""FastAPI surface. Run: uvicorn src.api:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval import retrieve
from src.generation import generate

app = FastAPI(title="Resume RAG")


class Query(BaseModel):
    query: str
    mode: str = "auto"  # "auto" | "filter" | "semantic" | "hybrid"
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


@app.get("/health")
def health():
    return {"status": "ok"}
