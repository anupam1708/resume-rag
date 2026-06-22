"""
Ingestion pipeline: CSV -> extracted JSON -> Postgres + pgvector.

Run: python -m src.ingestion
"""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL
from src.db import get_conn
from src.extraction import extract, build_chunks

DATA = Path(__file__).parent.parent / "data" / "resumes.csv"


def main():
    df = pd.read_csv(DATA)
    print(f"Loading embedder ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    with get_conn() as conn:
        cur = conn.cursor()
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
            cid = row["candidate_id"]
            text = str(row["resume_text"])

            # Idempotency: skip if already ingested
            cur.execute("SELECT 1 FROM candidates WHERE candidate_id = %s", (cid,))
            if cur.fetchone():
                continue

            try:
                extracted = extract(text)
            except Exception as e:
                print(f"\nExtraction failed for {cid}: {e}")
                continue

            cur.execute(
                """
                INSERT INTO candidates
                  (candidate_id, name, total_years_experience, skills, roles, education, raw_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    cid,
                    extracted.get("name"),
                    extracted.get("total_years_experience"),
                    json.dumps(extracted.get("skills", [])),
                    json.dumps(extracted.get("roles", [])),
                    json.dumps(extracted.get("education", [])),
                    text,
                ),
            )

            chunks = build_chunks(extracted, text)
            if not chunks:
                continue
            embeddings = embedder.encode([c["text"] for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO chunks (candidate_id, section, text, embedding) "
                    "VALUES (%s, %s, %s, %s)",
                    (cid, chunk["section"], chunk["text"], emb.tolist()),
                )

            conn.commit()

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
