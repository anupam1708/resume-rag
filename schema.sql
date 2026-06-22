-- Resume RAG schema
-- Run: psql $DATABASE_URL -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Structured candidate data extracted by LLM at ingest time.
-- Filter queries ("who knows Java") hit this table via JSONB ops.
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    name TEXT,
    total_years_experience NUMERIC,
    skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    roles JSONB NOT NULL DEFAULT '[]'::jsonb,
    education JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- GIN index on skills JSONB for fast filter queries.
CREATE INDEX IF NOT EXISTS idx_candidates_skills ON candidates USING GIN (skills);
CREATE INDEX IF NOT EXISTS idx_candidates_years ON candidates (total_years_experience);

-- Chunks for retrieval. One row per section of a resume.
-- Section-aware chunking preserves semantic boundaries.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    section TEXT NOT NULL,    -- 'summary' | 'experience' | 'skills' | 'education'
    text TEXT NOT NULL,
    embedding vector(384),    -- all-MiniLM-L6-v2 dim
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

-- HNSW index for fast cosine similarity (pgvector >= 0.5)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- GIN index on FTS for BM25-like ranking
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (fts);

-- Helper: get candidate -> all chunks
CREATE INDEX IF NOT EXISTS idx_chunks_candidate ON chunks (candidate_id);
