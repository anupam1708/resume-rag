-- Resume RAG — knowledge-graph schema (Graph RAG add-on)
-- Run AFTER schema.sql and AFTER ingestion: psql $DATABASE_URL -f graph.sql
--
-- The graph is DERIVED from the already-extracted `candidates` JSONB
-- (skills / roles / education). No new LLM calls are needed — src/graph_build.py
-- materializes nodes + edges from data that extraction.py already produced.
--
-- Node types : candidate | skill | company | domain | institution
-- Edge rels  : HAS_SKILL (candidate->skill)   weight = years
--              WORKED_AT (candidate->company)  weight = tenure years
--              IN_DOMAIN (company->domain)     weight = role count
--              STUDIED_AT(candidate->institution)
--              CO_OCCURS (skill<->skill)       weight = corpus co-occurrence count

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id   BIGSERIAL PRIMARY KEY,
    node_type TEXT NOT NULL,           -- candidate | skill | company | domain | institution
    key       TEXT NOT NULL,           -- canonical dedup key: candidate_id for candidates,
                                        -- lower(name) for everything else
    label     TEXT NOT NULL,           -- human-readable display name
    UNIQUE (node_type, key)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    src    BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    dst    BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    rel    TEXT NOT NULL,              -- HAS_SKILL | WORKED_AT | IN_DOMAIN | STUDIED_AT | CO_OCCURS
    weight NUMERIC NOT NULL DEFAULT 1,
    PRIMARY KEY (src, dst, rel)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges (src, rel);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges (dst, rel);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_key ON graph_nodes (node_type, key);

-- Undirected view: graph traversal in retrieval treats edges as bidirectional
-- (you reach candidates FROM a skill seed by following HAS_SKILL backwards).
-- A recursive CTE cannot self-reference inside a subquery, so we expose both
-- directions as a flat view the recursive step can join directly.
CREATE OR REPLACE VIEW graph_edges_undirected AS
    SELECT src, dst, rel, weight FROM graph_edges
    UNION ALL
    SELECT dst AS src, src AS dst, rel, weight FROM graph_edges;
