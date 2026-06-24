# Enterprise RAG Design — Heterogeneous Data (Manufacturing Context)

> How to build a RAG knowledge base over both structured databases (SQL/NoSQL) and unstructured documents (PDFs, Confluence, SharePoint) in a manufacturing environment.

## The Problem

A manufacturing department has data in two fundamentally different worlds:

| Data type | Examples | Current home |
|-----------|----------|-------------|
| **Structured** | Production line throughput, batch quality metrics, inventory levels, equipment maintenance schedules, supplier lead times | SAP/ERP (SQL Server), IoT time-series (MongoDB/InfluxDB), Data warehouse (Snowflake) |
| **Unstructured** | Standard Operating Procedures (SOPs), equipment manuals, safety datasheets, incident reports, troubleshooting guides | PDFs on SharePoint, Confluence wiki, email threads, scanned paper docs |
| **Semi-structured** | Work orders (JSON), lab test results (XML/CSV), regulatory compliance forms | Mixed — some in databases, some as files |

A plant operator asking *"What's the cleaning procedure for Line 3 after switching from Pepsi to Mountain Dew, and when was the last time we had a contamination event on that line?"* needs data from **both worlds** in a single answer.

---

## Architecture Overview

```
DATA SOURCES
├── Structured: SQL Server (ERP/SAP), MongoDB (IoT), Snowflake (DW)
└── Unstructured: PDFs (SOPs), Confluence (runbooks), SharePoint (reports), Email/Chat

        │
        ▼

INGESTION PIPELINE
  Source Connectors → Document Parsing → LLM Extraction → Chunking → Embedding
  └── Lineage Tracker: source_id, version, last_synced, parent_doc, access_control

        │
        ▼

UNIFIED KNOWLEDGE BASE (Single Postgres)
├── Vector Store (pgvector HNSW) — semantic similarity
├── Full-Text Index (tsvector) — keyword BM25
├── Structured Tables (JSONB) — SQL-queryable metadata and filters
└── Knowledge Graph (entity_relations) — multi-hop traversal via recursive CTEs

        │
        ▼

MULTI-STRATEGY RETRIEVAL
  Query Router (LLM classifies intent)
  → Text2SQL | Semantic | Keyword | Graph Walk | Hybrid
  → RRF Fusion → Re-rank
  → Access Control Filter

        │
        ▼

GENERATION + GROUNDING
  Claude Sonnet — grounded answer with source citations [doc_id:chunk_id]
  Guardrails: hallucination check, access control, confidence scoring, freshness
```

---

## Phase 1: Source Connectors — Getting the Data Out

This is the hardest engineering problem and the one most people skip in interviews.

### Structured Source Connectors

**Change Data Capture (CDC)** is the key pattern. You don't want to full-scan a 500 GB production database every night.

| Source | Connector approach | Why |
|--------|-------------------|-----|
| SQL Server (SAP/ERP) | **Debezium CDC** → Kafka → ingestion pipeline | Captures INSERT/UPDATE/DELETE in real-time from the transaction log. No polling, no load on the source DB |
| MongoDB (IoT sensor data) | **MongoDB Change Streams** → Kafka | Native CDC support, captures oplog events |
| Snowflake (Data Warehouse) | **Scheduled query export** (daily/hourly) via Snowflake Tasks → S3 → pipeline | DW data is already aggregated; CDC is overkill |

**What gets extracted from structured sources:**

```json
{
  "source_type": "structured",
  "source_id": "sap://PROD_BATCH/2024/B-10342",
  "entities": {
    "line": "Line 3",
    "product": "Mountain Dew",
    "batch_id": "B-10342",
    "start_time": "2024-03-15T06:00:00Z",
    "yield_pct": 97.2,
    "quality_grade": "A"
  },
  "text_representation": "Production batch B-10342 on Line 3 produced Mountain Dew on March 15, 2024. Yield was 97.2%, quality grade A. CIP cleaning completed at 05:45 before production start.",
  "lineage": { "table": "PROD_BATCHES", "pk": "B-10342", "extracted_at": "..." }
}
```

The critical insight: **structured data must be converted to text for embedding, but the original structured form must be preserved for SQL-style filtering.** You store both.

### Unstructured Source Connectors

| Source | Connector approach | Challenges |
|--------|-------------------|------------|
| PDFs (SOPs, manuals) | **S3 event trigger** — when a PDF lands in the docs bucket, invoke the parser | Layout-aware parsing needed: tables, headers, multi-column, diagrams |
| Confluence | **Confluence REST API** polling (or webhook on page update) | Pages have nested child pages; must crawl the tree. Rich formatting (macros, tables) |
| SharePoint | **Microsoft Graph API** with delta queries | Auth is complex (OAuth2 with tenant admin consent). Delta queries give you incremental changes |
| Email/Chat (incidents) | **Event-driven** — subscribe to mailbox via Graph API or Slack Events API | Need to handle threads as a unit, not individual messages |
| Scanned paper docs | PDF → **OCR** (AWS Textract or Azure Document Intelligence) → text | Quality varies wildly; need confidence scoring on OCR output |

**The connector must track:**

```python
{
    "source_id": "confluence://PLANT_OPS/Line-3-Cleaning-SOP",
    "version": 14,                    # Confluence page version
    "last_synced": "2024-03-15T...",
    "content_hash": "sha256:abc...",  # Detect no-op updates
    "access_control": ["plant_ops", "quality_team"],
}
```

**Why `content_hash`?** Confluence webhooks fire on metadata changes too (labels, comments). Hashing the content body lets you skip re-processing when only metadata changed.

---

## Phase 2: Document Parsing

This is where most RAG systems fail silently. Bad parsing → bad chunks → bad retrieval → hallucinated answers.

### PDF Parsing Strategy

Manufacturing PDFs are especially tricky — they contain:
- **Tables** (spec sheets, quality limits, ingredient lists)
- **Diagrams** (P&ID piping diagrams, process flows)
- **Multi-column layouts** (regulatory docs)
- **Headers/footers** (document IDs, revision numbers)

**Approach: Layout-aware parsing with table extraction**

```
PDF
 ├─ AWS Textract (or unstructured.io) for layout detection
 │   ├─ TITLE blocks → section headers
 │   ├─ TABLE blocks → structured rows (preserve as markdown tables)
 │   ├─ TEXT blocks → prose paragraphs
 │   └─ FIGURE blocks → caption text + "[Figure: description]" placeholder
 │
 └─ Reassemble into sections preserving document hierarchy:
      Chapter 4: CIP Cleaning Procedures
        4.1 Line 3 — Carbonated Beverages
          4.1.1 Product Changeover Protocol
            [prose paragraph]
            [table: cleaning agent concentrations]
          4.1.2 Post-Cleaning Verification
            [prose paragraph]
```

### Confluence Parsing

Confluence pages use a storage format (XHTML-like). Key challenges:
- **Macros** (`{code}`, `{panel}`, `{table-plus}`) need expansion
- **Page trees** — a runbook might span 15 child pages
- **Inline images** — extract alt text, skip decorative images

### Structured Data → Text Representation

A database row like:

```sql
SELECT * FROM quality_events WHERE line = 'Line 3' AND event_type = 'contamination';
-- Returns: {date: '2024-01-12', line: 'Line 3', contaminant: 'citric acid residue',
--           root_cause: 'Incomplete rinse cycle', resolution: 'Extended rinse to 45 min'}
```

Must become a **text chunk** that can be embedded:

> "On January 12, 2024, a contamination event occurred on Line 3. Citric acid residue was detected after a product changeover. Root cause analysis determined the rinse cycle was incomplete. Resolution: extended the rinse cycle duration to 45 minutes. This was classified as a minor quality deviation, no product was shipped."

**Use an LLM for this conversion.** A template-based approach ("On {date}, a {event_type} occurred...") works for simple cases, but an LLM produces more natural, contextually rich text that embeds better.

---

## Phase 3: Chunking Strategy — Section-Aware with Parent Context

### Chunking Rules by Content Type

| Content type | Chunking strategy | Chunk size | Why |
|-------------|-------------------|-----------|-----|
| **SOP sections** | By numbered section (4.1, 4.1.1, etc.) | Natural section boundaries | SOPs are already structured; respect their hierarchy |
| **Tables** | Entire table as one chunk + each row as a sub-chunk | Varies | Tables lose meaning when split mid-row |
| **Confluence pages** | By heading (h1/h2/h3) | ~500–1000 tokens | Wiki pages are already organized by heading |
| **Database rows** | By entity (one equipment, one batch, one incident) | One text repr per entity | Entities are atomic units of meaning |
| **Incident reports** | By section (timeline, root cause, resolution, action items) | Natural sections | Users ask about specific phases of an incident |

### Parent-Child Hierarchy

Every chunk gets **parent context prepended**:

```
Chunk ID: sop_line3_cleaning_v14_s4.1.1
Parent: "Chapter 4: CIP Cleaning Procedures > 4.1 Line 3 — Carbonated Beverages"
Section: "4.1.1 Product Changeover Protocol"
Text: "Before initiating a product changeover on Line 3, ensure the previous
       product has fully drained. Open valve V-302 to flush the line with
       hot water at 82°C for 15 minutes..."
Source: confluence://PLANT_OPS/Line-3-Cleaning-SOP (v14, synced 2024-03-15)
```

**Why parent context?** Without it, the chunk "Open valve V-302 to flush the line..." loses all context about *which* line, *which* procedure, *which* product. Embedding this orphaned text produces terrible retrieval. Prepending the breadcrumb gives the embedding model the context it needs.

---

## Phase 4: The Unified Knowledge Base Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table — the "source of truth" registry
CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,          -- "structured", "unstructured", "semi-structured"
    source_system   TEXT NOT NULL,          -- "sap", "confluence", "sharepoint", "mongodb"
    title           TEXT,
    doc_metadata    JSONB NOT NULL,
    entities        JSONB,                  -- extracted entities: {line, product, equipment, ...}
    access_control  TEXT[] NOT NULL,        -- groups that can see this doc
    content_hash    TEXT NOT NULL,
    version         INTEGER DEFAULT 1,
    last_synced     TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Chunks table — text + vectors + full-text search
CREATE TABLE chunks (
    chunk_id        BIGSERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section         TEXT,
    parent_context  TEXT,                   -- breadcrumb for contextual anchoring
    text            TEXT NOT NULL,
    embedding       vector(384) NOT NULL,
    fts             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    chunk_metadata  JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Entity relationship table — the knowledge graph layer
CREATE TABLE entity_relations (
    id              BIGSERIAL PRIMARY KEY,
    subject_id      TEXT NOT NULL,          -- "equipment://Line-3"
    predicate       TEXT NOT NULL,          -- "had_incident", "produces", "maintained_by"
    object_id       TEXT NOT NULL,          -- "incident://INC-2024-0042"
    source_doc_id   TEXT REFERENCES documents(doc_id),
    confidence      REAL DEFAULT 1.0,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_fts ON chunks USING gin (fts);
CREATE INDEX idx_chunks_doc ON chunks (doc_id);
CREATE INDEX idx_docs_entities ON documents USING gin (entities);
CREATE INDEX idx_docs_source ON documents (source_system, source_type);
CREATE INDEX idx_docs_access ON documents USING gin (access_control);
CREATE INDEX idx_relations_subject ON entity_relations (subject_id);
CREATE INDEX idx_relations_object ON entity_relations (object_id);
CREATE INDEX idx_relations_predicate ON entity_relations (predicate, subject_id);
```

**Key design decision: single Postgres for everything.** Vectors, full-text, structured JSONB, and graph traversal (via recursive CTEs) all in one store. No cross-store consistency problems. Split only when you hit scale limits (~10M+ chunks for HNSW, ~100M+ rows for FTS).

---

## Phase 5: Multi-Strategy Retrieval

Five retrieval strategies, orchestrated by an enhanced query router.

### Enhanced Query Router

```python
ROUTER_PROMPT = """Classify this manufacturing query. Return:
- strategies: list of retrieval strategies to use (1-3)
- sql_filter: structured filter if applicable
- search_query: rewritten query for semantic/keyword search
- graph_seed: entity to start graph traversal from, if applicable

Strategies:
  "text2sql"  — answerable from structured DB (metrics, counts, dates, aggregations)
  "semantic"  — needs fuzzy document matching (procedures, explanations, how-to)
  "keyword"   — exact term matching (equipment IDs, chemical names, part numbers)
  "graph"     — needs to connect related entities (incident → equipment → procedure)
  "hybrid"    — combine semantic + keyword with RRF fusion

Examples:
  "What's the OEE for Line 3 last month?" → ["text2sql"]
  "How do I clean Line 3 after switching products?" → ["semantic"]
  "Show me all incidents involving valve V-302" → ["keyword", "graph"]
  "What cleaning procedure should we follow for Line 3, and have there
   been contamination issues with it?" → ["semantic", "text2sql", "graph"]
"""
```

### Strategy Implementations

**Text2SQL** — LLM generates SQL against structured tables:

```python
def text2sql_retrieve(query, sql_filter, conn):
    schema_context = get_relevant_table_schemas(sql_filter)
    sql = llm_generate_sql(query, schema_context)
    # CRITICAL: validate and sandbox the SQL
    sql = validate_sql(sql, allowed_tables=QUERY_TABLES, max_rows=100)
    results = conn.execute(sql)
    return [{"type": "sql_result", "data": row, "text": row_to_text(row)}
            for row in results]
```

Safety: Never let the LLM run arbitrary SQL against production. Use a **read-only replica** with a **restricted role** that can only SELECT from designated views. Validate the generated SQL with an AST parser — reject DELETE, UPDATE, DROP, subqueries to system tables, etc.

**Graph Traversal** — walk entity relationships:

```python
def graph_retrieve(seed_entity, max_hops=2, conn):
    results = conn.execute("""
        WITH RECURSIVE graph AS (
            SELECT subject_id, predicate, object_id, 1 as depth, source_doc_id
            FROM entity_relations
            WHERE subject_id = %s OR object_id = %s
            UNION
            SELECT er.subject_id, er.predicate, er.object_id, g.depth + 1, er.source_doc_id
            FROM entity_relations er
            JOIN graph g ON (er.subject_id = g.object_id OR er.object_id = g.subject_id)
            WHERE g.depth < %s
        )
        SELECT DISTINCT * FROM graph
    """, [seed_entity, seed_entity, max_hops])
    doc_ids = {r.source_doc_id for r in results}
    return fetch_chunks_for_docs(doc_ids, conn)
```

**Semantic and Keyword** — same vector_search() and bm25_search() patterns.

**RRF Fusion** — same pattern, now merging results from 3-4 strategies instead of 2.

### Access Control Filtering

Every query is filtered by the user's permissions at retrieval time, **before** passing to the LLM:

```python
def retrieve(query, user_groups, conn):
    # ... run retrieval strategies ...
    # Filter by access control BEFORE passing to LLM
    results = [r for r in results
               if has_access(r.doc_id, user_groups, conn)]
    return results
```

If you pass restricted documents to the LLM and ask it to "ignore documents the user can't see," the LLM might still leak information in its reasoning. Filter before the LLM ever sees the content.

---

## Phase 6: Generation with Grounded Citations

```python
GENERATION_PROMPT = """You are a manufacturing knowledge assistant for PepsiCo.
Answer the question using ONLY the provided context. For every claim, cite the
source using [source_id] notation.

If the context includes both structured data (metrics, dates) and unstructured
data (procedures, explanations), synthesize them into a coherent answer.

If the context is insufficient, say so explicitly. Never guess at safety-critical
information (cleaning procedures, chemical concentrations, equipment settings).

Context:
{context}

Question: {query}
"""
```

### Manufacturing-Specific Guardrails

1. **Safety-critical answers require high confidence.** If the question is about a cleaning chemical concentration or equipment pressure setting, and the retrieved context has conflicting values or low similarity scores, flag it: *"I found conflicting specifications. Please verify with the latest SOP revision before proceeding."*

2. **Freshness matters.** An SOP from 2019 might be superseded. Include the document version and date in citations so the user can judge recency.

3. **Structured + unstructured synthesis.** The answer to "What cleaning procedure should we follow for Line 3, and have there been contamination issues?" combines:
   - SOP text (unstructured) → the procedure
   - Quality events table (structured) → contamination history
   - Entity graph → links Line 3 to specific incidents to their root causes

---

## Phase 7: Keeping the Knowledge Base Fresh

| Source type | Sync strategy | Frequency | Staleness tolerance |
|-------------|--------------|-----------|-------------------|
| ERP/SAP production data | CDC via Debezium (real-time) | Continuous | Minutes — operators need current batch status |
| IoT sensor data | Stream processing (Kafka → aggregate → ingest) | Near real-time | Seconds for alerts, hours for historical queries |
| SOPs/Manuals (PDFs) | S3 event trigger on upload | On change | Days — these change infrequently |
| Confluence pages | Webhook on page update + daily full crawl | On change + daily | Hours |
| Quality incidents | CDC from incident management system | Continuous | Minutes during active incidents |

### Versioning and Deduplication

```python
def should_reingest(doc_id, new_content_hash, conn):
    existing = conn.execute(
        "SELECT content_hash, version FROM documents WHERE doc_id = %s", [doc_id]
    ).fetchone()
    if not existing:
        return True, 1          # New document
    if existing.content_hash == new_content_hash:
        return False, existing.version  # No change, skip
    return True, existing.version + 1   # Changed, bump version
```

When a document is re-ingested: delete old chunks for that `doc_id` (CASCADE handles this), re-extract, re-chunk, re-embed, insert new chunks. The HNSW index updates automatically.

---

## Mapping to the Resume-RAG Codebase

| Resume-RAG component | Enterprise manufacturing equivalent |
|---------------------|--------------------------------------|
| `scripts/download_data.py` (CSV download) | Source connectors: CDC, API crawlers, S3 event triggers |
| `extraction.py` (LLM extracts from resume text) | LLM extraction from PDFs + structured-to-text conversion |
| `build_chunks()` (section-aware: summary, experience, skills) | Section-aware chunking per content type + parent-context prepending |
| `candidates` table (JSONB skills, roles) | `documents` table with JSONB `entities` + `entity_relations` graph |
| `chunks` table (vector + tsvector) | Same, plus access control filtering |
| `route_query()` (filter / semantic / hybrid) | Enhanced router: text2sql / semantic / keyword / graph / hybrid |
| `rrf_fuse()` (merge vector + BM25) | Same RRF, now merging 3-4 strategies |
| `generate()` (Claude with citations) | Same, plus safety-critical guardrails and freshness indicators |
| `run_evals.py` (recall, precision, hit) | Same metrics + domain-specific evals (safety answer accuracy, citation correctness) |

The architecture is the same skeleton — **ingest → extract → chunk → embed → retrieve → generate** — but each layer gets wider to handle heterogeneous data sources, and the retrieval layer adds Text2SQL and graph traversal for the structured data that pure vector search can't handle.
