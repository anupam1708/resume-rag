"""Centralized config. Loads from .env."""
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")

# Neo4j (Graph RAG add-on). Optional — only the graph/graph_hybrid modes and
# graph build/retrieval use these. Defaults match the docker-compose service.
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Use Sonnet for extraction (structured task) and generation (final answer)
# Use Haiku for query routing (cheap classification)
EXTRACT_MODEL = "claude-sonnet-4-6"
GEN_MODEL = "claude-sonnet-4-6"
ROUTER_MODEL = "claude-haiku-4-5-20251001"

CHUNK_SECTIONS = ("summary", "experience", "skills", "education")
