"""Centralized config. Loads from .env."""
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Use Sonnet for extraction (structured task) and generation (final answer)
# Use Haiku for query routing (cheap classification)
EXTRACT_MODEL = "claude-sonnet-4-6"
GEN_MODEL = "claude-sonnet-4-6"
ROUTER_MODEL = "claude-haiku-4-5-20251001"

CHUNK_SECTIONS = ("summary", "experience", "skills", "education")
