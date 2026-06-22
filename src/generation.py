"""Final answer generation with citations to specific candidates."""
import json
from anthropic import Anthropic
from src.config import ANTHROPIC_API_KEY, GEN_MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)


GEN_PROMPT = """You are a recruiting assistant. Answer the user's question using ONLY
the retrieved candidate profiles below. Every claim must cite the candidate ID
in square brackets, e.g. [c_0042].

If the retrieved candidates don't answer the question, say so plainly — do NOT
make up candidates or details.

Format:
- Lead with the direct answer
- Then 2-5 bullet candidates with one-line justifications

User question:
{query}

Retrieved candidates:
{candidates}
"""


def generate(query: str, candidates: list[dict]) -> str:
    candidates_str = "\n\n".join(
        f"[{c['id']}] {c.get('name') or '(no name)'}, "
        f"{c.get('years') or '?'} years experience\n"
        f"  Skills: {json.dumps(c.get('skills') or [])[:500]}\n"
        f"  Roles: {json.dumps(c.get('roles') or [])[:500]}"
        for c in candidates
    )
    msg = client.messages.create(
        model=GEN_MODEL,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": GEN_PROMPT.replace("{query}", query).replace("{candidates}", candidates_str)
        }]
    )
    return msg.content[0].text
