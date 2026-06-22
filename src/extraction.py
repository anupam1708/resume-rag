"""
LLM-based structured extraction of resume content.

This is the most important component for query quality. The richer and more
consistent the extracted structure, the better filter queries work without
falling back to vector similarity.
"""
import json
from anthropic import Anthropic
from src.config import ANTHROPIC_API_KEY, EXTRACT_MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)


EXTRACTION_PROMPT = """You are extracting structured data from a resume.

Output ONLY valid JSON matching this schema:
{
  "name": string | null,
  "total_years_experience": number,
  "skills": [
    {
      "name": string,           // canonical skill name (e.g. "Java", not "java programming")
      "years": number,           // estimated years of hands-on experience
      "last_used_year": number | null,  // most recent year you can infer usage
      "proficiency": "expert" | "advanced" | "intermediate" | "beginner"
    }
  ],
  "roles": [
    {
      "title": string,
      "company": string,
      "start_year": number,
      "end_year": number | null,    // null if current role
      "domain": string              // e.g. "fintech", "healthcare", "e-commerce", "automotive"
    }
  ],
  "education": [
    {"degree": string, "institution": string, "year": number | null}
  ],
  "summary": string                  // 2-3 sentence professional summary
}

Rules:
- Canonical skill names: "JavaScript" not "JS", "PostgreSQL" not "postgres",
  "Amazon Web Services" -> "AWS", "Spring Boot" not "springboot".
- Estimate years per skill from the role durations where that skill is mentioned.
  If unclear, lower-bound it conservatively.
- Domain: classify each role into one consistent set: fintech, banking, insurance,
  healthcare, retail, e-commerce, telecom, automotive, education, media,
  government, consulting, manufacturing, technology, other.
- If a field cannot be determined, omit it (do not invent).
- No prose, no markdown fences, JSON only.

Resume:
---
{resume_text}
---
"""


def extract(resume_text: str) -> dict:
    """Extract structured fields from a resume. Returns dict matching schema."""
    msg = client.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.replace("{resume_text}", resume_text[:15000])
        }]
    )
    raw = msg.content[0].text.strip()
    # Belt-and-suspenders: strip code fences if model added them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Self-correction: ask model to fix its JSON
        fix = client.messages.create(
            model=EXTRACT_MODEL,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"Fix this JSON to be valid. Output JSON only, no prose:\n{raw}"
            }]
        )
        return json.loads(fix.content[0].text.strip())


def build_chunks(extracted: dict, raw_text: str) -> list[dict]:
    """
    Build section-aware chunks for embedding.
    Returns list of {section, text} dicts.
    """
    chunks = []

    if extracted.get("summary"):
        chunks.append({"section": "summary", "text": extracted["summary"]})

    # One chunk per role: gives the embedding model role-level context.
    for role in extracted.get("roles", []):
        text = (
            f"{role.get('title', '')} at {role.get('company', '')} "
            f"({role.get('start_year', '')}-{role.get('end_year') or 'present'}). "
            f"Domain: {role.get('domain', '')}."
        )
        chunks.append({"section": "experience", "text": text})

    # Skills aggregated: helps with "who knows X" semantic fallback.
    if extracted.get("skills"):
        skill_text = "Skills: " + ", ".join(
            f"{s['name']} ({s.get('years', '?')}y)"
            for s in extracted["skills"]
        )
        chunks.append({"section": "skills", "text": skill_text})

    for edu in extracted.get("education", []):
        chunks.append({
            "section": "education",
            "text": f"{edu.get('degree', '')} from {edu.get('institution', '')}"
        })

    return chunks
