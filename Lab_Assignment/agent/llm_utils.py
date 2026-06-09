"""LLM factory — reuses Vertex AI config from the Day09 .env / vertex-key.json."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up two levels from this file to find the Day09 project root.
_HERE = Path(__file__).parent          # Lab_Assignment/agent/
_LAB_ROOT = _HERE.parent               # Lab_Assignment/
_DAY09_ROOT = _LAB_ROOT.parent         # Day09 project root

for _env_path in (_LAB_ROOT / ".env", _DAY09_ROOT / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
        break

# Auto-discover vertex-key.json if GOOGLE_APPLICATION_CREDENTIALS not set
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    for _kp in (_LAB_ROOT / "vertex-key.json", _DAY09_ROOT / "vertex-key.json"):
        if _kp.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_kp)
            break


def get_llm():
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(
        model=os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        project=os.getenv("VERTEX_PROJECT", "vinuni-project"),
        location=os.getenv("VERTEX_LOCATION", "asia-southeast1"),
        max_output_tokens=2048,
    )


def extract_text(content) -> str:
    """Convert Gemini list-format content to plain string."""
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content) if content else ""
