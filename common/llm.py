"""Shared LLM factory for all agents.

Uses Google Vertex AI with a service account key file.
"""

import os

from langchain_google_vertexai import ChatVertexAI


def extract_text(content) -> str:
    """Extract plain text from LLM content (handles Gemini list format)."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content) if content else ""


def get_llm() -> ChatVertexAI:
    """Return a ChatVertexAI client using service account credentials."""
    return ChatVertexAI(
        model=os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        project=os.getenv("VERTEX_PROJECT", "vinuni-project"),
        location=os.getenv("VERTEX_LOCATION", "asia-southeast1"),
        max_output_tokens=2048,
    )
