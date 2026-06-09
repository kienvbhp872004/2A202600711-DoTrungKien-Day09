"""Supervisor node — analyses intent once, then routes workers until done."""

from __future__ import annotations

import unicodedata

from .state import RAGState

# ── Intent keyword sets ───────────────────────────────────────────────────────
_LEGAL_TOKENS = {
    "điều", "khoản", "nghị định", "luật", "blhs",
    "hình phạt", "mức án", "khung hình phạt", "tội danh", "quy định",
    "dieu", "khoan", "nghi dinh", "luat", "hinh phat", "toi danh", "quy dinh",
}
_NEWS_TOKENS = {
    "nghệ sĩ", "ca sĩ", "người nổi tiếng", "bị bắt", "vụ án", "showbiz",
    "nghe si", "ca si", "nguoi noi tieng", "bi bat", "vu an",
    "long nhat", "son ngoc minh", "miu le",
}
_CONSEQUENCE_TOKENS = {
    "như thế nào", "bao lâu", "bao nhiêu", "xử lý", "phạt", "tội gì",
    "đi tù", "tù", "án", "khởi tố",
    "nhu the nao", "bao lau", "bao nhieu", "xu ly", "phat", "toi gi",
    "di tu", "tu", "an", "khoi to",
}


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _matches(tokens: set[str], q: str, q_norm: str) -> bool:
    for tok in tokens:
        if tok in q or _norm(tok) in q_norm:
            return True
    return False


def classify_intent(question: str) -> tuple[bool, bool]:
    """Return (needs_legal, needs_news) based on keyword heuristics."""
    q = question.lower()
    q_norm = _norm(question)

    has_legal = _matches(_LEGAL_TOKENS, q, q_norm)
    has_news = _matches(_NEWS_TOKENS, q, q_norm)
    has_consequence = _matches(_CONSEQUENCE_TOKENS, q, q_norm)

    needs_legal = has_legal or has_consequence
    needs_news = has_news

    # Celebrity + consequence → need both
    if has_news and has_consequence:
        needs_legal = True

    # Fallback: if nothing matched, retrieve everything
    if not needs_legal and not needs_news:
        needs_legal = True
        needs_news = True

    return needs_legal, needs_news


def _next_worker(
    needs_legal: bool,
    needs_news: bool,
    legal_chunks: list,
    news_chunks: list,
) -> str:
    if needs_legal and not legal_chunks:
        return "legal_research"
    if needs_news and not news_chunks:
        return "news_context"
    return "answer_generator"


def supervisor_node(state: RAGState) -> dict:
    """
    Supervisor logic:
    1. First call → classify intent, decide first worker.
    2. Subsequent calls → check what's already collected, route to next worker.
    """
    question = state.get("question", "")
    legal_chunks = state.get("legal_chunks") or []
    news_chunks = state.get("news_chunks") or []
    needs_legal = state.get("needs_legal")  # None = not yet classified

    if needs_legal is None:
        nl, nn = classify_intent(question)
        return {
            "needs_legal": nl,
            "needs_news": nn,
            "next_worker": _next_worker(nl, nn, [], []),
        }

    needs_news = state.get("needs_news", False)
    return {"next_worker": _next_worker(needs_legal, needs_news, legal_chunks, news_chunks)}
