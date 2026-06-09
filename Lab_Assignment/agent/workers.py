"""Three worker nodes for the Supervisor-Workers RAG pattern.

Worker 1 — LegalResearchWorker  : retrieves legal document chunks
Worker 2 — NewsContextWorker    : retrieves news article chunks
Worker 3 — AnswerGeneratorWorker: synthesises answer with citation

Each worker first tries the Day08 hybrid retrieval pipeline; if that is
unavailable (vector store not set up), it falls back to a direct Gemini call.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_utils import extract_text, get_llm
from .state import RAGState

# ── Optional Day08 retrieval ──────────────────────────────────────────────────
_DAY08_PATH = (
    Path(__file__).parent.parent.parent / "2A202600711-DoTrungKien-Day08"
)
_day08_retrieve = None


def _try_load_day08() -> None:
    global _day08_retrieve
    if not _DAY08_PATH.exists():
        return
    try:
        if str(_DAY08_PATH) not in sys.path:
            sys.path.insert(0, str(_DAY08_PATH))
        from src.task9_retrieval_pipeline import retrieve  # type: ignore

        _day08_retrieve = retrieve
    except Exception:
        pass


_try_load_day08()


# ── Worker 1: Legal Research ──────────────────────────────────────────────────

def legal_research_worker(state: RAGState) -> dict:
    """Retrieves legal document chunks relevant to the question."""
    question = state["question"]

    if _day08_retrieve is not None:
        try:
            all_chunks = _day08_retrieve(question, top_k=6)
            legal = [
                c
                for c in all_chunks
                if c.get("metadata", {}).get("type") == "legal"
            ]
            if legal:
                return {"legal_chunks": legal}
        except Exception:
            pass

    # Fallback: ask Gemini for relevant legal knowledge
    llm = get_llm()
    prompt = (
        f"Câu hỏi: {question}\n\n"
        "Hãy cung cấp thông tin pháp luật Việt Nam liên quan, bao gồm:\n"
        "• Điều khoản cụ thể từ Luật Phòng chống ma túy 2021 (Luật số 73/2021/QH15)\n"
        "• Điều khoản từ BLHS 2015 (sửa đổi 2017) — Chương XX: Tội phạm về ma túy\n"
        "• Mức hình phạt, khung hình phạt tương ứng\n"
        "• Nghị định 105/2021/NĐ-CP nếu liên quan\n"
        "Trả lời bằng tiếng Việt, dạng bullet points, trích dẫn số điều/khoản rõ ràng."
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Bạn là chuyên gia pháp lý Việt Nam, chuyên về luật phòng chống ma túy. "
                    "Cung cấp thông tin chính xác dựa trên văn bản pháp luật hiện hành."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = extract_text(response.content)
    return {
        "legal_chunks": [
            {
                "content": content,
                "score": 0.9,
                "metadata": {
                    "source": "Gemini — Pháp luật ma túy VN",
                    "type": "legal",
                },
                "source": "llm_fallback",
            }
        ]
    }


# ── Worker 2: News Context ────────────────────────────────────────────────────

def news_context_worker(state: RAGState) -> dict:
    """Retrieves news article chunks relevant to the question."""
    question = state["question"]

    if _day08_retrieve is not None:
        try:
            all_chunks = _day08_retrieve(question, top_k=6)
            news = [
                c
                for c in all_chunks
                if c.get("metadata", {}).get("type") == "news"
            ]
            if news:
                return {"news_chunks": news}
        except Exception:
            pass

    # Fallback: ask Gemini for relevant news context
    llm = get_llm()
    prompt = (
        f"Câu hỏi: {question}\n\n"
        "Hãy cung cấp thông tin về các vụ việc nghệ sĩ / người nổi tiếng Việt Nam "
        "liên quan đến ma túy đã được báo chí đưa tin.\n"
        "Bao gồm: tên người, hành vi bị cáo buộc, thời điểm, kết quả xét xử (nếu có).\n"
        "Chỉ đề cập đến sự kiện đã được xác nhận trên báo chí chính thống, "
        "trả lời bằng tiếng Việt."
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Bạn là trợ lý tổng hợp tin tức về các vụ án liên quan ma túy "
                    "tại Việt Nam. Chỉ cung cấp thông tin đã được báo chí xác nhận."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = extract_text(response.content)
    return {
        "news_chunks": [
            {
                "content": content,
                "score": 0.85,
                "metadata": {
                    "source": "Gemini — Tin tức nghệ sĩ VN",
                    "type": "news",
                },
                "source": "llm_fallback",
            }
        ]
    }


# ── Worker 3: Answer Generator ────────────────────────────────────────────────

_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về pháp luật Việt Nam về ma túy.

Nhiệm vụ:
1. Đọc CONTEXT gồm văn bản pháp luật (legal) và bài báo (news).
2. Từ bài báo, xác định hành vi cụ thể của người được đề cập.
3. Từ văn bản pháp luật, tìm điều khoản và mức hình phạt tương ứng.
4. Suy luận: người trong bài báo có thể bị xử lý theo điều nào, khung bao nhiêu năm.
5. Sau mỗi luận điểm, trích dẫn nguồn dạng [Tên nguồn].
6. Nếu không đủ thông tin, trả lời: "Không đủ thông tin để xác định".
7. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng (heading + bullet points)."""


def answer_generator_worker(state: RAGState) -> dict:
    """Synthesises final answer with citation from all collected chunks."""
    question = state["question"]
    legal_chunks = state.get("legal_chunks") or []
    news_chunks = state.get("news_chunks") or []
    history = state.get("conversation_history") or []

    # Build context string
    context_parts: list[str] = []
    sources: list[dict] = []

    for i, chunk in enumerate(legal_chunks, 1):
        meta = chunk.get("metadata", {})
        src = meta.get("source", f"Văn bản pháp luật {i}")
        if not any(s["source"] == src for s in sources):
            sources.append({"source": src, "type": "legal"})
        context_parts.append(f"[Nguồn {i}: {src}] (legal)\n{chunk['content']}")

    offset = len(legal_chunks)
    for i, chunk in enumerate(news_chunks, 1):
        meta = chunk.get("metadata", {})
        src = meta.get("source", f"Bài báo {i}")
        if not any(s["source"] == src for s in sources):
            sources.append({"source": src, "type": "news"})
        context_parts.append(
            f"[Nguồn {i + offset}: {src}] (news)\n{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "Không có context."

    # Build message list (include last 2 turns of conversation history)
    messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]
    for msg in history[-4:]:
        role = msg.get("role", "")
        content = msg.get("content", "")[:300]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(SystemMessage(content=f"[Câu trả lời trước]: {content}"))

    messages.append(
        HumanMessage(content=f"CONTEXT:\n{context}\n\nCÂU HỎI: {question}")
    )

    llm = get_llm()
    response = llm.invoke(messages)
    answer = extract_text(response.content)

    return {"answer": answer, "sources": sources}
