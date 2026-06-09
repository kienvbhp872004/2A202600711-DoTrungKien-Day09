"""
Lab Assignment — Day09
Improved RAG Chatbot using Supervisor-Workers pattern (LangGraph)

Improvement over Day08 monolithic pipeline:
  • Supervisor analyses intent → routes to specialised workers
  • Worker 1 — Legal Research   : retrieves / generates legal context
  • Worker 2 — News Context     : retrieves / generates news context
  • Worker 3 — Answer Generator : synthesises answer with citation
  • Multi-turn conversation memory
  • Live workflow visualisation via st.status()
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from agent.graph import get_graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — Supervisor-Workers",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ RAG Chatbot — Pháp luật Ma túy Việt Nam")
st.caption(
    "**Kiến trúc**: Supervisor-Workers (LangGraph) · "
    "**LLM**: Gemini 2.5 Flash (Vertex AI) · "
    "**Lab Assignment** Day09"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏗️ Kiến Trúc Supervisor-Workers")
    st.code(
        """\
User Question
      │
 ┌────▼────────┐
 │  SUPERVISOR │  phân tích intent
 └────┬────────┘
      │ routes
 ┌────▼────────────────────┐
 │ Worker 1: Legal Research │ → luật, điều khoản
 │ Worker 2: News Context   │ → tin tức, vụ án
 │ Worker 3: Answer Gen.    │ → tổng hợp + citation
 └─────────────────────────┘
      │
 Final Answer
""",
        language="text",
    )

    st.divider()
    st.header("⚙️ Cài đặt")
    show_sources = st.checkbox("Hiện nguồn tham khảo", value=True)
    show_workflow = st.checkbox("Hiện luồng xử lý", value=True)

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.conv_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Cải tiến so với Day08:**\n"
        "- Supervisor phân tích intent tự động\n"
        "- Workers chạy tuần tự theo nhu cầu\n"
        "- Dễ mở rộng thêm workers mới\n"
        "- Multi-turn conversation memory\n"
        "- Fallback về Gemini nếu vector store chưa có\n"
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_history" not in st.session_state:
    st.session_state.conv_history = []

# ── Node labels for display ───────────────────────────────────────────────────
_NODE_LABELS: dict[str, str] = {
    "supervisor": "🎯 Supervisor",
    "legal_research": "📘 Legal Research Worker",
    "news_context": "📰 News Context Worker",
    "answer_generator": "✍️ Answer Generator Worker",
}

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} nguồn tham khảo"):
                for src in msg["sources"]:
                    icon = "📘" if src.get("type") == "legal" else "📰"
                    st.markdown(f"{icon} {src.get('source', 'Unknown')}")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Hỏi về pháp luật ma tuý..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        sources: list[dict] = []

        # ── Run the LangGraph pipeline ─────────────────────────────────────
        initial_state = {
            "question": prompt,
            "conversation_history": st.session_state.conv_history,
            "needs_legal": None,
            "needs_news": None,
            "legal_chunks": [],
            "news_chunks": [],
            "next_worker": "",
            "answer": "",
            "sources": [],
        }

        if show_workflow:
            status_ctx = st.status("Đang xử lý...", expanded=True)
        else:
            status_ctx = None

        try:
            graph = get_graph()
            workflow_steps: list[str] = []
            accumulated: dict = dict(initial_state)

            for chunk in graph.stream(initial_state):
                for node_name, updates in chunk.items():
                    workflow_steps.append(node_name)
                    label = _NODE_LABELS.get(node_name, f"⚙️ {node_name}")

                    if status_ctx is not None:
                        # Show intent info after supervisor runs
                        if node_name == "supervisor" and updates.get("needs_legal") is not None:
                            nl = updates.get("needs_legal", False)
                            nn = updates.get("needs_news", False)
                            status_ctx.write(
                                f"✅ **{label}** → "
                                f"needs_legal=`{nl}` needs_news=`{nn}` "
                                f"→ next: `{updates.get('next_worker', '?')}`"
                            )
                        else:
                            status_ctx.write(f"✅ **{label}** hoàn thành")

                    # Accumulate state
                    for k, v in updates.items():
                        if v is not None:
                            accumulated[k] = v

            if status_ctx is not None:
                steps_str = " → ".join(
                    _NODE_LABELS.get(s, s) for s in workflow_steps
                )
                status_ctx.update(
                    label=f"✅ Hoàn thành — {steps_str}",
                    state="complete",
                    expanded=False,
                )

            answer = accumulated.get("answer", "Không có câu trả lời.")
            sources = accumulated.get("sources") or []

        except Exception as exc:
            answer = f"❌ Lỗi: {exc}"
            if status_ctx is not None:
                status_ctx.update(label="❌ Lỗi", state="error")

        st.markdown(answer)

        if show_sources and sources:
            with st.expander(f"📄 {len(sources)} nguồn tham khảo"):
                for src in sources:
                    icon = "📘" if src.get("type") == "legal" else "📰"
                    st.markdown(f"{icon} {src.get('source', 'Unknown')}")

    # Save to history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.conv_history.append({"role": "user", "content": prompt})
    st.session_state.conv_history.append({"role": "assistant", "content": answer})
