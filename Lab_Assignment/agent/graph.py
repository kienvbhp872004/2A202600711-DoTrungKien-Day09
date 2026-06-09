"""Build and compile the Supervisor-Workers LangGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .state import RAGState
from .supervisor import supervisor_node
from .workers import answer_generator_worker, legal_research_worker, news_context_worker


def _route_supervisor(state: RAGState) -> str:
    return state.get("next_worker", "answer_generator")


def build_graph():
    """
    Graph topology:

        START
          │
        supervisor  ──(intent analysis + routing)──┐
          │                                         │
          ├─── legal_research ───────────────► supervisor
          │
          ├─── news_context ─────────────────► supervisor
          │
          └─── answer_generator ─────────────► END
    """
    g = StateGraph(RAGState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("legal_research", legal_research_worker)
    g.add_node("news_context", news_context_worker)
    g.add_node("answer_generator", answer_generator_worker)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "legal_research": "legal_research",
            "news_context": "news_context",
            "answer_generator": "answer_generator",
        },
    )

    # Workers report back to Supervisor after completing
    g.add_edge("legal_research", "supervisor")
    g.add_edge("news_context", "supervisor")
    g.add_edge("answer_generator", END)

    return g.compile()


_compiled: object | None = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
