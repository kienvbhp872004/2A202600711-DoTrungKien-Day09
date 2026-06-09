from typing import TypedDict, Optional


class RAGState(TypedDict):
    # Input
    question: str
    conversation_history: list[dict]

    # Intent (set by Supervisor on first call, None = not yet analysed)
    needs_legal: Optional[bool]
    needs_news: Optional[bool]

    # Worker outputs
    legal_chunks: list[dict]
    news_chunks: list[dict]

    # Routing token written by Supervisor before each edge traversal
    next_worker: str  # "legal_research" | "news_context" | "answer_generator"

    # Final output (written by AnswerGenerator)
    answer: str
    sources: list[dict]
