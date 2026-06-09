"""Web UI backend — streams pipeline progress via SSE while A2A call runs."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")

app = FastAPI(title="Legal Multi-Agent UI")


class QuestionRequest(BaseModel):
    question: str


def event(stage: str, status: str, message: str = "", answer: str = ""):
    data = {"stage": stage, "status": status, "message": message, "answer": answer}
    return json.dumps(data)


async def call_a2a(question: str) -> str:
    """Call customer agent via A2A, return answer text."""
    async with httpx.AsyncClient(timeout=300.0) as http_client:
        card_resp = await http_client.get(f"{CUSTOMER_AGENT_URL}/.well-known/agent.json")
        card_resp.raise_for_status()

        from a2a.client import A2AClient
        from a2a.types import (
            AgentCard, Message, MessageSendParams,
            Part, Role, SendMessageRequest, TextPart,
        )

        agent_card = AgentCard.model_validate(card_resp.json())
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
            message_id=str(uuid4()),
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )
        response = await client.send_message(request)

        result_text = ""
        if hasattr(response, "root"):
            root = response.root
            if hasattr(root, "result"):
                result = root.result
                if hasattr(result, "artifacts") and result.artifacts:
                    for artifact in result.artifacts:
                        for part in artifact.parts:
                            p = part.root if hasattr(part, "root") else part
                            if hasattr(p, "text"):
                                result_text += p.text
                elif hasattr(result, "parts") and result.parts:
                    for part in result.parts:
                        p = part.root if hasattr(part, "root") else part
                        if hasattr(p, "text"):
                            result_text += p.text
                elif hasattr(result, "status") and result.status:
                    msg = result.status.message
                    if msg and msg.parts:
                        for part in msg.parts:
                            p = part.root if hasattr(part, "root") else part
                            if hasattr(p, "text"):
                                result_text += p.text

        return result_text or "Không nhận được phản hồi từ hệ thống."


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/ask/stream")
async def ask_stream(question: str, request: Request):
    """Stream pipeline progress events via SSE while A2A call runs."""

    async def generator():
        # Stage 1: Customer Agent nhận câu hỏi
        yield {"data": event("customer", "active", "Nhận câu hỏi từ người dùng...")}

        # Bắt đầu A2A call trong background
        a2a_task = asyncio.create_task(call_a2a(question))
        start_time = time.time()

        await asyncio.sleep(1.5)
        yield {"data": event("customer", "done", "Phân tích và chuyển tiếp câu hỏi")}

        # Stage 2: Law Agent
        yield {"data": event("law", "active", "Phân tích khía cạnh pháp lý tổng quát...")}
        await asyncio.sleep(3)

        # Stage 3: Routing + Specialists
        yield {"data": event("law", "done", "Xác định cần chuyên gia Tax + Compliance")}
        yield {"data": event("tax", "active", "Phân tích luật thuế, IRS, penalties...")}
        yield {"data": event("compliance", "active", "Phân tích SEC, SOX, FCPA, AML...")}

        # Chờ A2A call hoàn thành, ping mỗi 1s
        while not a2a_task.done():
            await asyncio.sleep(1)
            if await request.is_disconnected():
                a2a_task.cancel()
                return

        elapsed = round(time.time() - start_time, 1)

        try:
            answer = a2a_task.result()
            yield {"data": event("tax", "done", "Tax analysis hoàn thành")}
            yield {"data": event("compliance", "done", "Compliance analysis hoàn thành")}
            yield {"data": event("aggregate", "active", "Tổng hợp kết quả từ tất cả agents...")}
            await asyncio.sleep(0.5)
            yield {"data": event("aggregate", "done", f"Hoàn thành sau {elapsed}s")}
            yield {"data": event("result", "done", "", answer)}

        except Exception as exc:
            yield {"data": event("error", "error", str(exc))}

    return EventSourceResponse(generator())


@app.get("/status")
async def status():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CUSTOMER_AGENT_URL}/.well-known/agent.json")
            resp.raise_for_status()
            return {"online": True, "agent": resp.json().get("name", "Customer Agent")}
    except Exception:
        return {"online": False}


if __name__ == "__main__":
    port = int(os.getenv("UI_PORT", "8080"))
    print(f"Starting Legal AI UI at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
