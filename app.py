from __future__ import annotations

import json
import time
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError

load_dotenv()

app = FastAPI(title="OpenAI LLM Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

RATE_WINDOW_SEC = 60
RATE_MAX_REQ = 30
_hits: Dict[str, List[float]] = {}


def check_rate_limit(client_id: str) -> None:
    now = time.time()
    times = [t for t in _hits.get(client_id, []) if now - t < RATE_WINDOW_SEC]
    if len(times) >= RATE_MAX_REQ:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    times.append(now)
    _hits[client_id] = times


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant", "system"])
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_output_tokens: int = 500
    instructions: str | None = None


def trim_messages(messages: List[ChatMessage], max_chars: int = 8000) -> List[dict]:
    total = 0
    kept: List[dict] = []
    for m in reversed(messages):
        total += len(m.content)
        if total > max_chars:
            break
        kept.append({"role": m.role, "content": m.content})
    return list(reversed(kept))


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    input_msgs = trim_messages(req.messages)

    try:
        resp = client.responses.create(
            model=req.model,
            input=input_msgs,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            instructions=req.instructions,
            store=False,
        )
        return {"text": resp.output_text}

    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OPENAI_API_KEY")

    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    input_msgs = trim_messages(req.messages)

    try:
        stream = client.responses.create(
            model=req.model,
            input=input_msgs,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            instructions=req.instructions,
            stream=True,
            store=False,
        )
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OPENAI_API_KEY")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    def sse():
        try:
            for event in stream:
                ev_type = getattr(event, "type", None) or (
                    event.get("type") if isinstance(event, dict) else None
                )

                if ev_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None) or (
                        event.get("delta") if isinstance(event, dict) else ""
                    )
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

                elif ev_type == "error":
                    yield "data: {\"error\": true}\n\n"
                    break

            yield "data: {\"done\": true}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': True, 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
