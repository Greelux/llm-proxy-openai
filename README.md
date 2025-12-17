# OpenAI LLM Proxy (FastAPI)

Backend proxy for calling OpenAI safely from mobile/web apps (API key stays on server).

## Endpoints
- GET /health
- POST /chat (non-stream)
- POST /chat/stream (SSE streaming)

## Setup
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
