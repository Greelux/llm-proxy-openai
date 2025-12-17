# OpenAI LLM Proxy (FastAPI)

Backend proxy for calling OpenAI safely from mobile/web apps (API key stays on server).

## Endpoints
- GET /health
- POST /chat (non-stream)
- POST /chat/stream (SSE streaming)

## Run
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

## Test (non-stream)
```bat
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}]}"
```
## Test (stream / SSE)
```bat
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write 2 sentences about Flutter\"}]}"
```