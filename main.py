"""
main.py
-------
Production FastAPI entrypoint for Hyder Assistant.

Replaces streamlit_app.py as the app's front door. Consumes
rag_engine.generate_reply_stream — a native async generator that offloads
its own blocking work internally — directly over SSE, and adds the
production-hardening pieces a Streamlit app didn't need: CORS, per-IP
rate limiting, API key auth, a real readiness check, request-id/latency
logging, and structured error responses.
"""

import json
import os
import secrets
import time
import uuid

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from config import settings
from logger import get_logger
from rag_engine import generate_reply_stream, is_ready

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Startup sanity check
# --------------------------------------------------------------------------
# Refuse to boot into production without an API key configured, rather than
# silently serving an unauthenticated endpoint. Dev is left permissive so
# you don't need a key for local iteration.
if settings.ENVIRONMENT == "production" and not settings.API_KEY:
    raise RuntimeError(
        "settings.API_KEY must be set when ENVIRONMENT=production "
        "(refusing to serve an unauthenticated chat endpoint in prod)."
    )

# --------------------------------------------------------------------------
# CORS configuration
# --------------------------------------------------------------------------
# No ALLOWED_ORIGINS field exists in config.py yet, so this reads a
# comma-separated env var instead of touching Settings. If you'd rather have
# it validated/typed alongside the rest of your config, add:
#   ALLOWED_ORIGINS: str = "http://localhost:3000"
# to config.py and swap the line below for settings.ALLOWED_ORIGINS.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Hyder Assistant API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Request-ID + latency logging middleware
# --------------------------------------------------------------------------
# rag_engine.py already logs its own internal benchmark timings (embedding,
# retrieval, Gemini call) tagged with session_id, but nothing previously
# tied a specific HTTP request to those log lines. Stamping a request_id
# here (and echoing it back as a response header) lets you grep one
# request's full path through the logs, and the duration log gives you
# request-level latency independent of what rag_engine reports internally.
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_id=%s method=%s path=%s duration_ms=%.1f status=unhandled_exception",
            request_id, request.method, request.url.path, duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# --------------------------------------------------------------------------
# API key auth
# --------------------------------------------------------------------------
# Simple shared-secret check via header, meant to keep the endpoint from
# being called directly by arbitrary clients — CORS alone only stops
# browsers, not curl/scripts/other backends. Not a substitute for real
# per-user auth if you ever need to attribute requests to individual users;
# it's a "only our frontend talks to this" gate.
async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        # Dev mode: no key configured, auth is a no-op. The startup check
        # above guarantees this branch can't be reached in production.
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


# --------------------------------------------------------------------------
# Structured error handling
# --------------------------------------------------------------------------
def _error_body(error_type: str, message: str) -> dict:
    return {"error": {"type": error_type, "message": message}}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body("validation_error", "Request payload failed validation."),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body("http_error", str(exc.detail)),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body("internal_error", "Something went wrong. Please try again."),
    )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------
# Real readiness check: rag_engine.is_ready() reflects whether the
# embedding model / Chroma collection / Gemini client actually initialized
# successfully at import time, instead of always returning ok regardless
# of the engine's actual state.
@app.get("/health")
async def health():
    ready, error = is_ready()
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "detail": error},
        )
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Streaming chat endpoint
# --------------------------------------------------------------------------
async def _event_generator(request: Request, session_id: str, message: str):
    """Streams SSE events directly from generate_reply_stream.

    generate_reply_stream is now a native async generator: the blocking
    embedding/ChromaDB work is offloaded internally via asyncio.to_thread,
    and the Gemini call uses the SDK's async streaming client. Nothing it
    does blocks the event loop directly any more, so it's iterated here
    with a plain `async for` instead of being driven through
    iterate_in_threadpool.
    """
    try:
        async for chunk in generate_reply_stream(session_id, message):
            if await request.is_disconnected():
                logger.info(
                    "Client disconnected mid-stream", extra={"session_id": session_id}
                )
                break
            yield {
                "event": "message",
                "data": json.dumps({"chunk": chunk}, ensure_ascii=False),
            }
        else:
            yield {"event": "done", "data": "{}"}
    except Exception:
        logger.exception("Streaming failed", extra={"session_id": session_id})
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "error": "stream_failed",
                    "message": "Something went wrong while generating the response.",
                }
            ),
        }


@app.post("/api/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
async def chat_stream(request: Request, payload: ChatRequest):
    return EventSourceResponse(
        _event_generator(request, payload.session_id, payload.message),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx-style reverse proxies not to buffer the response,
            # which would otherwise hold chunks until a buffer threshold
            # is hit and defeat the point of streaming.
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)