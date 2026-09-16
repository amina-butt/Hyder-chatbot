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
from dotenv import load_dotenv
import json
import os
import secrets
import time
import uuid

import sys
import asyncio
from contextlib import asynccontextmanager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

import httpx
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
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
from rag_engine import (
    detect_escalation_trigger,
    detect_language,
    generate_reply,
    generate_reply_stream,
    is_ready,
    strip_hindi_characters,
    transcribe_audio_async,
)
# faq_router is imported here purely so main.py can log/verify its state at
# startup (see the startup_event below). rag_engine.py already imports and
# calls the same module-level `faq_router` singleton directly to intercept
# queries — main.py never calls faq_router.match() itself, since the
# interception point requested lives in generate_reply/generate_reply_stream,
# not here.
from faq_router import faq_router

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
# Chatwoot configuration
# --------------------------------------------------------------------------

CHATWOOT_BASE_URL = settings.CHATWOOT_BASE_URL.rstrip("/")
CHATWOOT_API_TOKEN = settings.CHATWOOT_API_TOKEN
CHATWOOT_ACCOUNT_ID = settings.CHATWOOT_ACCOUNT_ID

if not (CHATWOOT_BASE_URL and CHATWOOT_API_TOKEN and CHATWOOT_ACCOUNT_ID):
    logger.warning(
        "Chatwoot env vars incomplete (CHATWOOT_BASE_URL / CHATWOOT_API_TOKEN / "
        "CHATWOOT_ACCOUNT_ID) — /webhooks/chatwoot will still accept and ack "
        "requests, but every outbound Chatwoot API call will fail until these "
        "are set."
    )

# Chatwoot sends message_type as an int (0) over the API but as a string
# ("incoming") in some webhook payload variants — accept both so the
# recursive-echo guard doesn't accidentally let a bot/outgoing message
# through and re-trigger generate_reply on it.
_INCOMING_MESSAGE_TYPES = {0, "incoming"}

# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# --------------------------------------------------------------------------
# Lifespan: Chatwoot HTTP client + startup verification
# --------------------------------------------------------------------------
# Replaces the old module-level `_chatwoot_http_client` global and the
# `@app.on_event("startup"/"shutdown")` handlers. Building the
# httpx.AsyncClient here (inside the running event loop) rather than at
# import time avoids binding its connection pool to a loop that may not be
# the one actually serving requests. Stashed on app.state so the Chatwoot
# helper functions below can reach it without a module-level global.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        base_url=CHATWOOT_BASE_URL,
        headers={"api_access_token": CHATWOOT_API_TOKEN or ""},
        timeout=15.0,
    )

    # Startup verification: local FAQ router. faq_router is already a
    # module-level singleton (loaded once at import time in faq_router.py)
    # and rag_engine.py already imports/uses it directly — this doesn't
    # initialize anything new. It just surfaces, in the same structured
    # request/latency log stream as everything else, whether the
    # zero-cost local intercept actually has data to match against. An
    # empty/failed load (e.g. faqs.json missing or malformed) is NOT
    # fatal — FAQRouter fails closed and every query simply falls through
    # to the normal ChromaDB + Gemini pipeline — but it's worth knowing at
    # a glance whether that fallback is silently happening for 100% of
    # traffic.
    intent_count = len(faq_router._intents)
    model_count = len(faq_router._models)
    if intent_count == 0:
        logger.warning(
            "FAQRouter loaded 0 intents from %s — the local FAQ short-circuit "
            "is effectively disabled; every query will fall through to the "
            "full ChromaDB + Gemini pipeline.",
            faq_router._path,
        )
    else:
        logger.info(
            "FAQRouter ready: %d intents / %d models loaded from %s. "
            "Matching queries will be served with 0 API tokens used.",
            intent_count, model_count, faq_router._path,
        )

    yield

    await app.state.http_client.aclose()


app = FastAPI(title="Hyder Assistant API", version="1.0.0", lifespan=lifespan)
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
# Voice input constraints
# --------------------------------------------------------------------------

MAX_AUDIO_BYTES = settings.MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024


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

    Verified compatible with the local FAQ router short-circuit added in
    rag_engine.py: on a match, generate_reply_stream yields exactly one
    chunk (the canned template) and returns — from this loop's point of
    view that's indistinguishable from any other single-chunk stream, so
    it's wrapped in a normal "message" SSE event below and followed by
    the "done" event from the `else` branch of the `async for`, with no
    special-casing, no premature connection close, and no formatting
    glitches on the client side.
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
# Voice input endpoint
# --------------------------------------------------------------------------
# multipart/form-data (not JSON) since we're receiving an audio file
# alongside session_id, so this is a separate endpoint from /api/chat/stream
# rather than a variant of ChatRequest.
async def _transcript_event_generator(request: Request, session_id: str, transcript: str):
    """Same shape as _event_generator's stream, but starts with a
    "transcript" SSE event carrying the transcribed text, so the widget can
    render the user's own bubble (it never typed anything, so it doesn't
    otherwise know what was "said") before the assistant's reply starts
    streaming in.
    """
    yield {
        "event": "transcript",
        "data": json.dumps({"text": transcript}, ensure_ascii=False),
    }
    async for event in _event_generator(request, session_id, transcript):
        yield event


@app.post("/api/chat/audio", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat_audio(
    request: Request,
    session_id: str = Form(..., min_length=1),
    file: UploadFile = File(...),
):
    mime_type = file.content_type or "audio/webm"
    if mime_type not in settings.ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format: {mime_type}"
        )
    
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        transcript = await transcribe_audio_async(audio_bytes, mime_type)
        transcript = strip_hindi_characters(transcript)
    except Exception:
        logger.exception("Audio transcription failed", extra={"session_id": session_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while transcribing the audio.",
        )

    if not transcript.strip():
        # Not a server error — the audio just wasn't understandable speech
        # (silence, noise, unsupported language). SSE keeps this endpoint's
        # response shape consistent with /api/chat/stream (the widget's SSE
        # handler doesn't need a separate code path for a JSON error body).
        async def _empty_transcript_stream():
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "error": "empty_transcript",
                        "message": "Sorry, I couldn't make out what you said. Could you try again?",
                    }
                ),
            }

        return EventSourceResponse(
            _empty_transcript_stream(),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return EventSourceResponse(
        _transcript_event_generator(request, session_id, transcript),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# --------------------------------------------------------------------------
# Chatwoot async client
# --------------------------------------------------------------------------
async def send_chatwoot_msg(conversation_id: int | str, content: str, private: bool = False) -> None:
    """POST a message into a Chatwoot conversation. `private=True` is what
    send_chatwoot_private_note uses; kept as one function so both paths
    share the same error handling."""
    url = f"/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    try:
        resp = await app.state.http_client.post(
            url,
            json={"content": content, "message_type": "outgoing", "private": private},
        )
        resp.raise_for_status()
    except Exception:
        logger.exception(
            "Chatwoot send_message failed | conversation_id=%s private=%s",
            conversation_id, private,
        )


async def send_chatwoot_private_note(conversation_id: int | str, content: str) -> None:
    """Internal note visible only to agents in the inbox — never sent to
    the customer on WhatsApp."""
    await send_chatwoot_msg(conversation_id, content, private=True)


async def update_chatwoot_status(conversation_id: int | str, status: str = "open") -> None:
    """Toggle a conversation's status (open/resolved/pending/snoozed).
    Used after an escalation so the conversation actually surfaces in the
    agent inbox instead of sitting wherever it was before (e.g. resolved)."""
    url = f"/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/toggle_status"
    try:
        resp = await app.state.http_client.post(url, json={"status": status})
        resp.raise_for_status()
    except Exception:
        logger.exception(
            "Chatwoot update_status failed | conversation_id=%s status=%s",
            conversation_id, status,
        )


# --------------------------------------------------------------------------
# Chatwoot webhook — WhatsApp integration
# --------------------------------------------------------------------------
_AUDIO_EXTENSIONS = (".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm")


def _extract_incoming_text_and_audio(payload: dict) -> tuple[str, dict | None]:
    """Pulls user-facing text and (if present) the first audio attachment
    out of a Chatwoot `message_created` webhook payload."""
    content = (payload.get("content") or "").strip()
    audio_attachment = None
    for att in payload.get("attachments") or []:
        file_type = (att.get("file_type") or "").lower()
        data_url = (att.get("data_url") or "").lower()
        if file_type == "audio" or data_url.endswith(_AUDIO_EXTENSIONS):
            audio_attachment = att
            break
    return content, audio_attachment


async def process_chatwoot_webhook(payload: dict) -> None:
    """Background task: everything past the fast webhook ack.

    Runs the human-handoff check first (detect_escalation_trigger), then
    falls through to the normal generate_reply pipeline — which already
    contains the Tier-1 faq_router short-circuit and the Tier-2/3
    ChromaDB + Gemini path internally, so this function doesn't need to
    call either directly.
    """
    conversation = payload.get("conversation") or {}
    conversation_id = conversation.get("id") or payload.get("conversation_id")
    if not conversation_id:
        logger.warning("Chatwoot webhook payload had no conversation id — dropping.")
        return

    # One session per Chatwoot conversation, so conversation_memory keeps
    # WhatsApp follow-ups coherent the same way the web widget's
    # session_id does.
    session_id = f"chatwoot_{conversation_id}"

    try:
        content, audio_attachment = _extract_incoming_text_and_audio(payload)

        if audio_attachment is not None:
            try:
                async with httpx.AsyncClient(timeout=30.0) as fetch_client:
                    audio_resp = await fetch_client.get(audio_attachment["data_url"])
                    audio_resp.raise_for_status()
                audio_bytes = audio_resp.content
                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    raise ValueError(f"Audio attachment exceeds {MAX_AUDIO_BYTES} bytes")
                # WhatsApp voice notes arrive as ogg/opus near-universally;
                # the Chatwoot attachment payload doesn't reliably include a
                # real content-type, so this is a safe default rather than
                # a genuine sniff.
                transcript = await transcribe_audio_async(audio_bytes, "audio/ogg")
                user_input = strip_hindi_characters(transcript)
            except Exception:
                logger.exception("Chatwoot audio fetch/transcription failed", extra={"session_id": session_id})
                await send_chatwoot_msg(
                    conversation_id,
                    "Sorry, I couldn't process that voice note. Could you type your question instead?",
                )
                return
        else:
            user_input = content

        if not user_input.strip():
            logger.info("Chatwoot webhook had no usable text/audio content — dropping.", extra={"session_id": session_id})
            return

        triggered, reason = detect_escalation_trigger(user_input)
        if triggered:
            await send_chatwoot_private_note(
                conversation_id,
                f"[Auto-escalation: {reason}] Customer message: {user_input}",
            )
            await send_chatwoot_msg(
                conversation_id,
                "Thanks for reaching out — I'm connecting you with a member of our team who will follow up shortly.",
            )
            await update_chatwoot_status(conversation_id, "open")
            return

        # generate_reply is synchronous (blocking Gemini/Chroma calls) —
        # offload to a worker thread, same pattern rag_engine.py itself
        # uses for transcribe_audio_async, so it doesn't stall the event
        # loop other requests are running on.
        reply_text, handoff_triggered = await asyncio.to_thread(generate_reply, session_id, user_input)
        await send_chatwoot_msg(conversation_id, reply_text)

        if handoff_triggered:
            await update_chatwoot_status(conversation_id, "open")

    except Exception:
        logger.exception("process_chatwoot_webhook failed", extra={"session_id": session_id})


@app.post("/webhooks/chatwoot")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks):
    """Chatwoot webhook receiver — must ack fast (Chatwoot retries/backs
    off on slow or non-2xx responses), so all real work is deferred to
    process_chatwoot_webhook via BackgroundTasks. This handler only does
    cheap dict filtering before returning."""
    try:
        payload = await request.json()
    except Exception:
        return {"status": "received"}

# Guard 1: Message Type Check
    if payload.get("message_type") not in _INCOMING_MESSAGE_TYPES:
        return {"status": "received"}

    # Guard 2: Sender Check (Drop human agent replies; customer type is "contact")
    sender = payload.get("sender") or {}
    if sender.get("type") == "user":
        return {"status": "received"}

    # Guard 3: Private Note Check
    if payload.get("private") is True:
        return {"status": "received"}

    # Guard 4: Human Handoff Check (Do not process if conversation is open/handled by staff)
    conversation = payload.get("conversation") or {}
    if conversation.get("status") == "open":
        return {"status": "received"}

    background_tasks.add_task(process_chatwoot_webhook, payload)
    return {"status": "received"}

# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)