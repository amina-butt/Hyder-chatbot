from dotenv import load_dotenv
import json
import os
import secrets
import time
import uuid

import sys
import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

import httpx
import uvicorn
import re
from collections import OrderedDict
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
from memory import conversation_memory
from config import settings
from logger import get_logger
from rag_engine import (
    classify_handoff_reason,
    detect_language,
    generate_reply,
    generate_reply_stream,
    handoff_confirmation_prompt,
    handoff_connecting_message,
    handoff_reason_label,
    is_explicit_human_request,
    is_frustrated,
    is_ready,
    strip_hindi_characters,
    transcribe_audio_async,
)
from faq_router import faq_router

logger = get_logger(__name__)


def format_bot_response(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
    text = re.sub(r"^#{1,6}\s*(.*?)$", r"*\1*", text, flags=re.MULTILINE)
    text = text.replace("•", "• ")
    text = re.sub(r"•\s+", "• ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Contact Information Guardrail ---
PHONE_REDACTION = "[contact details withheld]"

# tel:/wa.me style links: always redacted.
_PHONE_LINK_REGEX = re.compile(
    r"(?:(?:tel|callto|sms):\s*\+?\d[\d\-\s]*"
    r"|(?:https?://)?(?:wa\.me/|api\.whatsapp\.com/send\?phone=)\+?\d+)",
    re.IGNORECASE,
)

# Runs of digits joined by spaces / dashes / dots / parentheses, optionally with
# a leading + (0300-1234567, +92 300 1234567, 0092-42-35761234, (042) 111-222-333).
# Commas are deliberately NOT separators, so prices like "PKR 1,240,000" survive;
# _redact_phone_match additionally requires >= 9 digits, so specs ("72V 30Ah"),
# "ELI 100" and short numbers are left alone. \d is Unicode-aware, so Urdu digits
# (۰۳۰۰...) are caught too.
PHONE_REGEX = re.compile(r"(?<![\w/.,])\(?\+?\d[\d\s\-().]{7,}\d(?!\w)")


def _redact_phone_match(m: "re.Match[str]") -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return PHONE_REDACTION if len(digits) >= 9 else m.group(0)


def redact_phone_numbers(text: str) -> str:
    if not text:
        return ""
    text = _PHONE_LINK_REGEX.sub(PHONE_REDACTION, text)
    return PHONE_REGEX.sub(_redact_phone_match, text)


def sanitize_bot_output(text: str) -> str:
    """Final gate for every outgoing bot message: strip phone numbers, then
    convert markdown to WhatsApp formatting."""
    return format_bot_response(redact_phone_numbers(text))

# --- Startup Sanity Check ---
if settings.ENVIRONMENT == "production" and not settings.API_KEY:
    raise RuntimeError(
        "settings.API_KEY must be set when ENVIRONMENT=production "
        "(refusing to serve an unauthenticated chat endpoint in prod)."
    )

# --- CORS Configuration ---
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

# --- Chatwoot Configuration ---
CHATWOOT_BASE_URL = settings.CHATWOOT_BASE_URL.rstrip("/")
CHATWOOT_API_TOKEN = settings.CHATWOOT_API_TOKEN
CHATWOOT_ACCOUNT_ID = settings.CHATWOOT_ACCOUNT_ID

HANDOFF_LABEL = "human_handoff"

if not (CHATWOOT_BASE_URL and CHATWOOT_API_TOKEN and CHATWOOT_ACCOUNT_ID):
    logger.warning(
        "Chatwoot env vars incomplete (CHATWOOT_BASE_URL / CHATWOOT_API_TOKEN / "
        "CHATWOOT_ACCOUNT_ID) — /webhooks/chatwoot will still accept and ack "
        "requests, but every outbound Chatwoot API call will fail until these "
        "are set."
    )

# --- Conversation Locks (refcounted, self-evicting) ---
_conversation_locks: dict[int | str, tuple[asyncio.Lock, int]] = {}
_locks_guard = asyncio.Lock()

@asynccontextmanager
async def conversation_lock(conversation_id: int | str):
    async with _locks_guard:
        lock, refcount = _conversation_locks.get(conversation_id, (None, 0))
        if lock is None:
            lock = asyncio.Lock()
        _conversation_locks[conversation_id] = (lock, refcount + 1)
    try:
        async with lock:
            yield
    finally:
        async with _locks_guard:
            lock, refcount = _conversation_locks[conversation_id]
            if refcount <= 1:
                del _conversation_locks[conversation_id]
            else:
                _conversation_locks[conversation_id] = (lock, refcount - 1)

_INCOMING_MESSAGE_TYPES = {0, "incoming"}

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        base_url=CHATWOOT_BASE_URL,
        headers={"api_access_token": CHATWOOT_API_TOKEN or ""},
        timeout=15.0,
    )

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


# --- Request Logging Middleware ---
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


# --- API Key Auth ---
async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


# --- Structured Error Handling ---
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


# --- Schemas ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


MAX_AUDIO_BYTES = settings.MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024


# --- Health Check ---
@app.get("/health")
async def health():
    ready, error = is_ready()
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "detail": error},
        )
    return {"status": "ok"}


# --- Streaming Chat Endpoint ---
async def _event_generator(request: Request, session_id: str, message: str):
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

# --- Standard Non-Streaming Chat Endpoint ---
@app.post("/api/chat")
async def chat_non_stream(request: Request, payload: ChatRequest):
    user_query = payload.message.strip()
    sid = payload.session_id

    # generate_reply handles Tier 1 FAQ + Tier 2 RAG internally
    reply_text, awaiting_confirmation = await asyncio.to_thread(
        generate_reply, sid, user_query
    )

    return {
        "reply": sanitize_bot_output(reply_text),
        "awaiting_confirmation": awaiting_confirmation,
    }

@app.post("/api/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
async def chat_stream(request: Request, payload: ChatRequest):
    return EventSourceResponse(
        _event_generator(request, payload.session_id, payload.message),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Session Lifecycle (custom web widget) ---
class SessionInitRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    new_chat: bool = False


@app.post("/api/session/init", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def session_init(request: Request, payload: SessionInitRequest):
    """Call on widget load. Tells the widget whether it may show the welcome
    greeting: True at most once per session, never for a session that already
    has messages, was already greeted, or is handed off. Pass new_chat=true to
    wipe any lingering state first (page reload into a new chat)."""
    sid = payload.session_id
    if payload.new_chat:
        conversation_memory.clear_session(sid)
    handed_off = conversation_memory.is_handed_off(sid)
    send_greeting = (not handed_off) and conversation_memory.claim_greeting(sid)
    return {
        "send_greeting": send_greeting,
        "is_handed_off": handed_off,
        "has_history": conversation_memory.has_history(sid),
    }


@app.post("/api/session/reset", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def session_reset(request: Request, payload: SessionInitRequest):
    conversation_memory.clear_session(payload.session_id)
    return {"status": "cleared"}


# --- Voice Input Endpoint ---
async def _transcript_event_generator(request: Request, session_id: str, transcript: str):
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

# --- Chatwoot Async Client ---
def _chatwoot_session_id(conversation_id: int | str) -> str:
    """Session memory is keyed strictly by Chatwoot conversation id."""
    return f"chatwoot_{conversation_id}"


def _conv_url(conversation_id: int | str, suffix: str = "") -> str:
    base = f"/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}"
    return f"{base}/{suffix}" if suffix else base


async def send_chatwoot_msg(conversation_id: int | str, content: str, private: bool = False) -> bool:
    """Post a message. Every public (non-private) message is sanitized here, so
    nothing the bot says can bypass the phone-number filter."""
    if not private:
        content = sanitize_bot_output(content)
    if not content:
        return False
    try:
        resp = await app.state.http_client.post(
            _conv_url(conversation_id, "messages"),
            json={"content": content, "message_type": "outgoing", "private": private},
        )
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception(
            "Chatwoot send_message failed | conversation_id=%s private=%s",
            conversation_id, private,
        )
        return False


async def send_chatwoot_msg_chunks(conversation_id: int | str, full_text: str) -> None:
    """Send a reply as one message (or 4000-char chunks if very long).

    Avoids splitting by double-newlines so WhatsApp users don't get flooded
    with multiple rapid-fire notifications. Sanitizes BEFORE splitting so a
    phone number can't be cut in half across a chunk boundary and slip through.
    """
    full_text = sanitize_bot_output(full_text)
    if not full_text:
        return
    chunks = [full_text[i : i + 4000] for i in range(0, len(full_text), 4000)]
    for idx, chunk in enumerate(chunks):
        await send_chatwoot_msg(conversation_id, chunk)
        if idx < len(chunks) - 1:
            await asyncio.sleep(0.1)


async def send_chatwoot_private_note(conversation_id: int | str, content: str) -> bool:
    """Internal note, visible to agents only."""
    return await send_chatwoot_msg(conversation_id, content, private=True)


async def update_chatwoot_status(conversation_id: int | str, status: str = "open") -> None:
    try:
        resp = await app.state.http_client.post(
            _conv_url(conversation_id, "toggle_status"), json={"status": status}
        )
        resp.raise_for_status()
    except Exception:
        logger.exception(
            "Chatwoot update_status failed | conversation_id=%s status=%s",
            conversation_id, status,
        )


def _parse_labels(raw_labels) -> set[str]:
    """Normalise labels from a webhook payload: comma string, list of strings,
    or list of dicts ({"title": ...} / {"name": ...})."""
    labels: set[str] = set()
    if isinstance(raw_labels, str):
        labels = {l.strip() for l in raw_labels.split(",") if l.strip()}
    elif isinstance(raw_labels, list):
        for item in raw_labels:
            if isinstance(item, str):
                if item.strip():
                    labels.add(item.strip())
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name")
                if title:
                    labels.add(title)
    return labels


async def get_chatwoot_labels(conversation_id: int | str) -> list[str] | None:
    """Live label list from Chatwoot, or None if the call failed."""
    try:
        resp = await app.state.http_client.get(_conv_url(conversation_id, "labels"))
        resp.raise_for_status()
        return list(resp.json().get("payload") or [])
    except Exception:
        logger.exception("Chatwoot get_labels failed | conversation_id=%s", conversation_id)
        return None


async def add_chatwoot_label(conversation_id: int | str, label: str) -> bool:
    """Attach `label`. Chatwoot's labels endpoint REPLACES the full list, so we
    fetch and merge to avoid clobbering labels a human agent added."""
    existing = await get_chatwoot_labels(conversation_id)
    if existing is None:
        logger.warning(
            "Could not fetch existing labels for conversation %s — setting '%s' "
            "anyway (may overwrite other labels).", conversation_id, label,
        )
        existing = []
    if label in existing:
        return True
    try:
        resp = await app.state.http_client.post(
            _conv_url(conversation_id, "labels"), json={"labels": [*existing, label]}
        )
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception(
            "Chatwoot add_label failed | conversation_id=%s label=%s", conversation_id, label
        )
        return False


async def remove_chatwoot_label(conversation_id: int | str, label_to_remove: str) -> bool:
    """Fetch labels, strip `label_to_remove`, post the rest back.

    Only writes when the label is actually present: every write fires a
    `conversation_updated` webhook, and a no-op write on a resolved
    conversation would re-trigger the reset forever.
    """
    existing = await get_chatwoot_labels(conversation_id)
    if existing is None:
        return False
    if label_to_remove not in existing:
        return True
    updated = [l for l in existing if l != label_to_remove]
    try:
        resp = await app.state.http_client.post(
            _conv_url(conversation_id, "labels"), json={"labels": updated}
        )
        resp.raise_for_status()
        logger.info("Removed label '%s' from conversation %s", label_to_remove, conversation_id)
        return True
    except Exception:
        logger.exception("Failed to remove Chatwoot label | conversation_id=%s", conversation_id)
        return False


async def get_chatwoot_conversation(conversation_id: int | str) -> dict | None:
    """Live conversation (status, labels, assignee) straight from Chatwoot, or
    None if the call failed. Webhook payloads can be stale; this cannot."""
    try:
        resp = await app.state.http_client.get(_conv_url(conversation_id))
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Chatwoot get_conversation failed | conversation_id=%s", conversation_id)
        return None


def _conv_has_assignee(conv: dict | None) -> bool:
    if not conv:
        return False  # unknown -> don't write (assignee no longer affects muting)
    return bool((conv.get("meta") or {}).get("assignee") or conv.get("assignee_id"))


async def unassign_chatwoot_conversation(conversation_id: int | str) -> None:
    """Unassign any human agent from the conversation."""
    try:
        resp = await app.state.http_client.post(
            _conv_url(conversation_id, "assignments"), json={"assignee_id": None}
        )
        resp.raise_for_status()
        logger.info("Unassigned agent from conversation %s", conversation_id)
    except Exception:
        logger.exception("Failed to unassign Chatwoot conversation | conversation_id=%s", conversation_id)


async def _reset_locked(conversation_id: int | str, conv: dict | None = None) -> None:
    """Reset body. Caller MUST already hold conversation_lock(conversation_id).

    Idempotent — resolved/updated webhooks are re-fired by our own label and
    assignment writes, so each step only writes if there is something to undo.
    """
    session_id = _chatwoot_session_id(conversation_id)
    if conv is None:
        conv = await get_chatwoot_conversation(conversation_id)
    await remove_chatwoot_label(conversation_id, HANDOFF_LABEL)
    if _conv_has_assignee(conv):
        await unassign_chatwoot_conversation(conversation_id)
    # Whatever comes next is a NEW session: wipe everything (handoff flag,
    # pending confirmation, greeted flag, history) so no old state bleeds in.
    conversation_memory.clear_session(session_id)
    logger.info("Conversation %s reset for AI agent.", conversation_id)


async def reset_chatwoot_conversation_for_bot(conversation_id: int | str) -> None:
    """Agent resolved (or the customer reopened) the ticket: give the
    conversation back to the bot — strip label, unassign, clear pending."""
    async with conversation_lock(conversation_id):
        await _reset_locked(conversation_id)


# --- Chatwoot Media / Payload Helpers ---
_AUDIO_EXTENSIONS = (".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm")
_AUDIO_FILE_TYPES = {"audio", "voice", "audio_clip"}


def _extract_incoming_text_and_audio(payload: dict) -> tuple[str, dict | None]:
    content = (payload.get("content") or "").strip()
    audio_attachment = None
    for att in payload.get("attachments") or []:
        file_type = (att.get("file_type") or "").lower()
        data_url = (att.get("data_url") or "").lower().split("?")[0]
        if file_type in _AUDIO_FILE_TYPES or data_url.endswith(_AUDIO_EXTENSIONS):
            audio_attachment = att
            break
    return content, audio_attachment


def _resolve_chatwoot_media_url(data_url: str) -> str:
    if not data_url:
        return data_url

    parsed = urlsplit(data_url)
    path = parsed.path if parsed.netloc else data_url.split("?", 1)[0]

    if path.startswith("/") and (
        "/rails/active_storage/" in path or not parsed.netloc
    ):
        base = urlsplit(CHATWOOT_BASE_URL)
        if parsed.netloc:
            return urlunsplit(parsed._replace(scheme=base.scheme, netloc=base.netloc))
        return f"{CHATWOOT_BASE_URL}{data_url}"

    return data_url


async def _media_get(url: str) -> httpx.Response:
    """GET a media URL without leaking the Chatwoot API token.

    app.state.http_client carries the api_access_token header on every request,
    and httpx only strips `Authorization` (not custom headers) on cross-origin
    redirects — so following a Chatwoot -> storage redirect automatically would
    send the token to the storage host. Redirects are followed manually and any
    non-Chatwoot host is fetched with a header-less client.
    """
    cw_host = urlsplit(CHATWOOT_BASE_URL).netloc
    for _ in range(5):
        if urlsplit(url).netloc == cw_host:
            resp = await app.state.http_client.get(url, timeout=30.0, follow_redirects=False)
        else:
            async with httpx.AsyncClient(timeout=30.0) as plain_client:
                resp = await plain_client.get(url)
        location = resp.headers.get("location")
        if resp.is_redirect and location:
            url = str(httpx.URL(url).join(location))
            continue
        return resp
    raise ValueError("Too many redirects while fetching media")


async def _fetch_audio_bytes(audio_attachment: dict, session_id: str) -> bytes:
    audio_url = _resolve_chatwoot_media_url(audio_attachment.get("data_url", ""))
    logger.info("Fetching Chatwoot audio attachment", extra={"session_id": session_id, "audio_url": audio_url})

    max_retries = 4
    retry_delay = 1.5
    audio_bytes = None
    for attempt in range(1, max_retries + 1):
        try:
            audio_resp = await _media_get(audio_url)
            audio_resp.raise_for_status()
            audio_bytes = audio_resp.content
            break
        except httpx.HTTPStatusError as err:
            if err.response.status_code == 404 and attempt < max_retries:
                logger.warning(
                    "Audio file not ready yet on attempt %d/%d (404). Retrying in %ss...",
                    attempt, max_retries, retry_delay,
                    extra={"session_id": session_id},
                )
                await asyncio.sleep(retry_delay)
            else:
                raise

    if not audio_bytes:
        raise ValueError("Failed to retrieve audio content from Chatwoot")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(f"Audio attachment exceeds {MAX_AUDIO_BYTES} bytes")
    return audio_bytes


async def _resolve_user_input(payload: dict, conversation_id: int | str, session_id: str) -> str | None:
    """Return the customer's text (typed or transcribed), or None if there is
    nothing usable. On voice-note failure the customer is told and None is returned."""
    content, audio_attachment = _extract_incoming_text_and_audio(payload)

    if audio_attachment is None:
        if not content:
            logger.info("Chatwoot webhook had no usable text/audio content — dropping.", extra={"session_id": session_id})
            return None
        return content

    try:
        audio_bytes = await _fetch_audio_bytes(audio_attachment, session_id)
        raw_transcript = await transcribe_audio_async(audio_bytes, "audio/ogg")
        logger.info("Raw audio transcript: '%s'", raw_transcript, extra={"session_id": session_id})
        transcript = strip_hindi_characters(raw_transcript)
        # transcribe_audio() swallows its own errors and returns "" — treat that as a failure too.
        if not transcript.strip():
            raise ValueError("Empty transcript")
        return transcript
    except Exception:
        logger.exception("Chatwoot audio fetch/transcription failed", extra={"session_id": session_id})
        await send_chatwoot_msg(
            conversation_id,
            "Sorry, I couldn't process that voice note. Could you type your question instead?",
        )
        return None


# --- Handoff State Machine ---
#   NORMAL --(explicit human request | frustration | RAG can't answer)--> AWAITING_CONFIRMATION
#   AWAITING_CONFIRMATION --(explicit yes)--> HANDED_OFF (label = mute)
#   AWAITING_CONFIRMATION --(anything else)--> NORMAL (pending cleared, message processed normally)
#   HANDED_OFF --(agent resolves)--> NORMAL (label removed, unassigned, pending cleared)
_AFFIRM_STRONG = {
    "yes", "yeah", "yep", "yup", "yea", "yess", "ya", "yah", "yaa", "yas", "y",
    "sure", "ok", "okay", "okey", "okie", "k", "kk", "done",
    "please", "pls", "plz", "connect", "absolutely", "definitely",
    "ji", "jee", "je", "jii", "g", "ge", "gee",
    "haan", "han", "hanji", "haanji", "ha", "haa", "hn",
    "zaroor", "zarur", "bilkul", "kardo", "karo", "kro", "krdo",
    "acha", "accha", "achha", "theek", "thik",
    "جی", "ہاں", "ضرور", "بالکل",
}
_AFFIRM_FILLER = {
    "me", "now", "to", "with", "a", "an", "the", "it", "do", "go", "ahead", "of",
    "course", "that", "would", "be", "great", "good", "sounds", "fine", "agent",
    "human", "representative", "team", "support", "person", "mujhe", "se", "abhi",
    "jaldi", "hai", "ho", "karen", "karein", "kijiye",
    "ٹھیک", "کریں", "کر", "دیں", "دو", "ابھی",
}
_AFFIRM_EMOJI = {"👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿", "✅"}


def _token_forms(tok: str) -> set[str]:
    """Spelling variants so elongated/typo'd chat replies still match:
    yesss -> yes, yupp -> yup, okkk -> ok, jeee -> jee."""
    return {
        tok,
        re.sub(r"(.)\1+", r"\1", tok),
        re.sub(r"(.)\1{2,}", r"\1\1", tok),
    }


def _is_affirmative(text: str) -> bool:
    """True for short, purely affirmative replies ("yes", "yupp", "ge", "yesss",
    "ji haan please"). Longer messages or ones with other content ("please tell
    me the price of ELI") are NOT consent."""
    t = text.strip().lower()
    if t in _AFFIRM_EMOJI:
        return True
    t = re.sub(r"\bkar\s+do\b", "kardo", t)
    tokens = re.findall(r"\w+", t)
    if not tokens or len(tokens) > 6:
        return False
    strong_seen = False
    for tok in tokens:
        forms = _token_forms(tok)
        if forms & _AFFIRM_STRONG:
            strong_seen = True
        elif not forms & _AFFIRM_FILLER:
            return False
    return strong_seen


_REASON_SUMMARIES = {
    "test_ride": "Customer wants to book a test ride.",
    "complaint": "Customer raised a complaint or refund request.",
    "discount": "Customer is asking about a discount / price negotiation.",
    "api_error": "The AI service failed while answering this question.",
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _earlier_topic(session_id: str, trigger: str) -> str | None:
    """Most recent real customer question before the trigger (skips greetings,
    'yes' replies and the trigger itself) — gives the agent context when the
    trigger is just 'connect me to an agent'."""
    skip = " ".join(trigger.split()).lower()
    for m in reversed(conversation_memory.get_history(session_id)):
        if m.role != "user":
            continue
        text = " ".join(m.content.split())
        if not text or text.lower() == skip:
            continue
        if _is_affirmative(text) or is_explicit_human_request(text) or _is_pure_greeting(text):
            continue
        return _clip(text, 120)
    return None


def build_handoff_summary(session_id: str, question: str, reason: str | None) -> str | None:
    """One-line customer-intent summary, or None when it would add nothing
    (e.g. a plain knowledge-base gap is already explained by the reason)."""
    if reason in _REASON_SUMMARIES:
        return _REASON_SUMMARIES[reason]
    if reason in ("explicit_request", "frustration"):
        earlier = _earlier_topic(session_id, question)
        if earlier:
            lead = (
                "Customer asked for a human agent"
                if reason == "explicit_request"
                else "Customer sounded frustrated"
            )
            return f'{lead}; earlier they asked: "{earlier}".'
    return None


def build_handoff_note(question: str, reason: str | None, summary: str | None = None) -> str:
    """Concise agent note: reason, the exact triggering question, optional summary."""
    lines = [
        "📌 **HUMAN HANDOFF REQUESTED**",
        f"• **Reason:** {handoff_reason_label(reason)}",
        f'• **Triggering Question:** "{_clip(question, 500)}"',
    ]
    if summary:
        lines.append(f"• **Summary:** {summary}")
    return "\n".join(lines)


# --- Greeting De-duplication ---
_GREETING_WORDS = {
    "hi", "hello", "hey", "helo", "hii", "hy", "salam", "salaam", "slam", "aoa", "oa",
    "assalam", "assalamu", "assalamualaikum", "asalam", "alaikum", "alikum", "walaikum",
    "walekum", "salamualaikum", "o", "good", "morning", "afternoon", "evening", "namaste",
    "السلام", "علیکم", "سلام", "ہیلو",
}
_REPEAT_GREETING_REPLY = {
    "english": "Hello! How can I help you today?",
    "roman_urdu": "Ji, main aap ki kya madad kar sakta hoon?",
    "urdu": "جی، میں آپ کی کیا مدد کر سکتا ہوں؟",
}


def _greeting_language(text: str) -> str:
    """Language for a greeting-only message. detect_language() is unreliable on
    one-word input ("hi" reads as a Roman Urdu particle), so decide by script /
    Islamic-greeting wording: Urdu script -> urdu, salam/aoa -> roman_urdu, else English."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "urdu"
    if re.search(r"\b(salam|salaam|slam|aoa|assalam\w*|asalam\w*|walaikum|walekum)\b", text.lower()):
        return "roman_urdu"
    return "english"


def _is_pure_greeting(text: str) -> bool:
    """True for greeting-only messages ("hi", "Assalam-o-Alaikum", "hiii")."""
    tokens = re.findall(r"\w+", text.strip().lower())
    if not tokens or len(tokens) > 5:
        return False
    return all(_token_forms(tok) & _GREETING_WORDS for tok in tokens)


# --- Webhook Delivery De-duplication ---
_SEEN_MESSAGE_IDS: "OrderedDict[str, None]" = OrderedDict()
_SEEN_MESSAGE_IDS_MAX = 5000


def _is_duplicate_delivery(message_id) -> bool:
    """Chatwoot retries webhooks; without this the same message is answered twice.
    No awaits inside, so it is atomic on the event loop."""
    if message_id is None:
        return False
    key = str(message_id)
    if key in _SEEN_MESSAGE_IDS:
        return True
    _SEEN_MESSAGE_IDS[key] = None
    while len(_SEEN_MESSAGE_IDS) > _SEEN_MESSAGE_IDS_MAX:
        _SEEN_MESSAGE_IDS.popitem(last=False)
    return False


async def _request_handoff_confirmation(conversation_id: int | str, session_id: str, user_input: str) -> None:
    """Step 1: ask, never hand off. Stores the customer's message (and why we
    are asking) as the pending handoff."""
    reason = classify_handoff_reason(user_input)  # explicit_request / frustration
    prompt = handoff_confirmation_prompt(detect_language(user_input))
    conversation_memory.add_message(session_id, "user", user_input)
    conversation_memory.add_message(session_id, "assistant", prompt)
    # Only arm the pending state if the customer actually received the question.
    if await send_chatwoot_msg(conversation_id, prompt):
        conversation_memory.set_pending_question(session_id, user_input, reason)


async def _execute_handoff(
    conversation_id: int | str,
    session_id: str,
    pending_question: str,
    reason: str | None,
    confirmation_text: str,
) -> None:
    """Step 2: customer said yes.

    The in-memory handed-off flag is set FIRST (before any network call) and
    clears the pending question in the same step, so even if a Chatwoot call
    below fails or a duplicate webhook arrives, the bot is already muted and
    the confirmation can't fire twice.
    """
    conversation_memory.mark_handed_off(session_id)

    conversation_memory.add_message(session_id, "user", confirmation_text)
    summary = build_handoff_summary(session_id, pending_question, reason)
    await send_chatwoot_private_note(
        conversation_id, build_handoff_note(pending_question, reason, summary)
    )
    if not await add_chatwoot_label(conversation_id, HANDOFF_LABEL):
        logger.error(
            "Handoff label could NOT be applied for conversation %s — bot is muted "
            "by the in-memory flag only (lost on restart/TTL); check Chatwoot API access.",
            conversation_id,
        )
    await update_chatwoot_status(conversation_id, "open")
    connecting = handoff_connecting_message(detect_language(pending_question))
    if await send_chatwoot_msg(conversation_id, connecting):
        conversation_memory.add_message(session_id, "assistant", connecting)


async def process_chatwoot_webhook(payload: dict) -> None:
    conversation = payload.get("conversation") or {}
    conversation_id = conversation.get("id") or payload.get("conversation_id")
    if not conversation_id:
        logger.warning("Chatwoot webhook payload had no conversation id — dropping.")
        return

    session_id = _chatwoot_session_id(conversation_id)

    async with conversation_lock(conversation_id):
        try:
            # Fast mute: session already handed off (covers duplicate deliveries and
            # messages queued behind the handoff). No API calls, no LLM, no reply.
            if conversation_memory.is_handed_off(session_id):
                logger.info("Session %s handed off — bot muted.", session_id)
                return

            # Mute guard, decided on LIVE Chatwoot state, not the webhook snapshot
            # (which can carry a stale human_handoff label if a resolve was missed).
            live = await get_chatwoot_conversation(conversation_id)
            if live is not None:
                live_status = live.get("status")
                labels = (
                    _parse_labels(live.get("labels"))
                    if "labels" in live
                    else _parse_labels(conversation.get("labels"))
                )
            else:  # Chatwoot unreachable: fall back to the payload snapshot
                live_status = None
                labels = _parse_labels(conversation.get("labels"))

            if live_status == "resolved":
                logger.info(
                    "Conversation %s is resolved but a customer message arrived — "
                    "resetting for AI.", conversation_id,
                )
                await _reset_locked(conversation_id, live)
                labels = set()

            if HANDOFF_LABEL in labels:
                logger.info("Conversation %s is in human handoff — bot muted.", conversation_id)
                return

            user_input = await _resolve_user_input(payload, conversation_id, session_id)
            if user_input is None:
                return

            # Step 2 of the handoff: customer is answering our confirmation question.
            pending = conversation_memory.get_pending_handoff(session_id)
            if pending:
                pending_question, pending_reason = pending
                if _is_affirmative(user_input) or is_explicit_human_request(user_input):
                    await _execute_handoff(
                        conversation_id, session_id, pending_question, pending_reason, user_input
                    )
                    return
                # "No", or a brand-new question: drop the pending state, process normally.
                conversation_memory.clear_pending_question(session_id)

            # Step 1 (explicit trigger): ask for confirmation instead of handing off.
            if is_explicit_human_request(user_input) or is_frustrated(user_input):
                await _request_handoff_confirmation(conversation_id, session_id, user_input)
                return

            # Greeting de-dupe: the welcome goes out at most once per session. A later
            # "hi" (or a greeting after a page reload into the same conversation)
            # gets a short prompt instead of the full welcome again.
            if _is_pure_greeting(user_input) and not conversation_memory.claim_greeting(session_id):
                reply = _REPEAT_GREETING_REPLY[_greeting_language(user_input)]
                conversation_memory.add_message(session_id, "user", user_input)
                conversation_memory.add_message(session_id, "assistant", reply)
                await send_chatwoot_msg(conversation_id, reply)
                return

            # Normal RAG turn. generate_reply() sets the pending question (with its
            # reason) itself when it can't answer, and appends the confirmation prompt.
            reply_text, awaiting_confirmation = await asyncio.to_thread(
                generate_reply, session_id, user_input
            )
            await send_chatwoot_msg_chunks(conversation_id, reply_text)
            if awaiting_confirmation:
                logger.info("Awaiting handoff confirmation", extra={"session_id": session_id})

        except Exception:
            logger.exception("process_chatwoot_webhook failed", extra={"session_id": session_id})
            # Never leave the customer in silence.
            await send_chatwoot_msg(
                conversation_id,
                "Sorry, something went wrong on our side. Please send your message again.",
            )


# --- Chatwoot Webhook (WhatsApp Integration) ---
_RESOLVE_EVENTS = {"conversation_resolved", "conversation_status_changed", "conversation_updated"}


def _is_reopen_event(payload: dict) -> bool:
    """True when the payload records a status change away from 'resolved'
    (customer wrote into a resolved chat, or an agent reopened it)."""
    changed = payload.get("changed_attributes")
    items = changed if isinstance(changed, list) else [changed] if isinstance(changed, dict) else []
    for item in items:
        st = item.get("status") if isinstance(item, dict) else None
        if isinstance(st, dict) and st.get("previous_value") == "resolved" \
                and st.get("current_value") != "resolved":
            return True
    return False


@app.post("/webhooks/chatwoot")
async def chatwoot_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_chatwoot_token: str | None = Header(default=None, alias="X-Chatwoot-Token"),
    token: str | None = None,
):
    # Guard 0: Webhook Token Verification
    webhook_secret = getattr(settings, "CHATWOOT_WEBHOOK_SECRET", None) or os.getenv("CHATWOOT_WEBHOOK_SECRET")
    if webhook_secret:
        provided_token = x_chatwoot_token or token
        if not provided_token or not secrets.compare_digest(provided_token, webhook_secret):
            logger.warning("Unauthorized Chatwoot webhook attempt with invalid or missing token.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing webhook token.",
            )

    try:
        payload = await request.json()
        event = payload.get("event")
        logger.debug("Chatwoot webhook payload received: event=%s", event)
    except Exception:
        return {"status": "received"}

    # Brand-new conversation (new chat / widget re-initialised after resolution):
    # start from a clean slate in case any state exists under this id.
    if event == "conversation_created":
        new_id = (payload.get("conversation") or payload).get("id") or payload.get("id")
        if new_id:
            conversation_memory.clear_session(_chatwoot_session_id(new_id))
        return {"status": "received"}

    # Conversation events: resolved OR reopened-from-resolved -> hand back to the bot.
    # (The reopen case covers a missed/delayed "resolved" webhook: the label is
    # still on the conversation when the customer writes back.)
    if event in _RESOLVE_EVENTS:
        conversation = payload.get("conversation") or payload
        status_val = conversation.get("status") or payload.get("status")
        conv_id = conversation.get("id") or payload.get("id")

        if conv_id and (
            event == "conversation_resolved"
            or status_val == "resolved"
            or _is_reopen_event(payload)
        ):
            logger.info("Conversation %s resolved/reopened. Triggering AI reset...", conv_id)
            background_tasks.add_task(reset_chatwoot_conversation_for_bot, conv_id)
        return {"status": "received"}

    # Only new messages past this point (message_updated etc. also carry
    # message_type=incoming and would make the bot answer the same message twice).
    if event != "message_created":
        return {"status": "received"}

    # Guard 1: Message Type Check
    if payload.get("message_type") not in _INCOMING_MESSAGE_TYPES:
        return {"status": "received"}

    # Guard 2: Sender Check (ignore human agents)
    sender = payload.get("sender") or {}
    if sender.get("type") == "user":
        return {"status": "received"}

    # Guard 3: Private Note Check
    if payload.get("private") is True:
        return {"status": "received"}

    conversation = payload.get("conversation") or {}
    conv_id = conversation.get("id") or payload.get("conversation_id")

    # Guard 3b: duplicate delivery of the same message
    if _is_duplicate_delivery(payload.get("id")):
        logger.info("Duplicate webhook delivery for message %s — ignoring.", payload.get("id"))
        return {"status": "received"}

    # Payload says resolved: reset FIRST. Background tasks run in order, so this
    # completes before process_chatwoot_webhook evaluates the mute guard.
    payload_resolved = bool(conv_id) and conversation.get("status") == "resolved"
    if payload_resolved:
        logger.info("Message on resolved conversation %s — resetting before processing.", conv_id)
        background_tasks.add_task(reset_chatwoot_conversation_for_bot, conv_id)

    # Mute Guard (early exit): handed-off session -> 200 OK immediately, nothing
    # scheduled, so no LLM / RAG / API call / auto-reply can happen.
    if conv_id and not payload_resolved and conversation_memory.is_handed_off(_chatwoot_session_id(conv_id)):
        logger.info("Session %s handed off — ignoring incoming message.", _chatwoot_session_id(conv_id))
        return {"status": "received"}

    # A payload label alone is NOT trusted here any more: a stale
    # human_handoff label there is exactly what kept the bot silent after
    # resolution. process_chatwoot_webhook checks live label + status instead.
    background_tasks.add_task(process_chatwoot_webhook, payload)
    return {"status": "received"}


# --- Entrypoint ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)