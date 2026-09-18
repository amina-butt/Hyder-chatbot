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
    detect_escalation_trigger,
    detect_language,
    generate_reply,
    generate_reply_stream,
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
async def send_chatwoot_msg(conversation_id: int | str, content: str, private: bool = False) -> None:
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

async def send_chatwoot_msg_chunks(conversation_id: int | str, full_text: str) -> None:
    chunks = [c.strip() for c in full_text.split("\n\n") if c.strip()]
    for chunk in chunks:
        await send_chatwoot_msg(conversation_id, chunk)
        await asyncio.sleep(0.5)

        
async def send_chatwoot_private_note(conversation_id: int | str, content: str) -> None:
    await send_chatwoot_msg(conversation_id, content, private=True)


async def update_chatwoot_status(conversation_id: int | str, status: str = "open") -> None:
    url = f"/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/toggle_status"
    try:
        resp = await app.state.http_client.post(url, json={"status": status})
        resp.raise_for_status()
    except Exception:
        logger.exception(
            "Chatwoot update_status failed | conversation_id=%s status=%s",
            conversation_id, status,
        )


async def add_chatwoot_label(conversation_id: int | str, label: str) -> None:
    """Attach `label` to a Chatwoot conversation's label set.

    Chatwoot's labels endpoint (POST .../conversations/{id}/labels) REPLACES
    the conversation's full label list rather than appending to it, so we
    fetch the existing labels first and merge, to avoid clobbering any
    labels a human agent has already added.
    """
    url = f"/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/labels"
    try:
        existing_labels: list[str] = []
        try:
            get_resp = await app.state.http_client.get(url)
            get_resp.raise_for_status()
            existing_labels = get_resp.json().get("payload") or []
        except Exception:
            logger.exception(
                "Chatwoot fetch existing labels failed | conversation_id=%s — "
                "proceeding to set label anyway (may overwrite other labels).",
                conversation_id,
            )

        if label in existing_labels:
            return

        resp = await app.state.http_client.post(
            url, json={"labels": [*existing_labels, label]}
        )
        resp.raise_for_status()
    except Exception:
        logger.exception(
            "Chatwoot add_label failed | conversation_id=%s label=%s",
            conversation_id, label,
        )


# --- Chatwoot Webhook (WhatsApp Integration) ---
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

_AFFIRMATIVE_RE = re.compile(
    r"^\s*(yes|yeah|yep|sure|yup|ok|okay|ji|ji haan|haan|ha|ji ha|connect|please|pls|yes please|جی|ہاں)\b",
    re.IGNORECASE,
)

def _is_affirmative(text: str) -> bool:
    return bool(_AFFIRMATIVE_RE.search(text.strip()))

async def process_chatwoot_webhook(payload: dict) -> None:
    conversation = payload.get("conversation") or {}
    conversation_id = conversation.get("id") or payload.get("conversation_id")
    if not conversation_id:
        logger.warning("Chatwoot webhook payload had no conversation id — dropping.")
        return

    session_id = f"chatwoot_{conversation_id}"

    async with conversation_lock(conversation_id):
        try:
            content, audio_attachment = _extract_incoming_text_and_audio(payload)

            if audio_attachment is not None:
                try:
                    audio_url = _resolve_chatwoot_media_url(audio_attachment.get("data_url", ""))
                    logger.info("Fetching Chatwoot audio attachment", extra={"session_id": session_id, "audio_url": audio_url})

                    audio_bytes = None
                    max_retries = 4
                    retry_delay = 1.5

                    for attempt in range(1, max_retries + 1):
                        try:
                            audio_resp = await app.state.http_client.get(
                                audio_url, timeout=30.0, follow_redirects=True
                            )
                            audio_resp.raise_for_status()
                            audio_bytes = audio_resp.content
                            break
                        except httpx.HTTPStatusError as err:
                            if err.response.status_code == 404 and attempt < max_retries:
                                logger.warning(
                                    f"Audio file not ready yet on attempt {attempt}/{max_retries} (404). Retrying in {retry_delay}s...",
                                    extra={"session_id": session_id},
                                )
                                await asyncio.sleep(retry_delay)
                            else:
                                raise

                    if not audio_bytes:
                        raise ValueError("Failed to retrieve audio content from Chatwoot")


                    if len(audio_bytes) > MAX_AUDIO_BYTES:
                        raise ValueError(f"Audio attachment exceeds {MAX_AUDIO_BYTES} bytes")

                    raw_transcript = await transcribe_audio_async(audio_bytes, "audio/ogg")
                    logger.info("Raw audio transcript: '%s'", raw_transcript, extra={"session_id": session_id})
                    
                    cleaned_transcript = strip_hindi_characters(raw_transcript)
                    
                    user_input = cleaned_transcript if cleaned_transcript.strip() else raw_transcript
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

            pending_question = conversation_memory.get_pending_question(session_id)
            if pending_question:
                if _is_affirmative(user_input):
                    summary_note = (
                        f"📌 **Human Handoff Requested by User**\n"
                        f"**Original Unanswered Question:** {pending_question}"
                    )
                    await send_chatwoot_private_note(conversation_id, summary_note)
                    await add_chatwoot_label(conversation_id, HANDOFF_LABEL)
                    await update_chatwoot_status(conversation_id, "open")
                    await send_chatwoot_msg(conversation_id, "Connecting you now...")
                    
                    conversation_memory.clear_pending_question(session_id)
                    return
                else:
                    # User asked a new question or said "No"; clear flag and let full flow run
                    conversation_memory.clear_pending_question(session_id)

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
                await add_chatwoot_label(conversation_id, HANDOFF_LABEL)
                await update_chatwoot_status(conversation_id, "open")
                return

            reply_text, handoff_triggered = await asyncio.to_thread(
                generate_reply, session_id, user_input
            )

            reply_text = format_bot_response(reply_text)
            await send_chatwoot_msg_chunks(conversation_id, reply_text)

            if handoff_triggered:
                summary_note = (
                    f"📌 **Human Handoff Triggered**\n"
                    f"**Customer Message:** {user_input}"
                )
                await send_chatwoot_private_note(conversation_id, summary_note)

                await add_chatwoot_label(conversation_id, HANDOFF_LABEL)
                await update_chatwoot_status(conversation_id, "open")
                
        except Exception:
            logger.exception("process_chatwoot_webhook failed", extra={"session_id": session_id})

@app.post("/webhooks/chatwoot")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        logger.debug(
            "Chatwoot webhook payload received: event=%s message_type=%s",
            payload.get("event"), payload.get("message_type"),
        )
    except Exception:
        return {"status": "received"}

    # Guard 1: Message Type Check
    if payload.get("message_type") not in _INCOMING_MESSAGE_TYPES:
        return {"status": "received"}

    # Guard 2: Sender Check
    sender = payload.get("sender") or {}
    if sender.get("type") == "user":
        return {"status": "received"}

    # Guard 3: Private Note Check
    if payload.get("private") is True:
        return {"status": "received"}

    # Guard 4: Human Handoff / Active Escalation Check
    conversation = payload.get("conversation") or {}
    raw_labels = conversation.get("labels") or []
    labels_set = set()
    if isinstance(raw_labels, str):
        labels_set = {l.strip() for l in raw_labels.split(",") if l.strip()}
    elif isinstance(raw_labels, list):
        for item in raw_labels:
            if isinstance(item, str):
                labels_set.add(item)
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name")
                if title:
                    labels_set.add(title)

    assignee = conversation.get("assignee")
    raw_labels = conversation.get("labels") or []
    labels_set = set()
    if isinstance(raw_labels, list):
        for item in raw_labels:
            if isinstance(item, str):
                labels_set.add(item)
            elif isinstance(item, dict):
                labels_set.add(item.get("title") or item.get("name", ""))

    if assignee is not None or "human_handoff" in labels_set:
        logger.info("Bot muting reply — assignee=%s labels=%s", assignee, labels_set)
        return {"status": "received"}

    background_tasks.add_task(process_chatwoot_webhook, payload)
    return {"status": "received"}

# --- Entrypoint ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)