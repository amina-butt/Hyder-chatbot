"""
faq_router.py
--------------
Zero-cost local intent router for simple, high-confidence informational
queries (price / specs / warranty / hours / location / contact).

This is a deliberately "dumb" keyword matcher, not a classifier. That's
the point: it must never guess. Its only job is to recognize the small
set of queries that are unambiguous enough to answer with a canned
template, and to get out of the way — returning None — for everything
else, so the existing ChromaDB + Gemini pipeline in rag_engine.py remains
the source of truth for anything even slightly nuanced.

Guardrail philosophy (see `FAQRouter.match`):
  1. A query must clear a minimum keyword-hit bar for a *single* intent.
  2. If it also hits keywords for a second, different intent, that's a
     compound/ambiguous question ("price and warranty?") — bail out.
  3. If a `requires_model` intent doesn't name exactly one model, bail
     out (zero models -> we don't know who to answer for; 2+ models ->
     that's a comparison, which is explicitly out of scope here and
     already has its own multi-model path in rag_engine.py).
  4. Any block_keyword (comparison words, negation, "why"/"explain",
     complaint language, etc.) on the matched intent immediately
     disqualifies that intent.
  5. Long queries (more word count than a simple lookup would ever need)
     are treated as inherently too complex for a static answer.
  6. Global complexity markers (negation, "why", "explain", "compare",
     conjunctions joining two asks) disqualify the whole message,
     regardless of which intent(s) it touched.

None of this is fuzzy — every rule is a hard cutoff. A false negative
here just costs one extra (already-existing) RAG round trip. A false
positive would mean confidently telling a customer something wrong with
no LLM in the loop to catch it, which is the failure mode this module
exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Resolved relative to this file, not the process's CWD, so it works the
# same whether the app is started from the repo root, a container
# WORKDIR, or a test runner. Override via env var if you want to point at
# a different file (e.g. a staging dataset) without touching code.
_DEFAULT_FAQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faqs.json")
FAQ_JSON_PATH = os.getenv("FAQ_JSON_PATH", _DEFAULT_FAQ_PATH)

# A simple lookup ("what's the ELi 100 price") is short. Anything
# noticeably longer than that is either compound, conditional, or
# genuinely needs the LLM to parse — so we refuse to local-match past
# this length rather than risk a wrong static answer on a nuanced ask.
_MAX_QUERY_WORDS = 14

# Words that, anywhere in the message, mean "this needs real reasoning,
# not a template" — comparisons, negation, causal/explanatory asks, and
# conjunctions that usually signal a compound question. Checked against
# the whole message regardless of which intent(s) matched, on top of each
# intent's own `block_keywords` (which are more targeted, e.g. "discount"
# only disqualifying the price intent specifically).
_GLOBAL_COMPLEXITY_MARKERS = {
    # comparison / relative
    "compare", "comparison", "vs", "versus", "difference", "better",
    "best", "cheaper", "cheapest", "behtar", "sabse acha", "sabse sasta",
    # negation
    "not", "isn't", "doesn't", "don't", "nahi", "nahin", "na", "bghair",
    "without",
    # causal / explanatory — these need a real answer, not a fact lookup
    "why", "kyun", "kyu", "kiun", "explain", "how does", "how come",
    # conditional — often hides a second, different question
    "what if", "agar", "magar",
    # complaint / support escalation — must not be silently auto-answered
    "problem", "issue", "complaint", "broken", "damaged", "kharab",
    "not working", "faulty",
}
_global_complexity_pattern = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _GLOBAL_COMPLEXITY_MARKERS) + r")\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zA-Z]+|[\u0600-\u06FF]+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Substring match for multi-word phrases; word-boundary match for
    single tokens, so e.g. the intent keyword "open" doesn't match inside
    "opening" in a way that changes meaning, while still letting short
    multi-word phrases like "how much" match as plain substrings."""
    phrase = phrase.lower().strip()
    if " " in phrase:
        return phrase in haystack
    return re.search(r"\b" + re.escape(phrase) + r"\b", haystack) is not None


@dataclass(frozen=True)
class FAQMatch:
    intent_key: str
    category: str
    model_key: Optional[str]
    language: str
    response_text: str
    matched_keywords: tuple[str, ...]


class FAQRouter:
    """Loads faqs.json once and evaluates incoming queries against it.

    Thread-safe for reads (the dataset is immutable after load); call
    `reload()` if you need to hot-swap faqs.json without a process
    restart (e.g. after replacing the mock data with production data).
    """

    def __init__(self, path: str = FAQ_JSON_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._models: dict = {}
        self._intents: list[dict] = []
        self._load()

    def _load(self) -> None:
        with self._lock:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._models = data.get("models", {})
                self._intents = data.get("intents", [])
                is_mock = data.get("_meta", {}).get("IS_MOCK_DATA", False)
                logger.info(
                    "FAQRouter loaded %d intents, %d models from %s (IS_MOCK_DATA=%s)",
                    len(self._intents), len(self._models), self._path, is_mock,
                )
            except Exception:
                # Same philosophy as rag_engine's own init: don't crash the
                # process over an optional optimization path. If faqs.json
                # is missing/malformed, the router just never matches
                # anything, and every query silently falls through to the
                # existing Gemini/ChromaDB pipeline.
                logger.exception(
                    "FAQRouter failed to load %s — router will pass every query "
                    "through to the normal RAG pipeline until this is fixed.",
                    self._path,
                )
                self._models = {}
                self._intents = []

    def reload(self) -> None:
        self._load()

    def _detect_model(self, normalized_query: str) -> tuple[Optional[str], int]:
        """Returns (model_key, how many distinct models were mentioned).

        The count is what matters for the ambiguity guard in `match()`:
        exactly one model found -> safe to answer; zero or 2+ -> bail.
        """
        found: list[str] = []
        for model_key, model_info in self._models.items():
            for alias in model_info.get("aliases", []):
                if _contains_phrase(normalized_query, alias):
                    found.append(model_key)
                    break
        unique = list(dict.fromkeys(found))  # de-dupe, preserve order
        if len(unique) == 1:
            return unique[0], 1
        return None, len(unique)

    def match(self, query: str, language_hint: str = "english") -> Optional[FAQMatch]:
        """Attempt a local, zero-cost match. Returns None on anything
        short of a confident, unambiguous, single-intent match — callers
        must treat None as "fall through to the normal pipeline", never
        as an error.
        """
        if not query or not self._intents:
            return None

        normalized = _normalize(query)
        word_count = len(normalized.split())

        # Guard 1: length. A real lookup is short; anything longer is
        # either compound or needs actual reasoning.
        if word_count == 0 or word_count > _MAX_QUERY_WORDS:
            return None

        # Guard 2: global complexity markers disqualify the whole
        # message, independent of which intent(s) it happens to touch.
        if _global_complexity_pattern.search(normalized):
            return None

        # Score every intent independently; keep only intents that
        # clear their own bar (min hits, no intent-specific block word).
        candidates: list[tuple[dict, tuple[str, ...]]] = []
        for intent in self._intents:
            block_keywords = intent.get("block_keywords", [])
            if any(_contains_phrase(normalized, bk) for bk in block_keywords):
                continue

            hits = tuple(
                kw for kw in intent.get("keywords", [])
                if _contains_phrase(normalized, kw)
            )
            min_hits = intent.get("min_keyword_hits", 1)
            if len(hits) >= min_hits:
                candidates.append((intent, hits))

        # Guard 3: exactly one intent must have matched. Two+ distinct
        # intents matching means a compound question ("price aur warranty
        # dono batao") — that needs real synthesis, not two templates
        # glued together.
        if len(candidates) != 1:
            return None

        intent, matched_keywords = candidates[0]
        intent_key = intent["intent_key"]
        category = intent.get("category", "general")
        requires_model = intent.get("requires_model", False)
        responses = intent.get("responses", {})

        model_key: Optional[str] = None
        if requires_model:
            model_key, model_count = self._detect_model(normalized)
            # Guard 4: zero models -> we don't know who to answer for.
            # Multiple models -> this is a comparison/multi-model ask;
            # rag_engine's existing _is_multi_model_query path already
            # handles that correctly, so we defer to it rather than
            # picking one model arbitrarily.
            if model_count != 1:
                return None
            template_bundle = responses.get(model_key)
        else:
            template_bundle = responses.get("default")

        if not template_bundle:
            # Data gap (e.g. a model was added to `models` but this
            # intent's `responses` wasn't updated for it yet) — fail
            # closed to the normal pipeline rather than guessing.
            return None

        response_text = template_bundle.get(language_hint) or template_bundle.get("english")
        if not response_text:
            return None

        return FAQMatch(
            intent_key=intent_key,
            category=category,
            model_key=model_key,
            language=language_hint,
            response_text=response_text,
            matched_keywords=matched_keywords,
        )


# Module-level singleton, loaded once at import time — same pattern as
# `conversation_memory` in memory.py. rag_engine.py imports this instance
# directly rather than constructing its own FAQRouter.
faq_router = FAQRouter()