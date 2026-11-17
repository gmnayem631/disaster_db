"""Hazard check: asks Qwen whether an article really reports a disaster.

The keyword filter in ingestion.py lets through many articles that only
mention a hazard word ("fire safety clearance", "set tyres on fire", "the
1970 cyclone"). Before the slow full extraction, Qwen reads the title and the
first part of the article and answers four short questions:

  reason          one sentence explaining the answer (asked first, so the
                  model thinks before choosing)
  report_type     what kind of article it is (see REPORT_TYPES)
  in_bangladesh   did the event happen in, or directly affect, Bangladesh?
  hazards         which hazards, from the list in data/hazard_keywords.txt

Python, not the model, then decides keep or drop (KEEP_REPORT_TYPES below).
run_pending() in extraction_service.py calls this before extracting an article.
"""
import json
import os
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from database.documents import utc_now

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_FILE = os.path.join(BASE_DIR, "data", "hazard_keywords.txt")

# Bump when the prompt, the answer format or the keep rule changes.
SCREENING_VERSION = "s1"

WORDS_SENT     = 400      # the title and first 400 words are enough to tell what an article is about
RAW_REPLY_KEPT = 2000

REPORT_TYPES = {
    "CURRENT_EVENT":   "reports a disaster that has just happened or is still happening",
    "FOLLOW_UP":       "reports the aftermath of a recent disaster: relief, damage, victims, recovery, investigation",
    "WARNING":         "an official warning of an approaching hazard (cyclone signal, flood warning, evacuation)",
    "PAST_REFERENCE":  "mentions a disaster only from the past (history, anniversary, background)",
    "GENERAL":         "about disasters in general: policy, preparedness, research, statistics, opinion, no specific new event",
    "NOT_DISASTER":    "not about a disaster at all (the hazard word is used another way)",
}
# Which answers are kept. DECISION for the author: WARNING is kept for now.
KEEP_REPORT_TYPES = {"CURRENT_EVENT", "FOLLOW_UP", "WARNING"}

SYSTEM_PROMPT = """You check Bangladeshi English news articles for a disaster database.
Read the article and decide whether it reports a real disaster event.

Report types:
{report_types}

Rules:
- A hazard word alone does not make a disaster report. "Fire safety clearance",
  "opened fire", "a flood of complaints", "a storm of criticism" are NOT_DISASTER.
- Burning tyres, vehicles or rubbish during a protest is NOT_DISASTER.
- Road, rail, boat and launch accidents with victims are disasters (ROAD_ACCIDENT,
  WATER_TRANSPORT_ACCIDENT).
- Disease outbreaks with reported cases or deaths are disasters (DISEASE_OUTBREAK).
- in_bangladesh is true only if the event happened in Bangladesh or directly
  affected people in Bangladesh.
- hazards: choose only from this list: {hazards}. Use OTHER for a disaster not in
  the list. Leave it empty for NOT_DISASTER.
- Use only what the article says. Write the reason in one short sentence.

Return only JSON in the required format."""


def hazard_names(path=KEYWORDS_FILE):
    """The hazard names from data/hazard_keywords.txt, in file order, plus OTHER."""
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                names.append(line.split(":", 1)[0].strip().upper())
    return names + ["OTHER"]


def answer_schema(hazards):
    """The JSON format Ollama is told to follow. The title lets FakeBackend
    recognise a hazard-check request."""
    return {
        "title": "HazardCheck",
        "type": "object",
        "properties": {
            "reason":        {"type": "string"},
            "report_type":   {"type": "string", "enum": list(REPORT_TYPES)},
            "in_bangladesh": {"type": "boolean"},
            "hazards":       {"type": "array", "items": {"type": "string", "enum": list(hazards)}},
        },
        "required": ["reason", "report_type", "in_bangladesh", "hazards"],
    }


class _Answer(BaseModel):
    reason: str
    report_type: str
    in_bangladesh: bool
    hazards: list[str]


def build_prompt(article, words=WORDS_SENT):
    text_words = (article.get("text") or "").split()
    excerpt = " ".join(text_words[:words])
    if len(text_words) > words:
        excerpt += " [...]"
    return f"Title: {article.get('title') or '(none)'}\n\nArticle:\n{excerpt}"


def needs_screening(article):
    """Only articles found by the daily scraper are checked. Articles from the
    annotated corpus are already known disaster reports."""
    return article.get("origin") == "daily" and article.get("screening") is None


@dataclass
class ScreeningResult:
    status: str                        # "ok" | "failed"
    record: dict | None = None         # stored on the article as `screening`
    error: dict | None = None
    attempts: int = 0
    seconds: float = 0.0
    raw_replies: list = field(default_factory=list)


def screen_article(article, backend, *, hazards=None, max_attempts=2, now=None):
    """Ask the LLM about one article. Returns a ScreeningResult.

    BackendUnavailable (Ollama not running) is NOT caught: the caller stops the run.
    """
    from extraction_service import BackendError, BackendUnavailable   # avoids a circular import

    hazards = hazards or hazard_names()
    system_prompt = SYSTEM_PROMPT.format(
        report_types="\n".join(f"- {k}: {v}" for k, v in REPORT_TYPES.items()),
        hazards=", ".join(hazards))
    schema = answer_schema(hazards)
    prompt = build_prompt(article)
    result = ScreeningResult(status="failed")
    started = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        raw = None
        try:
            response = backend.complete(system_prompt, prompt, schema)
            raw = response.content
            if not raw or not raw.strip():
                raise BackendError("Empty response from model")
            answer = _Answer.model_validate_json(raw)
            if answer.report_type not in REPORT_TYPES:
                raise ValueError(f"unknown report_type {answer.report_type!r}")
        except BackendUnavailable:
            raise
        except (ValidationError, ValueError) as exc:
            result.error = {"kind": "invalid_json", "message": str(exc)[:500],
                            "raw_reply": (raw or "")[:RAW_REPLY_KEPT]}
            continue
        except Exception as exc:
            result.error = {"kind": "request_failed", "message": f"{type(exc).__name__}: {exc}",
                            "raw_reply": (raw or "")[:RAW_REPLY_KEPT] or None}
            continue

        # A hazard outside the list becomes OTHER rather than being stored as-is.
        found = sorted({h if h in hazards else "OTHER" for h in answer.hazards})
        keep = answer.report_type in KEEP_REPORT_TYPES and answer.in_bangladesh
        result.status, result.error = "ok", None
        result.record = {
            "version":       SCREENING_VERSION,
            "keep":          keep,
            "report_type":   answer.report_type,
            "in_bangladesh": answer.in_bangladesh,
            "hazards":       found,
            "reason":        answer.reason.strip()[:500],
            "words_sent":    min(WORDS_SENT, len((article.get("text") or "").split())),
            "prompt_tokens": response.prompt_tokens,
            "llm":           backend.describe(),
            "at":            now or utc_now(),
        }
        break

    result.seconds = time.perf_counter() - started
    if result.record:
        result.record["seconds"] = round(result.seconds, 1)
    return result


def drop_reason(record):
    """The text stored on a dropped article as its filter reason."""
    where = "" if record["in_bangladesh"] else ", not in Bangladesh"
    return f"hazard check: {record['report_type']}{where}: {record['reason']}"


def fake_answer(report_type="CURRENT_EVENT", hazards=("FLOOD",), in_bangladesh=True,
                reason="fake answer"):
    """A well-formed answer, for tests."""
    return json.dumps({"reason": reason, "report_type": report_type,
                       "in_bangladesh": in_bangladesh, "hazards": list(hazards)})
