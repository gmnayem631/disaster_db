"""Runs Branch A and Branch B on ONE article and returns everything that happened.

Design: docs/design.md section 2. This file calls the existing code
(entity_linker, candidate_generator, claim_validator). It does not replace it
and it does not touch MongoDB. Saving the result is a separate, later step.

The LLM is reached through a "backend" object, so the model can be swapped.
FakeBackend below is a stand-in for tests: no Ollama needed.
"""
import json
import re
import time
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError

from database.documents import utc_now
from extraction.branch_b.candidate_generator import format_for_llm, generate_candidates
from extraction.branch_b.claim_validator import REVIEW_REQUIRED, validate_claims
from extraction.branch_b.schemas import BranchBResult

# Bump these when the prompt, schema, candidate generator or validator changes.
# They are stored with every claim so old and new results can be told apart.
EXTRACTION_VERSION = "b1"
PROMPT_VERSION     = "p1"

RAW_REPLY_KEPT = 2000      # characters of a bad reply kept in the error record
TRUNCATION_SHARE = 0.9     # prompt using 90%+ of the context window gets flagged


# ── The LLM backend ────────────────────────────────────────────────────────
class BackendError(Exception):
    """One request went wrong (timeout, empty reply...). The batch is retried."""


class BackendUnavailable(BackendError):
    """The LLM server cannot be reached at all. Retrying is pointless, so
    extract_article() lets this one through to stop the whole run."""


@dataclass
class LLMResponse:
    content: str                       # the model's reply as text (should be JSON)
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    seconds: float = 0.0


# A backend is any object with:
#   name, model                                    (two strings)
#   complete(system_prompt, user_prompt, json_schema) -> LLMResponse
#   describe() -> dict                             (model, options; saved for reproducibility)
# It returns TEXT ONLY. Parsing and checking happen below, the same way for
# every backend.


def auto_reply(user_prompt, impact_type="OTHER", skip=(), extra=()):
    """Build a well-formed reply that answers every candidate listed in a prompt.
    For tests: `skip` leaves candidates out, `extra` adds IDs that were never asked."""
    claims = []
    found = re.findall(r'^([QC]\d+) = ".*" in (S\d+)', user_prompt, flags=re.MULTILINE)
    for candidate_id, sentence_id in found + [(e, "S1") for e in extra]:
        if candidate_id in skip:
            continue
        claims.append({"candidate_id": candidate_id, "impact_type": impact_type,
                       "evidence_sentence_ids": [sentence_id], "location_ids": [],
                       "aggregation": "UNKNOWN", "certainty": "REPORTED", "unit": None})
    return {"claims": claims}


class FakeBackend:
    """Pretends to be an LLM. Used by tests.

    replies: a list used up one request at a time. Each item may be
      - a string: returned as the reply text, exactly as given
      - a dict:   turned into JSON text
      - an Exception: raised, to imitate a failure
      - a function(user_prompt) returning any of the above
    With no list, every request is answered with auto_reply().
    """
    name  = "fake"
    model = "fake-model"

    def __init__(self, replies=None, num_ctx=8192):
        self.replies = list(replies) if replies is not None else None
        self.num_ctx = num_ctx
        self.calls   = []          # every user prompt received, for tests to inspect

    def complete(self, system_prompt, user_prompt, json_schema):
        self.calls.append(user_prompt)
        if self.replies is None:
            reply = auto_reply(user_prompt)
        elif not self.replies:
            raise BackendError("FakeBackend has no reply left for this request")
        else:
            reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(user_prompt)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            reply = json.dumps(reply)
        return LLMResponse(content=reply, prompt_tokens=len(user_prompt) // 4,
                           output_tokens=len(reply) // 4, seconds=0.0)

    def describe(self):
        return {"backend": self.name, "model": self.model,
                "options": {"num_ctx": self.num_ctx}}


# ── What extract_article() returns ─────────────────────────────────────────
@dataclass
class BatchReport:
    index: int
    candidate_ids: list
    status: str = "failed"             # "ok" | "incomplete" (some unanswered) | "failed"
    attempts: int = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    seconds: float = 0.0
    error: dict | None = None          # {kind, message, raw_reply} of the last failed attempt
    unanswered: list = field(default_factory=list)    # candidates the LLM skipped
    foreign_ids: list = field(default_factory=list)   # IDs it returned that were not asked
    flags: list = field(default_factory=list)


@dataclass
class ExtractionResult:
    article_id: str
    status: str                        # "complete" | "partial" | "failed"
    entities: dict | None = None       # Branch A record
    hazard_types: list = field(default_factory=list)
    locations: list = field(default_factory=list)
    claims: list = field(default_factory=list)        # in the shape the claims collection expects
    batches: list = field(default_factory=list)       # one BatchReport per batch, failed ones included
    unanswered: list = field(default_factory=list)
    candidate_count: int = 0
    versions: dict = field(default_factory=dict)
    started_at: object = None
    finished_at: object = None
    seconds: float = 0.0
    error: dict | None = None

    def extraction_record(self):
        """The summary stored on the article as `extraction` (design section 1.1)."""
        return {
            "version": self.versions.get("extraction"), "status": self.status,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "prompt_version": self.versions.get("prompt"),
            "branch_a_model": self.versions.get("branch_a_model"),
            "llm": self.versions.get("llm"),
            "candidate_count": self.candidate_count, "claim_count": len(self.claims),
            "unanswered": list(self.unanswered),
            "batches": [asdict(b) for b in self.batches],
        }


# ── Small helpers ──────────────────────────────────────────────────────────
def _default_branch_a():
    """Load the real Branch A only when it is needed. Importing entity_linker
    loads the XLM-R model, which tests must be able to avoid."""
    import os
    import entity_linker
    return entity_linker.extract_entities, os.path.basename(entity_linker.MODEL_PATH)


def _system_prompt():
    # Imported here because local_llm.py imports the ollama package at the top.
    from extraction.branch_b.local_llm import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_locations(entities):
    """Branch A districts and upazilas become L1, L2, ... for the LLM prompt.
    Unions are left out for now (open question 9)."""
    found = (entities or {}).get("locations") or {}
    locations = []
    for key, label, level in (("districts", "BD_DISTRICT", "district"),
                              ("upazilas", "BD_UPAZILA", "upazila")):
        for item in found.get(key, []):
            locations.append({
                "location_id": f"L{len(locations) + 1}", "name": item["value"],
                "type": label, "level": level,
                # An upazila's parent district is not known from the text alone.
                "district": item["value"] if level == "district" else None,
            })
    return locations


def hazard_types_from(entities):
    """Normalise Branch A's free-text disaster types to FLOOD / FIRE."""
    words = " ".join(d["value"].lower() for d in (entities or {}).get("disaster_type", []))
    hazards = []
    if any(w in words for w in ("flood", "inundat", "waterlog", "deluge")):
        hazards.append("FLOOD")
    if any(w in words for w in ("fire", "blaze", "inferno")):
        hazards.append("FIRE")
    return hazards


def _error(stage, exc):
    return {"stage": stage, "kind": type(exc).__name__, "message": str(exc)}


# ── One batch ──────────────────────────────────────────────────────────────
def _run_batch(index, batch, sentences, locations, backend, system_prompt,
               json_schema, max_attempts, num_ctx):
    """Ask the LLM about one batch, retrying if needed.
    Returns (BatchReport, list of ImpactClaim that belong to this batch)."""
    wanted = [c.candidate_id for c in batch]
    report = BatchReport(index=index, candidate_ids=wanted)
    prompt = format_for_llm(sentences, batch, locations)
    best_claims, best_missing = [], list(wanted)

    for attempt in range(1, max_attempts + 1):
        report.attempts = attempt
        started = time.perf_counter()
        raw = None
        try:
            response = backend.complete(system_prompt, prompt, json_schema)
            raw = response.content
            report.prompt_tokens = response.prompt_tokens
            report.output_tokens = response.output_tokens
            if not raw or not raw.strip():
                raise BackendError("Empty response from model")
            parsed = BranchBResult.model_validate_json(raw)
        except BackendUnavailable:
            raise                                   # stop the whole run
        except ValidationError as exc:
            report.error = {"kind": "invalid_json", "message": str(exc)[:500],
                            "raw_reply": raw[:RAW_REPLY_KEPT]}
            continue
        except Exception as exc:                    # timeout, empty reply, anything else
            report.error = {"kind": "request_failed", "message": f"{type(exc).__name__}: {exc}",
                            "raw_reply": raw[:RAW_REPLY_KEPT] if raw else None}
            continue
        finally:
            report.seconds += time.perf_counter() - started

        # The reply parsed. Did it answer exactly what was asked?
        mine    = [c for c in parsed.claims if c.candidate_id in wanted]
        foreign = [str(c.candidate_id) for c in parsed.claims if c.candidate_id not in wanted]
        answered = {c.candidate_id for c in mine}
        missing = [cid for cid in wanted if cid not in answered]

        if len(missing) <= len(best_missing):       # keep the most complete attempt
            best_claims, best_missing = mine, missing
            report.foreign_ids = foreign
            report.error = None
        if not missing:
            break                                   # every candidate answered

    if best_claims or not best_missing:
        report.status = "ok" if not best_missing else "incomplete"
        report.unanswered = best_missing
    else:
        report.status = "failed"
        report.unanswered = list(wanted)
    if report.status == "incomplete" or (report.status == "failed" and report.error is None):
        report.error = {"kind": "unanswered_candidates",
                        "message": f"LLM did not answer {best_missing}", "raw_reply": None}
    if num_ctx and report.prompt_tokens and report.prompt_tokens >= TRUNCATION_SHARE * num_ctx:
        report.flags.append("POSSIBLE_TRUNCATION")
    return report, best_claims


# ── Turning validated claims into storable records ─────────────────────────
def _claim_records(llm_claims, validated, candidates, sentences, locations):
    """Replace article-local IDs (S2, L1) with the real sentence text and place
    names, so a stored claim can be read without the article."""
    candidates_by_id = {c.candidate_id: c for c in candidates}
    sentences_by_id  = {s.sentence_id: s for s in sentences}
    locations_by_id  = {loc["location_id"]: loc for loc in locations}
    records = []

    # validate_claims returns exactly one record per claim, in the same order.
    for llm_claim, checked in zip(llm_claims, validated):
        candidate = candidates_by_id[llm_claim.candidate_id]
        quantity  = checked.get("normalized_quantity") or {}
        flags     = list(checked["validation_flags"])
        status    = checked["validation_status"]

        sentence_ids = [sid for sid in llm_claim.evidence_sentence_ids if sid in sentences_by_id]
        if not sentence_ids:
            # The LLM named no usable sentence. Use the one the number sits in,
            # and say so, instead of storing a claim with no evidence.
            sentence_ids = [candidate.sentence_id]
            flags.append("EVIDENCE_DEFAULTED_TO_CANDIDATE_SENTENCE")
            if status == "ACCEPTED":
                status = REVIEW_REQUIRED

        records.append({
            "candidate_id":     candidate.candidate_id,
            "candidate_kind":   candidate.kind,
            "impact_type":      checked["impact_type"],
            "raw_value":        candidate.raw,
            # Numbers come from quantity_normalizer.py (inside the validator),
            # never from the LLM.
            "normalized_value": quantity.get("value"),
            "lower_bound":      quantity.get("lower_bound"),
            "upper_bound":      quantity.get("upper_bound"),
            "modifier":         quantity.get("modifier"),
            "is_qualitative":   bool(quantity.get("qualitative")),
            "unit":             llm_claim.unit,
            "aggregation":      checked["aggregation"],
            "certainty":        checked["certainty"],
            "locations": [{"name": locations_by_id[lid]["name"],
                           "level": locations_by_id[lid]["level"],
                           "district": locations_by_id[lid]["district"]}
                          for lid in llm_claim.location_ids if lid in locations_by_id],
            "evidence": [{"sentence_id": sid, "text": sentences_by_id[sid].text.strip(),
                          "start_char": sentences_by_id[sid].start_char,
                          "end_char": sentences_by_id[sid].end_char}
                         for sid in sentence_ids],
            "char_start":        candidate.start_char,
            "char_end":          candidate.end_char,
            "validation_status": status,
            "validation_flags":  flags,
        })
    return records


# ── The main function ──────────────────────────────────────────────────────
def extract_article(article, backend, *, branch_a=None, batch_size=6, max_batch_attempts=2):
    """Run Branch A + Branch B on one article.

    article: a dict with the text and, if available, _id/article_id, url, publish_date.
    backend: the LLM to use (FakeBackend in tests).
    branch_a: function(text, source_url, publish_date) -> Branch A record.
              Left out, the real entity_linker is loaded.

    Never raises for an ordinary failure: the returned result's `status` and
    `error` say what happened, and every batch has a report. The one exception
    is BackendUnavailable (the LLM server is not running).
    """
    clock = time.perf_counter()
    article_id = article.get("_id") or article.get("article_id")
    url = article.get("url") or article.get("source_url")
    result = ExtractionResult(article_id=article_id, status="failed", started_at=utc_now())
    result.versions = {"extraction": EXTRACTION_VERSION, "prompt": PROMPT_VERSION,
                       "branch_a_model": "injected", "llm": backend.describe()}

    def finish(status, error=None):
        result.status, result.error = status, error
        result.finished_at = utc_now()
        result.seconds = time.perf_counter() - clock
        return result

    text = article.get("text")
    if not text or not text.strip():
        return finish("failed", {"stage": "input", "kind": "no_text",
                                 "message": "article has no text"})

    # 1. Branch A: hazard type, places, agencies
    try:
        if branch_a is None:
            branch_a, result.versions["branch_a_model"] = _default_branch_a()
        result.entities = branch_a(text, url, article.get("publish_date"))
        result.hazard_types = hazard_types_from(result.entities)
        result.locations = build_locations(result.entities)
    except Exception as exc:
        return finish("failed", _error("branch_a", exc))

    # 2. Candidates: every number and every impact sentence
    try:
        sentences, candidates = generate_candidates(text)
    except Exception as exc:
        return finish("failed", _error("candidates", exc))
    result.candidate_count = len(candidates)

    # 3. The LLM, one batch at a time
    llm_claims = []
    if candidates:
        system_prompt = _system_prompt()
        json_schema = BranchBResult.model_json_schema()
        num_ctx = (backend.describe().get("options") or {}).get("num_ctx")
        for start in range(0, len(candidates), batch_size):
            report, claims = _run_batch(start // batch_size + 1, candidates[start:start + batch_size],
                                        sentences, result.locations, backend, system_prompt,
                                        json_schema, max_batch_attempts, num_ctx)
            result.batches.append(report)
            result.unanswered.extend(report.unanswered)
            llm_claims.extend(claims)

    # 4. Deterministic checks, then resolve IDs to text
    try:
        validated = validate_claims(BranchBResult(claims=llm_claims), candidates,
                                    sentences, result.locations)
        result.claims = _claim_records(llm_claims, validated, candidates, sentences,
                                       result.locations)
    except Exception as exc:
        return finish("failed", _error("validation", exc))

    # 5. Complete only if every batch answered every candidate
    bad = [b for b in result.batches if b.status != "ok"]
    if bad:
        return finish("partial", {
            "stage": "llm", "kind": "batches_not_ok",
            "message": f"batches {[b.index for b in bad]} did not complete; "
                       f"unanswered candidates: {result.unanswered}"})
    return finish("complete")
