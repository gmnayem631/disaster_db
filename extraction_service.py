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

import screening
from database.documents import utc_now
from extraction.branch_b.candidate_generator import (HEADLINE_ID, format_for_llm,
                                                     generate_candidates, headline_candidates)
from extraction.branch_b.claim_validator import REVIEW_REQUIRED, validate_claims
from extraction.branch_b.schemas import BranchBResult

# Bump these when the prompt, schema, candidate generator or validator changes.
# They are stored with every claim so old and new results can be told apart.
EXTRACTION_VERSION = "b5"   # b5: unit check word by word; b4: short OTHER answers; b3: headline numbers
PROMPT_VERSION     = "p5"   # p5: priority rules + examples from TRAIN articles; p4: OTHER id + type only

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
    found = re.findall(r'^([QCH]\d+) = ".*" in ([ST]\d+)', user_prompt, flags=re.MULTILINE)
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

    Hazard-check requests (screening.py) are answered separately, from
    `screen_replies` (same kinds of items), so they never use up `replies`.
    With no list, every article passes the check as a FLOOD report.
    """
    name  = "fake"
    model = "fake-model"

    def __init__(self, replies=None, num_ctx=8192, screen_replies=None):
        self.replies = list(replies) if replies is not None else None
        self.screen_replies = list(screen_replies) if screen_replies is not None else None
        self.num_ctx = num_ctx
        self.calls   = []          # every extraction prompt received, for tests to inspect
        self.screen_calls = []     # every hazard-check prompt received

    def complete(self, system_prompt, user_prompt, json_schema):
        if (json_schema or {}).get("title") == "HazardCheck":
            from screening import fake_answer
            self.screen_calls.append(user_prompt)
            if self.screen_replies is None:
                reply = fake_answer()
            elif not self.screen_replies:
                raise BackendError("FakeBackend has no hazard-check reply left")
            else:
                reply = self.screen_replies.pop(0)
        else:
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


class OllamaBackend:
    """The real LLM: a model served by Ollama on this machine.

    The settings below are the ones the project found necessary (CLAUDE.md
    section 7) and are always sent, because the model's own defaults are
    temperature 0.6 and thinking switched on:
      think=False    as its own argument, not inside options
      temperature 0  same input should give the same output
      num_gpu 0      CPU only; GPU offload on the lab PC produced garbage
      num_ctx        set explicitly so the prompt is not cut off silently
    """
    name = "ollama"

    def __init__(self, model="qwen3:8b", *, host="http://127.0.0.1:11434", num_ctx=8192,
                 num_predict=4096, timeout=900, keep_alive="10m", client=None):
        self.model      = model
        self.host       = host
        self.keep_alive = keep_alive
        self.options    = {"temperature": 0, "num_gpu": 0,
                           "num_ctx": num_ctx, "num_predict": num_predict}
        if client is None:
            import ollama                      # imported here so tests need no Ollama
            client = ollama.Client(host=host, timeout=timeout)
        self._client = client                  # tests pass a stand-in client
        self._digest = None

    def complete(self, system_prompt, user_prompt, json_schema):
        started = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                format=json_schema,
                options=dict(self.options),
                think=False,
                stream=False,
                keep_alive=self.keep_alive,
            )
        except ConnectionError as exc:
            # The ollama client raises this when the server is not running.
            raise BackendUnavailable(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                # Model not pulled. Asking again will not help.
                raise BackendUnavailable(
                    f"Ollama does not have model '{self.model}': {exc}") from exc
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

        return LLMResponse(
            content=response.message.content or "",
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
            seconds=time.perf_counter() - started,
        )

    def describe(self):
        """What was used, saved with every extraction for reproducibility."""
        if self._digest is None:
            try:
                for item in self._client.list().models:
                    if item.model == self.model:
                        self._digest = item.digest
            except Exception:
                pass                           # not worth failing an extraction over
        return {"backend": self.name, "model": self.model, "model_digest": self._digest,
                "host": self.host, "think": False, "options": dict(self.options)}


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
            # The LLM named no usable sentence. Use the one the number sits in.
            sentence_ids = [candidate.sentence_id]
            if checked["impact_type"] != "OTHER":
                # An impact claim without evidence: say so and let a person look.
                # (OTHER answers are one short line on purpose and give no evidence.)
                flags.append("EVIDENCE_DEFAULTED_TO_CANDIDATE_SENTENCE")
                if status == "ACCEPTED":
                    status = REVIEW_REQUIRED

        # Positions of a headline number are in the title, not in articles.text.
        in_headline = candidate.sentence_id == HEADLINE_ID
        if in_headline:
            flags.append("FROM_HEADLINE")

        def position(sid, which):
            return None if sid == HEADLINE_ID else getattr(sentences_by_id[sid], which)

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
                          "start_char": position(sid, "start_char"),
                          "end_char": position(sid, "end_char")}
                         for sid in sentence_ids],
            "char_start":        None if in_headline else candidate.start_char,
            "char_end":          None if in_headline else candidate.end_char,
            "validation_status": status,
            "validation_flags":  flags,
        })
    return records


# ── The main function ──────────────────────────────────────────────────────
def extract_article(article, backend, *, branch_a=None, branch_a_model=None,
                    batch_size=6, max_batch_attempts=2):
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
                       "branch_a_model": branch_a_model or "injected",
                       "llm": backend.describe()}

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
        # Numbers that only the headline gives (H1, H2...), asked first.
        headline, from_headline = headline_candidates(
            article.get("title"), candidates, sentences[0].sentence_id if sentences else None)
        if from_headline:
            sentences, candidates = [headline] + sentences, from_headline + candidates
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


# ── Many articles: read from the database, extract, save ──────────────────
@dataclass
class RunReport:
    """What one run did. Printed and logged at the end of every run."""
    started_at: object = None
    finished_at: object = None
    seconds: float = 0.0
    picked_up: int = 0             # articles that were waiting
    extracted: int = 0             # completed and saved
    screened_out: int = 0          # the hazard check said: not a disaster report
    failed: int = 0                # will be retried later
    dead: int = 0                  # failed too many times; a person must look
    claims_saved: int = 0
    not_attempted: int = 0         # left waiting because the run stopped early
    stopped_early: str | None = None    # why the run stopped, if it did
    articles: list = field(default_factory=list)   # one line per article handled

    def summary(self):
        lines = [f"Extraction run: {self.picked_up} waiting, {self.extracted} extracted "
                 f"({self.claims_saved} claims), {self.screened_out} dropped by the hazard "
                 f"check, {self.failed} failed, {self.dead} dead, "
                 f"{self.not_attempted} not attempted, {self.seconds:.0f} s"]
        if self.picked_up == 0:
            lines.append("  Nothing was waiting. If articles were expected, check the scraper.")
        if self.stopped_early:
            lines.append(f"  STOPPED EARLY: {self.stopped_early}")
        for a in self.articles:
            if a["outcome"] != "extracted":
                lines.append(f"  {a['outcome'].upper()} {a['article_id']} "
                             f"{(a['title'] or '')[:50]!r}: {a['error']}")
        return "\n".join(lines)


def run_pending(store, backend, *, limit=50, branch_a=None, batch_size=6,
                max_batch_attempts=2, screen=True, stop_at=None, progress=None, now=None):
    """Extract every article that is waiting, and record what happened to each.

    store:    database.store.Store
    backend:  OllamaBackend in production, FakeBackend in tests
    stop_at:  a time.monotonic() value; no new article is started after it.
              The rest stay waiting, with no retry used, for the next run.
    progress: function(str) for one line per article (e.g. print)
    now:      only for tests; leave it out normally

    For each waiting article (never tried, or failed earlier and due for retry):
      first, if it came from the daily scraper and has not been checked yet,
      the hazard check (screening.py) asks the LLM whether it really reports a
      disaster. No -> status `filtered_out` with the reason; no extraction.
      The answer is stored on the article as `screening` either way.
      Then the extraction:
      complete -> claims and entities are saved, status becomes `extracted`
      anything else -> NO claims are saved; the error and the batch reports are
                       stored on the article and a retry is scheduled.
                       The third failure makes it `dead`.

    The run stops at once, leaving the remaining articles untouched and their
    retries unused, when the problem is the machine and not the article:
    the LLM server is down, or the Branch A model cannot be loaded.
    """
    clock = time.perf_counter()
    report = RunReport(started_at=utc_now())

    def finish():
        report.finished_at = utc_now()
        report.seconds = time.perf_counter() - clock
        return report

    pending = store.get_pending("extraction", limit=limit, now=now)
    report.picked_up = len(pending)
    if not pending:
        return finish()

    # Load the Branch A model once, before touching any article.
    branch_a_model = None
    if branch_a is None:
        try:
            branch_a, branch_a_model = _default_branch_a()
        except Exception as exc:
            report.stopped_early = f"Branch A model could not be loaded: {type(exc).__name__}: {exc}"
            report.not_attempted = len(pending)
            return finish()

    say = progress or (lambda message: None)

    def done(line, position):
        line["seconds"] = time.perf_counter() - started
        report.articles.append(line)
        note = f" ({line['claims']} claims)" if line["outcome"] == "extracted" else f": {line['error']}"
        say(f"[{position + 1}/{len(pending)}] {line['outcome'].upper()} "
            f"{(line['title'] or '')[:60]}{note} [{line['seconds']:.0f} s]")

    for position, article in enumerate(pending):
        if stop_at is not None and time.monotonic() >= stop_at:
            report.stopped_early = "time limit reached; the rest wait for the next run"
            report.not_attempted = len(pending) - position
            return finish()
        article_id = article["_id"]
        line = {"article_id": article_id, "title": article.get("title"),
                "outcome": None, "claims": 0, "seconds": 0.0, "error": None}
        started = time.perf_counter()

        if screen and screening.needs_screening(article):
            try:
                check = screening.screen_article(article, backend, now=now)
            except BackendUnavailable as exc:
                report.stopped_early = f"LLM backend unavailable: {exc}"
                report.not_attempted = len(pending) - position
                return finish()
            if check.status != "ok":
                message = f"[screening] {check.error['message']}"
                status = store.mark_failed(article_id, stage="extraction",
                                           kind=check.error["kind"], message=message, now=now)
                line["outcome"] = "dead" if status == "dead" else "failed"
                line["error"] = message
                report.dead += status == "dead"
                report.failed += status != "dead"
                done(line, position)
                continue
            store.save_screening(article_id, check.record, now=now)
            if not check.record["keep"]:
                reason = screening.drop_reason(check.record)
                store.mark_filtered_out(article_id, reason, now=now)
                line["outcome"] = "screened_out"
                line["error"] = reason
                report.screened_out += 1
                done(line, position)
                continue

        try:
            result = extract_article(article, backend, branch_a=branch_a,
                                     branch_a_model=branch_a_model, batch_size=batch_size,
                                     max_batch_attempts=max_batch_attempts)
        except BackendUnavailable as exc:
            report.stopped_early = f"LLM backend unavailable: {exc}"
            report.not_attempted = len(pending) - position     # this article and all after it
            return finish()

        error = result.error
        if result.status == "complete":
            try:
                line["claims"] = store.save_extraction(
                    article_id, entities=result.entities, hazard_types=result.hazard_types,
                    extraction=result.extraction_record(), claims=result.claims, now=now)
            except ValueError as exc:
                # The store refused the result (for example a claim that does not
                # fit the schema). Nothing was written; treat it as a failure.
                error = {"stage": "save", "kind": "save_refused", "message": str(exc)}

        if result.status == "complete" and error is None:
            line["outcome"] = "extracted"
            report.extracted += 1
            report.claims_saved += line["claims"]
        else:
            status = store.mark_failed(
                article_id, stage="extraction", kind=error["kind"],
                message=f"[{error['stage']}] {error['message']}",
                extraction=result.extraction_record(), now=now)
            line["outcome"] = "dead" if status == "dead" else "failed"
            line["error"] = f"[{error['stage']}] {error['message']}"
            if status == "dead":
                report.dead += 1
            else:
                report.failed += 1

        done(line, position)

    return finish()
