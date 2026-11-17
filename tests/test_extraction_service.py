"""Tests for extraction_service.extract_article().

FakeBackend stands in for Qwen and fake_branch_a stands in for the XLM-R
model, so nothing here needs Ollama, the model files or MongoDB.
"""
import json

import pytest

from database.documents import build_article, build_claims
from extraction.branch_b.candidate_generator import generate_candidates
from extraction_service import (BackendError, BackendUnavailable, FakeBackend,
                                auto_reply, extract_article)

TEXT = ("Floods have devastated several districts of Bangladesh. "
        "The nationwide death toll has risen to 31 after three more people died in Feni district. "
        "At least 58 lakh people have been affected across 11 districts. "
        "Around 10,000 families remained marooned in Companiganj upazila.")

# 13 sentences, one number each -> 13 candidates -> batches of 6, 6 and 1
LONG_TEXT = " ".join(f"Officials said {n} people died in the area." for n in range(11, 24))


def fake_branch_a(text, source_url, publish_date=None):
    return {"source_url": source_url, "publish_date": publish_date,
            "disaster_type": [{"value": "Floods", "source": "model"}],
            "locations": {"districts": [{"value": "Feni", "source": "model"}],
                          "upazilas": [{"value": "Companiganj", "source": "model"}],
                          "unions": []},
            "agencies_involved": [], "raw_entities": []}


def article(text=TEXT):
    return {"_id": "abc123", "url": "https://www.thedailystar.net/news/x",
            "publish_date": None, "text": text}


def run(backend, text=TEXT, **options):
    return extract_article(article(text), backend, branch_a=fake_branch_a, **options)


def candidate(text, raw):
    """The candidate the real generator makes for this number."""
    _, candidates = generate_candidates(text)
    return next(c for c in candidates if c.raw == raw)


def reply_with(candidate_id, **fields):
    """A reply function: answers everything, with chosen fields for one candidate."""
    def build(prompt):
        reply = auto_reply(prompt)
        for c in reply["claims"]:
            if c["candidate_id"] == candidate_id:
                c.update(fields)
        return reply
    return build


# ── The normal case ────────────────────────────────────────────────────────
def test_complete_extraction():
    result = run(FakeBackend())
    assert result.status == "complete"
    assert result.error is None
    assert result.candidate_count == len(result.claims) > 0
    assert result.unanswered == []
    assert all(b.status == "ok" and b.attempts == 1 for b in result.batches)
    assert result.hazard_types == ["FLOOD"]
    assert [(l["location_id"], l["name"], l["level"]) for l in result.locations] == [
        ("L1", "Feni", "district"), ("L2", "Companiganj", "upazila")]


def test_thirteen_candidates_make_batches_of_6_6_1():
    backend = FakeBackend()
    result = run(backend, LONG_TEXT)
    assert result.candidate_count == 13
    assert [len(b.candidate_ids) for b in result.batches] == [6, 6, 1]
    assert len(backend.calls) == 3
    assert result.status == "complete"


def test_ids_are_resolved_to_sentence_text_and_place_names():
    toll = candidate(TEXT, "31")
    backend = FakeBackend([reply_with(toll.candidate_id, impact_type="FATALITIES",
                                      aggregation="CUMULATIVE", unit="people",
                                      location_ids=["L1"])])
    result = run(backend, batch_size=50)
    claim = next(c for c in result.claims if c["candidate_id"] == toll.candidate_id)

    assert claim["impact_type"] == "FATALITIES"
    assert claim["raw_value"] == "31"
    assert claim["normalized_value"] == 31
    assert claim["unit"] == "people"
    assert claim["locations"] == [{"name": "Feni", "level": "district", "district": "Feni"}]
    assert "death toll has risen to 31" in claim["evidence"][0]["text"]
    assert TEXT[claim["char_start"]:claim["char_end"]] == "31"
    assert claim["validation_status"] == "ACCEPTED"


def test_numbers_come_from_python_not_from_the_llm():
    lakh = candidate(TEXT, "At least 58 lakh")

    def reply(prompt):
        answer = auto_reply(prompt, impact_type="AFFECTED_PEOPLE")
        for c in answer["claims"]:
            c["normalized_value"] = 999          # the LLM "inventing" a number
            c["value"] = 999
        return answer

    result = run(FakeBackend([reply]), batch_size=50)
    claim = next(c for c in result.claims if c["candidate_id"] == lakh.candidate_id)
    assert claim["normalized_value"] == 5_800_000
    assert claim["modifier"] == "AT_LEAST"
    assert all(c["normalized_value"] != 999 for c in result.claims)


def test_claims_fit_the_database_layer():
    """The records must be accepted by database.documents.build_claims unchanged."""
    result = run(FakeBackend())
    stored = build_claims(build_article("https://www.thedailystar.net/news/x", text=TEXT),
                          result.versions["extraction"], result.claims)
    assert len(stored) == len(result.claims)


def test_extraction_record_is_plain_data():
    result = run(FakeBackend())
    record = result.extraction_record()
    assert record["status"] == "complete" and record["version"] == "b1"
    assert record["candidate_count"] == record["claim_count"] == len(result.claims)
    assert record["llm"]["model"] == "fake-model"
    assert record["batches"][0]["candidate_ids"] == result.batches[0].candidate_ids
    json.dumps(record, default=str)              # no objects MongoDB could not store


# ── Things that go wrong in a batch ────────────────────────────────────────
def test_batch_that_fails_once_then_succeeds():
    backend = FakeBackend([auto_reply, BackendError("timeout"), auto_reply, auto_reply])
    result = run(backend, LONG_TEXT)
    assert result.status == "complete"
    assert [b.attempts for b in result.batches] == [1, 2, 1]
    assert result.batches[1].error is None


def test_batch_that_always_fails_makes_the_article_partial():
    backend = FakeBackend([auto_reply, BackendError("timeout"), BackendError("timeout"), auto_reply])
    result = run(backend, LONG_TEXT)            # must not raise

    failed = result.batches[1]
    assert result.status == "partial"
    assert failed.status == "failed" and failed.attempts == 2
    assert len(failed.candidate_ids) == 6       # we know exactly which candidates were lost
    assert "timeout" in failed.error["message"]
    assert result.unanswered == failed.candidate_ids
    assert result.error["stage"] == "llm"
    # the other two batches still ran and are reported
    assert [b.status for b in result.batches] == ["ok", "failed", "ok"]
    assert len(result.claims) == 7


def test_reply_that_is_not_valid_json_is_kept_in_the_error():
    backend = FakeBackend(["0000000000 not json", "still not json"])
    result = run(backend, batch_size=50)
    report = result.batches[0]
    assert result.status == "partial"
    assert report.status == "failed" and report.attempts == 2
    assert report.error["kind"] == "invalid_json"
    assert report.error["raw_reply"] == "still not json"


def test_empty_reply_counts_as_a_failed_attempt():
    result = run(FakeBackend(["", auto_reply]), batch_size=50)
    assert result.status == "complete"
    assert result.batches[0].attempts == 2


def test_wrong_enum_value_is_a_failed_attempt():
    bad = lambda prompt: auto_reply(prompt, impact_type="DEATHS")     # not in the taxonomy
    result = run(FakeBackend([bad, bad]), batch_size=50)
    assert result.status == "partial"
    assert result.batches[0].error["kind"] == "invalid_json"
    assert result.claims == []


def test_skipped_candidate_is_retried_then_reported():
    toll = candidate(TEXT, "31")
    skip = lambda prompt: auto_reply(prompt, skip=[toll.candidate_id])
    backend = FakeBackend([skip, skip])
    result = run(backend, batch_size=50)

    assert len(backend.calls) == 2              # it was asked a second time
    assert result.status == "partial"
    assert result.unanswered == [toll.candidate_id]
    assert result.batches[0].status == "incomplete"
    assert result.batches[0].error["kind"] == "unanswered_candidates"
    assert toll.candidate_id not in [c["candidate_id"] for c in result.claims]


def test_skipped_candidate_answered_on_the_retry():
    toll = candidate(TEXT, "31")
    skip = lambda prompt: auto_reply(prompt, skip=[toll.candidate_id])
    result = run(FakeBackend([skip, auto_reply]), batch_size=50)
    assert result.status == "complete"
    assert result.unanswered == []
    assert result.batches[0].attempts == 2


def test_valid_reply_that_answers_nothing_is_a_failed_batch_with_a_reason():
    empty = {"claims": []}
    result = run(FakeBackend([empty, empty]), batch_size=50)
    report = result.batches[0]
    assert result.status == "partial"
    assert report.status == "failed"
    assert report.error["kind"] == "unanswered_candidates"
    assert report.unanswered == report.candidate_ids


def test_id_that_was_not_asked_is_dropped_and_counted():
    invent = lambda prompt: auto_reply(prompt, extra=["Q99"])
    result = run(FakeBackend([invent]), batch_size=50)
    assert result.status == "complete"
    assert result.batches[0].foreign_ids == ["Q99"]
    assert "Q99" not in [c["candidate_id"] for c in result.claims]


def test_id_from_another_batch_is_dropped():
    first_batch_id = generate_candidates(LONG_TEXT)[1][0].candidate_id
    replies = [auto_reply, lambda p: auto_reply(p, extra=[first_batch_id]), auto_reply]
    result = run(FakeBackend(replies), LONG_TEXT)
    assert result.batches[1].foreign_ids == [first_batch_id]
    assert len(result.claims) == 13             # not 14: no duplicate from the wrong batch


def test_missing_evidence_falls_back_to_the_numbers_own_sentence_and_is_flagged():
    toll = candidate(TEXT, "31")
    backend = FakeBackend([reply_with(toll.candidate_id, impact_type="FATALITIES",
                                      unit="people", evidence_sentence_ids=[])])
    result = run(backend, batch_size=50)
    claim = next(c for c in result.claims if c["candidate_id"] == toll.candidate_id)
    assert claim["evidence"][0]["sentence_id"] == toll.sentence_id
    assert "EVIDENCE_DEFAULTED_TO_CANDIDATE_SENTENCE" in claim["validation_flags"]
    assert claim["validation_status"] == "REVIEW_REQUIRED"


def test_prompt_close_to_the_context_limit_is_flagged():
    result = run(FakeBackend(num_ctx=50), batch_size=50)
    assert "POSSIBLE_TRUNCATION" in result.batches[0].flags
    assert run(FakeBackend(), batch_size=50).batches[0].flags == []


# ── Things that go wrong outside the LLM ───────────────────────────────────
def test_llm_server_down_is_raised_to_the_caller():
    with pytest.raises(BackendUnavailable):
        run(FakeBackend([BackendUnavailable("connection refused")]))


def test_article_with_no_candidates_never_calls_the_llm():
    backend = FakeBackend()
    result = run(backend, "The weather was calm and the river stayed quiet.")
    assert result.status == "complete"
    assert result.claims == [] and result.batches == []
    assert backend.calls == []


def test_article_without_text_fails_without_calling_anything():
    backend = FakeBackend()
    result = run(backend, "   ")
    assert result.status == "failed"
    assert result.error["kind"] == "no_text"
    assert backend.calls == []


def test_branch_a_crash_is_reported_not_raised():
    def broken(text, source_url, publish_date=None):
        raise RuntimeError("model file missing")

    backend = FakeBackend()
    result = extract_article(article(), backend, branch_a=broken)
    assert result.status == "failed"
    assert result.error == {"stage": "branch_a", "kind": "RuntimeError",
                            "message": "model file missing"}
    assert backend.calls == []
