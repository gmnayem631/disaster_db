"""Tests for the hazard check (screening.py) and how run_pending() uses it.

FakeBackend answers the hazard-check requests from `screen_replies`;
no Ollama is needed.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

import screening
from database import schema
from database.store import Store
from extraction_service import BackendError, BackendUnavailable, FakeBackend, run_pending
from screening import fake_answer, screen_article

NOW = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)


@pytest.fixture
def store(test_db):
    s = Store(test_db)
    s.setup()
    return s


def fake_branch_a(text, source_url, publish_date=None):
    return {"source_url": source_url, "publish_date": publish_date,
            "disaster_type": [{"value": "flood", "source": "model"}],
            "locations": {"districts": [{"value": "Feni", "source": "model"}],
                          "upazilas": [], "unions": []},
            "agencies_involved": [], "raw_entities": []}


def add(store, n, origin="daily"):
    article_id, _ = store.add_article(
        f"https://www.thedailystar.net/news/story-{n}", title=f"Story {n}",
        text=f"The death toll has risen to {30 + n} in Feni district.",
        publish_date="2026-11-16", origin=origin, now=NOW + timedelta(minutes=n))
    return article_id


def run(store, backend, **options):
    options.setdefault("now", NOW)
    return run_pending(store, backend, branch_a=fake_branch_a, **options)


ARTICLE = {"_id": "a1", "title": "Fire at Gazipur factory", "origin": "daily",
           "text": "A fire broke out at a garment factory in Gazipur, killing 3 workers."}


# ── The hazard list and the prompt ─────────────────────────────────────────
def test_hazard_names_come_from_the_keyword_file_plus_other():
    names = screening.hazard_names()
    assert names[0] == "FLOOD" and "ROAD_ACCIDENT" in names and names[-1] == "OTHER"


def test_prompt_holds_the_title_and_only_the_first_words():
    article = {"title": "Long story", "text": " ".join(f"w{i}" for i in range(1000))}
    prompt = screening.build_prompt(article, words=50)
    assert "Title: Long story" in prompt
    assert "w49" in prompt and "w50" not in prompt and prompt.endswith("[...]")


def test_only_unchecked_daily_articles_need_the_check():
    assert screening.needs_screening({"origin": "daily", "screening": None})
    assert not screening.needs_screening({"origin": "corpus", "screening": None})
    assert not screening.needs_screening({"origin": "daily", "screening": {"keep": True}})


# ── screen_article(): one article ──────────────────────────────────────────
def test_disaster_report_is_kept_and_the_answer_recorded():
    backend = FakeBackend(screen_replies=[fake_answer("CURRENT_EVENT", ["FIRE"])])
    result = screen_article(ARTICLE, backend, now=NOW)
    assert result.status == "ok"
    r = result.record
    assert r["keep"] is True and r["hazards"] == ["FIRE"] and r["report_type"] == "CURRENT_EVENT"
    assert r["version"] == screening.SCREENING_VERSION and r["llm"]["model"] == "fake-model"
    assert "Fire at Gazipur factory" in backend.screen_calls[0]
    assert backend.calls == []                      # the extraction replies were not used


@pytest.mark.parametrize("report_type", ["PAST_REFERENCE", "GENERAL", "NOT_DISASTER"])
def test_non_reports_are_dropped(report_type):
    backend = FakeBackend(screen_replies=[fake_answer(report_type, [])])
    assert screen_article(ARTICLE, backend).record["keep"] is False


@pytest.mark.parametrize("report_type", ["CURRENT_EVENT", "FOLLOW_UP", "WARNING"])
def test_reports_warnings_and_follow_ups_are_kept(report_type):
    backend = FakeBackend(screen_replies=[fake_answer(report_type)])
    assert screen_article(ARTICLE, backend).record["keep"] is True


def test_disaster_outside_bangladesh_is_dropped():
    backend = FakeBackend(screen_replies=[fake_answer("CURRENT_EVENT", ["EARTHQUAKE"],
                                                      in_bangladesh=False)])
    record = screen_article(ARTICLE, backend).record
    assert record["keep"] is False
    assert "not in Bangladesh" in screening.drop_reason(record)


def test_hazard_outside_the_list_becomes_other():
    backend = FakeBackend(screen_replies=[fake_answer(hazards=["VOLCANO", "FLOOD"])])
    assert screen_article(ARTICLE, backend).record["hazards"] == ["FLOOD", "OTHER"]


def test_bad_reply_is_asked_again():
    backend = FakeBackend(screen_replies=["not json", fake_answer()])
    result = screen_article(ARTICLE, backend)
    assert result.status == "ok" and result.attempts == 2


def test_two_bad_replies_fail_and_keep_the_reply():
    unknown_type = json.dumps({"reason": "x", "report_type": "MAYBE",
                               "in_bangladesh": True, "hazards": []})
    backend = FakeBackend(screen_replies=["not json", unknown_type])
    result = screen_article(ARTICLE, backend)
    assert result.status == "failed" and result.record is None
    assert result.error["kind"] == "invalid_json" and "MAYBE" in result.error["raw_reply"]


def test_request_error_fails_after_the_attempts():
    backend = FakeBackend(screen_replies=[BackendError("timeout"), BackendError("timeout")])
    result = screen_article(ARTICLE, backend)
    assert result.status == "failed" and result.error["kind"] == "request_failed"


def test_server_down_is_not_caught():
    backend = FakeBackend(screen_replies=[BackendUnavailable("connection refused")])
    with pytest.raises(BackendUnavailable):
        screen_article(ARTICLE, backend)


# ── Inside run_pending() ───────────────────────────────────────────────────
def test_kept_article_is_extracted_and_keeps_its_check(store):
    a = add(store, 1)
    backend = FakeBackend(screen_replies=[fake_answer("CURRENT_EVENT", ["FLOOD"])])
    report = run(store, backend)
    article = store.get_article(a)
    assert report.extracted == 1 and report.screened_out == 0
    assert article["processing_status"] == schema.EXTRACTED
    assert article["screening"]["keep"] is True and article["screening"]["hazards"] == ["FLOOD"]


def test_dropped_article_is_filtered_out_and_not_extracted(store):
    a = add(store, 1)
    backend = FakeBackend(screen_replies=[fake_answer("NOT_DISASTER", [], reason="protest")])
    report = run(store, backend)
    article = store.get_article(a)
    assert report.screened_out == 1 and report.extracted == 0
    assert backend.calls == []                      # no extraction request was sent
    assert article["processing_status"] == schema.FILTERED_OUT
    assert article["last_error"]["message"] == "hazard check: NOT_DISASTER: protest"
    assert article["screening"]["keep"] is False
    assert store.get_claims(a) == []
    assert "SCREENED_OUT" in report.summary() and "1 dropped by the hazard check" in report.summary()


def test_corpus_articles_skip_the_check(store):
    add(store, 1, origin="corpus")
    backend = FakeBackend(screen_replies=[])        # would fail if asked
    report = run(store, backend)
    assert report.extracted == 1 and backend.screen_calls == []


def test_check_can_be_switched_off(store):
    add(store, 1)
    backend = FakeBackend(screen_replies=[])
    assert run(store, backend, screen=False).extracted == 1


def test_failed_check_is_retried_on_a_later_run_and_not_repeated_after(store):
    a = add(store, 1)
    first = FakeBackend(screen_replies=["bad", "bad"])
    report = run(store, first)
    article = store.get_article(a)
    assert report.failed == 1 and "[screening]" in report.summary()
    assert article["processing_status"] == schema.EXTRACTION_FAILED
    assert article["screening"] is None

    # Next day: the check passes, but extraction fails.
    second = FakeBackend(replies=["bad", "bad"], screen_replies=[fake_answer()])
    run(store, second, now=NOW + DAY)
    assert store.get_article(a)["screening"]["keep"] is True

    # The day after: the article is extracted without asking the check again.
    third = FakeBackend(screen_replies=[])
    report = run(store, third, now=NOW + 4 * DAY)
    assert report.extracted == 1 and third.screen_calls == []


def test_check_saying_no_on_a_retry_drops_the_article(store):
    a = add(store, 1)
    run(store, FakeBackend(screen_replies=["bad", "bad"]))
    run(store, FakeBackend(screen_replies=[fake_answer("GENERAL", [])]), now=NOW + DAY)
    assert store.get_article(a)["processing_status"] == schema.FILTERED_OUT


def test_server_down_during_the_check_stops_the_run_and_spends_no_retries(store):
    a, b = add(store, 1), add(store, 2)
    backend = FakeBackend(screen_replies=[fake_answer(), BackendUnavailable("refused")])
    report = run(store, backend)
    assert report.extracted == 1 and report.not_attempted == 1
    assert "LLM backend unavailable" in report.stopped_early
    article = store.get_article(b)
    assert article["processing_status"] == schema.CLEANED and article["retry_count"] == 0
    assert article["screening"] is None
