"""Tests for extraction_service.run_pending(): extraction + database together.

mongomock stands in for MongoDB, FakeBackend for Qwen and fake_branch_a for
the XLM-R model. Nothing here needs Atlas, Ollama or the model files.
"""
from datetime import datetime, timedelta, timezone

import pytest

import extraction_service
from database import schema
from database.store import Store
from extraction_service import (BackendError, BackendUnavailable, FakeBackend,
                                auto_reply, run_pending)

NOW = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)


@pytest.fixture
def store(test_db):
    # mongomock normally; real `disaster_db_test` with RUN_MONGO=1 (see conftest.py)
    s = Store(test_db)
    s.setup()
    return s


def fake_branch_a(text, source_url, publish_date=None):
    return {"source_url": source_url, "publish_date": publish_date,
            "disaster_type": [{"value": "flood", "source": "model"}],
            "locations": {"districts": [{"value": "Feni", "source": "model"}],
                          "upazilas": [], "unions": []},
            "agencies_involved": [], "raw_entities": []}


def add(store, n, text=None, minutes=0):
    """A cleaned article waiting for extraction. One sentence -> one batch."""
    article_id, _ = store.add_article(
        f"https://www.thedailystar.net/news/flood-{n}", title=f"Flood report {chr(65 + n)}",
        text=text or f"The death toll has risen to {30 + n} in Feni district.",
        publish_date="2024-08-27", now=NOW + timedelta(minutes=minutes))
    return article_id


def run(store, backend, **options):
    options.setdefault("now", NOW)
    return run_pending(store, backend, branch_a=fake_branch_a, **options)


class DownAfter(FakeBackend):
    """Answers normally for the first `good` requests, then the server is gone."""

    def __init__(self, good):
        super().__init__()
        self.good = good

    def complete(self, system_prompt, user_prompt, json_schema):
        if len(self.calls) >= self.good:
            raise BackendUnavailable("connection refused")
        return super().complete(system_prompt, user_prompt, json_schema)


# ── Success ────────────────────────────────────────────────────────────────
def test_waiting_article_is_extracted_and_saved(store):
    article_id = add(store, 1)
    report = run(store, FakeBackend())

    article = store.get_article(article_id)
    claims = store.get_claims(article_id)
    assert (report.picked_up, report.extracted, report.failed, report.dead) == (1, 1, 0, 0)
    assert report.claims_saved == len(claims) == 1
    assert article["processing_status"] == schema.EXTRACTED
    assert article["hazard_types"] == ["FLOOD"]
    assert article["entities"]["locations"]["districts"][0]["value"] == "Feni"
    assert article["extraction"]["status"] == "complete"
    assert article["extraction"]["llm"]["model"] == "fake-model"
    assert article["extraction"]["batches"][0]["status"] == "ok"
    assert claims[0]["raw_value"] == "31"
    assert "death toll has risen to 31" in claims[0]["evidence"][0]["text"]


def test_running_again_does_nothing_and_duplicates_nothing(store):
    article_id = add(store, 1)
    run(store, FakeBackend())
    backend = FakeBackend()
    second = run(store, backend)

    assert second.picked_up == 0
    assert backend.calls == []
    assert len(store.get_claims(article_id)) == 1
    assert "Nothing was waiting" in second.summary()


def test_limit_is_respected_and_oldest_go_first(store):
    ids = [add(store, n, minutes=n) for n in range(5)]
    report = run(store, FakeBackend(), limit=2)
    assert [a["article_id"] for a in report.articles] == ids[:2]
    assert [store.get_article(i)["processing_status"] for i in ids] == (
        [schema.EXTRACTED] * 2 + [schema.CLEANED] * 3)


# ── Failure and retry ──────────────────────────────────────────────────────
def test_failed_article_saves_no_claims_and_keeps_the_batch_report(store):
    article_id = add(store, 1)
    report = run(store, FakeBackend([BackendError("timeout"), BackendError("timeout")]))

    article = store.get_article(article_id)
    assert (report.extracted, report.failed) == (0, 1)
    assert store.claims.count_documents({}) == 0
    assert article["processing_status"] == schema.EXTRACTION_FAILED
    assert article["retry_count"] == 1
    assert article["last_error"]["kind"] == "batches_not_ok"
    assert article["next_retry_at"] > NOW
    # which batch, which candidates, and why: all on record
    batch = article["extraction"]["batches"][0]
    assert batch["status"] == "failed" and batch["candidate_ids"] == ["Q1"]
    assert "timeout" in batch["error"]["message"]
    assert "FAILED" in report.summary() and article_id in report.summary()


def test_skipped_candidate_fails_the_article_and_is_named(store):
    article_id = add(store, 1, text="Floods killed 31 people and injured 12 others in Feni.")
    skip = lambda prompt: auto_reply(prompt, skip=["Q2"])
    run(store, FakeBackend([skip, skip]))

    article = store.get_article(article_id)
    assert article["processing_status"] == schema.EXTRACTION_FAILED
    assert article["extraction"]["unanswered"] == ["Q2"]
    assert store.claims.count_documents({}) == 0      # not even the claim that was answered


def test_failed_article_is_not_retried_too_soon(store):
    add(store, 1)
    run(store, FakeBackend([BackendError("x"), BackendError("x")]))
    backend = FakeBackend()
    report = run(store, backend, now=NOW + timedelta(hours=1))
    assert report.picked_up == 0 and backend.calls == []


def test_failed_article_succeeds_on_the_next_days_run(store):
    article_id = add(store, 1)
    run(store, FakeBackend([BackendError("x"), BackendError("x")]))
    report = run(store, FakeBackend(), now=NOW + DAY)

    article = store.get_article(article_id)
    assert report.extracted == 1
    assert article["processing_status"] == schema.EXTRACTED
    assert article["retry_count"] == 0 and article["last_error"] is None
    assert len(store.get_claims(article_id)) == 1


def test_third_failure_makes_the_article_dead_and_reports_it(store):
    article_id = add(store, 1)
    always_fail = lambda: FakeBackend([BackendError("x"), BackendError("x")])
    run(store, always_fail())
    run(store, always_fail(), now=NOW + DAY)
    report = run(store, always_fail(), now=NOW + 5 * DAY)

    assert (report.failed, report.dead) == (0, 1)
    assert store.get_article(article_id)["processing_status"] == schema.DEAD
    assert "DEAD" in report.summary()
    assert run(store, FakeBackend(), now=NOW + 60 * DAY).picked_up == 0


def test_one_bad_article_does_not_stop_the_others(store):
    ids = [add(store, n, minutes=n) for n in range(3)]
    backend = FakeBackend([auto_reply, BackendError("x"), BackendError("x"), auto_reply])
    report = run(store, backend)

    assert (report.extracted, report.failed) == (2, 1)
    assert [store.get_article(i)["processing_status"] for i in ids] == [
        schema.EXTRACTED, schema.EXTRACTION_FAILED, schema.EXTRACTED]


def test_article_with_no_numbers_is_extracted_with_zero_claims(store):
    article_id = add(store, 1, text="The river stayed calm and the weather was quiet.")
    backend = FakeBackend()
    report = run(store, backend)
    assert report.extracted == 1 and report.claims_saved == 0
    assert backend.calls == []
    assert store.get_article(article_id)["processing_status"] == schema.EXTRACTED


def test_result_the_store_refuses_is_recorded_as_a_failure(store):
    article_id = add(store, 1)

    def refuse(*args, **kwargs):
        raise ValueError("claim impact_type='DEATHS' is not allowed")
    store.save_extraction = refuse

    report = run(store, FakeBackend())
    article = store.get_article(article_id)
    assert report.failed == 1
    assert article["processing_status"] == schema.EXTRACTION_FAILED
    assert article["last_error"]["kind"] == "save_refused"
    assert store.claims.count_documents({}) == 0


# ── Problems with the machine, not the article ─────────────────────────────
def test_llm_server_going_down_stops_the_run_and_spends_no_retries(store):
    ids = [add(store, n, minutes=n) for n in range(5)]
    report = run(store, DownAfter(good=1))

    assert report.extracted == 1
    assert report.not_attempted == 4
    assert "unavailable" in report.stopped_early
    assert "STOPPED EARLY" in report.summary()
    assert store.get_article(ids[0])["processing_status"] == schema.EXTRACTED
    for article_id in ids[1:]:
        article = store.get_article(article_id)
        assert article["processing_status"] == schema.CLEANED
        assert article["retry_count"] == 0 and article["last_error"] is None

    # the next run, with the server back, finishes the rest
    assert run(store, FakeBackend()).extracted == 4


def test_missing_branch_a_model_stops_before_touching_anything(store, monkeypatch):
    article_id = add(store, 1)

    def cannot_load():
        raise OSError("models/xlmroberta_v3 not found")
    monkeypatch.setattr(extraction_service, "_default_branch_a", cannot_load)

    backend = FakeBackend()
    report = run_pending(store, backend, now=NOW)        # no branch_a given -> tries the real one

    assert "Branch A model could not be loaded" in report.stopped_early
    assert report.not_attempted == 1 and backend.calls == []
    article = store.get_article(article_id)
    assert article["processing_status"] == schema.CLEANED and article["retry_count"] == 0


def test_branch_a_crash_on_one_article_is_that_articles_failure(store):
    bad_id = add(store, 0, text="This text breaks the model. It mentions 5 people.")
    good_id = add(store, 1, minutes=1)

    def fragile(text, source_url, publish_date=None):
        if "breaks the model" in text:
            raise RuntimeError("tokenizer error")
        return fake_branch_a(text, source_url, publish_date)

    report = run_pending(store, FakeBackend(), branch_a=fragile, now=NOW)
    assert (report.extracted, report.failed) == (1, 1)
    assert store.get_article(bad_id)["last_error"]["message"] == "[branch_a] tokenizer error"
    assert store.get_article(good_id)["processing_status"] == schema.EXTRACTED


# ── The old collection ─────────────────────────────────────────────────────
def test_the_old_collection_is_not_touched(store):
    legacy = store.db[schema.LEGACY]
    legacy.insert_one({"_id": 1, "source_url": "https://www.thedailystar.net/news/flood-1"})
    before = list(legacy.find())
    add(store, 1)
    add(store, 2, minutes=1)
    run(store, FakeBackend([auto_reply, BackendError("x"), BackendError("x")]))
    assert list(legacy.find()) == before
