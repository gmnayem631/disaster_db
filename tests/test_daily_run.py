"""Tests for daily_run.py: both stages together, logging, exit codes and the lock.

Ingestion is replaced by a stand-in that adds articles (no internet), the LLM
by FakeBackend and Branch A by fake_branch_a. No Atlas, Ollama or model needed.
"""
import os
import time
from datetime import datetime, timezone

import pytest

import daily_run
import ingestion
from database import schema
from database.store import Store
from extraction_service import BackendUnavailable, FakeBackend
from screening import fake_answer

NOW = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)


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


class Lines:
    """Collects log lines instead of printing them."""
    def __init__(self):
        self.lines = []

    def __call__(self, message=""):
        self.lines.extend(str(message).splitlines())

    def text(self):
        return "\n".join(self.lines)


def fake_ingestion(articles=2, crash=False):
    def run_ingestion(store, **options):
        if crash:
            raise RuntimeError("sitemap parser exploded")
        for n in range(articles):
            store.add_article(f"https://www.thedailystar.net/news/flood-{n}", title=f"Flood {n}",
                              text=f"The death toll has risen to {30 + n} in Feni district.",
                              publish_date="2026-11-16", now=options.get("now"))
        options["progress"]("KEPT something")
        report = ingestion.IngestReport()
        report.cleaned = articles
        return report
    return run_ingestion


def go(store, backend=None, **options):
    log = Lines()
    options.setdefault("now", NOW)
    report = daily_run.daily_run(store, backend=backend or FakeBackend(), log=log,
                                 sources=[], hazard_keywords={"FLOOD": []},
                                 branch_a=fake_branch_a, **options)
    return report, log


def test_new_articles_are_collected_checked_and_extracted(store, monkeypatch):
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(articles=2))
    report, log = go(store)
    assert report.exit_code == 0
    assert report.extraction.extracted == 2
    statuses = {a["processing_status"] for a in store.articles.find()}
    assert statuses == {schema.MATCHED}                   # stage 3 put them into events
    assert store.events.count_documents({}) == 1          # both report the Feni flood
    text = log.text()
    assert "Ingestion run" in text and "Extraction run" in text
    assert "KEPT something" in text                       # ingestion progress reaches the log
    assert "[1/2] EXTRACTED Flood 0" in text              # one line per extracted article
    assert "Event matching: 2 articles, 1 joined an existing event, 1 new events" in text
    assert report.summaries.rebuilt == 1 and "Event summaries: 1 rebuilt" in text
    assert "waiting for extraction now: 0" in text


def test_hazard_check_drops_are_logged(store, monkeypatch):
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(articles=1))
    backend = FakeBackend(screen_replies=[fake_answer("NOT_DISASTER", [], reason="protest")])
    report, log = go(store, backend)
    assert report.exit_code == 0 and report.extraction.screened_out == 1
    assert "SCREENED_OUT Flood 0: hazard check: NOT_DISASTER: protest" in log.text()


def test_ingestion_crash_is_logged_and_waiting_articles_are_still_extracted(store, monkeypatch):
    store.add_article("https://www.thedailystar.net/news/old", title="Old",
                      text="The death toll has risen to 5 in Feni district.", now=NOW)
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(crash=True))
    report, log = go(store)
    assert report.exit_code == 1 and report.problems == ["ingestion crashed"]
    assert "sitemap parser exploded" in log.text()
    assert report.extraction.extracted == 1


def test_llm_down_gives_exit_code_1_and_loses_nothing(store, monkeypatch):
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(articles=2))
    backend = FakeBackend(screen_replies=[BackendUnavailable("connection refused")])
    report, log = go(store, backend)
    assert report.exit_code == 1
    assert "LLM backend unavailable" in log.text()
    for article in store.articles.find():
        assert article["processing_status"] == schema.CLEANED and article["retry_count"] == 0
    assert "waiting for extraction now: 2" in log.text()


def test_time_limit_leaves_the_rest_waiting(store, monkeypatch):
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(articles=2))
    report, log = go(store, extract_minutes=0)
    assert report.extraction.not_attempted == 2 and report.extraction.extracted == 0
    assert "time limit reached" in log.text()
    assert report.exit_code == 1


def test_stages_can_be_skipped(store, monkeypatch):
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(crash=True))
    report, log = go(store, skip_ingestion=True, skip_extraction=True, skip_matching=True)
    assert report.exit_code == 0 and report.ingestion is None and report.extraction is None
    assert report.matching is None and "Event matching: skipped" in log.text()
    assert "Ingestion: skipped" in log.text() and "Extraction: skipped" in log.text()


def test_the_old_collection_is_not_touched(store, monkeypatch):
    store.db[schema.LEGACY].insert_one({"_id": "old", "x": 1})
    monkeypatch.setattr(ingestion, "run_ingestion", fake_ingestion(articles=1))
    go(store)
    assert list(store.db[schema.LEGACY].find()) == [{"_id": "old", "x": 1}]


# ── Lock and log file ──────────────────────────────────────────────────────
def test_second_lock_is_refused_until_released(tmp_path):
    lock = str(tmp_path / "daily_run.lock")
    daily_run.take_lock(lock)
    with pytest.raises(daily_run.AlreadyRunning):
        daily_run.take_lock(lock)
    daily_run.release_lock(lock)
    daily_run.take_lock(lock)                       # free again
    daily_run.release_lock(lock)


def test_lock_left_by_a_crashed_run_is_replaced(tmp_path):
    lock = str(tmp_path / "daily_run.lock")
    daily_run.take_lock(lock)
    later = time.time() + (daily_run.STALE_LOCK_HOURS + 1) * 3600
    daily_run.take_lock(lock, now=later)            # no error
    daily_run.release_lock(lock)


def test_main_does_nothing_while_another_run_holds_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_run, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(daily_run, "LOCK_FILE", str(tmp_path / "daily_run.lock"))
    daily_run.take_lock()
    assert daily_run.main(["--db", "disaster_db_test"]) == 2
    log_files = [f for f in os.listdir(tmp_path) if f.endswith(".log")]
    assert len(log_files) == 1
    with open(tmp_path / log_files[0], encoding="utf-8") as f:
        assert "Not started" in f.read()
    assert os.path.exists(tmp_path / "daily_run.lock")   # the other run's lock is left alone


def test_log_writes_timestamped_lines_to_the_file(tmp_path, capsys):
    path = tmp_path / "x.log"
    log = daily_run.Log(str(path))
    log("one\ntwo")
    log.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].endswith("  one") and lines[0][2] == ":"
    assert "two" in capsys.readouterr().out


def test_log_survives_a_console_that_cannot_show_a_character(tmp_path, monkeypatch):
    def narrow_print(text, **kwargs):
        text.encode("cp1252")                         # raises for Bengali, like a Windows console
    monkeypatch.setattr("builtins.print", narrow_print)
    path = tmp_path / "x.log"
    log = daily_run.Log(str(path))
    log("বন্যা flood")
    log.close()
    assert "বন্যা flood" in path.read_text(encoding="utf-8")
