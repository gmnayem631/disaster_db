"""Tests for database/store.py.

They use mongomock, an in-memory imitation of MongoDB. Nothing here connects
to Atlas, reads .env, loads the NER model or calls the LLM.
"""
from datetime import datetime, timedelta, timezone

import pytest

from database import schema
from database.store import Store

NOW = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)
URL_A = "https://www.thedailystar.net/news/flood-a"
URL_B = "https://www.dhakatribune.com/bangladesh/flood-b"
URL_C = "https://www.newagebd.net/article/flood-c"


@pytest.fixture
def db(test_db):
    # mongomock normally; real `disaster_db_test` with RUN_MONGO=1 (see conftest.py)
    return test_db


@pytest.fixture
def store(db):
    s = Store(db)
    s.setup()
    return s


def claim(candidate_id="Q1", **changes):
    base = {"candidate_id": candidate_id, "candidate_kind": "QUANTITY",
            "impact_type": "FATALITIES", "raw_value": "31", "normalized_value": 31.0,
            "unit": "people", "aggregation": "CUMULATIVE", "certainty": "REPORTED",
            "validation_status": "ACCEPTED",
            "evidence": [{"sentence_id": "S2", "text": "The death toll has risen to 31.",
                          "start_char": 40, "end_char": 71}]}
    base.update(changes)
    return base


def entities(*districts):
    return {"disaster_type": [{"value": "flood", "source": "model"}],
            "locations": {"districts": [{"value": d, "source": "model"} for d in districts],
                          "upazilas": [], "unions": []},
            "agencies_involved": [], "raw_entities": []}


def run_info(version="b1", status="complete"):
    return {"version": version, "status": status, "llm_model": "fake",
            "batches": [{"index": 1, "candidate_ids": ["Q1", "Q2"], "status": "ok"}]}


def add_extracted(store, url, *districts, text=None, publish_date="2024-08-27", now=NOW):
    """Helper: an article that has gone through a successful extraction."""
    article_id, _ = store.add_article(url, title="Flood", text=text or f"text of {url}",
                                      publish_date=publish_date, scraped_at=now, now=now)
    store.save_extraction(article_id, entities=entities(*districts), hazard_types=["FLOOD"],
                          extraction=run_info(), claims=[claim("Q1"), claim("Q2", raw_value="58 lakh")],
                          now=now)
    return article_id


# ── Indexes ────────────────────────────────────────────────────────────────
def test_setup_twice_gives_the_same_indexes(store, db):
    before = {c: sorted(db[c].index_information()) for c in schema.INDEXES}
    store.setup()
    after = {c: sorted(db[c].index_information()) for c in schema.INDEXES}
    assert before == after
    assert "url_unique" in after[schema.ARTICLES]


# ── Adding articles ────────────────────────────────────────────────────────
def test_same_url_is_stored_once(store):
    first_id, created = store.add_article(URL_A, text="original text", now=NOW)
    again_id, created_again = store.add_article(URL_A + "/?utm_source=fb", text="different", now=NOW)
    assert created is True and created_again is False
    assert first_id == again_id
    assert store.articles.count_documents({}) == 1
    assert store.get_article(first_id)["text"] == "original text"


def test_same_text_from_two_papers_is_marked_as_a_copy(store):
    first, _ = store.add_article(URL_A, text="wire story", now=NOW)
    second, _ = store.add_article(URL_B, text="wire story", now=NOW + timedelta(hours=1))
    assert store.get_article(first)["duplicate_of"] is None
    assert store.get_article(second)["duplicate_of"] == first


def test_discovered_then_text_arrives(store):
    article_id, _ = store.add_article(URL_A, now=NOW)
    assert store.get_pending("extraction", now=NOW) == []
    assert [a["_id"] for a in store.get_pending("scrape", now=NOW)] == [article_id]

    store.set_text(article_id, title="Flood", text="Body text.", publish_date="2024-08-27",
                   publish_date_source="meta_tag", now=NOW)
    article = store.get_article(article_id)
    assert article["processing_status"] == schema.CLEANED
    assert article["scraped_at"] == NOW
    assert [a["_id"] for a in store.get_pending("extraction", now=NOW)] == [article_id]


def test_filtered_out_is_final(store):
    article_id, _ = store.add_article(URL_A, text="cricket", now=NOW)
    store.mark_filtered_out(article_id, "not flood or fire related", now=NOW)
    assert store.get_pending("extraction", now=NOW + timedelta(days=30)) == []


# ── Failure and retry ──────────────────────────────────────────────────────
def test_failed_article_waits_then_retries_then_dies(store):
    article_id, _ = store.add_article(URL_A, text="text", now=NOW)
    failed_run = run_info(status="partial")
    failed_run["batches"] = [{"index": 2, "candidate_ids": ["Q7", "Q8"], "status": "failed",
                              "error": "Empty response from model"}]

    status = store.mark_failed(article_id, stage="extraction", kind="batch_failed",
                               message="batch 2 failed", extraction=failed_run, now=NOW)
    article = store.get_article(article_id)
    assert status == schema.EXTRACTION_FAILED
    assert article["retry_count"] == 1
    assert article["last_error"]["message"] == "batch 2 failed"
    # the failed batch and its candidates are on record
    assert article["extraction"]["batches"][0]["candidate_ids"] == ["Q7", "Q8"]
    # and no claims were written
    assert store.claims.count_documents({}) == 0

    # not due yet one hour later, due the next day
    assert store.get_pending(now=NOW + timedelta(hours=1)) == []
    assert len(store.get_pending(now=NOW + timedelta(hours=24))) == 1

    second = store.mark_failed(article_id, stage="extraction", kind="k", message="m",
                               now=NOW + timedelta(days=1))
    third = store.mark_failed(article_id, stage="extraction", kind="k", message="m",
                              now=NOW + timedelta(days=4))
    assert second == schema.EXTRACTION_FAILED
    assert third == schema.DEAD
    assert store.get_pending(now=NOW + timedelta(days=365)) == []


def test_scrape_failure_uses_its_own_status(store):
    article_id, _ = store.add_article(URL_A, now=NOW)
    assert store.mark_failed(article_id, stage="scrape", kind="http", message="404",
                             now=NOW) == schema.SCRAPE_FAILED
    assert len(store.get_pending("scrape", now=NOW + timedelta(days=1))) == 1


def test_untried_articles_are_not_touched_when_a_run_stops_early(store):
    """If Ollama is down the service stops and simply does not call the store.
    The remaining articles must still be waiting, with no retries used."""
    ids = [store.add_article(u, text=f"text {u}", now=NOW)[0] for u in (URL_A, URL_B, URL_C)]
    store.save_extraction(ids[0], entities=entities("Feni"), hazard_types=["FLOOD"],
                          extraction=run_info(), claims=[claim()], now=NOW)
    waiting = store.get_pending(now=NOW)
    assert [a["_id"] for a in waiting] == ids[1:]
    assert all(a["retry_count"] == 0 for a in waiting)


# ── Saving an extraction ───────────────────────────────────────────────────
def test_successful_extraction(store):
    article_id = add_extracted(store, URL_A, "Feni", "Cumilla")
    article = store.get_article(article_id)
    claims = store.get_claims(article_id)

    assert article["processing_status"] == schema.EXTRACTED
    assert article["hazard_types"] == ["FLOOD"]
    assert article["extraction"]["version"] == "b1"
    assert article["retry_count"] == 0 and article["last_error"] is None
    assert [c["candidate_id"] for c in claims] == ["Q1", "Q2"]
    assert claims[0]["evidence"][0]["text"] == "The death toll has risen to 31."
    assert claims[0]["article_date"] == datetime(2024, 8, 27, tzinfo=timezone.utc)
    assert store.get_pending(now=NOW) == []


def test_success_after_a_failure_clears_the_error(store):
    article_id, _ = store.add_article(URL_A, text="text", now=NOW)
    store.mark_failed(article_id, stage="extraction", kind="k", message="m", now=NOW)
    store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                          extraction=run_info(), claims=[claim()], now=NOW + timedelta(days=1))
    article = store.get_article(article_id)
    assert article["processing_status"] == schema.EXTRACTED
    assert article["retry_count"] == 0 and article["last_error"] is None


def test_partial_extraction_is_refused_and_writes_nothing(store):
    article_id, _ = store.add_article(URL_A, text="text", now=NOW)
    with pytest.raises(ValueError):
        store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                              extraction=run_info(status="partial"), claims=[claim()], now=NOW)
    assert store.claims.count_documents({}) == 0
    assert store.get_article(article_id)["processing_status"] == schema.CLEANED


def test_one_bad_claim_means_no_claims_are_written(store):
    article_id, _ = store.add_article(URL_A, text="text", now=NOW)
    with pytest.raises(ValueError):
        store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                              extraction=run_info(),
                              claims=[claim("Q1"), claim("Q2", impact_type="DEATHS")], now=NOW)
    assert store.claims.count_documents({}) == 0
    assert store.get_article(article_id)["processing_status"] == schema.CLEANED


def test_article_without_text_cannot_be_extracted(store):
    article_id, _ = store.add_article(URL_A, now=NOW)
    with pytest.raises(ValueError):
        store.save_extraction(article_id, entities=entities(), hazard_types=[],
                              extraction=run_info(), claims=[], now=NOW)


def test_article_with_no_candidates_is_extracted_with_zero_claims(store):
    article_id, _ = store.add_article(URL_A, text="text", now=NOW)
    count = store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                                  extraction=run_info(), claims=[], now=NOW)
    assert count == 0
    assert store.get_article(article_id)["processing_status"] == schema.EXTRACTED


def test_saving_the_same_run_twice_does_not_duplicate(store):
    article_id = add_extracted(store, URL_A, "Feni")
    first_created = store.get_claims(article_id)[0]["created_at"]
    store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                          extraction=run_info(), claims=[claim("Q1")],
                          now=NOW + timedelta(days=1))
    claims = store.get_claims(article_id)
    assert [c["candidate_id"] for c in claims] == ["Q1"]      # Q2 from the first run is gone
    assert claims[0]["created_at"] == first_created
    assert store.claims.count_documents({}) == 1


def test_a_persons_review_survives_a_repeat_run(store):
    article_id = add_extracted(store, URL_A, "Feni")
    claim_id = store.get_claims(article_id)[0]["_id"]
    store.claims.update_one({"_id": claim_id}, {"$set": {"review": {"decision": "correct"}}})
    store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                          extraction=run_info(), claims=[claim("Q1"), claim("Q2")], now=NOW)
    assert store.claims.find_one({"_id": claim_id})["review"] == {"decision": "correct"}


def test_new_version_keeps_old_claims_but_marks_them_replaced(store):
    article_id = add_extracted(store, URL_A, "Feni")
    store.save_extraction(article_id, entities=entities("Feni"), hazard_types=["FLOOD"],
                          extraction=run_info(version="b2"),
                          claims=[claim("Q1", impact_type="MISSING")], now=NOW + timedelta(days=1))

    current = store.get_claims(article_id)
    everything = store.get_claims(article_id, include_superseded=True)
    assert [(c["extraction_version"], c["impact_type"]) for c in current] == [("b2", "MISSING")]
    assert len(everything) == 3
    assert sum(c["superseded"] for c in everything) == 2


# ── Events ─────────────────────────────────────────────────────────────────
def test_attaching_articles_builds_the_event(store):
    a = add_extracted(store, URL_A, "Feni", "Cumilla", publish_date="2024-08-22")
    b = add_extracted(store, URL_B, "Feni", "Noakhali", publish_date="2024-08-27")
    event_id = store.create_event("FLOOD", now=NOW)

    assert store.attach_article(a, event_id, method="rules_v1", score=1.0, now=NOW) is True
    assert store.attach_article(b, event_id, method="rules_v1", score=0.5,
                                features={"district_overlap": 0.5, "day_gap": 5}, now=NOW) is True

    event = store.get_event(event_id)
    assert event["article_ids"] == [a, b]
    assert event["start_date"] == datetime(2024, 8, 22, tzinfo=timezone.utc)
    assert event["end_date"] == datetime(2024, 8, 27, tzinfo=timezone.utc)
    assert event["date_basis"] == "publish_date"
    assert event["source_count"] == 2
    assert {"name": "Feni", "articles": 2} in event["locations"]["districts"]
    assert {"name": "Cumilla", "articles": 1} in event["locations"]["districts"]

    article = store.get_article(b)
    assert article["processing_status"] == schema.MATCHED
    assert article["event_match"]["features"]["day_gap"] == 5
    assert article["event_match"]["decided_by"] == "auto"
    # every claim of both articles now belongs to the event
    assert len(store.get_event_claims(event_id)) == 4
    assert len(store.get_event_claims(event_id, impact_type="FATALITIES")) == 4


def test_event_says_when_a_date_is_only_the_download_time(store):
    a = add_extracted(store, URL_A, "Feni", publish_date=None)
    event_id = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, event_id, method="rules_v1", now=NOW)
    event = store.get_event(event_id)
    assert event["date_basis"] == "scraped_at"
    assert event["start_date"] == NOW


def test_copied_wire_story_is_not_a_second_source(store):
    a = add_extracted(store, URL_A, "Feni", text="same wire story")
    b = add_extracted(store, URL_B, "Feni", text="same wire story", now=NOW + timedelta(hours=1))
    event_id = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, event_id, method="rules_v1", now=NOW)
    store.attach_article(b, event_id, method="rules_v1", now=NOW)
    event = store.get_event(event_id)
    assert len(event["article_ids"]) == 2
    assert event["source_count"] == 1


def test_borderline_match_flags_the_event(store):
    a = add_extracted(store, URL_A, "Feni")
    event_id = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, event_id, method="rules_v1", score=0.3, needs_review=True, now=NOW)
    assert store.get_event(event_id)["needs_review"] is True


def test_automatic_match_does_not_overwrite_a_persons_decision(store):
    a = add_extracted(store, URL_A, "Feni")
    chosen = store.create_event("FLOOD", now=NOW)
    other = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, chosen, method="manual", decided_by="human", now=NOW)

    assert store.attach_article(a, other, method="rules_v1", now=NOW) is False
    assert store.get_article(a)["event_id"] == chosen
    assert store.get_event(other)["article_ids"] == []


def test_a_person_can_move_an_article_to_another_event(store):
    a = add_extracted(store, URL_A, "Feni")
    first = store.create_event("FLOOD", now=NOW)
    second = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, first, method="rules_v1", now=NOW)
    store.attach_article(a, second, method="manual", decided_by="human", now=NOW)

    assert store.get_event(first)["article_ids"] == []
    assert store.get_event(first)["source_count"] == 0
    assert store.get_event(second)["article_ids"] == [a]
    assert store.get_event_claims(first) == []
    assert len(store.get_event_claims(second)) == 2


def test_re_extracting_a_matched_article_keeps_it_in_its_event(store):
    a = add_extracted(store, URL_A, "Feni")
    event_id = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, event_id, method="rules_v1", now=NOW)
    store.save_extraction(a, entities=entities("Feni"), hazard_types=["FLOOD"],
                          extraction=run_info(version="b2"), claims=[claim("Q1")], now=NOW)
    assert store.get_article(a)["processing_status"] == schema.MATCHED
    assert [c["extraction_version"] for c in store.get_event_claims(event_id)] == ["b2"]


def test_unextracted_article_cannot_be_attached(store):
    article_id, _ = store.add_article(URL_A, text="text", now=NOW)
    event_id = store.create_event("FLOOD", now=NOW)
    with pytest.raises(ValueError):
        store.attach_article(article_id, event_id, method="rules_v1", now=NOW)


# ── The old collection ─────────────────────────────────────────────────────
def test_the_old_collection_is_never_changed(store, db):
    legacy = db[schema.LEGACY]
    legacy.insert_one({"_id": 1, "event_id": "old-uuid", "source_url": URL_A,
                       "fatalities": {"value": 31, "source": "model"}})
    before = list(legacy.find())
    indexes_before = legacy.index_information()

    store.setup()
    a = add_extracted(store, URL_A, "Feni")
    event_id = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, event_id, method="rules_v1", now=NOW)
    store.mark_failed(store.add_article(URL_B, text="t", now=NOW)[0], stage="extraction",
                      kind="k", message="m", now=NOW)

    assert list(legacy.find()) == before
    assert legacy.index_information() == indexes_before
