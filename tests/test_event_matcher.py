"""Tests for event_matcher.py: which articles become one event.

Articles are put straight into the `extracted` state with chosen hazards,
places and dates, so every rule can be checked without Qwen or XLM-R.
"""
from datetime import datetime, timedelta, timezone

import pytest

import event_matcher
from database import schema
from database.store import Store, event_title
from event_matcher import match_pending

NOW = datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)
JUNE = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(test_db):
    s = Store(test_db)
    s.setup()
    return s


_counter = {"n": 0}


def add(store, *, hazards=("FLOOD",), districts=("Sylhet",), upazilas=(), date=JUNE,
        origin="daily", screened=True, branch_a_hazards=(), text=None, scraped_at=None):
    """An extracted article with exactly these hazards, places and date."""
    _counter["n"] += 1
    n = _counter["n"]
    article_id, _ = store.add_article(
        f"https://www.thedailystar.net/news/story-{n}", title=f"Story {n}",
        text=text or f"Unique text of story number {n}.", publish_date=date,
        scraped_at=scraped_at, origin=origin, now=NOW)
    if screened:
        store.save_screening(article_id, {"keep": True, "hazards": list(hazards)}, now=NOW)
    entities = {"locations": {"districts": [{"value": d} for d in districts],
                              "upazilas": [{"value": u} for u in upazilas], "unions": []}}
    store.save_extraction(article_id, entities=entities, hazard_types=list(branch_a_hazards),
                          extraction={"status": "complete", "version": "b2"}, claims=[], now=NOW)
    return article_id


def event_of(store, article_id):
    return store.get_event(store.get_article(article_id)["event_id"])


def decision(store, article_id):
    return store.get_article(article_id)["event_match"]["features"]["decision"]


def run(store):
    return match_pending(store, now=NOW)


# ── Settings are complete and consistent ───────────────────────────────────
def test_every_hazard_has_a_rule_and_a_place_in_the_order():
    assert set(event_matcher.HAZARD_RULES) == schema.HAZARD_TYPES
    assert set(event_matcher.PRIMARY_ORDER) == schema.HAZARD_TYPES


def test_every_keyword_hazard_is_a_known_hazard_type():
    from screening import hazard_names
    assert set(hazard_names()) <= schema.HAZARD_TYPES


# ── Spread hazards (floods) ────────────────────────────────────────────────
def test_first_article_starts_an_event_with_a_title(store):
    a = add(store, districts=("Sylhet", "Sunamganj", "Moulvibazar"))
    report = run(store)
    event = event_of(store, a)
    assert report.new_events == 1 and decision(store, a) == "new"
    assert store.get_article(a)["processing_status"] == schema.MATCHED
    assert event["hazard_type"] == "FLOOD" and event["article_ids"] == [a]
    assert event["title"] == "Flood: Moulvibazar, Sunamganj and 1 more districts, Jun 2026"
    assert event["needs_review"] is False


def test_flood_reports_sharing_districts_become_one_event(store):
    a = add(store, districts=("Sylhet", "Sunamganj"))
    b = add(store, districts=("Sunamganj", "Sylhet", "Habiganj"), date=JUNE + 5 * DAY)
    report = run(store)
    assert report.new_events == 1 and report.attached == 1
    assert event_of(store, a)["_id"] == event_of(store, b)["_id"]
    event = event_of(store, a)
    assert event["source_count"] == 1 and len(event["article_ids"]) == 2
    assert event["end_date"] == JUNE + 5 * DAY
    assert decision(store, b) == "clear"


def test_flood_report_after_the_window_starts_a_new_event(store):
    a = add(store)
    b = add(store, date=JUNE + 20 * DAY)
    run(store)
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]


def test_window_counts_from_the_events_latest_article(store):
    # A long flood keeps growing: each report is within 14 days of the previous one.
    ids = [add(store, date=JUNE + d * DAY) for d in (0, 10, 20, 30)]
    run(store)
    assert len({event_of(store, i)["_id"] for i in ids}) == 1


def test_different_hazard_in_the_same_district_is_a_different_event(store):
    a = add(store, hazards=("FLOOD",))
    b = add(store, hazards=("LANDSLIDE",))
    run(store)
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]
    assert event_of(store, b)["hazard_type"] == "LANDSLIDE"


def test_no_shared_district_is_a_different_event(store):
    a = add(store, districts=("Sylhet",))
    b = add(store, districts=("Feni",))
    run(store)
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]


def test_small_district_overlap_joins_but_is_flagged(store):
    add(store, districts=("Sylhet", "Sunamganj", "Habiganj", "Moulvibazar"))
    b = add(store, districts=("Sylhet", "Feni", "Noakhali", "Cumilla"), date=JUNE + DAY)
    report = run(store)
    assert decision(store, b) == "borderline"
    assert event_of(store, b)["needs_review"] is True and report.needs_review == 1


# ── Spot hazards (fires, accidents) ────────────────────────────────────────
def test_fire_reports_from_the_same_upazila_are_one_event(store):
    a = add(store, hazards=("FIRE",), districts=("Gazipur",), upazilas=("Kaliakair",))
    b = add(store, hazards=("FIRE",), districts=("Gazipur",), upazilas=("Kaliakair",),
            date=JUNE + DAY)
    run(store)
    assert event_of(store, a)["_id"] == event_of(store, b)["_id"]


def test_fires_in_different_upazilas_are_different_events(store):
    a = add(store, hazards=("FIRE",), districts=("Gazipur",), upazilas=("Kaliakair",))
    b = add(store, hazards=("FIRE",), districts=("Gazipur",), upazilas=("Sreepur",))
    run(store)
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]
    assert event_of(store, b)["needs_review"] is False


def test_fire_with_unknown_upazila_gets_its_own_flagged_event(store):
    a = add(store, hazards=("FIRE",), districts=("Gazipur",), upazilas=("Kaliakair",))
    b = add(store, hazards=("FIRE",), districts=("Gazipur",))
    run(store)
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]
    assert decision(store, b) == "new_possible_match"
    assert event_of(store, b)["needs_review"] is True
    features = store.get_article(b)["event_match"]["features"]
    assert features["possible_match"]["event_id"] == event_of(store, a)["_id"]


def test_road_accidents_two_days_apart_in_one_upazila_are_different(store):
    a = add(store, hazards=("ROAD_ACCIDENT",), districts=("Dhaka",), upazilas=("Savar",))
    b = add(store, hazards=("ROAD_ACCIDENT",), districts=("Dhaka",), upazilas=("Savar",),
            date=JUNE + 3 * DAY)
    run(store)
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]


# ── Ambiguity, missing information, copies ─────────────────────────────────
def test_article_fitting_two_events_is_attached_once_and_both_are_flagged(store):
    # Two events that should probably have been one (built before the bridge article).
    a = add(store, districts=("Sylhet",))
    b = add(store, districts=("Feni",))
    run(store)
    c = add(store, districts=("Sylhet", "Feni"), date=JUNE + DAY)
    run(store)
    assert decision(store, c) == "clear_but_ambiguous"
    assert event_of(store, a)["needs_review"] and event_of(store, b)["needs_review"]
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]       # never merged


def test_corpus_article_without_a_date_is_not_matched_automatically(store):
    a = add(store, origin="corpus", screened=False, branch_a_hazards=("FLOOD",), date=None)
    b = add(store, origin="corpus", screened=False, branch_a_hazards=("FLOOD",))
    run(store)
    assert decision(store, a) == "cannot_match"
    assert store.get_article(a)["event_match"]["features"]["missing"] == ["date"]
    assert event_of(store, a)["needs_review"] is True
    assert event_of(store, a)["_id"] != event_of(store, b)["_id"]


def test_daily_article_without_publish_date_uses_the_download_time(store):
    a = add(store)
    b = add(store, date=None, scraped_at=JUNE + DAY)
    run(store)
    assert event_of(store, a)["_id"] == event_of(store, b)["_id"]


def test_article_without_a_district_is_flagged(store):
    a = add(store, districts=())
    run(store)
    assert decision(store, a) == "cannot_match" and event_of(store, a)["needs_review"]


def test_syndicated_copy_joins_the_original_event(store):
    a = add(store, text="Same agency text about the flood.")
    run(store)
    b = add(store, districts=("Feni",), text="Same agency text about the flood.")
    run(store)
    assert store.get_article(b)["duplicate_of"] == a
    assert decision(store, b) == "same_text"
    assert event_of(store, b)["source_count"] == 1                      # a copy is not a second source


# ── Which hazard ───────────────────────────────────────────────────────────
def test_hazard_check_is_preferred_over_branch_a(store):
    a = add(store, hazards=("ROAD_ACCIDENT",), branch_a_hazards=("FIRE",))
    run(store)
    assert event_of(store, a)["hazard_type"] == "ROAD_ACCIDENT"


def test_corpus_article_uses_branch_a_hazards(store):
    a = add(store, origin="corpus", screened=False, branch_a_hazards=("FIRE",))
    run(store)
    assert event_of(store, a)["hazard_type"] == "FIRE"


def test_cyclone_article_mentioning_floods_starts_a_cyclone_event(store):
    a = add(store, hazards=("FLOOD", "CYCLONE"))
    run(store)
    assert event_of(store, a)["hazard_type"] == "CYCLONE"


def test_multi_hazard_article_can_join_an_event_of_either_hazard(store):
    a = add(store, hazards=("FLOOD",))
    b = add(store, hazards=("CYCLONE", "FLOOD"), date=JUNE + DAY)
    run(store)
    assert event_of(store, a)["_id"] == event_of(store, b)["_id"]


def test_unknown_hazard_becomes_other(store):
    a = add(store, hazards=())
    run(store)
    assert event_of(store, a)["hazard_type"] == "OTHER"
    assert store.get_article(a)["event_match"]["features"]["missing"] == ["hazard"]


# ── Runs ───────────────────────────────────────────────────────────────────
def test_articles_are_matched_oldest_date_first(store):
    late = add(store, date=JUNE + 10 * DAY)
    early = add(store, date=JUNE)
    run(store)
    assert decision(store, early) == "new" and decision(store, late) == "clear"


def test_running_again_changes_nothing(store):
    a = add(store)
    run(store)
    before = event_of(store, a)
    report = run(store)
    assert report.picked_up == 0 and event_of(store, a) == before


def test_a_persons_decision_is_kept(store):
    a = add(store)
    run(store)
    other = store.create_event("FLOOD", now=NOW)
    store.attach_article(a, other, method="manual", decided_by="human", now=NOW)
    assert event_matcher.match_article(store, store.get_article(a), now=NOW)
    assert store.get_article(a)["event_id"] == other


def test_one_bad_article_does_not_stop_the_others(store, monkeypatch):
    a, b = add(store), add(store, districts=("Feni",))
    real = event_matcher.match_article

    def flaky(store_, article, now=None):
        if article["_id"] == a:
            raise RuntimeError("boom")
        return real(store_, article, now=now)

    monkeypatch.setattr(event_matcher, "match_article", flaky)
    report = run(store)
    assert report.failed == 1 and report.new_events == 1 and "boom" in report.summary()
    assert store.get_article(a)["processing_status"] == schema.EXTRACTED   # tried again next run


def test_person_typed_title_is_kept(store):
    a = add(store)
    run(store)
    event = event_of(store, a)
    store.events.update_one({"_id": event["_id"]},
                            {"$set": {"title": "Sylhet flash flood 2026", "title_locked": True}})
    add(store, date=JUNE + DAY)
    run(store)
    assert event_of(store, a)["title"] == "Sylhet flash flood 2026"


def test_event_title_shapes():
    d = lambda name, n=1: {"name": name, "articles": n}
    assert event_title("ROAD_ACCIDENT", [d("Dhaka")], JUNE) == "Road accident: Dhaka, Jun 2026"
    assert event_title("FLOOD", [d("Feni"), d("Sylhet", 3)], None) == "Flood: Sylhet and Feni"
    assert event_title("FIRE", [], JUNE) == "Fire: district unknown, Jun 2026"


def test_the_old_collection_is_not_touched(store):
    store.db[schema.LEGACY].insert_one({"_id": "old", "x": 1})
    add(store)
    run(store)
    assert list(store.db[schema.LEGACY].find()) == [{"_id": "old", "x": 1}]


# ── District spellings ─────────────────────────────────────────────────────
@pytest.mark.parametrize("written, expected", [
    ("Chittagong", "Chattogram"), ("Chattogram", "Chattogram"), ("COMILLA", "Cumilla"),
    ("Cox's Bazar", "Cox's Bazar"), ("Coxsbazar", "Cox's Bazar"), ("cox’s bazar", "Cox's Bazar"),
    ("Sirajgonj", "Sirajganj"), ("Kishorgonj", "Kishoreganj"), ("Moulvi Bazar", "Moulvibazar"),
    ("Jessore", "Jashore"), ("Bogra", "Bogura"), ("Narayanganj", "Narayanganj"),
    ("Atlantis", "Atlantis"),
])
def test_district_spellings_become_one_name(written, expected):
    from database.documents import canonical_district
    assert canonical_district(written) == expected


def test_all_64_districts_are_known():
    from database.documents import DISTRICTS
    assert len(DISTRICTS) == len(set(DISTRICTS)) == 64


def test_old_and_new_spellings_match_the_same_event(store):
    a = add(store, hazards=("CYCLONE",), districts=("Chittagong", "Cox's Bazar"))
    b = add(store, hazards=("CYCLONE",), districts=("Chattogram",), date=JUNE + DAY)
    run(store)
    event = event_of(store, a)
    assert event["_id"] == event_of(store, b)["_id"]
    assert [d["name"] for d in event["locations"]["districts"]] == ["Chattogram", "Cox's Bazar"]
