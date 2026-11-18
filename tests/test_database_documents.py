"""Tests for database/documents.py. No MongoDB, no model, no LLM."""
from datetime import datetime, timedelta, timezone

import pytest

from database import schema
from database.documents import (build_article, build_claims, build_event,
                                canonical_url, make_article_id, source_name, to_utc)

URL = "https://www.thedailystar.net/environment/news/flood-feni-3690000"


def claim(**changes):
    base = {"candidate_id": "Q1", "candidate_kind": "QUANTITY", "impact_type": "FATALITIES",
            "raw_value": "31", "normalized_value": 31.0, "unit": "people",
            "aggregation": "CUMULATIVE", "certainty": "REPORTED",
            "validation_status": "ACCEPTED"}
    base.update(changes)
    return base


# ── URLs ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("variant", [
    URL + "/",
    URL + "#comments",
    URL + "?utm_source=facebook&fbclid=abc",
    "HTTPS://WWW.TheDailyStar.net/environment/news/flood-feni-3690000",
])
def test_url_variants_are_the_same_article(variant):
    assert canonical_url(variant) == URL
    assert make_article_id(variant) == make_article_id(URL)


def test_real_query_parameters_are_kept():
    assert canonical_url("https://example.com/story?id=42&utm_medium=x") == "https://example.com/story?id=42"


def test_not_a_url_is_refused():
    with pytest.raises(ValueError):
        canonical_url("flood in feni")


@pytest.mark.parametrize("url, name", [
    (URL, "The Daily Star"),
    ("https://en.prothomalo.com/bangladesh/abc", "Prothom Alo English"),
    ("https://today.thefinancialexpress.com.bd/x/y", "The Financial Express"),
    ("https://www.unknown-paper.com/a", "unknown-paper.com"),
])
def test_source_name(url, name):
    assert source_name(url) == name


# ── Dates ──────────────────────────────────────────────────────────────────
def test_date_only_is_not_shifted():
    assert to_utc("2024-08-27") == datetime(2024, 8, 27, tzinfo=timezone.utc)


def test_old_timestamp_is_bangladesh_time():
    # 06:34 in Dhaka is 00:34 UTC
    assert to_utc("2026-08-24 06:34:54") == datetime(2026, 8, 24, 0, 34, 54, tzinfo=timezone.utc)


def test_zone_aware_datetime_is_converted():
    ist = timezone(timedelta(hours=5, minutes=30))
    assert to_utc(datetime(2024, 8, 27, 5, 30, tzinfo=ist)) == datetime(2024, 8, 27, tzinfo=timezone.utc)


def test_missing_and_unreadable_dates():
    assert to_utc(None) is None
    assert to_utc("") is None
    with pytest.raises(ValueError):
        to_utc("last Tuesday")


# ── Articles ───────────────────────────────────────────────────────────────
def test_article_without_text_is_discovered():
    doc = build_article(URL)
    assert doc["processing_status"] == schema.DISCOVERED
    assert doc["content_hash"] is None
    assert doc["source"] == "The Daily Star"


def test_article_with_text_is_cleaned():
    doc = build_article(URL, title="Flood", text="Some text.", publish_date="2024-08-27",
                        publish_date_source="meta_tag")
    assert doc["processing_status"] == schema.CLEANED
    assert len(doc["content_hash"]) == 64
    assert doc["publish_date_source"] == "meta_tag"
    assert doc["retry_count"] == 0 and doc["event_id"] is None


def test_date_source_is_dropped_when_there_is_no_date():
    assert build_article(URL, publish_date_source="meta_tag")["publish_date_source"] is None


# ── Claims ─────────────────────────────────────────────────────────────────
ARTICLE = build_article(URL, text="t", publish_date="2024-08-27")


def test_claim_id_and_copied_fields():
    [doc] = build_claims(ARTICLE, "b1", [claim()])
    assert doc["_id"] == f"{ARTICLE['_id']}:b1:Q1"
    assert doc["article_id"] == ARTICLE["_id"]
    assert doc["source"] == "The Daily Star"
    assert doc["article_date"] == ARTICLE["publish_date"]
    assert doc["superseded"] is False
    assert doc["extractor"] == "branch_b"
    assert doc["normalized_value"] == 31.0


def test_two_claims_for_one_candidate_get_different_ids():
    docs = build_claims(ARTICLE, "b1", [claim(), claim(impact_type="MISSING")])
    assert [d["_id"].split(":", 2)[2] for d in docs] == ["Q1", "Q1:2"]


def test_scope_is_event_without_locations_and_local_with():
    place = {"name": "Feni", "level": "district", "district": "Feni"}
    whole, local = build_claims(ARTICLE, "b1", [claim(), claim(candidate_id="Q2", locations=[place])])
    assert whole["scope"] == "event"
    assert local["scope"] == "local"


@pytest.mark.parametrize("changes", [
    {"impact_type": "DEATHS"},
    {"aggregation": "TOTAL"},
    {"certainty": "MAYBE"},
    {"validation_status": "OK"},
    {"candidate_kind": "NUMBER"},
    {"candidate_id": ""},
    {"confidence": 0.93},          # not in the schema: there is no confidence score
])
def test_bad_claims_are_refused(changes):
    with pytest.raises(ValueError):
        build_claims(ARTICLE, "b1", [claim(**changes)])


def test_missing_required_field_is_refused():
    bad = claim()
    del bad["aggregation"]
    with pytest.raises(ValueError):
        build_claims(ARTICLE, "b1", [bad])


def test_one_bad_claim_refuses_the_whole_list():
    with pytest.raises(ValueError):
        build_claims(ARTICLE, "b1", [claim(), claim(candidate_id="Q2", impact_type="DEATHS")])


# ── Events ─────────────────────────────────────────────────────────────────
def test_new_event_is_empty_and_open():
    doc = build_event("FLOOD")
    assert doc["status"] == "open" and doc["article_ids"] == [] and doc["summary"] == {}


def test_unknown_hazard_is_refused():
    with pytest.raises(ValueError):
        build_event("VOLCANO")
