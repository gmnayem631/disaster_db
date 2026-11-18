"""Tests for event_summary.py: headline numbers of an event, built from its claims.

Most tests call summarize() directly with hand-made claims (no database).
The last ones go through the Store, the matcher and refresh_summaries().
"""
from datetime import datetime, timedelta, timezone

import pytest

import event_summary
from database.store import Store
from event_matcher import match_pending
from event_summary import refresh_summaries, summarize

NOW = datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)
JUNE = datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)

_n = {"i": 0}


def claim(value, impact="FATALITIES", article="a1", *, aggregation="CUMULATIVE", unit="people",
          places=(), status="ACCEPTED", certainty="REPORTED", flags=(), modifier=None,
          raw=None, low=None, high=None, qualitative=False):
    _n["i"] += 1
    return {"_id": f"{article}:b2:Q{_n['i']}", "article_id": article, "impact_type": impact,
            "raw_value": raw or str(value), "normalized_value": value, "unit": unit,
            "aggregation": aggregation, "certainty": certainty, "validation_status": status,
            "validation_flags": list(flags), "modifier": modifier, "is_qualitative": qualitative,
            "lower_bound": low, "upper_bound": high,
            "locations": [{"name": p, "level": "district", "district": p} for p in places]}


def articles(*specs):
    """articles(("a1", JUNE, "The Daily Star"), ...) -> the mapping summarize() expects."""
    return {aid: {"date": date, "source": source} for aid, date, source in specs}


ONE = articles(("a1", JUNE, "The Daily Star"))


def entries(summary, impact="FATALITIES"):
    return summary["impacts"].get(impact, [])


# ── Headline: the latest figure, never a sum ───────────────────────────────
def test_latest_figure_is_the_headline_and_nothing_is_added():
    arts = articles(("a1", JUNE, "Daily Star"), ("a2", JUNE + DAY, "Prothom Alo"))
    summary, conflicts = summarize([claim(20, article="a1"), claim(31, article="a2")], arts)
    [entry] = entries(summary)
    assert entry["headline"] == 31 and entry["display"] == "31"     # not 51
    assert entry["as_of"] == JUNE + DAY and entry["sources"] == 1
    assert len(entry["earlier_claim_ids"]) == 1 and conflicts == []


def test_same_day_agreement_counts_sources():
    arts = articles(("a1", JUNE, "Daily Star"), ("a2", JUNE, "Prothom Alo"))
    summary, conflicts = summarize([claim(31, article="a1"), claim(31, article="a2")], arts)
    [entry] = entries(summary)
    assert entry["headline"] == 31 and entry["sources"] == 2 and conflicts == []


def test_same_day_disagreement_becomes_a_range_and_a_conflict():
    arts = articles(("a1", JUNE, "Daily Star"), ("a2", JUNE, "Prothom Alo"))
    summary, conflicts = summarize([claim(27, article="a1"), claim(31, article="a2")], arts)
    [entry] = entries(summary)
    assert entry["headline"] is None and (entry["low"], entry["high"]) == (27, 31)
    assert entry["display"] == "27 to 31"
    assert conflicts[0]["kind"] == "same_day_disagreement" and conflicts[0]["values"] == [27, 31]


def test_small_difference_is_not_a_conflict_except_for_deaths_and_missing():
    arts = articles(("a1", JUNE, "Daily Star"), ("a2", JUNE, "Prothom Alo"))
    affected = [claim(100000, "AFFECTED_PEOPLE", "a1"), claim(105000, "AFFECTED_PEOPLE", "a2")]
    assert summarize(affected, arts)[1] == []                       # 5% apart
    missing = [claim(10, "MISSING", "a1"), claim(11, "MISSING", "a2")]
    assert summarize(missing, arts)[1][0]["kind"] == "same_day_disagreement"


def test_a_figure_that_goes_down_is_flagged():
    arts = articles(("a1", JUNE, "Daily Star"), ("a2", JUNE + DAY, "Prothom Alo"))
    summary, conflicts = summarize([claim(40, article="a1"), claim(31, article="a2")], arts)
    assert entries(summary)[0]["headline"] == 31
    assert conflicts[0]["kind"] == "decreasing" and conflicts[0]["values"] == [31, 40]


# ── Groups are never mixed ─────────────────────────────────────────────────
def test_places_units_and_types_are_kept_apart():
    claims = [claim(31), claim(17, places=["Feni"]), claim(5, places=["Comilla"]),
              claim(300, "RELIEF", unit="tonnes"), claim(4000, "RELIEF", unit="packets"),
              claim(500, "DISPLACED", unit="families")]
    summary, conflicts = summarize(claims, ONE)
    deaths = {(e["scope"], e["place"]): e["headline"] for e in entries(summary)}
    assert deaths == {("event", None): 31, ("local", "Feni"): 17, ("local", "Cumilla"): 5}
    assert {e["unit"] for e in entries(summary, "RELIEF")} == {"tonnes", "packets"}
    assert entries(summary, "DISPLACED")[0]["unit"] == "families"   # never turned into people
    assert conflicts == []


def test_a_figure_for_two_districts_together_stays_together():
    summary, _ = summarize([claim(12, places=["Feni", "Noakhali"])], ONE)
    assert entries(summary)[0]["place"] == "Feni + Noakhali"


def test_event_wide_figures_come_before_local_ones():
    summary, _ = summarize([claim(17, places=["Feni"]), claim(31)], ONE)
    assert [e["scope"] for e in entries(summary)] == ["event", "local"]


# ── What does not give a headline ──────────────────────────────────────────
@pytest.mark.parametrize("kwargs, reason", [
    ({"status": "REVIEW_REQUIRED"}, "unverified"),
    ({"aggregation": "UNKNOWN"}, "unknown_aggregation"),
    ({"certainty": "PREDICTED"}, "uncertain"),
    ({"certainty": "POSSIBLE"}, "uncertain"),
    ({"flags": ["HISTORICAL_REFERENCE"]}, "historical_reference"),
    ({"flags": ["FORECAST_DETECTED"]}, "forecast_detected"),
    ({"flags": ["NEGATION_DETECTED"]}, "negation_detected"),
])
def test_doubtful_claims_are_listed_but_not_headlines(kwargs, reason):
    summary, _ = summarize([claim(31, **kwargs)], ONE)
    assert entries(summary) == []
    assert summary["unverified"][0]["reason"] == reason
    assert summary["not_counted"] == {reason: 1}


def test_rejected_and_other_claims_are_ignored():
    summary, _ = summarize([claim(5, status="REJECTED"), claim(2026, "OTHER")], ONE)
    assert summary["impacts"] == {} and summary["unverified"] == [] and summary["not_counted"] == {}


def test_incremental_claims_go_to_the_timeline_only():
    arts = articles(("a1", JUNE, "Daily Star"), ("a2", JUNE + DAY, "Prothom Alo"))
    claims = [claim(31, article="a1"), claim(3, article="a2", aggregation="INCREMENTAL", raw="three more")]
    summary, _ = summarize(claims, arts)
    assert entries(summary)[0]["headline"] == 31                    # not 34
    assert summary["timeline"][0]["raw_value"] == "three more"


def test_qualitative_claims_are_kept_as_text():
    summary, _ = summarize([claim(None, "HOUSING_DAMAGE", raw="hundreds", unit="houses",
                                  qualitative=True)], ONE)
    assert summary["impacts"] == {} and summary["qualitative"][0]["raw_value"] == "hundreds"


def test_range_keeps_both_bounds_and_modifier_is_shown():
    summary, _ = summarize([claim(None, "AFFECTED_PEOPLE", aggregation="RANGE", low=5000, high=7000),
                            claim(4000, "DISPLACED", modifier="nearly")], ONE)
    affected = entries(summary, "AFFECTED_PEOPLE")[0]
    assert (affected["low"], affected["high"], affected["display"]) == (5000, 7000, "5,000 to 7,000")
    assert entries(summary, "DISPLACED")[0]["display"] == "nearly 4,000"


def test_copies_of_another_article_are_not_counted():
    summary, _ = summarize([claim(31, article="copy")], ONE)       # "copy" is not in the mapping
    assert summary["impacts"] == {}


def test_articles_without_a_date_still_give_a_figure():
    summary, _ = summarize([claim(9)], articles(("a1", None, "Daily Star")))
    assert entries(summary)[0]["headline"] == 9 and entries(summary)[0]["as_of"] is None


def test_summary_carries_the_unverified_label():
    assert summarize([], ONE)[0]["label"] == event_summary.LABEL


# ── Through the database ───────────────────────────────────────────────────
@pytest.fixture
def store(test_db):
    s = Store(test_db)
    s.setup()
    return s


def stored_claim(value, impact="FATALITIES", **kwargs):
    c = claim(value, impact, **kwargs)
    c.pop("_id"), c.pop("article_id")
    c["candidate_id"] = f"Q{_n['i']}"
    c["candidate_kind"] = "QUANTITY"
    return c


def add(store, n, claims, *, date=JUNE, text=None, version="b2"):
    article_id, _ = store.add_article(f"https://www.thedailystar.net/news/flood-{n}", title=f"Flood {n}",
                                      text=text or f"Story number {n}.", publish_date=date, now=NOW)
    store.save_screening(article_id, {"keep": True, "hazards": ["FLOOD"]}, now=NOW)
    store.save_extraction(article_id, entities={"locations": {"districts": [{"value": "Feni"}],
                                                              "upazilas": [], "unions": []}},
                          hazard_types=["FLOOD"], extraction={"status": "complete", "version": version},
                          claims=claims, now=NOW)
    return article_id


def test_matched_event_gets_its_summary(store):
    add(store, 1, [stored_claim(20)])
    add(store, 2, [stored_claim(31)], date=JUNE + DAY)
    match_pending(store, now=NOW)
    report = refresh_summaries(store, now=NOW)
    [event] = list(store.events.find())
    assert report.rebuilt == 1 and report.failed == 0
    assert event["summary"]["impacts"]["FATALITIES"][0]["headline"] == 31
    assert event["reconciliation_version"] == event_summary.SUMMARY_VERSION
    assert refresh_summaries(store, now=NOW).rebuilt == 0           # nothing changed since


def test_a_new_article_or_re_extraction_makes_the_summary_stale(store):
    a1 = add(store, 1, [stored_claim(20)])
    match_pending(store, now=NOW)
    refresh_summaries(store, now=NOW)
    later = NOW + DAY
    store.save_extraction(a1, entities={"locations": {"districts": [{"value": "Feni"}],
                                                      "upazilas": [], "unions": []}},
                          hazard_types=["FLOOD"], extraction={"status": "complete", "version": "b3"},
                          claims=[stored_claim(25)], now=later)
    assert refresh_summaries(store, now=later).rebuilt == 1
    [event] = list(store.events.find())
    assert event["summary"]["impacts"]["FATALITIES"][0]["headline"] == 25   # old claim superseded


def test_syndicated_copy_does_not_add_a_source(store):
    add(store, 1, [stored_claim(31)], text="Same text in two papers.")
    add(store, 2, [stored_claim(31)], text="Same text in two papers.")
    match_pending(store, now=NOW)
    refresh_summaries(store, now=NOW)
    [event] = list(store.events.find())
    assert event["summary"]["impacts"]["FATALITIES"][0]["sources"] == 1
