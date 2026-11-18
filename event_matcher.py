"""Event matching: several articles about one disaster become one event.

Design: docs/design.md section 3.2. Rules only (no embeddings), so every
decision can be explained in one sentence and is stored on the article.

For each extracted article (oldest article date first):
  1. Find open events with the same hazard, a date within the time window,
     and at least one district in common.
  2. Score them: how many districts they share, how many days apart,
     whether they share an upazila.
  3. Decide:
       clear match       -> attach the article to that event
       borderline match  -> "spread" hazards: attach and flag the event for review
                            "spot" hazards:   new event, flagged, possible match noted
       two clear matches -> attach to the best, flag both (possible duplicate events)
       no match          -> new event
     Never merges two events automatically. A person's decision is never
     overwritten (Store.attach_article checks that).

PROVISIONAL: the time windows and thresholds below are the author's to decide
(open decision 1, "what counts as one event"). Change them here only.
"""
import time
from dataclasses import dataclass, field
from datetime import timedelta

from database import schema
from database.documents import canonical_district, utc_now

MATCHER_VERSION = "rules_v1"

# ── Settings (PROVISIONAL, see above) ─────────────────────────────────────
# "spread" hazards cover many districts for days or weeks: matched on shared districts.
# "spot" hazards happen at one place at one time: matched on district AND upazila.
HAZARD_RULES = {
    "FLOOD":                    {"kind": "spread", "window_days": 14},
    "CYCLONE":                  {"kind": "spread", "window_days": 7},
    "STORM":                    {"kind": "spread", "window_days": 3},
    "RIVERBANK_EROSION":        {"kind": "spread", "window_days": 14},
    "HEATWAVE":                 {"kind": "spread", "window_days": 7},
    "COLD_WAVE":                {"kind": "spread", "window_days": 7},
    "DROUGHT":                  {"kind": "spread", "window_days": 30},
    "DISEASE_OUTBREAK":         {"kind": "spread", "window_days": 14},   # waits for the dengue decision
    "EARTHQUAKE":               {"kind": "spread", "window_days": 3},
    "OTHER":                    {"kind": "spread", "window_days": 7},
    "FIRE":                     {"kind": "spot",   "window_days": 3},
    "BUILDING_COLLAPSE":        {"kind": "spot",   "window_days": 3},
    "ROAD_ACCIDENT":            {"kind": "spot",   "window_days": 2},
    "WATER_TRANSPORT_ACCIDENT": {"kind": "spot",   "window_days": 3},
    "LANDSLIDE":                {"kind": "spot",   "window_days": 3},
    "LIGHTNING":                {"kind": "spot",   "window_days": 1},
}
CLEAR_OVERLAP = 0.5     # spread hazards: share at least half of the smaller district list

# When an article names several hazards (a cyclone article that mentions
# flooding), its new event gets the first one in this order: causes before effects.
PRIMARY_ORDER = ["CYCLONE", "EARTHQUAKE", "FLOOD", "LANDSLIDE", "RIVERBANK_EROSION", "STORM",
                 "BUILDING_COLLAPSE", "FIRE", "WATER_TRANSPORT_ACCIDENT", "ROAD_ACCIDENT",
                 "LIGHTNING", "HEATWAVE", "COLD_WAVE", "DROUGHT", "DISEASE_OUTBREAK", "OTHER"]


# ── Reading what the matcher needs from an article ─────────────────────────
def article_hazards(article):
    """Hazards from the Qwen hazard check if there is one (it knows every
    hazard), else from Branch A (flood and fire only). OTHER only if nothing else."""
    found = list((article.get("screening") or {}).get("hazards") or [])
    if not found:
        found = list(article.get("hazard_types") or [])
    specific = [h for h in found if h != "OTHER"]
    hazards = specific or found
    return sorted(set(hazards), key=lambda h: PRIMARY_ORDER.index(h) if h in PRIMARY_ORDER else 99)


def article_date(article):
    """publish_date, or the download time for daily articles (downloaded within
    a day or two of publication). Old corpus articles without a publish date
    have no usable date."""
    if article.get("publish_date"):
        return article["publish_date"]
    if article.get("origin") == "daily":
        return article.get("scraped_at")
    return None


def _places(article, level):
    found = ((article.get("entities") or {}).get("locations") or {}).get(level, [])
    names = {item["value"] for item in found}
    return {canonical_district(n) for n in names} if level == "districts" else names


def _event_places(event, level):
    return {item["name"] for item in (event.get("locations") or {}).get(level, [])}


def rule_for(hazard):
    return HAZARD_RULES.get(hazard, HAZARD_RULES["OTHER"])


# ── Scoring one candidate event ────────────────────────────────────────────
def score(article, event, date):
    """Features of one article/event pair, and the verdict for its hazard kind:
    "clear", "borderline" or "no"."""
    a_districts, e_districts = _places(article, "districts"), _event_places(event, "districts")
    a_upazilas, e_upazilas = _places(article, "upazilas"), _event_places(event, "upazilas")
    shared = a_districts & e_districts
    overlap = len(shared) / min(len(a_districts), len(e_districts)) if shared else 0.0
    if event["start_date"] <= date <= event["end_date"]:
        gap = 0.0
    else:
        gap = min(abs((date - event["start_date"]).total_seconds()),
                  abs((date - event["end_date"]).total_seconds())) / 86400
    same_upazila = bool(a_upazilas & e_upazilas)
    upazila_known = bool(a_upazilas) and bool(e_upazilas)

    if not shared:
        verdict = "no"
    elif rule_for(event["hazard_type"])["kind"] == "spread":
        verdict = "clear" if overlap >= CLEAR_OVERLAP else "borderline"
    elif same_upazila:
        verdict = "clear"
    elif upazila_known:
        verdict = "no"            # same district, different upazila: two separate incidents
    else:
        verdict = "borderline"    # same district, upazila not known on one side

    return {"event_id": event["_id"], "hazard": event["hazard_type"], "verdict": verdict,
            "district_overlap": round(overlap, 3), "shared_districts": sorted(shared),
            "day_gap": round(gap, 1), "same_upazila": same_upazila}


def _rank(s):
    """Best candidate first: clear before borderline, then shared upazila,
    then more district overlap, then fewer days apart."""
    return (s["verdict"] != "clear", not s["same_upazila"], -s["district_overlap"], s["day_gap"])


def find_candidates(store, article, hazards, date):
    districts = sorted(_places(article, "districts"))
    if not districts:
        return []
    widest = max(rule_for(h)["window_days"] for h in hazards)
    query = {
        "hazard_type": {"$in": hazards}, "status": "open",
        "locations.districts.name": {"$in": districts},
        "end_date": {"$gte": date - timedelta(days=widest)},
        "start_date": {"$lte": date + timedelta(days=widest)},
    }
    scored = []
    for event in store.events.find(query):
        if event["_id"] == article.get("event_id"):
            continue
        s = score(article, event, date)
        if s["verdict"] != "no" and s["day_gap"] <= rule_for(event["hazard_type"])["window_days"]:
            scored.append(s)
    return sorted(scored, key=_rank)


# ── One article ────────────────────────────────────────────────────────────
def match_article(store, article, now=None):
    """Attach one extracted article to an event (existing or new).
    Returns a dict describing the decision."""
    article_id = article["_id"]
    hazards = article_hazards(article)
    date = article_date(article)
    districts = sorted(_places(article, "districts"))
    features = {"hazards": hazards, "districts": districts, "article_date": date}

    def attach(event_id, decision, score_value, needs_review, **extra):
        features.update(extra, decision=decision)
        store.attach_article(article_id, event_id, method=MATCHER_VERSION, score=score_value,
                             features=features, needs_review=needs_review, now=now)
        return {"article_id": article_id, "event_id": event_id, "decision": decision,
                "needs_review": needs_review, "title": article.get("title")}

    def new_event(decision, needs_review, **extra):
        event_id = store.create_event(hazards[0] if hazards else "OTHER", now=now)
        return attach(event_id, decision, None, needs_review, **extra)

    # Same text as an article already in an event (syndicated copy).
    if article.get("duplicate_of"):
        original = store.get_article(article["duplicate_of"])
        if original and original.get("event_id"):
            return attach(original["event_id"], "same_text", 1.0, False)

    # Without a hazard, a date or a district the rules cannot work. A person decides.
    missing = [name for name, value in (("hazard", hazards), ("date", date),
                                        ("district", districts)) if not value]
    if missing:
        return new_event("cannot_match", True, missing=missing)

    candidates = find_candidates(store, article, hazards, date)
    clear = [c for c in candidates if c["verdict"] == "clear"]
    if clear:
        best = clear[0]
        if len(clear) > 1:
            # Two events both fit: they may be the same disaster. Flag them, never merge.
            others = [c["event_id"] for c in clear[1:]]
            store.events.update_many({"_id": {"$in": others}}, {"$set": {"needs_review": True}})
            return attach(best["event_id"], "clear_but_ambiguous", best["district_overlap"], True,
                          best=best, possible_same_event=others)
        return attach(best["event_id"], "clear", best["district_overlap"], False, best=best)

    if candidates:
        best = candidates[0]
        if rule_for(best["hazard"])["kind"] == "spread":
            return attach(best["event_id"], "borderline", best["district_overlap"], True, best=best)
        return new_event("new_possible_match", True, possible_match=best)

    return new_event("new", False)


# ── Many articles ──────────────────────────────────────────────────────────
@dataclass
class MatchReport:
    picked_up: int = 0
    attached: int = 0              # joined an existing event
    new_events: int = 0
    needs_review: int = 0
    failed: int = 0
    seconds: float = 0.0
    decisions: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)

    def summary(self):
        lines = [f"Event matching: {self.picked_up} articles, {self.attached} joined an existing "
                 f"event, {self.new_events} new events, {self.needs_review} flagged for review, "
                 f"{self.failed} failed, {self.seconds:.0f} s"]
        if self.decisions:
            lines.append("  decisions: " + ", ".join(f"{k} {v}" for k, v in sorted(self.decisions.items())))
        lines.extend(f"  {p}" for p in self.problems)
        return "\n".join(lines)


def _sort_key(article):
    """Oldest article date first, so events grow in the order things happened.
    Articles with no date go last."""
    date = article_date(article)
    return (date is None, date or article["created_at"], article["_id"])


def match_pending(store, *, limit=0, progress=None, now=None):
    """Match every article waiting with status `extracted`. Returns a MatchReport."""
    clock = time.perf_counter()
    report = MatchReport()
    say = progress or (lambda message: None)
    waiting = sorted(store.articles.find({"processing_status": schema.EXTRACTED}), key=_sort_key)
    if limit:
        waiting = waiting[:limit]
    report.picked_up = len(waiting)

    for position, article in enumerate(waiting, 1):
        try:
            result = match_article(store, article, now=now)
        except Exception as exc:
            # One bad article must not stop the others. It stays `extracted`
            # and is tried again on the next run.
            report.failed += 1
            report.problems.append(f"FAILED {article['_id']}: {type(exc).__name__}: {exc}")
            continue
        decision = result["decision"]
        report.decisions[decision] = report.decisions.get(decision, 0) + 1
        if decision.startswith("new") or decision == "cannot_match":
            report.new_events += 1
        else:
            report.attached += 1
        report.needs_review += result["needs_review"]
        flag = " (review)" if result["needs_review"] else ""
        say(f"[{position}/{len(waiting)}] {decision}{flag}: {(result['title'] or '')[:60]}")

    report.seconds = time.perf_counter() - clock
    return report
