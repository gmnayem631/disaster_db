"""Event summaries: the headline numbers of an event, built from its claims.

Design: docs/design.md section 3.3 (cross-source reconciliation). In short:

  * NOTHING IS EVER ADDED TOGETHER. Not across articles, not across districts,
    not "3 more died" onto a total.
  * The headline is the figure from the most recent article date. If the
    sources on that date disagree, it is shown as a range and the
    disagreement is listed in `conflicting_claims`. No winner is chosen.
  * A figure that seems to go down (an earlier report was higher) is also
    listed as a conflict, for a person to look at.
  * Figures for different places, units or impact types are never compared.

The summary only reads claims and never changes them. It can be thrown away
and rebuilt at any time (refresh_summaries does that when an event changes).
Everything here is automatically extracted and not checked by a person.
"""
import time
from dataclasses import dataclass, field

from database.documents import canonical_district, utc_now

SUMMARY_VERSION = "recon_v1"
LABEL = "Automatically extracted from news reports. Not checked by a person."

# ── Settings (docs/design.md 3.3) ──────────────────────────────────────────
# Aggregations that can give a headline. INCREMENTAL goes to the timeline only.
HEADLINE_AGGREGATIONS = {"CUMULATIVE", "POINT_ESTIMATE", "LOWER_BOUND", "UPPER_BOUND", "RANGE"}
UNCERTAIN = {"EXPECTED", "PREDICTED", "POSSIBLE"}             # never a headline
WARNING_FLAGS = {"HISTORICAL_REFERENCE", "FORECAST_DETECTED", "NEGATION_DETECTED"}
# How far same-day figures may differ before it counts as a conflict.
# Deaths and missing people: any difference. Others: more than 10% (initial guess).
CONFLICT_TOLERANCE = {"FATALITIES": 0.0, "MISSING": 0.0}
DEFAULT_TOLERANCE = 0.10


# ── Sorting one claim ──────────────────────────────────────────────────────
def claim_place(claim):
    """("event", None) for a whole-event figure, else ("local", place name).
    A figure given for several places together ("in Feni and Noakhali") keeps
    them together; it is never split or shared out."""
    names = set()
    for loc in claim.get("locations") or []:
        if loc.get("level") == "district" or loc.get("district"):
            names.add(canonical_district(loc.get("district") or loc["name"]))
        else:
            names.add(loc["name"])
    if not names:
        return "event", None
    return "local", " + ".join(sorted(names))


def claim_unit(claim):
    return (claim.get("unit") or "unspecified").strip().lower()


def claim_bounds(claim):
    """(low, high) of a numeric claim, or None if it has no usable number."""
    if claim.get("aggregation") == "RANGE" and claim.get("lower_bound") is not None:
        return claim["lower_bound"], claim.get("upper_bound") or claim["lower_bound"]
    value = claim.get("normalized_value")
    return None if value is None else (value, value)


def why_not_counted(claim):
    """None if the claim may give a headline, else the reason it may not.
    "skip" means ignore completely; "unverified" means list it for a person."""
    if claim.get("validation_status") == "REJECTED" or claim.get("impact_type") == "OTHER":
        return "skip"
    if claim.get("certainty") in UNCERTAIN:
        return "uncertain"
    flags = {f.split(":")[0] for f in claim.get("validation_flags") or []}
    if flags & WARNING_FLAGS:
        return sorted(flags & WARNING_FLAGS)[0].lower()
    if claim.get("validation_status") == "REVIEW_REQUIRED":
        return "unverified"
    if claim.get("aggregation") == "UNKNOWN":
        return "unknown_aggregation"
    if claim.get("is_qualitative"):
        return "qualitative"
    if claim.get("aggregation") == "INCREMENTAL":
        return "incremental"
    if claim_bounds(claim) is None:
        return "no_number"
    return None


def _number(value):
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}".rstrip("0")


def show(low, high, modifiers):
    """Text for the website: "31", "27 to 31", "nearly 5,000"."""
    text = _number(low) if low == high else f"{_number(low)} to {_number(high)}"
    return f"{modifiers[0]} {text}" if len(modifiers) == 1 else text


# ── One event ──────────────────────────────────────────────────────────────
def summarize(claims, articles):
    """Build (summary, conflicting_claims) from an event's current claims.

    `articles` maps article_id -> {"date", "source"} and holds only the
    articles that count (copies of another article's text are left out).
    Pure function: no database, easy to test.
    """
    groups, timeline, qualitative, unverified, not_counted = {}, [], [], [], {}

    for claim in claims:
        article = articles.get(claim["article_id"])
        if article is None:
            continue                                   # a copy of another article
        reason = why_not_counted(claim)
        if reason == "skip":
            continue
        scope, place = claim_place(claim)
        base = {"impact_type": claim["impact_type"], "raw_value": claim["raw_value"],
                "unit": claim_unit(claim), "place": place, "date": article["date"],
                "source": article["source"], "claim_id": claim["_id"]}
        if reason is None:
            key = (claim["impact_type"], claim_unit(claim), scope, place)
            groups.setdefault(key, []).append((claim, article))
            continue
        not_counted[reason] = not_counted.get(reason, 0) + 1
        if reason == "incremental":
            timeline.append({**base, "value": claim.get("normalized_value")})
        elif reason == "qualitative":
            qualitative.append(base)
        else:
            unverified.append({**base, "reason": reason})

    impacts, conflicts = {}, []
    for (impact, unit, scope, place), members in sorted(groups.items(), key=_group_order):
        entry, found = _headline(impact, unit, scope, place, members)
        impacts.setdefault(impact, []).append(entry)
        conflicts.extend(found)

    by_date = lambda item: (item["date"] is None, item["date"] or 0, item["claim_id"])
    summary = {
        "label": LABEL,
        "impacts": impacts,
        "timeline": sorted(timeline, key=by_date),
        "qualitative": sorted(qualitative, key=by_date),
        "unverified": sorted(unverified, key=by_date),
        "not_counted": not_counted,
    }
    return summary, conflicts


def _group_order(item):
    (impact, unit, scope, place), _ = item
    return (impact, scope != "event", place or "", unit)


def _headline(impact, unit, scope, place, members):
    """The summary entry of one group, and its conflicts (if any)."""
    dated = [m for m in members if m[1]["date"] is not None]
    if dated:
        latest = max(article["date"] for _, article in dated)
        today = [m for m in dated if m[1]["date"] == latest]
        earlier = [m for m in dated if m[1]["date"] < latest]
    else:
        latest, today, earlier = None, members, []    # no dates at all: all count as "latest"

    lows, highs = zip(*(claim_bounds(c) for c, _ in today))
    low, high = min(lows), max(highs)
    modifiers = sorted({c["modifier"] for c, _ in today if c.get("modifier")})
    ids = [c["_id"] for c, _ in today]
    entry = {
        "scope": scope, "place": place, "unit": unit,
        "headline": low if low == high else None,      # None: sources disagree, see low/high
        "low": low, "high": high, "display": show(low, high, modifiers),
        "modifiers": modifiers, "as_of": latest,
        "sources": len({a["source"] for _, a in today}),
        "claim_ids": ids,
        "earlier_claim_ids": [c["_id"] for c, _ in earlier],
    }

    conflicts = []
    where = {"impact_type": impact, "unit": unit, "scope": scope, "place": place}
    tolerance = CONFLICT_TOLERANCE.get(impact, DEFAULT_TOLERANCE)
    if high > 0 and (high - low) / high > tolerance:
        conflicts.append({**where, "kind": "same_day_disagreement", "date": latest,
                          "values": sorted({v for pair in zip(lows, highs) for v in pair}),
                          "claim_ids": ids})
    higher_before = [(c, a) for c, a in earlier if claim_bounds(c)[1] > high]
    if higher_before:
        conflicts.append({**where, "kind": "decreasing", "date": latest,
                          "values": sorted({claim_bounds(c)[1] for c, _ in higher_before} | {high}),
                          "claim_ids": [c["_id"] for c, _ in higher_before] + ids})
    return entry, conflicts


def event_articles(store, event):
    """Date and newspaper of each article that counts. A copy of another
    article's text in the same event is left out (it is not a second source)."""
    from event_matcher import article_date
    found = list(store.articles.find({"_id": {"$in": event.get("article_ids") or []}}))
    ids = {a["_id"] for a in found}
    return {a["_id"]: {"date": article_date(a), "source": a.get("source")}
            for a in found if a.get("duplicate_of") not in ids}


def build_event_summary(store, event_id, now=None):
    """Rebuild and save one event's summary. Returns (summary, conflicts)."""
    now = now or utc_now()
    event = store.get_event(event_id)
    summary, conflicts = summarize(store.get_event_claims(event_id), event_articles(store, event))
    store.events.update_one({"_id": event_id}, {"$set": {
        "summary": summary, "conflicting_claims": conflicts,
        "reconciliation_version": SUMMARY_VERSION, "summary_built_at": now}})
    return summary, conflicts


# ── Many events ────────────────────────────────────────────────────────────
@dataclass
class SummaryReport:
    rebuilt: int = 0
    with_conflicts: int = 0
    failed: int = 0
    seconds: float = 0.0
    problems: list = field(default_factory=list)

    def summary(self):
        lines = [f"Event summaries: {self.rebuilt} rebuilt, {self.with_conflicts} with "
                 f"conflicting figures, {self.failed} failed, {self.seconds:.0f} s"]
        lines.extend(f"  {p}" for p in self.problems)
        return "\n".join(lines)


def is_stale(event):
    """True if the summary is missing, older than the event, or made by older rules."""
    built = event.get("summary_built_at")
    return (built is None or event.get("reconciliation_version") != SUMMARY_VERSION
            or (event.get("last_updated") is not None and event["last_updated"] > built))


def refresh_summaries(store, *, everything=False, progress=None, now=None):
    """Rebuild the summary of every event that changed since its last summary
    (or of every event, with everything=True). Returns a SummaryReport."""
    clock = time.perf_counter()
    report = SummaryReport()
    say = progress or (lambda message: None)
    fields = {"title": 1, "last_updated": 1, "summary_built_at": 1, "reconciliation_version": 1}
    events = list(store.events.find({"status": {"$ne": "merged"}}, fields))
    for event in [e for e in events if everything or is_stale(e)]:
        try:
            _, conflicts = build_event_summary(store, event["_id"], now=now)
        except Exception as exc:
            report.failed += 1
            report.problems.append(f"FAILED {event['_id']}: {type(exc).__name__}: {exc}")
            continue
        report.rebuilt += 1
        if conflicts:
            report.with_conflicts += 1
            say(f"conflicting figures: {event.get('title')}")
    report.seconds = time.perf_counter() - clock
    return report
