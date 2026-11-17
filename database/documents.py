"""Builds the dictionaries that get stored in MongoDB.

Everything here is a plain function: values in, dictionary out. No database
connection is used, so these can be tested without MongoDB.
"""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from extraction.branch_b.schemas import AggregationType, CertaintyType, ImpactType
from database import schema

# Old date strings in this project are local Bangladesh time with no zone.
# Bangladesh is UTC+6 all year (no daylight saving).
DHAKA = timezone(timedelta(hours=6))

TRACKING_PARAMS = {"fbclid", "gclid"}

SOURCE_NAMES = {
    "thedailystar.net":           "The Daily Star",
    "dhakatribune.com":           "Dhaka Tribune",
    "newagebd.net":               "New Age",
    "bdnews24.com":               "bdnews24",
    "tbsnews.net":                "The Business Standard",
    "prothomalo.com":             "Prothom Alo English",
    "thefinancialexpress.com.bd": "The Financial Express",
    "unb.com.bd":                 "UNB",
    "bangladeshpost.net":         "Bangladesh Post",
    "observerbd.com":             "Daily Observer",
    "daily-sun.com":              "The Daily Sun",
    "bssnews.net":                "BSS",
}


# ── Dates ──────────────────────────────────────────────────────────────────
def _to_ms(dt):
    """MongoDB keeps milliseconds, not microseconds. Cut them off here so a
    value read back equals the value written."""
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)


def utc_now():
    return _to_ms(datetime.now(timezone.utc))


def to_utc(value):
    """Turn a date from the old files into a UTC datetime.

    None                    -> None
    "2024-08-27"            -> 2024-08-27 00:00 UTC (a calendar date; not shifted)
    "2024-08-27 09:30:00"   -> treated as Bangladesh time, converted to UTC
    datetime without a zone -> treated as Bangladesh time
    datetime with a zone    -> converted to UTC
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 10:
            day = datetime.strptime(text, "%Y-%m-%d")
            return day.replace(tzinfo=timezone.utc)
        value = datetime.fromisoformat(text)      # raises ValueError if unreadable
    if not isinstance(value, datetime):
        raise TypeError(f"Not a date: {value!r}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=DHAKA)
    return _to_ms(value.astimezone(timezone.utc))


# ── URLs and IDs ───────────────────────────────────────────────────────────
def canonical_url(url):
    """One spelling per article address: lower-case host, no #fragment,
    no tracking parameters, no trailing slash."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Not a web address: {url!r}")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), urlencode(query), ""))


def make_article_id(url):
    """MD5 of the canonical URL. The same URL always gives the same ID."""
    return hashlib.md5(canonical_url(url).encode("utf-8")).hexdigest()


def make_content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_name(url):
    """Newspaper name from the web address. Unknown sites keep their domain."""
    host = urlsplit(canonical_url(url)).netloc
    for domain, name in SOURCE_NAMES.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host[4:] if host.startswith("www.") else host


# ── Articles ───────────────────────────────────────────────────────────────
def build_article(url, *, title=None, text=None, publish_date=None,
                  publish_date_source=None, scraped_at=None,
                  legacy_event_id=None, origin="daily", keyword_hazards=None,
                  keyword_hits=None, now=None):
    """A new article document. With text it starts as `cleaned`,
    without text as `discovered`."""
    now = now or utc_now()
    clean_url = canonical_url(url)
    return {
        "_id":                 make_article_id(url),
        "url":                 clean_url,
        "original_url":        url.strip(),   # articles/cleaned files are named by the MD5 of this
        "source":              source_name(url),
        "origin":              origin,        # "daily", "corpus" or "legacy"
        "title":               title,
        "text":                text,
        "content_hash":        make_content_hash(text) if text else None,
        "publish_date":        to_utc(publish_date),
        "publish_date_source": publish_date_source if publish_date else None,
        "scraped_at":          to_utc(scraped_at),
        "hazard_types":        [],
        # Which hazard keywords the article matched when it was ingested
        # (data/hazard_keywords.txt). A filter result, not a classification.
        "keyword_hazards":     sorted(keyword_hazards or []),
        "keyword_hits":        sorted(keyword_hits or []),
        "screening":           None,   # the LLM hazard check (screening.py)
        "entities":            None,
        "processing_status":   schema.CLEANED if text else schema.DISCOVERED,
        "retry_count":         0,
        "next_retry_at":       None,
        "last_error":          None,
        "extraction":          None,
        "event_id":            None,
        "event_match":         None,
        "duplicate_of":        None,
        "legacy_event_id":     legacy_event_id,
        "created_at":          now,
        "updated_at":          now,
        "schema_version":      schema.SCHEMA_VERSION,
    }


# ── Claims ─────────────────────────────────────────────────────────────────
# Fields the extraction service must supply, and optional ones with defaults.
# Anything else is refused, so a misspelled or invented field (for example a
# made-up "confidence") cannot slip into the database.
CLAIM_REQUIRED = ("candidate_id", "candidate_kind", "impact_type", "raw_value",
                  "aggregation", "certainty", "validation_status")
CLAIM_DEFAULTS = {
    "normalized_value": None, "lower_bound": None, "upper_bound": None,
    "modifier": None, "is_qualitative": False, "unit": None,
    "locations": [], "evidence": [], "char_start": None, "char_end": None,
    "validation_flags": [], "extractor": "branch_b", "review": None,
}


def _check_choice(value, allowed, field):
    if value not in allowed:
        raise ValueError(f"claim {field}={value!r} is not one of {sorted(allowed)}")


def build_claims(article, extraction_version, claims, now=None):
    """Turn the service's claim records into documents for the claims collection.

    Checks every claim first. If one is wrong, ValueError is raised and no
    list is returned, so the caller writes nothing.
    """
    now = now or utc_now()
    docs, times_seen = [], {}

    for claim in claims:
        missing = [f for f in CLAIM_REQUIRED if f not in claim]
        if missing:
            raise ValueError(f"claim is missing {missing}: {claim!r}")
        unknown = set(claim) - set(CLAIM_REQUIRED) - set(CLAIM_DEFAULTS)
        if unknown:
            raise ValueError(f"claim has fields that are not in the schema: {sorted(unknown)}")
        if not claim["candidate_id"]:
            raise ValueError("claim has an empty candidate_id")
        _check_choice(claim["candidate_kind"], schema.CANDIDATE_KINDS, "candidate_kind")
        _check_choice(claim["impact_type"], {e.value for e in ImpactType}, "impact_type")
        _check_choice(claim["aggregation"], {e.value for e in AggregationType}, "aggregation")
        _check_choice(claim["certainty"], {e.value for e in CertaintyType}, "certainty")
        _check_choice(claim["validation_status"], schema.VALIDATION_STATUSES, "validation_status")

        # ID: article + extraction version + candidate. A second claim for
        # the same candidate gets ":2", a third ":3".
        cid = claim["candidate_id"]
        times_seen[cid] = times_seen.get(cid, 0) + 1
        claim_id = f"{article['_id']}:{extraction_version}:{cid}"
        if times_seen[cid] > 1:
            claim_id += f":{times_seen[cid]}"

        doc = {"_id": claim_id, "article_id": article["_id"],
               "event_id": article.get("event_id")}
        for field in CLAIM_REQUIRED:
            doc[field] = claim[field]
        for field, default in CLAIM_DEFAULTS.items():
            value = claim.get(field, default)
            doc[field] = list(value) if isinstance(value, list) else value
        doc.update({
            # A claim with no place is about the whole event.
            "scope":              "local" if doc["locations"] else "event",
            "extraction_version": extraction_version,
            "superseded":         False,
            "article_date":       article.get("publish_date"),
            "source":             article.get("source"),
            "created_at":         now,
            "schema_version":     schema.SCHEMA_VERSION,
        })
        docs.append(doc)
    return docs


# ── Events ─────────────────────────────────────────────────────────────────
def build_event(hazard_type, title=None, now=None):
    """A new, empty event. Dates, locations and counts are filled in from its
    articles by Store.refresh_event()."""
    _check_choice(hazard_type, schema.HAZARD_TYPES, "hazard_type")
    now = now or utc_now()
    return {
        "_id":                    str(uuid.uuid4()),
        "hazard_type":            hazard_type,
        "title":                  title,
        "start_date":             None,
        "end_date":               None,
        "date_basis":             None,
        "locations":              {"districts": [], "upazilas": [], "unions": []},
        "article_ids":            [],
        "source_count":           0,
        "summary":                {},     # built later by reconciliation (not part of this layer)
        "conflicting_claims":     [],
        "status":                 "open",
        "merged_into":            None,
        "needs_review":           False,
        "reconciliation_version": None,
        "created_at":             now,
        "last_updated":           now,
        "schema_version":         schema.SCHEMA_VERSION,
    }
