"""Reads and writes the articles, claims and events collections.

The Store is given a database object. In production that is the real
MongoDB database; in tests it is an in-memory imitation (mongomock).
The code is the same for both.

The old `disaster_events` collection is never used here.
"""
import os
from datetime import timedelta

from pymongo.errors import DuplicateKeyError

from database import schema
from database.documents import (build_article, build_claims, build_event,
                                make_article_id, make_content_hash, to_utc, utc_now)


def get_database(name="disaster_db"):
    """Connect using MONGODB_URI from .env (the same variable db_writer.py uses)."""
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        # Without this check pymongo would quietly try a database on this PC instead.
        raise RuntimeError("MONGODB_URI is not set. Put it in .env in the project root.")
    return MongoClient(uri, tz_aware=True)[name]


class Store:
    def __init__(self, db):
        self.db       = db
        self.articles = db[schema.ARTICLES]
        self.claims   = db[schema.CLAIMS]
        self.events   = db[schema.EVENTS]

    def setup(self):
        """Create the indexes. Safe to call every time the program starts."""
        schema.ensure_indexes(self.db)

    # ── Articles ───────────────────────────────────────────────────────────
    def add_article(self, url, *, now=None, **fields):
        """Record a URL. Returns (article_id, created).
        created is False when the URL is already in the database; nothing is changed then."""
        doc = build_article(url, now=now, **fields)
        if doc["text"]:
            doc["duplicate_of"] = self._earlier_copy(doc["content_hash"], doc["_id"])
        try:
            self.articles.insert_one(doc)
        except DuplicateKeyError:
            return doc["_id"], False
        return doc["_id"], True

    def set_text(self, article_id, *, title, text, publish_date=None,
                 publish_date_source=None, scraped_at=None, keyword_hazards=(),
                 keyword_hits=(), now=None):
        """Store the scraped, cleaned text: discovered/scrape_failed -> cleaned."""
        now = now or utc_now()
        self._require(article_id, {schema.DISCOVERED, schema.SCRAPE_FAILED})
        content_hash = make_content_hash(text)
        self.articles.update_one({"_id": article_id}, {"$set": {
            "title": title, "text": text, "content_hash": content_hash,
            "duplicate_of": self._earlier_copy(content_hash, article_id),
            "publish_date": to_utc(publish_date),
            "publish_date_source": publish_date_source if publish_date else None,
            "scraped_at": to_utc(scraped_at) or now,
            "keyword_hazards": sorted(keyword_hazards), "keyword_hits": sorted(keyword_hits),
            "processing_status": schema.CLEANED,
            "retry_count": 0, "next_retry_at": None, "last_error": None,
            "updated_at": now,
        }})

    def mark_filtered_out(self, article_id, reason, now=None):
        """Not a flood/fire article, or too short. Final; it is not retried."""
        now = now or utc_now()
        self._require(article_id, {schema.DISCOVERED, schema.SCRAPE_FAILED, schema.CLEANED})
        self.articles.update_one({"_id": article_id}, {"$set": {
            "processing_status": schema.FILTERED_OUT,
            "last_error": {"stage": "filter", "kind": "filtered_out", "message": reason, "at": now},
            "next_retry_at": None, "updated_at": now,
        }})

    def mark_failed(self, article_id, *, stage, kind, message, extraction=None, now=None):
        """Record a failed scrape or extraction and schedule the retry.
        Returns the new status. After MAX_RETRIES failures the article is `dead`.

        `extraction` is the run metadata including the batch reports, so the
        reason (which batch, which candidates) can be read later.
        No claims are written for a failed article.
        """
        if stage not in ("scrape", "extraction"):
            raise ValueError("stage must be 'scrape' or 'extraction'")
        now = now or utc_now()
        allowed = ({schema.DISCOVERED, schema.SCRAPE_FAILED} if stage == "scrape"
                   else {schema.CLEANED, schema.EXTRACTION_FAILED})
        article = self._require(article_id, allowed)

        retries = article["retry_count"] + 1
        if retries >= schema.MAX_RETRIES:
            status, next_retry = schema.DEAD, None
        else:
            status = schema.SCRAPE_FAILED if stage == "scrape" else schema.EXTRACTION_FAILED
            next_retry = now + timedelta(hours=schema.RETRY_DELAY_HOURS[retries - 1])

        changes = {
            "processing_status": status, "retry_count": retries,
            "next_retry_at": next_retry, "updated_at": now,
            "last_error": {"stage": stage, "kind": kind, "message": message, "at": now},
        }
        if extraction is not None:
            changes["extraction"] = extraction
        self.articles.update_one({"_id": article_id}, {"$set": changes})
        return status

    def get_pending(self, stage="extraction", limit=50, now=None):
        """Articles waiting for this stage: never tried, or failed and due for retry.
        Oldest first."""
        now = now or utc_now()
        fresh, failed = ((schema.DISCOVERED, schema.SCRAPE_FAILED) if stage == "scrape"
                         else (schema.CLEANED, schema.EXTRACTION_FAILED))
        query = {"$or": [
            {"processing_status": fresh},
            {"processing_status": failed, "next_retry_at": {"$lte": now}},
        ]}
        return list(self.articles.find(query).sort("created_at", 1).limit(limit))

    def get_article(self, article_id):
        return self.articles.find_one({"_id": article_id})

    def get_article_by_url(self, url):
        return self.get_article(make_article_id(url))

    # ── Extraction results ─────────────────────────────────────────────────
    def save_extraction(self, article_id, *, entities, hazard_types, extraction, claims, now=None):
        """Store a COMPLETE extraction: Branch A entities, run metadata and all claims.
        Returns the number of claims stored.

        Order matters. Claims are written first and the article's status is
        changed last. If the program stops halfway, the article still looks
        unfinished, the next run repeats it, and because claim IDs are
        predictable the repeat overwrites instead of duplicating.
        """
        now = now or utc_now()
        if extraction.get("status") != "complete":
            raise ValueError("save_extraction only accepts status 'complete'. "
                             "Use mark_failed for a partial or failed run.")
        version = extraction.get("version")
        if not version:
            raise ValueError("extraction['version'] is required")
        unknown = set(hazard_types) - schema.HAZARD_TYPES
        if unknown:
            raise ValueError(f"unknown hazard types: {sorted(unknown)}")
        article = self._require(article_id, {schema.CLEANED, schema.EXTRACTION_FAILED,
                                             schema.EXTRACTED, schema.MATCHED})

        # Every claim is checked here, before the first write.
        docs = build_claims(article, version, claims, now)

        for doc in docs:
            fields = dict(doc)
            claim_id = fields.pop("_id")
            # created_at and a person's review survive a repeat run.
            keep_if_exists = {"created_at": fields.pop("created_at"),
                              "review": fields.pop("review")}
            self.claims.update_one({"_id": claim_id},
                                   {"$set": fields, "$setOnInsert": keep_if_exists},
                                   upsert=True)
        # A repeat of the same version that produced fewer claims: remove leftovers.
        self.claims.delete_many({"article_id": article_id, "extraction_version": version,
                                 "_id": {"$nin": [d["_id"] for d in docs]}})
        # Claims from an older version are kept but marked as replaced.
        self.claims.update_many({"article_id": article_id, "extraction_version": {"$ne": version}},
                                {"$set": {"superseded": True}})

        self.articles.update_one({"_id": article_id}, {"$set": {
            "entities": entities, "hazard_types": sorted(hazard_types), "extraction": extraction,
            "processing_status": schema.MATCHED if article.get("event_id") else schema.EXTRACTED,
            "retry_count": 0, "next_retry_at": None, "last_error": None, "updated_at": now,
        }})
        return len(docs)

    def get_claims(self, article_id, include_superseded=False):
        query = {"article_id": article_id}
        if not include_superseded:
            query["superseded"] = False
        return list(self.claims.find(query).sort("_id", 1))

    # ── Events ─────────────────────────────────────────────────────────────
    def create_event(self, hazard_type, title=None, now=None):
        doc = build_event(hazard_type, title, now)
        self.events.insert_one(doc)
        return doc["_id"]

    def get_event(self, event_id):
        return self.events.find_one({"_id": event_id})

    def attach_article(self, article_id, event_id, *, method, score=None, features=None,
                       decided_by="auto", needs_review=False, now=None):
        """Attach an extracted article (and its claims) to an event.

        Returns False, changing nothing, when a person already decided this
        article's event and this call is automatic. This layer only records
        the decision; choosing the event is the matcher's job.
        """
        if decided_by not in schema.DECIDED_BY:
            raise ValueError("decided_by must be 'auto' or 'human'")
        now = now or utc_now()
        article = self._require(article_id, {schema.EXTRACTED, schema.MATCHED})
        event = self.get_event(event_id)
        if event is None or event["status"] == "merged":
            raise ValueError(f"event {event_id} does not exist or was merged into another")

        previous = article.get("event_match") or {}
        if previous.get("decided_by") == "human" and decided_by == "auto":
            return False

        old_event_id = article.get("event_id")
        self.articles.update_one({"_id": article_id}, {"$set": {
            "event_id": event_id, "processing_status": schema.MATCHED, "updated_at": now,
            "event_match": {"method": method, "score": score, "features": features or {},
                            "decided_by": decided_by, "decided_at": now},
        }})
        self.claims.update_many({"article_id": article_id}, {"$set": {"event_id": event_id}})
        if needs_review:
            self.events.update_one({"_id": event_id}, {"$set": {"needs_review": True}})

        self.refresh_event(event_id, now)
        if old_event_id and old_event_id != event_id:
            self.refresh_event(old_event_id, now)
        return True

    def refresh_event(self, event_id, now=None):
        """Recalculate an event's article list, dates, places and source count
        from its articles. These fields are always derived, never typed in."""
        now = now or utc_now()
        articles = list(self.articles.find({"event_id": event_id}).sort("created_at", 1))

        dates = [a["publish_date"] or a["scraped_at"] for a in articles]
        dates = [d for d in dates if d is not None]
        if not articles:
            date_basis = None
        elif all(a["publish_date"] for a in articles):
            date_basis = "publish_date"
        else:
            date_basis = "scraped_at"     # at least one date is only when we downloaded it

        locations = {}
        for level in ("districts", "upazilas", "unions"):
            counts = {}
            for a in articles:
                found = ((a.get("entities") or {}).get("locations") or {}).get(level, [])
                for name in {item["value"] for item in found}:
                    counts[name] = counts.get(name, 0) + 1
            locations[level] = [{"name": n, "articles": c} for n, c in sorted(counts.items())]

        # The same text printed by two papers counts once.
        sources = {a["source"] for a in articles if not a.get("duplicate_of")}

        self.events.update_one({"_id": event_id}, {"$set": {
            "article_ids": [a["_id"] for a in articles],
            "start_date": min(dates) if dates else None,
            "end_date": max(dates) if dates else None,
            "date_basis": date_basis, "locations": locations,
            "source_count": len(sources), "last_updated": now,
        }})

    def get_event_claims(self, event_id, impact_type=None):
        """Current claims of an event, for building its summary."""
        query = {"event_id": event_id, "superseded": False}
        if impact_type:
            query["impact_type"] = impact_type
        return list(self.claims.find(query).sort([("article_date", 1), ("_id", 1)]))

    # ── Helpers ────────────────────────────────────────────────────────────
    def _require(self, article_id, allowed_statuses):
        """Fetch an article and refuse to continue if it is in the wrong state."""
        article = self.get_article(article_id)
        if article is None:
            raise KeyError(f"no article with id {article_id}")
        if article["processing_status"] not in allowed_statuses:
            raise ValueError(f"article {article_id} is '{article['processing_status']}', "
                             f"expected one of {sorted(allowed_statuses)}")
        return article

    def _earlier_copy(self, content_hash, article_id):
        """ID of an earlier article with exactly the same text, or None."""
        other = self.articles.find_one(
            {"content_hash": content_hash, "_id": {"$ne": article_id}, "duplicate_of": None},
            sort=[("created_at", 1)])
        return other["_id"] if other else None
