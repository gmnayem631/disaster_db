"""Names, allowed values and indexes for the three new collections.

This file contains no logic except ensure_indexes(). It is the one place to
look up "what is the collection called?" or "which statuses exist?".
"""

SCHEMA_VERSION = 1

# Collection names
ARTICLES = "articles"
CLAIMS   = "claims"
EVENTS   = "events"
# The old collection. Listed only so tests can prove we never change it.
LEGACY   = "disaster_events"

# Article processing_status values (docs/design.md section 2.4)
DISCOVERED        = "discovered"         # URL known, not downloaded yet
SCRAPE_FAILED     = "scrape_failed"      # download failed, will retry
FILTERED_OUT      = "filtered_out"       # not flood/fire or too short. Final.
CLEANED           = "cleaned"            # text stored, waiting for extraction
EXTRACTION_FAILED = "extraction_failed"  # extraction failed, will retry
EXTRACTED         = "extracted"          # claims stored, waiting for event matching
MATCHED           = "matched"            # attached to an event
DEAD              = "dead"               # failed too many times, needs a person

# Claim validation_status values (same strings as claim_validator.py)
VALIDATION_STATUSES = {"ACCEPTED", "REVIEW_REQUIRED", "REJECTED"}

CANDIDATE_KINDS = {"QUANTITY", "CUE"}
HAZARD_TYPES    = {"FLOOD", "FIRE"}
DECIDED_BY      = {"auto", "human"}

# Retry policy. These are the suggested defaults from open question 8,
# not a tested choice. 20 hours (not 24) so tomorrow's run, which starts at
# about the same clock time, does pick the article up.
MAX_RETRIES       = 3
RETRY_DELAY_HOURS = [20, 72]   # wait after the 1st failure, after the 2nd failure

# Indexes: collection -> list of (keys, options).
# An index is a lookup table MongoDB keeps so a query does not scan everything.
INDEXES = {
    ARTICLES: [
        ([("url", 1)], {"name": "url_unique", "unique": True}),
        ([("processing_status", 1), ("next_retry_at", 1)], {"name": "status_retry"}),
        ([("event_id", 1)], {"name": "event"}),
        ([("content_hash", 1)], {"name": "content_hash"}),
        ([("publish_date", -1)], {"name": "publish_date"}),
        ([("entities.locations.districts.value", 1), ("publish_date", -1)],
         {"name": "district_date"}),
    ],
    CLAIMS: [
        ([("article_id", 1), ("superseded", 1)], {"name": "article_current"}),
        ([("event_id", 1), ("impact_type", 1), ("superseded", 1)], {"name": "event_impact"}),
        ([("validation_status", 1), ("created_at", -1)], {"name": "review_queue"}),
        ([("impact_type", 1), ("article_date", -1)], {"name": "impact_date"}),
    ],
    EVENTS: [
        ([("hazard_type", 1), ("end_date", -1)], {"name": "hazard_recent"}),
        ([("locations.districts.name", 1)], {"name": "district"}),
        ([("status", 1), ("last_updated", -1)], {"name": "status_updated"}),
    ],
}


def ensure_indexes(db):
    """Create every index above. Safe to run again: existing ones are left alone."""
    for collection, indexes in INDEXES.items():
        for keys, options in indexes:
            db[collection].create_index(keys, **options)
