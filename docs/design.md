# Design: articles, claims and events

Status: **proposal, nothing here is built yet.** Written 2026-11-17 from `CLAUDE.md` and the code on branch `hybrid-local-llm` (commit `af06181`).

Every threshold and number in sections 3 and 4 is a starting guess that must be tuned on labelled data. They are marked "initial guess". Do not report them as results.

Contents:

1. [MongoDB schemas and indexes](#1-mongodb-schemas-and-indexes)
2. [`extraction_service.py` interface](#2-extraction_servicepy-interface)
3. [Event matching and cross-source reconciliation](#3-event-matching-and-cross-source-reconciliation)
4. [Migration plan](#4-migration-plan)
5. [Test plan](#5-test-plan)
6. [Open questions](#6-open-questions)

## What changes, in one paragraph

Today one article becomes one MongoDB document in `disaster_events`, with one number per field. The new design splits that into three collections. An **article** is what a newspaper published. A **claim** is one impact statement inside an article ("death toll rose to 31"), kept with its evidence sentence. An **event** is one real-world disaster that many articles describe. The website shows events. The old `disaster_events` collection stays exactly as it is.

```
articles (1) ──< claims (many)
    │                │
    └──> events (1) <┘      an event has many articles and many claims
```

## Facts from the code that shaped this design

These were checked in the repo, not assumed.

| Fact | Where | Consequence |
| --- | --- | --- |
| `disaster_events` documents do not contain the article text. | `entity_linker.py:121-137`, `pipeline.py:263-272` | Old records cannot be re-extracted from the database alone. Text must come from `articles/cleaned/` or a re-scrape (section 4). |
| `pipeline.py` never saves article text to disk. Only `scraper.py` does. | `pipeline.py:242-272` | Articles that entered through the daily pipeline have no stored text anywhere. |
| 281 of the 345 files in `articles/cleaned/` have `publish_date` null. | counted on the lab PC | Event matching needs a date. Most of the existing corpus cannot be matched automatically until dates are recovered. |
| Validated claims carry only article-local IDs (`S2`, `L1`, `Q3`), not the sentence text or place name. | `claim_validator.py:203-222` | The service must resolve IDs to text before storing. `L1` means nothing outside one run. |
| For a cue-only candidate (no number), the unit the LLM gave is dropped. | `claim_validator.py:140-141, 211` | Store `unit` at the top level of the claim, taken from the LLM claim. |
| Nothing checks that the LLM answered every candidate in a batch. | `local_llm.py:86-90` | A candidate the LLM skips disappears without an error. The service must check (section 2.5). |
| The validator accepts a candidate ID from a different batch. | `claim_validator.py:85, 96` | Check each batch against its own candidate IDs. |
| Location list passed to the LLM contains districts and upazilas, not unions. | `test_branch_b_pipeline.py:46-61` | Decide whether to add unions (open question 9). |
| The LLM prompt tells Qwen that DISPLACED includes "marooned, stranded, evacuated", but the schema has separate `MAROONED` and `EVACUATED` types. The prompt describes 5 of the 14 types. | `local_llm.py:14-19`, `schemas.py:6-20` | **Resolved 2026-11-17:** one class DISPLACED; prompt `p2` describes every type. |
| `num_ctx` is not set and `prompt_eval_count` is not logged. | `local_llm.py:47-51` | Silent prompt truncation would not be noticed. The service logs token counts per batch. |
| 9 upazila names are shared by more than one upazila (e.g. Companiganj, Kaliganj). 289 union names are shared. 216 union names are also upazila names. | `data/gazetteer/*.csv` | A place name alone is not an identifier. Store the parent district when it is known. |
| `entity_linker.py` loads the XLM-R model when it is imported. | `entity_linker.py:15-16` | The service must import it lazily so tests can run without the model. |

---

## 1. MongoDB schemas and indexes

Database: `disaster_db` (same as now). New collections: `articles`, `claims`, `events`. Old collection `disaster_events` is not modified.

Conventions for all three:

- Dates are stored as real MongoDB dates in UTC, not strings. (The old collection stores local-time strings like `"2026-08-24 06:34:54"`.)
- Every document has `schema_version` (integer, starts at 1) so later changes can be migrated.
- Enumerated values are the exact strings from `extraction/branch_b/schemas.py`.

### 1.1 `articles`

One document per URL.

| Field | Type | Notes |
| --- | --- | --- |
| `_id` | string | `article_id` = MD5 of the canonical URL. Same rule `scraper.py` already uses for file names, so files and documents line up. |
| `url` | string | Canonical URL (no fragment, no tracking parameters, no trailing slash). |
| `source` | string | Newspaper name, e.g. `"The Daily Star"`. Derived from the domain. |
| `title` | string | |
| `text` | string | Cleaned text. Sentence and character offsets in claims refer to this exact string. |
| `content_hash` | string | SHA-256 of `text`. Detects the same wire story printed by two papers, and detects a changed article on re-scrape. |
| `publish_date` | date or null | |
| `publish_date_source` | string or null | `"newspaper3k"`, `"meta_tag"`, `"url"`, `"dateline"`, `"manual"`. Says how reliable the date is. |
| `scraped_at` | date | |
| `hazard_types` | list of string | Normalised: `"FLOOD"`, `"FIRE"`. From Branch A `DISASTER_TYPE` plus the keyword filter. |
| `entities` | object | Branch A output: `disaster_type[]`, `locations{districts[], upazilas[], unions[]}`, `agencies_involved[]`, `raw_entities[]`. Same shape as today. |
| `processing_status` | string | See the status list in section 2.4. |
| `retry_count` | int | Failed attempts so far for the current stage. |
| `next_retry_at` | date or null | Do not retry before this time. |
| `last_error` | object or null | `{stage, kind, message, at}`. |
| `extraction` | object or null | Metadata of the latest extraction run (below). |
| `event_id` | string or null | Set by the event matcher. |
| `event_match` | object or null | `{method, score, features, decided_by, decided_at}`. `decided_by` is `"auto"` or `"human"`. Keeps the reason an article was attached to an event. |
| `duplicate_of` | string or null | `article_id` of an earlier article with the same `content_hash`. |
| `legacy_event_id` | string or null | `event_id` of the matching document in `disaster_events`, if any. |
| `created_at`, `updated_at` | date | |
| `schema_version` | int | |

`extraction` sub-document (written by the service, for reproducibility):

```
extraction: {
  version:            "b1",                    // bumped when prompt, schema, generator or validator changes
  status:             "complete" | "partial" | "failed",
  started_at, finished_at,
  branch_a_model:     "xlmroberta_v3",
  llm_backend:        "ollama",
  llm_model:          "qwen3:8b",
  llm_model_digest:   "500a1f067a9f",          // from `ollama list`
  llm_options:        {temperature: 0, num_gpu: 0, num_ctx: 8192, think: false},
  prompt_version:     "p1",
  candidate_count:    25,
  claim_count:        25,
  batches: [
    { index: 1, candidate_ids: ["Q1",...,"Q6"], status: "ok" | "failed",
      attempts: 1, prompt_tokens: 812, output_tokens: 340, seconds: 41.2, error: null }
  ]
}
```

The `batches` list is what makes a failed batch visible later. It records which candidates were in the batch that failed.

Indexes:

| Index | Why |
| --- | --- |
| `{url: 1}` unique | Duplicate check (replaces the `find_one` on `source_url` in `db_writer.py`). |
| `{processing_status: 1, next_retry_at: 1}` | The daily job asks "what is waiting or due for retry?" |
| `{event_id: 1}` | All articles of an event. |
| `{content_hash: 1}` | Syndicated-copy detection. Not unique. |
| `{publish_date: -1}` | Timeline and the matcher's date window. |
| `{"entities.locations.districts.value": 1, publish_date: -1}` | Matcher candidate lookup by district. |

### 1.2 `claims`

One document per impact statement. Claims are never edited after writing, except `event_id`, `superseded` and the review fields.

| Field | Type | Notes |
| --- | --- | --- |
| `_id` | string | `"{article_id}:{extraction_version}:{candidate_id}"`. Deterministic, so a re-run cannot create duplicates. If the LLM returns two claims for one candidate, append `:2`. |
| `article_id` | string | |
| `event_id` | string or null | Copied from the article when it is matched. |
| `candidate_id` | string | `Q7` or `C2`. Article-local. |
| `candidate_kind` | string | `"QUANTITY"` or `"CUE"`. |
| `impact_type` | string | 14 values in `ImpactType`. |
| `raw_value` | string | Exactly as in the text: `"58 lakh"`. |
| `normalized_value` | number or null | From `quantity_normalizer.py`. Python computes this, never the LLM. |
| `lower_bound`, `upper_bound` | number or null | For ranges. |
| `modifier` | string or null | `"AT_LEAST"`, `"NEARLY"`, ... |
| `is_qualitative` | bool | `"several"`, `"hundreds"`. |
| `unit` | string or null | From the LLM claim. |
| `aggregation` | string | 7 values in `AggregationType`. |
| `certainty` | string | 6 values in `CertaintyType`. |
| `locations` | list of object | Resolved from `L1`...: `{name, level, district}`. `level` is `district`, `upazila` or `union`. `district` is the parent district when known, else null. |
| `scope` | string | `"event"` when `locations` is empty (a nationwide or whole-event figure), else `"local"`. Used by reconciliation. |
| `evidence` | list of object | `{sentence_id, text, start_char, end_char}`. The sentence text is copied in, so the claim can be read without the article. |
| `char_start`, `char_end` | int | Position of the number in `articles.text`. |
| `validation_status` | string | `ACCEPTED`, `REVIEW_REQUIRED`, `REJECTED`. |
| `validation_flags` | list of string | As produced by `claim_validator.py`. |
| `extractor` | string | `"branch_b"`. Leaves room for `"branch_a_legacy"` or `"manual"`. |
| `extraction_version` | string | Same value as `articles.extraction.version`. |
| `superseded` | bool | Set to true when the article is re-extracted with a newer version. Old claims are kept, not deleted. |
| `review` | object or null | `{decision, corrected_fields, reviewer, at}` if a person checks it. |
| `article_date` | date or null | Copy of the article's `publish_date`. Saves a join when building timelines. |
| `source` | string | Copy of the article's newspaper name. |
| `created_at` | date | |
| `schema_version` | int | |

There is **no confidence field**. spaCy and Qwen do not give a usable one, and `CLAUDE.md` forbids inventing it. `validation_status` and `validation_flags` are what the system honestly knows.

Indexes:

| Index | Why |
| --- | --- |
| `{article_id: 1, superseded: 1}` | Current claims of an article. |
| `{event_id: 1, impact_type: 1, superseded: 1}` | Building an event summary. |
| `{validation_status: 1, created_at: -1}` | Review queue. |
| `{impact_type: 1, article_date: -1}` | Researcher queries such as "all fatality claims in August 2024". |

`_id` already guarantees uniqueness, so no extra unique index is needed.

### 1.3 `events`

One document per real-world disaster. Everything in `summary` is **derived** from claims and can be rebuilt at any time. Nothing in an event is the only copy of a fact.

| Field | Type | Notes |
| --- | --- | --- |
| `_id` | string | UUID. |
| `hazard_type` | string | `"FLOOD"` or `"FIRE"`. |
| `title` | string | Generated label such as `"Flood, Feni and 10 more districts, Aug 2024"`. Editable. |
| `start_date`, `end_date` | date | Earliest and latest article date. This is the reporting period, not the physical start and end of the disaster. |
| `date_basis` | string | `"publish_date"`, or `"scraped_at"` if any article had no publish date. |
| `locations` | object | `{districts[], upazilas[], unions[]}`: union of the articles' Branch A locations, with a count of how many articles mention each. |
| `article_ids` | list of string | |
| `source_count` | int | Number of distinct newspapers, after removing syndicated copies. |
| `summary` | object | Built by reconciliation, section 3.3. |
| `conflicting_claims` | list of object | Section 3.3. |
| `status` | string | `"open"` (may still gain articles), `"closed"`, `"merged"`. |
| `merged_into` | string or null | If a person merges two events, the old one points to the new one and is kept. |
| `needs_review` | bool | True if any article was attached with a borderline score. |
| `reconciliation_version` | string | Version of the rules that built `summary`. |
| `created_at`, `last_updated` | date | |
| `schema_version` | int | |

Indexes:

| Index | Why |
| --- | --- |
| `{hazard_type: 1, end_date: -1}` | Matcher: recent events of the same hazard. Also the website list. |
| `{"locations.districts.name": 1}` | Matcher and map filter. |
| `{status: 1, last_updated: -1}` | Open events; "recently updated" on the website. |

District coordinates are already in `data/gazetteer/districts.csv` (latitude and longitude columns). A Leaflet map can use them directly. A MongoDB geospatial index is not needed for 64 points.

### 1.4 Size check

About 15 articles a day, each perhaps 6 KB of text plus 25 claims of roughly 1 KB, is roughly 0.5 MB a day, or under 200 MB a year. That fits the Atlas free tier (512 MB) for about two years. This is an estimate, not a measurement. Measure after the first month.

---

## 2. `extraction_service.py` interface

One new module at the project root. It **calls** the existing code. It does not replace `entity_linker.py` or anything in `extraction/branch_b/`.

It has two layers, kept separate on purpose:

- `extract_article()` takes an article and returns a result. It does not touch MongoDB. This is the part that is easy to test.
- `run_pending()` reads waiting articles from the database, calls `extract_article()`, and records status and retries.

### 2.1 The LLM backend (the swappable part)

```python
class LLMResponse:            # a small dataclass
    content: str              # raw text returned by the model (should be JSON)
    prompt_tokens: int | None
    output_tokens: int | None
    seconds: float

class LLMBackend(Protocol):
    name: str                 # "ollama"
    model: str                # "qwen3:8b"

    def complete(self, system_prompt: str, user_prompt: str, json_schema: dict) -> LLMResponse:
        """One request. Raises BackendUnavailable if the server cannot be reached.
        Raises BackendError for anything else (timeout, empty response)."""

    def describe(self) -> dict:
        """Model name, digest, options. Stored in articles.extraction."""
```

Three implementations:

| Backend | Purpose |
| --- | --- |
| `OllamaBackend` | Wraps the call that is in `local_llm.py` today: `think=False` top-level, `temperature 0`, `num_gpu 0`, and `num_ctx` set explicitly. |
| `FakeBackend` | For tests. Returns prepared answers, or raises on purpose. No Ollama needed. |
| (later) any other | A different local model, or a hosted API for comparison. Only this class changes. |

The backend returns **text only**. Parsing it with `BranchBResult.model_validate_json` happens in the service, so every backend goes through the same schema check and the same validator. A backend can never bypass the rules "the LLM does not convert numbers" and "the LLM does not invent IDs".

### 2.2 Extracting one article

```python
def extract_article(
    article: dict,                  # needs: article_id, text, url, publish_date
    backend: LLMBackend,
    *,
    branch_a: Callable | None = None,   # default: entity_linker.extract_entities, imported lazily
    batch_size: int = 6,
    max_batch_attempts: int = 2,
) -> ExtractionResult
```

```python
class ExtractionResult:
    article_id: str
    status: str                  # "complete" | "partial" | "failed"
    entities: dict               # Branch A record
    locations: list[dict]        # [{location_id: "L1", name, level, district}]
    claims: list[dict]           # validated, with IDs resolved to text (the shape of section 1.2)
    batches: list[BatchReport]   # one per batch, including failed ones
    unanswered: list[str]        # candidate IDs the LLM never returned
    versions: dict               # extraction version, prompt version, backend.describe()
    seconds: float
    error: dict | None           # {stage, kind, message} when status is not "complete"
```

`extract_article()` does not raise for an ordinary failure. It returns a result whose `status` says what happened. That way the caller always has something to record. The only exception it lets through is `BackendUnavailable` (see 2.4).

`branch_a` is a parameter so tests can pass a small fake function and never load the 1 GB model.

### 2.3 Steps inside `extract_article()`

1. Branch A: `extract_entities(text, url, publish_date)`.
2. Build the location list `L1, L2, ...` from Branch A districts and upazilas.
3. `generate_candidates(text)` gives sentences and candidates.
4. Split candidates into batches of 6. For each batch: build the prompt with `format_for_llm`, call `backend.complete`, parse, check (2.5). Retry a failed batch up to `max_batch_attempts`.
5. `validate_claims(...)` on everything that came back.
6. Resolve IDs to text (sentence text, place names) and build claim records.
7. Set `status`.

### 2.4 Article status and retry

`processing_status` values:

| Status | Meaning | Next step |
| --- | --- | --- |
| `discovered` | URL found, not yet downloaded. | scrape |
| `scrape_failed` | Download or parse failed. | retry later |
| `filtered_out` | Not flood/fire related, or too short. Final. | none |
| `cleaned` | Text stored, waiting for extraction. | extract |
| `extraction_failed` | Extraction did not complete. | retry later |
| `extracted` | Branch A and B complete, claims stored. | match to an event |
| `matched` | Attached to an event. Final (until re-extraction). | none |
| `dead` | Failed 3 times. Needs a person. | none, listed in the run report |

This fixes known problem 4 in `CLAUDE.md`: today a URL is added to `seen_urls` before anything succeeds, so a failed article is never tried again. In the new design the URL is recorded as `discovered` first, and a failure leaves a status that the next run picks up.

```python
def run_pending(store, backend, *, limit=50, max_retries=3) -> RunReport
```

Rules:

- **Pick up** articles with status `cleaned`, or `extraction_failed` whose `next_retry_at` has passed.
- **On success:** write claims, write `articles.entities` and `articles.extraction`, set `extracted`.
- **On failure:** set `extraction_failed`, add 1 to `retry_count`, store `last_error`, set `next_retry_at` (next daily run; later 3 days). After `max_retries`, set `dead`.
- **If Ollama is not running** (`BackendUnavailable`): stop the whole run immediately. Do **not** mark the remaining articles as failed and do not use up their retries. They stay `cleaned` and are processed next time. The run report says the backend was down.
- **Writes are all-or-nothing per article.** Claims are written to `claims` only when every batch succeeded. An article is never left looking finished with part of its claims missing.
- `store` is a small object with methods like `get_pending()`, `save_result()`, `mark_failed()`. In production it wraps pymongo. In tests it is an in-memory fake.

`RunReport` is printed and logged at the end of every run: counts per status, the list of failed and dead article IDs with reasons, total seconds, and seconds per article. A run that processed zero articles is reported as such, so a broken scraper is noticed.

### 2.5 Failed batches are never lost silently

There are four ways a batch can lose information. Each one must leave a record.

| What goes wrong | How the service notices | What it records |
| --- | --- | --- |
| The request fails, times out, or returns empty text. | Exception from the backend. | `BatchReport{status: "failed", error, candidate_ids}` after the retries. Article `status` is `partial`. |
| The reply is not valid JSON for the schema. | `model_validate_json` raises. | Same as above. The raw reply (first 2,000 characters) is kept in the error. |
| The LLM skips a candidate. | Compare returned `candidate_id`s with the batch's own IDs. | One retry of the batch. If still missing, the ID goes into `unanswered` and the article is `partial`. |
| The LLM returns an ID that was not in this batch. | Same comparison. | The claim is dropped and counted in the batch report as `foreign_ids`. |

`partial` is treated as a failure by `run_pending()`: nothing is written to `claims`, the article goes to `extraction_failed`, and the batch reports are saved in `articles.extraction.batches` so the reason can be read later.

In the first version a retry re-runs the **whole article**. Re-running only the failed batches would save CPU time, but it adds bookkeeping. It can be added later because the batch reports already hold the candidate IDs (open question 7).

Token counts are logged for every batch. If `prompt_tokens` is close to `num_ctx`, the batch is flagged `POSSIBLE_TRUNCATION`.

### 2.6 What stays out of the service

- No number conversion. `quantity_normalizer.py` does that.
- No event matching. That is a separate step after `extracted`.
- No scheduling. A small entry script calls `run_pending()`; Windows Task Scheduler starts the script.
- No change to `pipeline.py` on `main`.

---

## 3. Event matching and cross-source reconciliation

### 3.1 Approach: rules first, embeddings only if measured to be needed

`CLAUDE.md` lists "hazard + date + district overlap + embedding similarity". This design starts with the first three only. Reasons: they can be explained to a reviewer in one sentence, they need no extra model, and there is no labelled data yet to show that embeddings help. Embeddings can be added as a fourth feature if the measured errors call for it.

### 3.2 Matching one article to an event

Input: an article with status `extracted`. It needs a hazard type, an article date, and at least one district.

**Article date** = `publish_date`, or `scraped_at` if the publish date is missing. `scraped_at` is only trusted for articles collected by the daily job, where scraping happens within a day or so of publication. For the historical corpus, scraped months later, `scraped_at` says nothing. **An article with no usable date is not matched automatically.** It gets an event of its own with `needs_review: true`.

Step 1, find candidate events (a cheap database query):

- same `hazard_type`
- event `end_date` within the time window of the article date
- at least one district in common

Step 2, score each candidate event:

- district overlap = (districts in both) / (districts in the smaller of the two lists). 1.0 when the article's districts are all inside the event.
- day gap = days between the article date and the event's nearest date.
- upazila overlap, used for fires and as a tie-breaker.

Step 3, decide:

| Result | Action |
| --- | --- |
| Clear match | Attach the article to the event. `decided_by: "auto"`. |
| Borderline | Attach to the best event, set `needs_review: true` on the event and keep the score. |
| No match | Create a new event from this article. |
| Two events both match clearly | Attach to the better one and flag both for a possible merge. Never merge automatically. |

Initial guesses for the windows and thresholds, **to be tuned, not reported**:

| | Flood | Fire |
| --- | --- | --- |
| Time window | 10 days from the event's last article | 3 days |
| Clear match | district overlap ≥ 0.5 | same district and same upazila |
| Borderline | overlap > 0 but < 0.5 | same district, upazila unknown or different |

Floods and fires need different rules because a flood covers many districts for weeks, while a fire is one place on one day. Two fires in Dhaka in the same week are two events.

The score, the features and the decision are saved in `articles.event_match`. A person can change the event later; that is stored as `decided_by: "human"` and is not overwritten by a re-run.

**How this will be evaluated.** Label a set of article pairs as "same event" or "different event" (a few hundred pairs, drawn from articles that share a hazard and a month). Tune the thresholds on one part and report pairwise precision, recall and F1 on the rest. The same event labels give the event-grouped train/test split that `CLAUDE.md` section 5 asks for, so this labelling work pays for itself twice.

**Known limits, stated honestly:**

- A long flood with a quiet gap in reporting may be split into two events.
- Two separate floods in overlapping districts in the same fortnight may be joined.
- Branch A district recall is high (F1 96.5) but not perfect. A missed district lowers the overlap.
- Follow-up articles weeks later (rehabilitation, compensation) may start a new event.

### 3.3 Cross-source reconciliation rules

Reconciliation builds `events.summary` from the event's claims. It runs each time an event gains an article. It only reads claims and never changes them.

**Which claims are used**

1. Only current claims (`superseded: false`).
2. Only `validation_status: ACCEPTED` feed the headline numbers. `REVIEW_REQUIRED` claims are listed separately as "unverified". `REJECTED` claims are ignored.
3. `impact_type: OTHER` is ignored.
4. Claims with certainty `EXPECTED`, `PREDICTED` or `POSSIBLE`, and claims flagged `HISTORICAL_REFERENCE`, `FORECAST_DETECTED` or `NEGATION_DETECTED`, never enter a headline number.
5. Articles with `duplicate_of` set (same text as an earlier article) do not count as a separate source.

**How claims are grouped**

6. Claims are compared only inside the same group: same `impact_type`, same `unit`, same `scope`. A figure for Feni district is never compared with a figure for the whole event. Local figures are grouped per district.
7. `families` and `people` are different units. The system does not multiply families by a household size.

**How a group becomes a summary entry**

8. **Nothing is ever added together.** Not across articles, not across districts, not incremental onto cumulative. This is the rule from `CLAUDE.md` and it has no exceptions.
9. For `CUMULATIVE`, `POINT_ESTIMATE`, `LOWER_BOUND` and `UPPER_BOUND` claims: the **headline** is the value from the most recent article date. The summary also stores how many sources reported on that date, and the lowest and highest value among them.
10. If sources on the same latest date disagree, the headline is shown as a range (lowest to highest), and an entry is added to `conflicting_claims` with every claim ID. The system does not choose a winner. For `FATALITIES` and `MISSING` any difference counts as a conflict. For other types a difference above 10% counts (initial guess).
11. An earlier, higher figure than the latest one (a toll that appears to go down) is recorded as a conflict of kind `"decreasing"`. It is usually a correction or a different scope, and a person should look.
12. `INCREMENTAL` claims ("three more died") are shown in the event timeline only. They are never used to compute a total.
13. `RANGE` claims keep both bounds. Modifiers (`AT_LEAST`, `NEARLY`) are kept and displayed with the number.
14. Qualitative claims ("hundreds of houses") are listed as text and never turned into a number.
15. `aggregation: UNKNOWN` claims are listed as unverified and do not feed a headline.

**What the summary looks like**

```
summary: {
  FATALITIES: {
    event: { unit: "people", headline: 31, low: 31, high: 31, modifier: null,
             as_of: 2024-08-27, sources: 3, claim_ids: [...] },
    by_district: { "Feni": { headline: 17, ... } }
  },
  AFFECTED_PEOPLE: { event: { unit: "people", headline: 5800000, low: 5600000, high: 5800000, ... } }
}

conflicting_claims: [
  { impact_type: "FATALITIES", scope: "event", kind: "same_day_disagreement",
    date: 2024-08-27, values: [31, 27], claim_ids: [...] }
]
```

Every number in `summary` carries the claim IDs behind it. The website can show the evidence sentence and a link to the source article for each figure.

**What reconciliation cannot fix.** If Qwen gives a wrong but plausible `impact_type` or `aggregation`, the validator accepts it and it reaches the summary. The gold comparison measures how often that happens. Until that number exists, event summaries should be labelled as automatically extracted and unverified.

---

## 4. Migration plan

Principle: **add, never change.** `disaster_events` is kept, is never written by new code, and is never deleted. Every phase can be undone by dropping the three new collections.

| Phase | What happens | How to check | How to undo |
| --- | --- | --- | --- |
| 0. Backup | Export `disaster_events` to a dated JSON file with `mongoexport`. Keep it outside the repo. Record the document count. | File opens; count matches Atlas. | Nothing to undo. |
| 1. Create | A script creates `articles`, `claims`, `events` and their indexes. Running it twice changes nothing. | List the indexes. | Drop the three collections. |
| 2. Backfill articles | For each `disaster_events` document, create an `articles` document: URL, title, dates, the old Branch A fields as `entities`, and `legacy_event_id`. Text is filled in from `articles/cleaned/<md5 of url>.json` when that file exists. | Count of `articles` with `legacy_event_id` equals count of `disaster_events`. Report how many have text. | Drop `articles`. |
| 3. Re-extract | Backfilled articles that have text get status `cleaned` and go through `run_pending()` like new ones. Articles without text get status `discovered` so they are re-scraped; if the page is gone they end as `dead`, and still keep their legacy link. | Run report. | Delete claims with that `extraction_version`. |
| 4. Run side by side | The existing `pipeline.py` keeps writing `disaster_events` unchanged. The new job writes the new collections. Run both for at least two weeks. | Compare daily: same URLs found? Extraction completes overnight? | Stop the new job. |
| 5. Switch readers | The API and website (not built yet) read only the new collections. | | Point readers back. |
| 6. Freeze | Stop the old `pipeline.py` job. `disaster_events` stays in the database, read-only, as the record of the baseline system. | | Start the old job again. |

Decisions inside this plan:

- **Old impact numbers are not converted into claims.** A `fatalities: {value: 31, source: "model"}` field has no evidence sentence, no aggregation and no location. Turning it into a claim would create records that look like Branch B output but are not. The old numbers stay where they are, reachable through `legacy_event_id`.
- **Old `event_id` values are not reused as new event IDs.** Each old "event" is really one article. The old ID is kept as `articles.legacy_event_id`.
- **The 276 annotated articles are a separate question.** They are training and test data. Loading them into the production database is useful for event-pair labelling, but they must be marked (for example `origin: "corpus"`) so they are never mistaken for daily-pipeline output, and test-split articles must not be used to tune anything.
- **Dates.** Old date strings are parsed to UTC dates on the new documents. The old strings are in local time with no time zone; assume Asia/Dhaka and say so in the migration notes.
- **Nothing in this plan touches `main`.** New files only, on `hybrid-local-llm`.

Not known yet: how many documents `disaster_events` holds now, and how many of them have a matching file in `articles/cleaned/` (open question 2).

---

## 5. Test plan

All tests live in `tests/` and run with `venv\Scripts\python -m pytest tests`. Today there are 12 tests there and all pass (checked 2026-11-17).

Three pytest markers split the slow or machine-dependent tests from the rest:

- `@pytest.mark.qwen`: needs Ollama running with `qwen3:8b`. Minutes per test on CPU.
- `@pytest.mark.model`: needs `models/xlmroberta_v3/`. Seconds per test.
- `@pytest.mark.mongo`: needs a MongoDB connection. Uses a separate database named `disaster_db_test`, never `disaster_db`.

The default run skips all three, so it works on the 8 GB laptop in a few seconds.

One housekeeping point: `test_cpu.py` and `test_branch_b_pipeline.py` at the project root are scripts, not pytest tests, but their names start with `test_`. Running plain `pytest` from the root would try to collect them and start Ollama. Either always run `pytest tests`, or add a `pytest.ini` with `testpaths = tests`.

### 5.1 Tests that do NOT need Qwen

**Already exist**

| Area | File | Covers |
| --- | --- | --- |
| Quantity normalizer | `tests/test_quantity_normalizer.py` | lakh/crore, word numbers, `5, 40,510`, ranges keep both bounds |
| Validator | `tests/test_claim_validator.py` | non-person noun after a number, function-word stop, Feni case not rejected |

**To add: existing code**

| Area | Cases |
| --- | --- |
| Normalizer | modifiers (`at least 31`), qualitative (`several`), decimals (`4.52 crore`), `one hundred and twenty five thousand`, a year (`2022`), empty string |
| Candidate generator | Bengali grouping `12,27,554`; overlap removal (`58 lakh` is one candidate, not `58` plus `58 lakh`); sentence IDs and character offsets point at the right text; cue-only candidate for a sentence with no number; negation/historical/forecast flags |
| Validator | invalid candidate ID rejected; invalid sentence ID rejected; invalid location ID flagged; unit mismatch flagged; negation, historical, forecast flagged; cue candidate keeps its unit (currently fails, see the facts table) |

**To add: extraction service, using `FakeBackend` and a fake `branch_a`**

| Case | Expected |
| --- | --- |
| 13 candidates, batch size 6 | 3 requests of 6, 6 and 1 candidates |
| All batches succeed | `status: complete`, one claim per candidate, IDs resolved to sentence text and place names |
| Batch 2 fails once, then succeeds | `complete`, batch report shows `attempts: 2` |
| Batch 2 fails every time | `status: partial`, batch report lists its candidate IDs, no exception raised |
| Reply is not valid JSON | Treated as a failed attempt; raw reply kept in the error |
| LLM skips candidate `Q4` | Batch retried; then `Q4` in `unanswered`, `status: partial` |
| LLM returns `Q99`, not in the batch | Claim dropped, counted as `foreign_ids` |
| Article with zero candidates | `complete`, zero claims, backend never called |
| Backend raises `BackendUnavailable` | The exception reaches the caller |
| Number conversion | `normalized_value` comes from the normalizer even if the fake LLM reply contains a different number |

**To add: status and retry, using an in-memory fake store**

| Case | Expected |
| --- | --- |
| Successful article | `cleaned` → `extracted`, claims saved |
| Failed article | `extraction_failed`, `retry_count` 1, `last_error` set, **no claims saved** |
| Third failure | `dead`, listed in the run report |
| `next_retry_at` in the future | Not picked up |
| Backend unavailable on the 2nd of 5 articles | Run stops; articles 2 to 5 still `cleaned`, `retry_count` unchanged |
| Run again after success | No duplicate claims |

**To add: event matcher (plain functions on small hand-written dictionaries)**

- same flood, next day, same districts → same event
- same districts, 40 days later → new event
- flood and fire in the same district on the same day → different events
- two fires in one district, different upazilas, same day → different events
- article with no date → own event, `needs_review`
- article with no district → own event, `needs_review`
- human-decided match is not overwritten by a re-run

**To add: reconciliation (plain functions)**

- three sources, cumulative 27, 29, 31 on three days → headline 31
- two sources on the same day, 31 and 27 → range 27 to 31 and one conflict entry
- incremental claim present → does not change the headline
- district figure and event figure → separate entries, not compared
- `REVIEW_REQUIRED`, `REJECTED`, `OTHER`, forecast and historical claims → not in the headline
- families and people → separate entries
- same `content_hash` from two newspapers → counted as one source
- toll goes down → `"decreasing"` conflict
- **no rule ever produces a number that is not in some claim** (check every headline value against the claim values)

**To add: migration (`mongo` marker, test database)**

- running the create script twice leaves the same indexes
- a second article with the same URL is refused by the unique index
- backfill of 3 sample legacy documents: 3 articles, legacy link set, legacy collection unchanged (compare before and after)

**Needs the XLM-R model but not Qwen (`model` marker)**

- `extract_entities` on the built-in sample text finds Sylhet, Sunamganj, Feni as districts
- the service builds `L1, L2, ...` from real Branch A output

### 5.2 Tests that NEED Qwen (`qwen` marker)

Run these on the lab PC, by hand or overnight. They check the model and the machine, not the Python logic.

| Test | Pass condition |
| --- | --- |
| Smoke test (what `test_cpu.py` does now) | 3 plain and 3 JSON replies are non-empty; JSON parses |
| CPU check | `ollama ps` reports `100% CPU` after a call |
| Four-sentence prompt (the one in `local_llm.py`) | Reply is valid against `BranchBResult`; every candidate ID answered; no invented IDs |
| One real article through `OllamaBackend` | `status: complete`, every batch `ok`, no `unanswered` |
| Context check | `prompt_tokens` below `num_ctx` for every batch of the longest article in `articles/cleaned/` |
| Repeatability | Same article twice at temperature 0 gives the same claims. If it does not, record how many differ. This is a finding to report, not a reason to hide the test. |
| Timing | Seconds per batch and per article, saved to a file. No pass/fail. |

**Not a test: the gold comparison.** Comparing Branch B output with `data/gold/gold_verified.csv` produces precision, recall and accuracy figures. It is an evaluation run with its own script and saved outputs. It must not be a pass/fail test, because then there would be pressure to adjust the code until it passes on the test split.

---

## 6. Open questions

Each has a suggested answer where one is possible. The ones marked **blocks build** need a decision before coding starts.

**Decisions about data**

1. **What is one "event"?** Is the August 2024 eastern flood across 11 districts one event, or one per district or river basin? The suggestion in this design is one event per connected flood (shared districts, continuous reporting). This is a disaster-science question, and it changes the matcher and what DDM sees. **Blocks build of the matcher.**
2. **How many documents are in `disaster_events` now, and which have text on disk?** Needed to size phases 2 and 3 of the migration. A read-only count would answer it. I did not connect to the database because `.env` is off limits.
3. **Recovering publish dates.** 281 of 345 cleaned articles have none. Options: parse the dateline in the text, parse the URL (`scraper.py` already tries), re-fetch the page for meta tags, or enter them by hand for the 42 test articles first. Without dates, the existing corpus cannot be matched to events or used for an event-grouped split.
4. **Should the 276 annotated articles go into the production database?** Suggestion: yes, marked `origin: "corpus"`, because event-pair labelling needs them.

**Decisions about extraction**

5. **The prompt and the taxonomy disagree** (the prompt folds marooned and evacuated into DISPLACED; the schema has them separate; 9 of 14 types have no description in the prompt). Which is intended? This must be settled **before** correcting the gold set, because the gold labels have to use the same definitions. Related: the gold draft uses the five Branch A labels (`RELIEF_INFO`, not `RELIEF`).
6. **What should happen to a claim the validator marks `REVIEW_REQUIRED`?** Suggestion: store it, keep it out of headline numbers, show it as "unverified". Is there anyone who will actually review them? If not, the review queue only grows.
7. **Retry the whole article, or only the failed batches?** Suggestion: whole article first (simpler). Revisit once real timing is known.
8. **How many retries, how far apart?** Suggestion: 3 attempts; next day, then 3 days later, then `dead`.
9. **Should unions be given to the LLM as locations?** Branch A union F1 is 61.9 and 289 union names are ambiguous. Suggestion: not yet.
10. **`num_ctx` value.** `CLAUDE.md` suggests 8192. It is not set in `local_llm.py` today. Confirm on the lab PC that 8192 fits in memory on CPU, and record the timing change.

**Decisions about the product**

11. **May the full article text be stored in a cloud database and shown on a public website?** Newspaper text is copyrighted. Suggestion: store the text for research, show only the evidence sentence and a link on the website. Worth asking the supervisor.
12. **Who is allowed to correct data** (merge events, fix a claim), and should corrections be logged with a name? This decides whether the website needs logins.
13. **Does DDM need Bangla place names?** They are already in `data/gazetteer/*.csv`.
14. **Is "reporting period" acceptable as the event date range?** The system knows when newspapers wrote about a disaster, not when it physically started.

**Repository housekeeping found while writing this**

15. **`README.md` on this branch is not the project README.** Commit `fbde06a` replaced it with the README of the ChatGPT patch that was not adopted. It describes files that are not in the repo (`reliable_extractor.py`, `ollama_smoke_test.py`). The original is in commit `b6d3137`. Restore it?
16. **`config_bert.cfg` is not the XLM-R training config.** It is a CNN config (`max_steps = 20000`, `learn_rate = 0.001`, no transformer). The real settings survive only inside `models/xlmroberta_v3/config.cfg`, which is not in Git. Copy that file into the repo for reproducibility?
17. **`requirements.txt`** is a full `pip freeze` saved as UTF-16. It includes Label Studio and Django, and it does not include `pytest`. The venv on the lab PC runs Python 3.13.15, while `CLAUDE.md` says 3.11.9. Which Python version is the reference?
18. **`build_gold_draft.py`** writes a fixed `gold_draft.csv` and does not skip articles already in `gold_verified.csv`. `CLAUDE.md` described both features as existing. Add them, or keep the simpler behaviour?
19. **Name of the entry script** that Task Scheduler will run. The rejected patch used `daily_pipeline.py`; a different name avoids confusion. Suggestion: `run_daily.py`.
