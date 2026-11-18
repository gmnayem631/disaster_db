# Project context for AI assistants

Read this fully before changing anything. Last checked against the code: 2026-11-17 (branch `hybrid-local-llm`, commit `af06181`).
Items marked **VERIFY** have not been confirmed in the repo. Check them before relying on them.

More detail lives in `docs/`:

- `docs/design.md`: proposed MongoDB schemas, `extraction_service.py` interface, event matching, migration, test plan, open questions. **Proposal only, not built.**
- `docs/runbook.md`: how to run every script, and the current data formats.
- `docs/background.md`: project history, the Ollama incident, achievements, risk table, glossary.

## 1. What this project is

**AI-Powered Digital Disaster Database in Bangladesh.** A system that reads English-language Bangladeshi newspaper articles, extracts structured disaster information (hazard type, locations, deaths, displaced/affected people, missing, relief, agencies), and stores it in MongoDB so researchers and policymakers can query it. It is meant to run daily and automatically.

- Author: Md. Gulam Mustafa Nayem (Dept. of Disaster Science and Climate Resilience, University of Dhaka). Comfortable with Python and frontend work, **new to NLP/ML**. Wants to understand every change.
- Repo: https://github.com/gmnayem631/disaster_db
- Branches: `main` (working baseline, do not break) and `hybrid-local-llm` (experimental Branch B work).
- Status: an academic report was written and defended on the first version. A journal submission is planned, and **rejections are expected, so the work must be reproducible and honestly evaluated.** The supervisor also wants a usable system for the Department of Disaster Management (DDM) and possibly a service later.
- Scope so far: floods (prototype). Fire articles are being added next. English only.

## 2. How to work with the author (important)

- Make **one small change at a time**. Explain in plain English what it does and why, before the code.
- Tell the author exactly which file and which lines to change. Run or reason through a test after each change.
- Be honest. Say "I don't know" or "unverified" instead of guessing. Do not oversell results.
- Do **not** rewrite files wholesale or add new architectures unprompted. Earlier in the project, overlapping instructions from two chatbots confused the author.
- Do **not** modify `main`, delete models, delete annotations, or touch `.env`.
- Windows machine. Use **PowerShell** with the venv (`venv\Scripts\activate`). An MSYS2 bash shell on the lab PC resolved `python` to a different interpreter with no pip. Run scripts from the project root.
- Python: the lab PC venv runs **3.13.15** (checked). Earlier notes said 3.11.9. Which one is the reference is undecided (`docs/design.md` question 17).
- Never commit secrets. `.env` holds `MONGODB_URI` and is gitignored.

## 3. Repository layout

**In Git:** code, `docs/`, `tests/`, `data/` (annotations, `.spacy` splits, gazetteer, gold draft, URL lists), and some old models (`models/gazetteer_model`, `models/trained_ner*`) that were committed before `models/` was gitignored.
**Not in Git (local or pen drive):** `articles/`, `models/xlmroberta_*` and the other transformer models, `.env`, `venv/`, `logs/`.

```
scraper.py                 reads data/urls.txt, newspaper3k, saves JSON to articles/raw/
preprocessor.py            cleans text, flood/fire keyword filter, min length -> articles/cleaned/
gazetteer_builder.py       builds data/gazetteer/bangladesh_gazetteer.json from CSVs
gazetteer_ner.py           early gazetteer experiment (not used in production)
label_studio_import.py     original 90 articles -> Label Studio tasks
pre_annotator.py           runs the NER model on new articles to pre-annotate for Label Studio
convert_annotations.py     Label Studio export -> train/dev/test .spacy (70/15/15, seed 42)
entity_linker.py           PRODUCTION extractor (Branch A): XLM-R + gazetteer + regex fallback
db_writer.py               MongoDB insert with duplicate check on source_url
pipeline.py                scrape -> clean -> extract -> MongoDB, run daily with `schedule` (flood keywords only)
evaluator.py               per-entity P/R/F1 on a .spacy file + timing
test_cpu.py                Ollama CPU-only sanity test (plain + JSON). A script, not a pytest test.
test_branch_b_pipeline.py  runs Branch A + candidate generator + Qwen + validator on one article. A script.
build_gold_draft.py        drafts a claim-level evaluation CSV from test-split annotations
gold_helpers.py            guesses unit/aggregation for the draft CSV
ner_test.py, pipeline_test.py   early one-off scripts
config.cfg, config_bert.cfg     old CNN training configs. NOT the XLM-R config (see section 5.8).

extraction/branch_b/
  schemas.py               Pydantic: ImpactType, AggregationType, CertaintyType, ImpactClaim, BranchBResult, NormalizedQuantity
  impact_cues.py           keyword lists: impact, negation, historical, forecast
  quantity_normalizer.py   deterministic number parsing (lakh/crore, word numbers, ranges, modifiers)
  candidate_generator.py   sentence split + finds every number and impact-cue sentence
  local_llm.py             Ollama call (qwen3:8b), structured JSON output, batching
  claim_validator.py       deterministic checks on the LLM's claims

database/                  NEW collections only (articles, claims, events). Never touches disaster_events.
  schema.py                collection names, status values, retry policy, index list, ensure_indexes()
  documents.py             pure functions that build article/claim/event documents (IDs, dates, checks)
  store.py                 Store class: add articles, pending/retry, save extraction, events

extraction_service.py      extract_article(): Branch A + B on one article via a swappable LLM backend.
                           OllamaBackend (real Qwen) and FakeBackend (tests).
                           run_pending(): extracts every waiting article in the new `articles` collection,
                           saves claims or records the failure and retry.
ingestion.py               NEW daily collection: reads data/newspapers.txt (news sitemaps / RSS), downloads
                           politely (robots.txt, random 5-15 s per site, crawl-delay, backs off on 429/403/503),
                           cleans, keeps articles matching data/hazard_keywords.txt -> `articles` as `cleaned`.
                           Failed downloads retried; unrelated articles stored without text as `filtered_out`.
screening.py               Qwen hazard check run by run_pending() before extraction: is this a real
                           disaster report in Bangladesh, which hazards. Drops keyword false matches.
event_matcher.py           articles -> events by rules (hazard, date window, shared district, upazila for
                           fires/accidents). Settings at the top are PROVISIONAL (author's decision 1).
                           Never merges events; unclear cases set needs_review. District spellings unified
                           by documents.canonical_district (Chittagong -> Chattogram).
daily_run.py               the daily job: ingestion, then run_pending(), then event matching. Log per run in logs/, lock file
                           so two runs never overlap, exit code for Task Scheduler. --db is required.
data/newspapers.txt        one line per newspaper: name | method | url. 7 enabled (checked 2026-11-18).
data/hazard_keywords.txt   one line per hazard. PROPOSED hazard list, awaiting the author's decision.

tests/                     pytest, 137 tests pass without Atlas, the NER model or the LLM
                           (mongomock for the database, FakeBackend / a stand-in client for the LLM).
                           1 test marked `qwen` needs real Qwen and is skipped unless RUN_QWEN=1.
                           With RUN_MONGO=1 the database tests use real Atlas, database
                           `disaster_db_test` only (41 passed on 2026-11-18; disaster_db unchanged).
docs/                      design.md, runbook.md, background.md
```

Run tests with `venv\Scripts\python -m pytest tests`. Always pass `tests`: plain `pytest` would also collect the root scripts and try to load the model and call Ollama.

The folders for a planned restructure (`ingestion/`, `preprocessing/`, `events/`, `database/`, `api/`, `evaluation/`) do not exist in the repo. Branch-B scripts use a `sys.path.insert` hack so they run from any folder.

`README.md` on this branch is **not** the project README. Commit `fbde06a` overwrote it with the README of a rejected ChatGPT patch. The original is in commit `b6d3137`. Not yet restored.

## 4. Baseline system (`main`)

Pipeline: scraper -> preprocessor -> `entity_linker.extract_entities()` -> MongoDB (`disaster_events`, db `disaster_db`).

- **Model:** XLM-RoBERTa base fine-tuned for NER with spaCy 3.8 + spacy-transformers. Local copy at `models/xlmroberta_v3/` (not in Git).
- **Gazetteer:** 64 districts, 483 distinct upazila names, 4,060 distinct union names from `nuhil/bangladesh-geocode`, added as `EntityRuler` patterns (9,214 incl. lowercase). The CSVs have 494 upazila rows and 4,540 union rows; the difference is repeated names. **Report "distinct names", not "483 upazilas".**
- **10 labels:** DISASTER_TYPE, BD_DISTRICT, BD_UPAZILA, BD_UNION, FATALITIES, DISPLACED, AFFECTED_PEOPLE, MISSING, RELIEF_INFO, AGENCIES_INVOLVED.
- **Hybrid inside Branch A:** if the model finds no number for fatalities/displaced/affected, regex patterns try (`source: "rule_based"` vs `"model"`). Contribution was about 3% on a 20-article sample.
- **Dataset:** 276 annotated flood articles, split 193 train / 41 dev / 42 test, **randomly by article, not by disaster event**. `articles/cleaned/` now holds 345 files (it includes unannotated and fire articles, and sources beyond the original four, e.g. TBS, Prothom Alo English, Financial Express).

**Current numbers (`models/xlmroberta_v3`, `test.spacy`, 42 articles):** overall F1 **85.03%**, ~2.5 s/article.

| Entity            | F1    |     | Entity          | F1    |
| ----------------- | ----- | --- | --------------- | ----- |
| BD_DISTRICT       | 96.51 |     | AFFECTED_PEOPLE | 68.35 |
| DISASTER_TYPE     | 95.29 |     | MISSING         | 63.64 |
| BD_UPAZILA        | 85.59 |     | DISPLACED       | 62.96 |
| AGENCIES_INVOLVED | 78.02 |     | BD_UNION        | 61.90 |
| FATALITIES        | 74.68 |     | RELIEF_INFO     | 39.71 |

The model-architecture comparison (XLM-R 79.21 etc.) was done **before** an annotation-conversion bug was fixed and has not been re-run. The submitted report still has the old figures. Details in `docs/background.md`.

## 5. Known problems in the baseline (not yet fixed)

1. `evaluator.py:149` hard-codes `"dev_set_size": 56`, but the 85.03% was measured on the 42-article test set. The saved `data/evaluation_results.json` carries the wrong label. Rerun and correct before submission.
2. Split is random by article. Reports of the same flood can land in train and test. Needs an event-grouped split for the paper.
3. `publish_date` is null for 281 of 345 cleaned articles. `scraped_at` is stored instead.
4. `pipeline.py` adds URLs to `seen_urls` _before_ scraping/extraction succeeds, so failed articles are never retried.
5. Each article becomes its own MongoDB "event" (UUID per article). Several articles about one flood are not linked.
6. Number handling in `entity_linker.py` picks one value per field with a keyword heuristic. It loses incremental vs cumulative information and has no location link or evidence sentence.
7. No true per-entity confidence is available from spaCy transformers. The `source` field is used instead. **Do not fabricate confidence scores.**
8. The XLM-R training config is not in the repo. `config_bert.cfg` is a CNN config despite its name. The real one is in `models/xlmroberta_v3/config.cfg` (not in Git).
9. `requirements.txt` is a full `pip freeze` in UTF-16, and includes Label Studio/Django. `pytest` and `mongomock` were added by hand. Rebuild cleanly.
10. `pipeline.py` and `disaster_events` do not store article text, so pipeline-ingested articles cannot be re-extracted without re-scraping.
11. `pipeline.py` filters on flood keywords only. `preprocessor.py` already includes fire keywords.

## 6. Branch B (`hybrid-local-llm`)

**Why:** NER is weak on impact numbers (what does "31", "58 lakh", "10,000" refer to?). Conventional code finds candidates, a local LLM decides only what each number _means_, then deterministic code validates.

```
article -> Branch A (XLM-R + gazetteer: hazard, locations, agencies)
        -> candidate_generator (every number + impact-cue sentences, with sentence IDs)
        -> local LLM (Qwen3 8B via Ollama): impact_type, aggregation, certainty, unit, location IDs
        -> claim_validator (deterministic checks)
        -> [planned, docs/design.md] store claims with evidence -> match to events -> MongoDB
```

Design rules (keep):

- The LLM **never converts numbers** and never invents IDs/URLs. Python does lakh/crore, word numbers, ranges, modifiers.
- Preserve raw text, evidence sentence, and individual claims. Never sum cumulative figures. Keep conflicting reports.
- Impact taxonomy: FATALITIES, INJURED, MISSING, AFFECTED_PEOPLE, DISPLACED (includes marooned, stranded, evacuated; author's decision 2026-11-17), HOUSING_DAMAGE, INFRASTRUCTURE_DAMAGE, AGRICULTURAL_DAMAGE, LIVESTOCK_LOSS, ECONOMIC_LOSS, RELIEF, OTHER. Aggregation: CUMULATIVE, INCREMENTAL, POINT_ESTIMATE, RANGE, LOWER_BOUND, UPPER_BOUND, UNKNOWN.

**Confirmed in the code and by `pytest tests` (2026-11-17):** normalizer (range bounds, `"5, 40,510"` -> 540510, whole word-number phrases, word number + lakh/crore), candidate generator (word-number phrases, Bengali commas, stray-space numbers), validator (invalid IDs, unit compatibility, negation/historical/forecast flags, non-person noun after a people count), `local_llm.py` raises on a failed batch after 2 attempts.

**VERIFY:** the full Qwen run (`test_branch_b_pipeline.py`) has not been re-run since commits `76adeac` and `af06181`, nor on Ollama 0.40.16.

**Known weaknesses:**

- Qwen can give a plausible-looking wrong meaning (e.g. "people took shelter" labelled RELIEF). The validator cannot catch semantic errors. The gold comparison measures this.
- Fixed 2026-11-17 (prompt `p2`, extraction `b2`): the system prompt now describes all 12 impact types, all aggregations and certainties, for all hazards. MAROONED and EVACUATED were removed from the schema; DISPLACED covers them. Gold CSV rows using them must be relabelled DISPLACED.
- Nothing checks that the LLM answered every candidate in a batch. A skipped candidate disappears without an error.
- `num_ctx` is not set in `local_llm.py` and `prompt_eval_count` is not logged, so silent truncation would go unnoticed.
- For cue-only candidates the validator drops the LLM's `unit`. Validated claims hold only article-local IDs (`S2`, `L1`), not the sentence text.
- Only districts and upazilas are passed to the LLM as locations, not unions.
- Cue-only candidates are created only for sentences with no number (so "houses washed away" next to a number can be missed). Deferred until recall is measured.
- Years like "2022" become candidates (labelled OTHER by the LLM).
- `build_gold_draft.py` writes a fixed `data/gold/gold_draft.csv`, overwrites it each run, and does **not** skip verified articles.

## 7. Local LLM: required call shape

- `think=False` as a top-level argument to `ollama.chat()` (not in `options`); `temperature 0`; `num_gpu 0` (CPU only). Batches of 6 candidates. Raise on a failed batch.
- Why CPU only: on the lab PC (AMD RX 550) partial GPU offload produced empty or garbage output. `num_gpu 0` fixed it. Full story in `docs/background.md`.
- The model's own defaults are temperature 0.6 and thinking on, so these options must always be passed.
- Set `num_ctx` explicitly for full-article work (8192 is a sensible start) and log `prompt_eval_count`. Not done yet.
- Lab PC now has Ollama **0.40.16**, `qwen3:8b` ID `500a1f067a9f` (Q4_K_M). The fix was tested on 0.40.11.
- A ChatGPT-generated patch (`reliable_extractor.py`, `daily_pipeline.py`, `ollama_smoke_test.py`) was **not adopted**. Do not build on it.

## 8. Current plan (in order)

1. **Done:** confirm and fix Ollama CPU inference.
2. **Done in code, unit-tested:** number-finder and validator fixes. **Remaining:** re-run `test_branch_b_pipeline.py` end to end.
3. **In progress:** claim-level gold set. Draft exists (279 rows, 20 test articles, 114 flagged `review`). The author corrects `impact_type`, `aggregation`, `unit` and saves `data/gold/gold_verified.csv` (does not exist yet). Plan: a second person independently checks ~20% of rows. Report unit accuracy on a checked sample, not the full rule-derived column.
4. **Next:** compare on the same articles: (A) NER baseline, (B) candidate-routed Qwen + validator, (C) optionally full-article Qwen. Metrics: impact-type P/R/F1, value accuracy, aggregation accuracy, unsupported/invented claims, missed claims, runtime, failed requests. Do not present baseline span-F1 (85.03%) and claim-level F1 as directly comparable.
5. Fix evaluator reporting (5.1); event-grouped split.
6. Add fire articles (71 pre-annotated, annotation pending in Label Studio, **VERIFY** count), retrain, extend keywords in `pipeline.py`.
7. Build what `docs/design.md` proposes, after the author answers its open questions: `extraction_service.py`, `articles` / `claims` / `events` collections, event matcher, reconciliation, retry with article status, OS scheduler instead of the `schedule` loop, then FastAPI + website with a Leaflet map.
8. Journal submission (suggested: IJDRR, IJGI/MDPI, or an NLP crisis workshop). Do not fine-tune the LLM until the zero-shot baseline is measured.

## 9. Things to avoid

- Fabricated confidence scores, fabricated citations, or claims of improvement without a held-out comparison.
- Using training-split articles for the evaluation set, or tuning anything on test-split articles.
- Letting the LLM do arithmetic or invent identifiers.
- Adding numbers together across articles or across claims.
- Merging `hybrid-local-llm` into `main` before Branch A + B are integrated and tested end to end.
- Writing to or deleting the `disaster_events` collection. New work goes in new collections.
