# Runbook: how to run things, and current data formats

Moved out of `CLAUDE.md` on 2026-11-17. Run everything from the project root in PowerShell with the venv active.

## Environment on a new machine

```
git clone https://github.com/gmnayem631/disaster_db
cd disaster_db
git checkout hybrid-local-llm
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then copy from the author (not in Git): `articles/`, `models/xlmroberta_v3/`, and create `.env` with `MONGODB_URI=...`. `data/` is in Git.

Notes on `requirements.txt` (checked 2026-11-17): it is a full `pip freeze` saved as UTF-16, and it includes Label Studio and its Django dependencies. It does list the test tools `pytest` and `mongomock` (added 2026-11-17). `mongomock` is an in-memory imitation of MongoDB used only by the database tests.

For the LLM: install Ollama and `ollama pull qwen3:8b`. Verify with `python test_cpu.py`; `ollama ps` should show `100% CPU`.

## Tests

```
venv\Scripts\python -m pytest tests
```

To also run the test that talks to real Qwen (Ollama must be running; minutes on CPU):

```
$env:RUN_QWEN = "1"; venv\Scripts\python -m pytest tests -m qwen -s
```

To run the database tests against the real MongoDB server from `.env` instead of the in-memory imitation (about 2 minutes). They use only the database `disaster_db_test`, empty it before and after each test, and print the document counts of the real `disaster_db` before and after as a check:

```
$env:RUN_MONGO = "1"; venv\Scripts\python -m pytest tests/test_database_store.py tests/test_run_pending.py -s; Remove-Item Env:RUN_MONGO
```

Always pass `tests`. The root folder has scripts named `test_cpu.py`, `test_branch_b_pipeline.py`, `pipeline_test.py` and `ner_test.py`. They are not pytest tests, but plain `pytest` would collect them and try to load the model and call Ollama.

## Collect articles automatically (new system)

```
venv\Scripts\python ingestion.py --db disaster_db_test --limit 20
```

- Reads the newspapers in `data/newspapers.txt`. To add one, add a line `name | method | url` and do a small run like the one above to check it.
- Keeps articles that match a keyword in `data/hazard_keywords.txt`, stores them in the `articles` collection as `cleaned`, ready for `run_pending()`.
- `--db` has no default on purpose. Use `disaster_db_test` until the daily job is set up.
- Polite: obeys robots.txt and each site's crawl-delay, waits a random 5 to 15 seconds between requests to the same site, and stops using a site for the run if it answers 403, 429 or 503.
- A full daily run downloads several hundred pages and takes roughly an hour (estimate).

### Hazard check (Qwen)

Keywords let through many non-disasters ("fire safety clearance", "set tyres on fire").
`run_pending()` therefore first asks Qwen about each new daily article (`screening.py`):
title + first 400 words -> report type, in Bangladesh or not, hazards.
Kept: `CURRENT_EVENT`, `FOLLOW_UP`, `WARNING` in Bangladesh. Anything else becomes
`filtered_out` with the reason, and is not extracted. The answer is stored on the
article as `screening`. Corpus articles (`origin` not `daily`) are not checked.

## Daily job (collect + hazard check + extract)

```
venv\Scripts\python daily_run.py --db disaster_db_test
```

- Runs ingestion, then `run_pending()` (hazard check, Branch A, Qwen), then event matching (`event_matcher.py`), then event summaries (`event_summary.py`, only events that changed; seconds). One log file per run in `logs/daily_<date>_<time>.log`.
- Event summary: `events.summary.impacts.<TYPE>` is a list of entries, one per unit and place (`scope` "event" or "local"), each with `headline` (None if sources disagree), `low`, `high`, `display`, `as_of`, `sources`, `claim_ids`. Also `timeline` (INCREMENTAL claims), `qualitative`, `unverified` (with a reason), `not_counted`. Disagreements are in `events.conflicting_claims`. To rebuild every summary: `python -c "from database.store import Store, get_database; from event_summary import refresh_summaries; print(refresh_summaries(Store(get_database('disaster_db_test')), everything=True).summary())"`.
- Limits: `--scrape-limit 1000`, `--extract-limit 200`, `--extract-minutes 300` (no new extraction starts after 5 hours; the rest wait for tomorrow, no retry used).
- `--skip-ingestion` / `--skip-extraction` / `--skip-matching` leave a stage out.
- Event matching rules (time windows, spread vs spot hazards) are settings at the top of `event_matcher.py`. They are provisional until the author decides what counts as one event. Events marked `needs_review: true` need a person to look.
- Exit code 0 = all ran, 1 = a stage stopped or crashed (e.g. Ollama not running; nothing is lost), 2 = another run was still going.
- Ollama must be running (it starts with Windows when installed normally).

Schedule it (Windows Task Scheduler, every day at 06:00, only while you are logged in so Ollama is running).
Run once in PowerShell from the project folder:

```
$action  = New-ScheduledTaskAction -Execute "$PWD\venv\Scripts\python.exe" -Argument "daily_run.py --db disaster_db_test" -WorkingDirectory "$PWD"
$trigger = New-ScheduledTaskTrigger -Daily -At 6am
Register-ScheduledTask -TaskName "DisasterDB daily" -Action $action -Trigger $trigger
```

Check it: `Get-ScheduledTaskInfo -TaskName "DisasterDB daily"` (`LastTaskResult` is the exit code). Run it now: `Start-ScheduledTask -TaskName "DisasterDB daily"`. Remove it: `Unregister-ScheduledTask -TaskName "DisasterDB daily"`.
Change `--db` to the real database only when the author decides to go live.

## Collect and clean articles by hand (old way, used for the training corpus)

1. Append URLs to `data/urls.txt`.
2. `python scraper.py` (skips URLs already scraped).
3. `python preprocessor.py` (skips articles already cleaned).

## Annotate (Label Studio)

1. `python pre_annotator.py` writes `data/annotations/pre_annotated_<timestamp>.json` for unannotated articles. Import **that one file once** (project "Disaster NER Annotation"). Importing the same file twice creates duplicate tasks.
2. Correct the annotations in Label Studio (`label-studio start`, http://localhost:8080).
3. Export JSON to `data/annotations/annotations_export.json`.
4. `python convert_annotations.py` writes `train.spacy`, `dev.spacy`, `test.spacy` (70/15/15, seed 42). Labels listed in `IGNORE_LABELS` are dropped.

## Train and evaluate (Google Colab T4, no Google Drive)

1. Install: `pip install spacy spacy-transformers torch transformers`. Upload the three `.spacy` files and write the config (`xlm-roberta-base`, 2,000 steps, lr 5e-5, warmup 250, seed 42, `gpu_allocator = "pytorch"`).
2. `python -m spacy train config_bert.cfg --output /content/xlmroberta_vN --gpu-id 0`
3. `python -m spacy evaluate /content/xlmroberta_vN/model-best /content/test.spacy --gpu-id 0`
4. Zip `model-best`, download it **before closing the session** (an idle session wipes it), and unzip to `models/xlmroberta_vN/`.
5. Point `MODEL_PATH` in `entity_linker.py` and `evaluator.py` at it. `python evaluator.py` reads `data/annotations/test.spacy`.

**The XLM-R training config is not in the repo.** The file `config_bert.cfg` in the repo is a CNN config (20,000 steps, lr 0.001), despite its name. The settings actually used are inside `models/xlmroberta_v3/config.cfg`, which is not in Git. Copy it into the repo for reproducibility.

## Run the daily pipeline (baseline)

`python pipeline.py` runs once immediately, then every day at 09:00 via `schedule` (stop with Ctrl+C). It uses `entity_linker.extract_entities()` only, with flood keywords only. Branch B is not wired into it.

## Test Branch B

```
python extraction/branch_b/candidate_generator.py     # prints candidates for a built-in test text
python test_branch_b_pipeline.py                       # full Branch A + B run on one real article
```

Qwen on CPU takes minutes for an article with 25 candidates, so do not cancel early.

## Claim-level gold set

1. `python build_gold_draft.py` writes `data/gold/gold_draft.csv` from `test.spacy` articles (`MAX_ARTICLES = 20`). **It overwrites the file each run and does not skip articles already verified.** The current draft has 279 rows from 20 articles, 114 of them flagged `review`.
2. Correct `impact_type`, `aggregation`, `unit` in Excel. Rows with a `review` note need attention first.
3. Save as `data/gold/gold_verified.csv` (CSV UTF-8). Do not re-run step 1 over an edited draft.

## Data formats

**Raw / cleaned article JSON** (`articles/raw`, `articles/cleaned`): `title`, `text`, `publish_date` (mostly null), `source_url`, `scraped_at`, and `preprocessed_at` for cleaned files. Files are named by the MD5 hash of the URL.

**Current MongoDB document** (db `disaster_db`, collection `disaster_events`): `event_id` (a UUID per article), `title`, `source_url`, `publish_date`, `scraped_at`, `extraction_timestamp`, `disaster_type[]`, `locations{districts[], upazilas[], unions[]}`, `fatalities`, `displaced`, `affected_people`, `missing` (each `{value, source}`), `relief_info[]`, `agencies_involved[]`, `raw_entities[]`. `source` is `"model"` or `"rule_based"`. The article text is **not** stored. Dates are local-time strings.

**Branch B LLM output:** `{"claims": [{candidate_id, impact_type, evidence_sentence_ids, location_ids, aggregation, certainty, unit}]}`. Validated claims add `raw_value`, `normalized_quantity {raw, value, lower_bound, upper_bound, modifier, unit, qualitative}`, `validation_status` (ACCEPTED / REVIEW_REQUIRED / REJECTED) and `validation_flags`.

**Planned collections** (`articles`, `claims`, `events`): see `docs/design.md`.

**Gazetteer CSVs** (`data/gazetteer/`, no header row): `id, parent_id, name, bengali_name, [lat, lon for districts], website`.
