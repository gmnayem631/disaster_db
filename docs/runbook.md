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

Always pass `tests`. The root folder has scripts named `test_cpu.py`, `test_branch_b_pipeline.py`, `pipeline_test.py` and `ner_test.py`. They are not pytest tests, but plain `pytest` would collect them and try to load the model and call Ollama.

## Collect and clean articles

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
