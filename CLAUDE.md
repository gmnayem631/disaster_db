# Project context for AI assistants

Read this fully before changing anything. Last updated: November 2026.
Items marked **VERIFY** were stated in conversation but not confirmed in the repo. Check them before relying on them.

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
- Windows machine. Use **PowerShell** with the venv (`venv\Scripts\activate`). An MSYS2 bash shell on the lab PC resolved `python` to a different interpreter with no pip. Run scripts from the project root. Python 3.11.9.
- Never commit secrets. `.env` holds `MONGODB_URI` and is gitignored.

## 3. Repository layout

Committed to Git: code, `requirements.txt`, `README.md`.
**Not in Git (kept local or copied by pen drive):** `articles/`, `data/`, `models/`, `.env`, `venv/`, `logs/`.

```
scraper.py                 reads data/urls.txt, newspaper3k, saves JSON to articles/raw/
preprocessor.py            cleans text, flood/fire keyword filter, min length -> articles/cleaned/
gazetteer_builder.py       builds data/gazetteer/bangladesh_gazetteer.json from CSVs
gazetteer_ner.py           early gazetteer experiment (not used in production)
label_studio_import.py     original 90 articles -> Label Studio tasks
pre_annotator.py           runs the NER model on new articles to pre-annotate for Label Studio
convert_annotations.py     Label Studio export -> train/dev/test .spacy (70/15/15, seed 42)
entity_linker.py           PRODUCTION extractor (Branch A style): XLM-R + gazetteer + regex fallback
db_writer.py               MongoDB insert with duplicate check on source_url
pipeline.py                scrape -> clean -> extract -> MongoDB, run daily with `schedule`
evaluator.py               per-entity P/R/F1 on a .spacy file + timing
test_cpu.py                Ollama CPU-only sanity test (plain + JSON)
test_branch_b_pipeline.py  runs Branch A + candidate generator + Qwen + validator on one article
build_gold_draft.py        drafts a claim-level evaluation CSV from test-split annotations
gold_helpers.py            guesses unit/aggregation for the draft CSV

extraction/branch_b/       (hybrid-local-llm branch)
  schemas.py               Pydantic: ImpactType, AggregationType, CertaintyType, ImpactClaim, BranchBResult, NormalizedQuantity
  impact_cues.py           keyword lists: impact, negation, historical, forecast
  quantity_normalizer.py   deterministic number parsing (lakh/crore, word numbers, ranges, modifiers)
  candidate_generator.py   sentence split + finds every number and impact-cue sentence
  local_llm.py             Ollama call (qwen3:8b), structured JSON output, batching
  claim_validator.py       deterministic checks on the LLM's claims
```

Empty folders exist for the planned restructure (`ingestion/`, `preprocessing/`, `events/`, `database/`, `api/`, `evaluation/`, `tests/`). Existing code has not been moved into them. Branch-B scripts use a `sys.path.insert` hack so they run from any folder.

## 4. Baseline system (`main`): what exists and how well it works

Pipeline: scraper -> preprocessor -> `entity_linker.extract_entities()` -> MongoDB (`disaster_events`, db `disaster_db`).

- **Model:** XLM-RoBERTa base fine-tuned for NER with spaCy 3.8 + spacy-transformers, config trained on Colab T4 (2,000 steps, lr 5e-5, Adam, warmup 250). Local copy at `models/xlmroberta_v3/` (not in Git).
- **Gazetteer:** 64 districts, 483 upazilas, 4,060 unions from `nuhil/bangladesh-geocode` (and OCHA COD-AB for reference), added as an `EntityRuler` patterns (9,214 patterns incl. lowercase). It was originally missing from the production extractor and was wired into `entity_linker.py` later.
- **10 labels:** DISASTER_TYPE, BD_DISTRICT, BD_UPAZILA, BD_UNION, FATALITIES, DISPLACED, AFFECTED_PEOPLE, MISSING, RELIEF_INFO, AGENCIES_INVOLVED. (INJURED, EVENT_DATE, SECONDARY_HAZARD, INFRASTRUCTURE_DAMAGED, CROPS_AFFECTED, HOUSES_DAMAGED were dropped: too rare or inconsistent.)
- **Hybrid inside Branch A:** if the model finds no number for fatalities/displaced/affected, regex patterns try (`source: "rule_based"` vs `"model"`). Rule-based contribution was small (~3%).
- **Dataset:** 276 annotated flood articles (Daily Star, Dhaka Tribune, New Age, bdnews24, plus UNB/Bangladesh Post), split 193 train / 41 dev / 42 test, **randomly by article, not by disaster event**.

**Current numbers (`models/xlmroberta_v3`, evaluated on `test.spacy`, 42 articles):** overall F1 **85.03%**, ~2.5 s/article.

| Entity            | F1    |     | Entity          | F1    |
| ----------------- | ----- | --- | --------------- | ----- |
| BD_DISTRICT       | 96.51 |     | AFFECTED_PEOPLE | 68.35 |
| DISASTER_TYPE     | 95.29 |     | MISSING         | 63.64 |
| BD_UPAZILA        | 85.59 |     | DISPLACED       | 62.96 |
| AGENCIES_INVOLVED | 78.02 |     | BD_UNION        | 61.90 |
| FATALITIES        | 74.68 |     | RELIEF_INFO     | 39.71 |

**History worth knowing (for honest reporting):**

- Model comparison on the first split (220/56, before the bug fix below): CNN 60.47 (72 articles), DistilBERT 67.05 (72), RoBERTa 78.16, DistilBERT 77.37, mBERT 77.52, **XLM-RoBERTa 79.21** (all 220). The submitted report and defense used these.
- **Bug found and fixed:** `convert_annotations.py` used `doc.char_span()` with strict alignment and silently dropped ~45% of annotations. Fixed with `alignment_mode="contract"` then `"expand"`; training spans rose from about 5,197 to 7,290. Overall F1 rose to 85.12% (dev, v2) and 85.03% (test, v3). **The model-architecture ranking was not re-run after this fix.** Re-validate before claiming XLM-R is best.
- The submitted report still contains the older 79.21% figures. They need updating.

## 5. Known problems in the baseline (not yet fixed)

1. `evaluator.py` hard-codes `"dev_set_size": 56` in saved results, but the 85.03% was measured on the 42-article test set. The score looks right, but the label is wrong. Rerun and correct before submission.
2. Split is random by article. Reports of the same flood can land in train and test. Needs an event-grouped split for the paper.
3. `publish_date` is `null` for most articles. A meta-tag fallback exists in `pipeline.py` but the 276 training articles have no dates. `scraped_at` is stored instead.
4. `pipeline.py` adds URLs to `seen_urls` _before_ scraping/extraction succeeds, so failed articles are never retried.
5. Each article becomes its own MongoDB "event" (UUID per article). Several articles about one flood are not linked.
6. Number handling in `entity_linker.py` picks one value per field with a keyword heuristic for "cumulative" phrases. It loses incremental vs cumulative information and has no location link or evidence sentence.
7. No true per-entity confidence is available from spaCy transformers. The `source` field (`model`/`rule_based`) is used instead. **Do not fabricate confidence scores.**
8. `requirements.txt` may be incomplete. **VERIFY** and rebuild cleanly.
9. Label Studio has duplicate tasks (about 531) because a pre-annotated file was imported more than once. The author chose not to delete them (annotations would be lost). Use timestamped import files from now on.

## 6. Branch B (`hybrid-local-llm`): the redesign in progress

**Why:** NER is weak on impact numbers (what does "31", "58 lakh", "10,000" refer to?). Idea: let conventional code find candidates, let a local LLM decide only what each number _means_, then validate deterministically.

```
article -> Branch A (XLM-R + gazetteer: hazard, locations, agencies)
        -> candidate_generator (every number + impact-cue sentences, with sentence IDs)
        -> local LLM (Qwen3 8B via Ollama): impact_type, aggregation, certainty, unit, location IDs
        -> claim_validator (deterministic checks)
        -> [planned] store claims with evidence -> match to events -> MongoDB
```

Design rules (keep):

- The LLM **never converts numbers** and never invents IDs/URLs. Python does lakh/crore, word numbers, ranges, modifiers.
- Preserve raw text, evidence sentence, and individual claims. Never sum cumulative figures. Keep conflicting reports.
- Impact taxonomy: FATALITIES, INJURED, MISSING, AFFECTED_PEOPLE, DISPLACED, EVACUATED, MAROONED, HOUSING_DAMAGE, INFRASTRUCTURE_DAMAGE, AGRICULTURAL_DAMAGE, LIVESTOCK_LOSS, ECONOMIC_LOSS, RELIEF, OTHER. Aggregation: CUMULATIVE, INCREMENTAL, POINT_ESTIMATE, RANGE, LOWER_BOUND, UPPER_BOUND, UNKNOWN.

**What works (tested):** quantity normalizer, candidate generator (incl. Bengali commas like `12,27,554`), validator checks (invalid candidate/evidence/location IDs, unit compatibility, negation/historical/forecast flags), end-to-end run on a 25-candidate article with all batches completing.

**Recent changes** (given to the author on the date above; **VERIFY** they are applied and tested):

- Normalizer: range bounds (`lower_bound`/`upper_bound`), `"5, 40,510"` -> 540510, whole word-number phrases.
- Candidate generator: whole word-number phrases ("one hundred and twenty five"), stray-space Bengali numbers.
- Validator: rejects people-count claims where the number is followed by a non-person noun (districts, upazilas, shelter centres, animals, teams...).
- `build_gold_draft.py` + `gold_helpers.py`: draft evaluation CSV.

**Known remaining weaknesses:**

- Qwen can still give a plausible-looking wrong meaning (e.g. "people took shelter" labelled RELIEF). The validator cannot catch semantic errors. This is what the gold comparison measures.
- Candidate routing cannot recover a claim the generator never surfaces. Cue-only candidates are created only for sentences with no number (so "houses washed away" next to a number can be missed). Deliberately deferred until recall is measured.
- Years like "2022" become candidates (labelled OTHER by the LLM).

## 7. Local LLM setup and the empty-response incident

- Lab PC: Windows 11, 32 GB RAM, Xeon W-1250, **AMD Radeon RX 550 (4 GB)**. Dev laptop: 8 GB RAM, no usable GPU.
- Ollama 0.40.11, model `qwen3:8b`. Ollama partly offloaded the model to the RX 550 (59% CPU / 41% GPU). The model then produced garbage (`0000...`) and the server log showed `Unexpected empty grammar stack after accepting piece`, which surfaced in Python as "Empty response from model" / JSON EOF.
- **Fix confirmed by test:** `options={"num_gpu": 0}` (CPU only). `test_cpu.py` passed 6/6 (plain + JSON) and `ollama ps` showed `100% CPU`. Then the full 25-candidate article completed all batches. The GPU explanation is strongly supported but not formally proven. A cross-check (a second AI reviewer) said the grammar error could also come from schema/template interactions.
- Required call shape: `think=False` as a top-level argument to `ollama.chat()` (not in `options`); `temperature 0`; `num_gpu 0`. Batches of 6 candidates. `local_llm.py` must raise on a failed batch and not silently skip it (**VERIFY** this edit is in).
- Context: Ollama showed `num_ctx` 4096. For full-article tests set `num_ctx` explicitly (8192 is a sensible start) and log `prompt_eval_count` so silent truncation is detected.
- A ChatGPT-generated patch (`reliable_extractor.py`, `daily_pipeline.py`, `ollama_smoke_test.py`) exists outside the repo. **It was not adopted.** It duplicates Branch B and has no `num_gpu` setting.
- Record Ollama version and `ollama show qwen3:8b` output for reproducibility.

## 8. Current plan (in order)

1. **Done:** confirm and fix Ollama CPU inference.
2. **In progress:** number-finder and validator fixes (section 6). Verify they pass `test_branch_b_pipeline.py`.
3. **In progress:** claim-level gold set. `build_gold_draft.py` drafts rows from `test.spacy` articles (default 20 articles, ~280 rows) into `data/gold/gold_draft_<timestamp>.csv`. The author corrects `impact_type`, `aggregation`, `unit` (only `review`-flagged rows need real attention) and saves as `data/gold/gold_verified.csv`. The script skips articles already in that file. Plan: a second person independently checks ~20% of rows. Unit accuracy should be reported on a checked sample, not the full rule-derived column.
4. **Next:** compare on the same articles: (A) NER baseline, (B) candidate-routed Qwen + validator, (C) optionally full-article Qwen. Metrics: impact-type P/R/F1, value accuracy, aggregation accuracy, unsupported/invented claims, missed claims, runtime, failed requests. Do not present baseline span-F1 (85.03%) and claim-level F1 as directly comparable.
5. Fix evaluator reporting (section 5.1); event-grouped split.
6. Add fire articles (71 pre-annotated, annotation pending in Label Studio), retrain, and extend keywords in `pipeline.py`.
7. Then: MongoDB `articles` / `claims` / `events` collections (design agreed, not built), event matcher, cross-source reconciliation, retry queue with article status, OS scheduler (Task Scheduler/cron) instead of the infinite `schedule` loop, FastAPI + website with a Leaflet map.
8. Journal submission (suggested: IJDRR, IJGI/MDPI, or an NLP crisis workshop). Do not fine-tune the LLM until the zero-shot baseline is measured.

## 9. Deployment idea

Only the daily extraction job needs the LLM. Run it on a PC the lab controls on a schedule. The database (MongoDB Atlas) and the website can be in the cloud. Atlas free tier (512 MB) is far more than needed. Expected about 10 to 20 relevant articles per day.

## 10. Achievements so far

- Built an end-to-end automated pipeline: scrape -> clean -> extract -> MongoDB, with duplicate detection. First automated run inserted 22 records from 4 newspapers.
- Built and manually annotated a 276-article Bangladeshi flood corpus (10 entity types) in Label Studio.
- Built a Bangladesh administrative gazetteer (64 / 483 / 4,060) and integrated it into extraction.
- Trained and compared five architectures; reached **85.03% overall F1** on a held-out test set (BD_DISTRICT 96.5, DISASTER_TYPE 95.3, BD_UPAZILA 85.6).
- Found and fixed a silent data-loss bug in annotation conversion (about 2,093 spans recovered).
- Diagnosed a local-LLM failure (Ollama GPU offload on an unsupported AMD card) with server-log evidence and fixed it with CPU-only inference.
- Built the Branch B toolkit: deterministic quantity normalizer (lakh/crore/words/ranges), candidate generator, structured-output LLM call, and a validator with evidence, unit, negation, historical and forecast checks.
- Started the evaluation dataset for claim-level comparison.

## 11. Potential challenges and proposed solutions

| Challenge                                                             | Why it matters                        | Proposed solution                                                                                                                        |
| --------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Few examples of rare entities (MISSING, BD_UNION, RELIEF_INFO)        | Low F1 regardless of model            | More data, other hazards; report per-class results honestly; consider merging classes                                                    |
| Annotation inconsistency (span boundaries, labels)                    | Noisy training and evaluation         | Written guidelines, second annotator on a 20% sample, report agreement                                                                   |
| Number meaning: cumulative vs incremental vs different scope          | Wrong values in the database          | Branch B claims with `aggregation`; never sum across articles; keep all claims                                                           |
| LLM gives a plausible but wrong meaning or invents a value            | Research data would be untrustworthy  | Python converts numbers, validator checks evidence/units, gold-set comparison; no fake confidence scores                                 |
| CPU-only inference is slow (minutes per article with many candidates) | Daily run must finish                 | Batching, keep model loaded, run overnight, measure real time per article; compare against a full-article call; consider a smaller model |
| Candidate generator misses a real claim                               | The LLM never sees it                 | Measure recall against the gold set; add cue candidates for sentences that also contain numbers if recall is low                         |
| Many articles describe one event                                      | Database overcounts events            | Event matcher (hazard + date + district overlap + embedding similarity), thresholds tuned on labelled article pairs                      |
| Missing publication dates                                             | No timeline analysis                  | Parse HTML meta tags and datelines; fall back to `scraped_at` and label it as such                                                       |
| Scraper breaks when a site changes                                    | Silent data gaps                      | Prefer RSS where available, log per-source counts, alert on zero results, retry queue and article status                                 |
| Lab PC must be on at run time                                         | Missed days                           | OS scheduler plus retry of failed/missed articles; cloud runner later                                                                    |
| Random train/test split leaks same-event articles                     | Inflated scores                       | Event-grouped split before the paper                                                                                                     |
| Reproducibility                                                       | Reviewers will ask                    | Record versions, seeds, prompt/schema versions, `ollama show` digest; save the Colab training config in the repo                         |
| Journal rejection                                                     | Likely at least once                  | Honest evaluation, state limitations, use reviewer feedback; keep a clean reproducible repo                                              |
| Adding fire and other hazards                                         | Labels and keywords differ per hazard | Add hazard-specific cues and labels, evaluate per hazard and combined, do not mix silently                                               |

## 12. How to run things

Run everything from the project root in PowerShell with the venv active.

**Environment on a new machine**

```
git clone https://github.com/gmnayem631/disaster_db
cd disaster_db
git checkout hybrid-local-llm
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then copy from the author (not in Git): `articles/`, `data/`, `models/xlmroberta_v3/`, and create `.env` with `MONGODB_URI=...`. For the LLM: install Ollama and `ollama pull qwen3:8b`. Verify with `python test_cpu.py`; `ollama ps` should show `100% CPU`.

**Collect and clean articles**

1. Append URLs to `data/urls.txt`.
2. `python scraper.py` (skips URLs already scraped).
3. `python preprocessor.py` (skips articles already cleaned).

**Annotate (Label Studio)**

1. `python pre_annotator.py` writes a pre-annotated JSON of unannotated articles. Import **that one file once** (project "Disaster NER Annotation"). Importing the same file twice creates duplicate tasks.
2. Correct the annotations in Label Studio (`label-studio start`, http://localhost:8080).
3. Export JSON to `data/annotations/annotations_export.json`.
4. `python convert_annotations.py` writes `train.spacy`, `dev.spacy`, `test.spacy`. Labels listed in `IGNORE_LABELS` are dropped.

**Train and evaluate (Google Colab T4, no Google Drive)**

1. Install: `pip install spacy spacy-transformers torch transformers`. Upload the three `.spacy` files and write the config (`xlm-roberta-base`, 2,000 steps, lr 5e-5, seed 42, `gpu_allocator = "pytorch"`).
2. `python -m spacy train config_bert.cfg --output /content/xlmroberta_vN --gpu-id 0`
3. `python -m spacy evaluate /content/xlmroberta_vN/model-best /content/test.spacy --gpu-id 0`
4. Zip `model-best`, download it **before closing the session** (an idle session wipes it), and unzip to `models/xlmroberta_vN/`.
5. Point `MODEL_PATH` in `entity_linker.py` and `evaluator.py` at it. `python evaluator.py` reads `data/annotations/test.spacy`.

The Colab training config is not stored in the repo. Save a copy in the repo for reproducibility.

**Run the daily pipeline (baseline)**: `python pipeline.py` runs once immediately, then every day at 09:00 via `schedule` (stop with Ctrl+C). It uses `entity_linker.extract_entities()` only. Branch B is not wired into it yet.

**Test Branch B**

```
python extraction/branch_b/candidate_generator.py     # prints candidates for a built-in test text
python test_branch_b_pipeline.py                       # full Branch A + B run on one real article
```

Qwen on CPU takes minutes for an article with 25 candidates, so do not cancel early.

**Claim-level gold set**

1. `python build_gold_draft.py` writes `data/gold/gold_draft_<timestamp>.csv` from `test.spacy` articles (`MAX_ARTICLES = 20`).
2. Correct `impact_type`, `aggregation`, `unit` in Excel. Rows with a `review` note need attention first.
3. Save as `data/gold/gold_verified.csv` (CSV UTF-8). Later runs skip articles already in this file.

## 13. Data formats and schemas

**Raw / cleaned article JSON** (`articles/raw`, `articles/cleaned`): `title`, `text`, `publish_date` (mostly null), `source_url`, `scraped_at`, and `preprocessed_at` for cleaned files. Files are named by the MD5 hash of the URL.

**Current MongoDB document** (db `disaster_db`, collection `disaster_events`): `event_id` (a UUID per article), `title`, `source_url`, `publish_date`, `scraped_at`, `extraction_timestamp`, `disaster_type[]`, `locations{districts[], upazilas[], unions[]}`, `fatalities`, `displaced`, `affected_people`, `missing` (each `{value, source}`), `relief_info[]`, `agencies_involved[]`, `raw_entities[]`. `source` is `"model"` or `"rule_based"`.

**Planned collections (design agreed, not built):**

- `articles`: article_id, source, url, title, publish_date, scraped_at, content_hash, text, processing_status, retry_count, last_error, event_id, extraction metadata (model names, prompt/schema versions).
- `claims`: claim_id, article_id, event_id, impact_type, raw_value, normalized_value (or lower/upper bound), unit, modifier, aggregation, certainty, location_ids, evidence_sentence, extractor, validation_status and flags.
- `events`: event_id, hazard_type, start/end date, locations, article_ids, current_summary, conflicting_claims, last_updated. The website should show events, not articles.

**Branch B LLM output:** `{"claims": [{candidate_id, impact_type, evidence_sentence_ids, location_ids, aggregation, certainty, unit}]}`. Validated claims add `normalized_quantity {raw, value, lower_bound, upper_bound, modifier, unit, qualitative}`, `validation_status` (ACCEPTED / REVIEW_REQUIRED / REJECTED) and `validation_flags`.

## 14. Glossary

- **NER**: finding and labelling spans of text (a district name, a death count phrase).
- **Span**: a start and end position in the text. **F1**: the balance of precision and recall.
- **Gazetteer**: a list of known place names used to help recognise locations.
- **Candidate**: any number or impact phrase that might matter. Code finds them; it does not decide their meaning.
- **Claim**: one impact statement from an article (type, value, unit, evidence).
- **Cumulative / incremental**: a running total ("toll rose to 31") versus a new addition ("three more died").
- **Lakh / crore**: South Asian units, 100,000 and 10,000,000. Bengali digit grouping looks like `12,27,554` (= 1,227,554).
- **Upazila / union / district**: Bangladesh administrative levels (district > upazila > union).
- **Branch A / Branch B**: stable-entity extraction (XLM-R + gazetteer) / impact-claim extraction (LLM + validator). Not Git branches.

## 15. Things to avoid

- Fabricated confidence scores, fabricated citations, or claims of improvement without a held-out comparison.
- Using training-split articles for the evaluation set.
- Letting the LLM do arithmetic or invent identifiers.
- Merging `hybrid-local-llm` into `main` before Branch A + B are integrated and tested end to end.
