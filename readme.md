# Local Ollama disaster extraction patch

This is a **standalone experimental patch**, not a replacement for the published 85% F1 baseline. It keeps claims, evidence and original article metadata. It does not perform cross-article event reconciliation or certify semantic correctness of claims.

## Setup (Windows PowerShell, in your Python 3.11 environment)

```powershell
python -m pip install -U ollama 'pydantic>=2,<3' 'spacy>=3.7,<4' pytest
ollama --version
ollama list
ollama ps
ollama show --modelfile qwen3:8b
python ollama_smoke_test.py
python -m pytest -q tests
```

If smoke test fails, capture the returned `thinking_length`, `done_reason`, `eval_count`, Ollama version, and the relevant lines from `%LOCALAPPDATA%\\Ollama\\server.log`. Test the tiny schema before the article schema.

## Run on a cleaned article

From the original disaster_db repository root (so the baseline entity_linker can be found), copy `reliable_extractor.py` here, then:

```powershell
python reliable_extractor.py path\to\cleaned_article.json --model qwen3:8b --output extracted_article.json
```

Optional Branch A baseline output:

```powershell
python reliable_extractor.py path\to\cleaned_article.json --use-branch-a --output extracted_article.json
```

For full 500-word article on CPU, default context 8192, max output 2048. Longer articles automatically split into overlapping 550-word windows. If the schema response fails, one JSON-mode retry is attempted and that window is recursively split. Unrecoverable sections set extraction_status=INCOMPLETE and the process exits nonzero; **do not insert incomplete results as complete**.

Set `--max-words 350` for less CPU pressure, or `--num-ctx 4096` for short windows. Check `ollama_diagnostics.jsonl` for the reason for blank outputs. The accepted claims here only pass structural evidence checks; calibrate semantic accuracy against human-verified article claims before public deployment.

## Recommended deployment changes

1. Keep `main` untouched; use a separate feature branch.
2. Archive article JSON and all extracted claims with source URLs. Never sum cumulative numbers across repeated articles.
3. Store original NER span-F1 and new claim/field-level performance separately; evaluate both systems on the same gold articles.
4. Introduce an independent event matcher and claim reconciler **before** publishing national event totals.
5. Do not rely on the original `seen_urls.add(url)` before success; queue failed URLs for retry.
6. Pin Ollama version and model digest in experiment metadata; sample with a few fixed seeds; inspect omitted claims and unsupported evidence.

## One-shot daily processing (uses existing cleaner output)

From the hybrid branch root, with both `reliable_extractor.py` and `daily_pipeline.py` available:

```powershell
python daily_pipeline.py --cleaned-dir articles\cleaned --out-dir extracted_articles --limit 20
```

When validated on your own data, and after setting `MONGODB_URI`, write to **new** `articles` and `claims` collections without touching the old database:

```powershell
python -m pip install pymongo
python daily_pipeline.py --cleaned-dir articles\cleaned --out-dir extracted_articles --limit 20 --mongo
```

The daily runner skips unchanged articles already marked COMPLETE, retries failed or incomplete extractions, and does not treat one article as one event. Schedule the one-shot command via Windows Task Scheduler. Implement a separate event matcher/reconciler before creating events or publishing accumulated national totals. It does not automatically rescrape news - keep and schedule the existing scraper too.

Use `--use-branch-a` in the daily runner to load the existing `entity_linker.py` once and pass only its hazard, locations, and agency output as optional evidence hints to the local LLM:

```powershell
python daily_pipeline.py --cleaned-dir articles\cleaned --out-dir extracted_articles --limit 20 --use-branch-a --mongo
```

This continues to execute the old ten-label XLM-RoBERTa model internally. A future optimization is to train a stable-entities-only Branch A model; that is not included here. The model's location suggestions are **hints**, not gold truth.
