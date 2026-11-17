# Background: history, the Ollama incident, achievements, risks, glossary

Moved out of `CLAUDE.md` on 2026-11-17. This is reference material for writing the paper and for understanding why things are the way they are.

## Baseline history (for honest reporting)

- **Training setup:** XLM-RoBERTa base fine-tuned for NER with spaCy 3.8 + spacy-transformers on Colab T4 (2,000 steps, lr 5e-5, Adam, warmup 250, seed 42).
- **Model comparison on the first split (220/56, before the bug fix below):** CNN 60.47 (72 articles), DistilBERT 67.05 (72), RoBERTa 78.16, DistilBERT 77.37, mBERT 77.52, **XLM-RoBERTa 79.21** (all 220). The submitted report and defense used these.
- **Bug found and fixed:** `convert_annotations.py` used `doc.char_span()` with strict alignment and silently dropped about 45% of annotations. Fixed with `alignment_mode="contract"` then `"expand"`; training spans rose from about 5,197 to 7,290. Overall F1 rose to 85.12% (dev, v2) and 85.03% (test, v3). **The model-architecture ranking was not re-run after this fix.** Re-validate before claiming XLM-R is best.
- The submitted report still contains the older 79.21% figures. They need updating.
- Dropped labels (too rare or inconsistent): INJURED, EVENT_DATE, SECONDARY_HAZARD, INFRASTRUCTURE_DAMAGED, CROPS_AFFECTED, HOUSES_DAMAGED.
- The gazetteer was originally missing from the production extractor and was wired into `entity_linker.py` later.
- Rule-based fallback contribution: 1 of 37 numeric fields on a 20-article sample (`data/evaluation_results.json`), about 3%.
- Label Studio has about 531 duplicate tasks because a pre-annotated file was imported more than once. The author chose not to delete them (annotations would be lost). `pre_annotator.py` now writes timestamped files.

## Local LLM setup and the empty-response incident

- Lab PC: Windows 11, 32 GB RAM, Xeon W-1250, **AMD Radeon RX 550 (4 GB)**. Dev laptop: 8 GB RAM, no usable GPU.
- At the time of the incident: Ollama 0.40.11, model `qwen3:8b`. Ollama partly offloaded the model to the RX 550 (59% CPU / 41% GPU). The model then produced garbage (`0000...`) and the server log showed `Unexpected empty grammar stack after accepting piece`, which surfaced in Python as "Empty response from model" / JSON EOF.
- **Fix confirmed by test:** `options={"num_gpu": 0}` (CPU only). `test_cpu.py` passed 6/6 (plain + JSON) and `ollama ps` showed `100% CPU`. Then the full 25-candidate article completed all batches. The GPU explanation is strongly supported but not formally proven. A second AI reviewer said the grammar error could also come from schema/template interactions.
- A ChatGPT-generated patch (`reliable_extractor.py`, `daily_pipeline.py`, `ollama_smoke_test.py`) exists outside the repo. **It was not adopted.** It duplicates Branch B and has no `num_gpu` setting. Its README was committed over the project README by mistake in `fbde06a`.

**Reproducibility record, lab PC, 2026-11-17:**

```
ollama version 0.40.16        (was 0.40.11 during the incident; the fix has not been re-tested on 0.40.16)
qwen3:8b   ID 500a1f067a9f   5.2 GB
architecture qwen3, parameters 8.2B, quantization Q4_K_M, context length 40960
model defaults: temperature 0.6, top_k 20, top_p 0.95, thinking default true
python ollama client 0.6.3, pydantic 2.12.5, pymongo 4.17.0, spaCy 3.8.13
venv Python 3.13.15
```

The model defaults matter: the code must keep passing `temperature 0` and `think=False`, because the defaults are 0.6 and thinking on.

## Achievements so far

- Built an end-to-end automated pipeline: scrape -> clean -> extract -> MongoDB, with duplicate detection. First automated run inserted 22 records from 4 newspapers.
- Built and manually annotated a 276-article Bangladeshi flood corpus (10 entity types) in Label Studio.
- Built a Bangladesh administrative gazetteer and integrated it into extraction.
- Trained and compared five architectures; reached **85.03% overall F1** on a held-out test set (BD_DISTRICT 96.5, DISASTER_TYPE 95.3, BD_UPAZILA 85.6).
- Found and fixed a silent data-loss bug in annotation conversion (about 2,093 spans recovered).
- Diagnosed a local-LLM failure (Ollama GPU offload on an unsupported AMD card) with server-log evidence and fixed it with CPU-only inference.
- Built the Branch B toolkit: deterministic quantity normalizer (lakh/crore/words/ranges), candidate generator, structured-output LLM call, and a validator with evidence, unit, negation, historical and forecast checks. 12 pytest regression tests.
- Started the evaluation dataset for claim-level comparison (279 draft rows from 20 test articles).

## Potential challenges and proposed solutions

| Challenge | Why it matters | Proposed solution |
| --- | --- | --- |
| Few examples of rare entities (MISSING, BD_UNION, RELIEF_INFO) | Low F1 regardless of model | More data, other hazards; report per-class results honestly; consider merging classes |
| Annotation inconsistency (span boundaries, labels) | Noisy training and evaluation | Written guidelines, second annotator on a 20% sample, report agreement |
| Number meaning: cumulative vs incremental vs different scope | Wrong values in the database | Branch B claims with `aggregation`; never sum across articles; keep all claims |
| LLM gives a plausible but wrong meaning or invents a value | Research data would be untrustworthy | Python converts numbers, validator checks evidence/units, gold-set comparison; no fake confidence scores |
| CPU-only inference is slow (minutes per article with many candidates) | Daily run must finish | Batching, keep model loaded, run overnight, measure real time per article; compare against a full-article call; consider a smaller model |
| Candidate generator misses a real claim | The LLM never sees it | Measure recall against the gold set; add cue candidates for sentences that also contain numbers if recall is low |
| Many articles describe one event | Database overcounts events | Event matcher, thresholds tuned on labelled article pairs (`docs/design.md` section 3) |
| Missing publication dates | No timeline analysis, no event matching | Parse HTML meta tags and datelines; fall back to `scraped_at` and label it as such |
| Scraper breaks when a site changes | Silent data gaps | Prefer RSS where available, log per-source counts, alert on zero results, retry queue and article status |
| Lab PC must be on at run time | Missed days | OS scheduler plus retry of failed/missed articles; cloud runner later |
| Random train/test split leaks same-event articles | Inflated scores | Event-grouped split before the paper |
| Reproducibility | Reviewers will ask | Record versions, seeds, prompt/schema versions, `ollama show` digest; save the training config in the repo |
| Journal rejection | Likely at least once | Honest evaluation, state limitations, use reviewer feedback; keep a clean reproducible repo |
| Adding fire and other hazards | Labels and keywords differ per hazard | Add hazard-specific cues and labels, evaluate per hazard and combined, do not mix silently |

## Deployment idea

Only the daily extraction job needs the LLM. Run it on a PC the lab controls on a schedule. The database (MongoDB Atlas) and the website can be in the cloud. Atlas free tier is 512 MB. Expected about 10 to 20 relevant articles per day.

## Glossary

- **NER**: finding and labelling spans of text (a district name, a death count phrase).
- **Span**: a start and end position in the text. **F1**: the balance of precision and recall.
- **Gazetteer**: a list of known place names used to help recognise locations.
- **Candidate**: any number or impact phrase that might matter. Code finds them; it does not decide their meaning.
- **Claim**: one impact statement from an article (type, value, unit, evidence).
- **Cumulative / incremental**: a running total ("toll rose to 31") versus a new addition ("three more died").
- **Lakh / crore**: South Asian units, 100,000 and 10,000,000. Bengali digit grouping looks like `12,27,554` (= 1,227,554).
- **Upazila / union / district**: Bangladesh administrative levels (district > upazila > union).
- **Branch A / Branch B**: stable-entity extraction (XLM-R + gazetteer) / impact-claim extraction (LLM + validator). Not Git branches.
- **Backend**: the piece of code that talks to one particular LLM. Swapping the backend changes the model without changing the rest.
