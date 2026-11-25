"""Accuracy of the impact numbers: XLM-R (old) vs the new system (XLM-R + Qwen).

Both are compared with the SAME human annotations (Label Studio export in a
.spacy file) on the SAME articles, with the SAME matching rule:

    a prediction is correct when its class equals the annotated class and its
    characters overlap the annotated span (one prediction per annotation).

Two steps, because Qwen is slow (~3 min per article on this CPU):

    python score_extraction.py run   [--limit 3]   # runs Qwen, saves its answers (resumable)
    python score_extraction.py score               # compares, prints the table, instant

`run` never writes to MongoDB. Answers and results go to data/eval/.
Nothing here may be used to tune the prompt: these are held-out test articles.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLD = os.path.join(BASE_DIR, "data", "annotations", "test.spacy")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
# The model behind the 85.03% result (same path as evaluator.py and entity_linker.py).
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmroberta_v3")

# The impact classes that were annotated for XLM-R. Qwen's other classes
# (INJURED, HOUSING_DAMAGE, ...) were never annotated here, so they are not scored.
SCORED_CLASSES = ["FATALITIES", "DISPLACED", "AFFECTED_PEOPLE", "MISSING", "RELIEF_INFO"]
QWEN_TO_GOLD = {"RELIEF": "RELIEF_INFO"}      # Qwen's name -> the annotation's name

MAIN_STATUSES = ("ACCEPTED",)                  # what the system really uses
WITH_REVIEW = ("ACCEPTED", "REVIEW_REQUIRED")  # shown as a second view


# ── Spans ─────────────────────────────────────────────────────────────────
# A span is a dict: start, end, label, text, and for predictions "kind"
# ("QUANTITY" for a number, "CUE" for a whole impact sentence) and "status".

def gold_spans(doc, classes=SCORED_CLASSES):
    return [{"start": e.start_char, "end": e.end_char, "label": e.label_, "text": e.text}
            for e in doc.ents if e.label_ in classes]


def qwen_spans(claims, statuses=MAIN_STATUSES, classes=SCORED_CLASSES):
    """Qwen claims -> predicted spans. Only the scored classes and the given
    validation statuses count. Headline claims have no position and are skipped."""
    spans = []
    for c in claims:
        label = QWEN_TO_GOLD.get(c["impact_type"], c["impact_type"])
        if label not in classes or c["validation_status"] not in statuses:
            continue
        if c.get("char_start") is None:
            continue
        spans.append({"start": c["char_start"], "end": c["char_end"], "label": label,
                      "text": c.get("raw_value") or "", "kind": c.get("candidate_kind", "QUANTITY")})
    return spans


def overlap(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def match(gold, predicted):
    """One-to-one matching within each class. Numbers are matched before whole
    cue sentences, and bigger overlaps first, so a cue sentence cannot take an
    annotation that a number in it already explains.
    Returns (pairs, unmatched_gold, unmatched_predicted)."""
    pairs, used_gold, used_pred = [], set(), set()
    for kind in ("QUANTITY", "CUE"):
        options = []
        for pi, p in enumerate(predicted):
            if pi in used_pred or p.get("kind", "QUANTITY") != kind:
                continue
            for gi, g in enumerate(gold):
                if gi not in used_gold and g["label"] == p["label"] and overlap(g, p) > 0:
                    options.append((-overlap(g, p), gi, pi))
        for _, gi, pi in sorted(options):
            if gi in used_gold or pi in used_pred:
                continue
            used_gold.add(gi)
            used_pred.add(pi)
            pairs.append((gold[gi], predicted[pi]))
    return (pairs,
            [g for i, g in enumerate(gold) if i not in used_gold],
            [p for i, p in enumerate(predicted) if i not in used_pred])


def exact_match(gold, predicted):
    """The original evaluator.py rule: identical start, end and label."""
    g = {(s["start"], s["end"], s["label"]) for s in gold}
    p = {(s["start"], s["end"], s["label"]) for s in predicted}
    return len(g & p), len(p - g), len(g - p)


# ── Metrics ───────────────────────────────────────────────────────────────
def prf(tp, fp, fn):
    """Precision, recall, F1 in percent. None (shown as n/a) when there is
    nothing to divide by: e.g. a class with no annotations has no recall."""
    p = tp / (tp + fp) * 100 if tp + fp else None
    r = tp / (tp + fn) * 100 if tp + fn else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (
        0.0 if p is not None and r is not None else None)
    return p, r, f


def score(articles, classes=SCORED_CLASSES, rule="overlap"):
    """articles: list of {"id", "gold": [spans], "pred": [spans]}.
    Returns per-class counts and metrics, micro and macro averages, and the
    list of mistakes."""
    # "inside": extra predictions within an annotation that is already matched;
    # neither right nor wrong, reported only.
    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "inside": 0} for c in classes}
    errors = []
    for art in articles:
        if rule == "exact":
            for c in classes:
                tp, fp, fn = exact_match([s for s in art["gold"] if s["label"] == c],
                                         [s for s in art["pred"] if s["label"] == c])
                counts[c]["tp"] += tp; counts[c]["fp"] += fp; counts[c]["fn"] += fn
            continue
        pairs, missed, extra = match(art["gold"], art["pred"])
        for g, _ in pairs:
            counts[g["label"]]["tp"] += 1
        for g in missed:
            counts[g["label"]]["fn"] += 1
            # What did the model say here instead (any class, any status)?
            said = [p["label"] for p in art.get("other", art["pred"]) if overlap(g, p) > 0]
            errors.append({"article": art["id"], "error": "missed", "annotated": g["label"],
                           "predicted": " / ".join(said) or "nothing", "text": g["text"],
                           "start": g["start"]})
        matched_gold = [g for g, _ in pairs]
        for p in extra:
            # Several numbers inside ONE annotated phrase ("1,700 packets of dry food and
            # 650 tons of rice"): the first matches, the others are not wrong either.
            if any(g["label"] == p["label"] and overlap(g, p) > 0 for g in matched_gold):
                counts[p["label"]]["inside"] += 1
                continue
            counts[p["label"]]["fp"] += 1
            said = [g["label"] for g in art["gold"] if overlap(g, p) > 0]
            errors.append({"article": art["id"], "error": "wrong", "annotated": " / ".join(said),
                           "predicted": p["label"], "text": p["text"][:120], "start": p["start"]})

    per_class = {}
    for c in classes:
        k = counts[c]
        p, r, f = prf(k["tp"], k["fp"], k["fn"])
        per_class[c] = {**k, "support": k["tp"] + k["fn"], "precision": p, "recall": r, "f1": f}
    total = {x: sum(counts[c][x] for c in classes) for x in ("tp", "fp", "fn", "inside")}
    mp, mr, mf = prf(total["tp"], total["fp"], total["fn"])
    measurable = [per_class[c]["f1"] for c in classes
                  if per_class[c]["support"] and per_class[c]["f1"] is not None]
    macro = sum(measurable) / len(measurable) if measurable else None
    return {"per_class": per_class,
            "micro": {**total, "support": total["tp"] + total["fn"],
                      "precision": mp, "recall": mr, "f1": mf},
            "macro_f1": macro, "errors": sorted(errors, key=lambda e: (e["article"], e["start"]))}


def every_answer(claims):
    """For the mistake list: everything Qwen said, any class and status, so a
    miss can show e.g. 'INJURED (ACCEPTED)' or 'FATALITIES (REJECTED)'."""
    return [{"start": c["char_start"], "end": c["char_end"],
             "label": f'{c["impact_type"]} ({c["validation_status"]})'}
            for c in claims if c.get("char_start") is not None]


# ── Loading the annotated test set ────────────────────────────────────────
def load_gold(path, nlp=None):
    import spacy
    from spacy.tokens import DocBin
    vocab = (nlp or spacy.blank("en")).vocab
    docs = list(DocBin().from_disk(path).get_docs(vocab))
    prefix = os.path.splitext(os.path.basename(path))[0]      # "test", "dev"
    return [{"id": f"{prefix}-{i + 1:02d}", "text": d.text,
             "sha": hashlib.sha1(d.text.encode("utf-8")).hexdigest()[:12], "doc": d}
            for i, d in enumerate(docs)]


def cache_path(name, what):
    return os.path.join(EVAL_DIR, f"{name}_{what}.json")


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)       # never leaves a half-written file behind


# ── Step 1: run the new system (slow) ─────────────────────────────────────
def run_qwen(gold_path, name, limit=None, model="qwen3:8b"):
    import entity_linker
    from extraction_service import (EXTRACTION_VERSION, PROMPT_VERSION, BackendUnavailable,
                                    OllamaBackend, extract_article)
    articles = load_gold(gold_path)
    path = cache_path(name, "qwen")
    cache = read_json(path, {"articles": {}})
    backend = OllamaBackend(model=model)
    version = {"extraction": EXTRACTION_VERSION, "prompt": PROMPT_VERSION, "model": model}
    if cache.get("version") and cache["version"] != version:
        raise SystemExit(f"{path} holds answers from {cache['version']}, not {version}. "
                         f"Use another --name or move that file away.")
    cache["version"] = version
    cache["llm"] = backend.describe()

    todo = [a for a in articles
            if cache["articles"].get(a["id"], {}).get("sha") != a["sha"]
            or cache["articles"][a["id"]].get("status") != "complete"]
    print(f"{len(articles)} test articles: {len(articles) - len(todo)} already done, "
          f"{len(todo)} still to do.")
    if limit:
        todo = todo[:limit]
        print(f"--limit {limit}: running {len(todo)} now.")
    for n, art in enumerate(todo, 1):
        started = time.perf_counter()
        try:
            # No title: these articles were annotated without their headlines.
            result = extract_article({"article_id": art["id"], "text": art["text"]}, backend,
                                     branch_a=entity_linker.extract_entities,
                                     branch_a_model=os.path.basename(entity_linker.MODEL_PATH))
        except BackendUnavailable as exc:
            raise SystemExit(f"Stopped: {exc}. Answers so far are saved in {path}.")
        cache["articles"][art["id"]] = {
            "sha": art["sha"], "status": result.status, "error": result.error,
            "seconds": round(result.seconds, 1), "candidates": result.candidate_count,
            "unanswered": result.unanswered,
            "claims": [{k: c[k] for k in ("candidate_id", "candidate_kind", "impact_type",
                                          "raw_value", "aggregation", "char_start", "char_end",
                                          "validation_status", "validation_flags")}
                       for c in result.claims]}
        cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(path, cache)
        print(f"  [{n}/{len(todo)}] {art['id']} {result.status:8} {result.candidate_count:3} "
              f"numbers  {time.perf_counter() - started:5.0f} s  | {art['text'][:50]!r}")
    print(f"Saved: {path}")


# ── XLM-R on its own (the old system), fast ───────────────────────────────
def run_xlmr(articles, name):
    """The plain XLM-R model, exactly as evaluator.py loads it (no gazetteer)."""
    import spacy
    path = cache_path(name, "xlmr")
    cache = read_json(path, {})
    if all(cache.get(a["id"], {}).get("sha") == a["sha"] for a in articles):
        return cache
    print("Running plain XLM-R on the test articles...")
    nlp = spacy.load(MODEL_PATH)
    cache = {}
    for art in articles:
        doc = nlp(art["text"])
        cache[art["id"]] = {"sha": art["sha"], "ents": [
            {"start": e.start_char, "end": e.end_char, "label": e.label_, "text": e.text}
            for e in doc.ents]}
    write_json(path, cache)
    return cache


def hybrid_spans(xlmr_pred, qwen_pred):
    """The fallback design: XLM-R decides the class of every number it marks;
    Qwen adds only numbers XLM-R did not mark at all. (In the real system
    Qwen would still give running total vs new, place, and the new classes.)"""
    added = [q for q in qwen_pred if not any(overlap(q, x) > 0 for x in xlmr_pred)]
    return list(xlmr_pred) + added


# ── Step 2: compare and report (instant once answers exist) ──────────────
def fmt(x):
    return "  n/a" if x is None else f"{x:5.1f}"


def report(gold_path, name):
    articles = load_gold(gold_path)
    qwen = read_json(cache_path(name, "qwen"), {"articles": {}})
    done = [a for a in articles
            if qwen["articles"].get(a["id"], {}).get("sha") == a["sha"]]
    if not done:
        raise SystemExit("No Qwen answers yet. Run:  python score_extraction.py run")
    if len(done) < len(articles):
        print(f"NOTE: Qwen has answered only {len(done)} of {len(articles)} articles. "
              f"Both models are scored on those {len(done)} only.")
    xlmr = run_xlmr(done, name)

    sets = {}
    for key, statuses in (("qwen", MAIN_STATUSES), ("qwen_with_review", WITH_REVIEW)):
        sets[key] = [{"id": a["id"], "gold": gold_spans(a["doc"]),
                      "pred": qwen_spans(qwen["articles"][a["id"]]["claims"], statuses),
                      "other": every_answer(qwen["articles"][a["id"]]["claims"])}
                     for a in done]
    sets["xlmr"] = [{"id": a["id"], "gold": gold_spans(a["doc"]),
                     "pred": [s for s in xlmr[a["id"]]["ents"] if s["label"] in SCORED_CLASSES]}
                    for a in done]

    sets["hybrid"] = [{"id": x["id"], "gold": x["gold"], "pred": hybrid_spans(x["pred"], q["pred"])}
                      for x, q in zip(sets["xlmr"], sets["qwen"])]

    results = {
        "xlmr": score(sets["xlmr"]),
        "qwen": score(sets["qwen"]),
        "hybrid": score(sets["hybrid"]),
        "qwen_with_review": score(sets["qwen_with_review"]),
        "xlmr_exact": score(sets["xlmr"], rule="exact"),
    }

    statuses = {}
    for a in done:
        s = qwen["articles"][a["id"]]["status"]
        statuses[s] = statuses.get(s, 0) + 1

    # The table
    line = "-" * 86
    print(f"\nACCURACY OF IMPACT NUMBERS  ({len(done)} test articles, {gold_path})")
    print("Rule: same class + overlapping characters. P = precision, R = recall, F1 in %.")
    print(line)
    print(f"{'Class':<17}{'Count':>6} | {'XLM-R  P     R    F1':>22} | "
          f"{'XLM-R + Qwen  P     R    F1':>29}")
    print(line)
    for c in SCORED_CLASSES:
        x, q = results["xlmr"]["per_class"][c], results["qwen"]["per_class"][c]
        print(f"{c:<17}{x['support']:>6} | {fmt(x['precision'])} {fmt(x['recall'])} {fmt(x['f1'])} | "
              f"       {fmt(q['precision'])} {fmt(q['recall'])} {fmt(q['f1'])}")
    print(line)
    x, q = results["xlmr"]["micro"], results["qwen"]["micro"]
    print(f"{'OVERALL (micro)':<17}{x['support']:>6} | {fmt(x['precision'])} {fmt(x['recall'])} "
          f"{fmt(x['f1'])} |        {fmt(q['precision'])} {fmt(q['recall'])} {fmt(q['f1'])}")
    print(f"{'Macro F1':<17}{'':>6} | {'':>11}{fmt(results['xlmr']['macro_f1'])} |        "
          f"{'':>12}{fmt(results['qwen']['macro_f1'])}")
    print(line)
    print("Hybrid (XLM-R decides the class, Qwen adds numbers XLM-R did not mark), F1:")
    print("   " + "   ".join(f"{c} {fmt(results['hybrid']['per_class'][c]['f1']).strip()}"
                             for c in SCORED_CLASSES))
    h = results["hybrid"]["micro"]
    print(f"   OVERALL: P {fmt(h['precision'])}  R {fmt(h['recall'])}  F1 {fmt(h['f1'])}   "
          f"macro F1 {fmt(results['hybrid']['macro_f1'])}")
    print(line)
    w, e = results["qwen_with_review"]["micro"], results["xlmr_exact"]["micro"]
    print(f"Qwen incl. 'needs review' answers: P {fmt(w['precision'])}  R {fmt(w['recall'])}  "
          f"F1 {fmt(w['f1'])}")
    print(f"XLM-R with the original exact rule: P {fmt(e['precision'])}  R {fmt(e['recall'])}  "
          f"F1 {fmt(e['f1'])}")
    print(f"Extra numbers inside an already matched annotation (not counted): "
          f"XLM-R {results['xlmr']['micro']['inside']}, Qwen {results['qwen']['micro']['inside']}")
    print(f"Qwen article runs: {statuses}  (unanswered numbers count as not found)")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = {"created": stamp, "gold": os.path.relpath(gold_path, BASE_DIR),
           "articles_scored": len(done), "articles_total": len(articles),
           "qwen_version": qwen.get("version"), "llm": qwen.get("llm"),
           "qwen_article_status": statuses, "rule": "same class + character overlap, one-to-one",
           "classes": SCORED_CLASSES,
           "results": {k: {kk: vv for kk, vv in v.items() if kk != "errors"}
                       for k, v in results.items()}}
    write_json(cache_path(name, "report"), out)
    errors_path = os.path.join(EVAL_DIR, f"{name}_errors.csv")
    with open(errors_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "article", "error", "annotated", "predicted", "text", "start"])
        for model in ("xlmr", "qwen"):
            for err in results[model]["errors"]:
                writer.writerow([model, err["article"], err["error"], err["annotated"],
                                 err["predicted"], err["text"], err["start"]])
    print(f"\nSaved: {cache_path(name, 'report')}\n       {errors_path}  (every mistake, both models)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["run", "score"])
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="annotated .spacy file")
    parser.add_argument("--name", default="flood_test", help="name used for the files in data/eval/")
    parser.add_argument("--limit", type=int, help="run: only this many articles now")
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args(argv)
    if args.step == "run":
        run_qwen(args.gold, args.name, args.limit, args.model)
    else:
        report(args.gold, args.name)


if __name__ == "__main__":
    main()
