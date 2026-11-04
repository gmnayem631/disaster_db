"""
Build a DRAFT gold-standard CSV from your existing Label Studio annotations.
You then correct the draft in Excel. Nothing here is final truth.
Only articles from the held-out test split (test.spacy) are used.
"""
import csv
import json
import os
import sys

import spacy
from spacy.tokens import DocBin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extraction.branch_b.candidate_generator import generate_candidates
from gold_helpers import guess_unit, guess_aggregation, review_reason

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EXPORT_FILE = os.path.join(BASE_DIR, "data", "annotations", "annotations_export.json")
TEST_FILE   = os.path.join(BASE_DIR, "data", "annotations", "test.spacy")
OUT_DIR     = os.path.join(BASE_DIR, "data", "gold")
OUT_FILE    = os.path.join(OUT_DIR, "gold_draft.csv")

IMPACT_LABELS = {"FATALITIES", "DISPLACED", "AFFECTED_PEOPLE", "MISSING", "RELIEF_INFO"}
MAX_ARTICLES  = 20   # start small; raise later


def load_test_texts():
    nlp = spacy.blank("en")
    docs = DocBin().from_disk(TEST_FILE).get_docs(nlp.vocab)
    return {d.text for d in docs}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_texts = load_test_texts()

    with open(EXPORT_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    rows, articles_used, no_number_spans = [], 0, 0

    for task in tasks:
        text = task["data"].get("text", "")
        if text not in test_texts or not task.get("annotations"):
            continue
        if articles_used >= MAX_ARTICLES:
            break
        articles_used += 1

        sentences, candidates = generate_candidates(text)
        sent_by_id = {s.sentence_id: s for s in sentences}
        quantities = [c for c in candidates if c.kind == "QUANTITY"]

        for item in task["annotations"][0].get("result", []):
            v = item.get("value", {})
            labels = v.get("labels", [])
            if not labels or labels[0] not in IMPACT_LABELS:
                continue
            start, end = v["start"], v["end"]
            inside = [c for c in quantities if c.start_char >= start and c.end_char <= end]

            if not inside:
                no_number_spans += 1
                inside = [None]   # qualitative claim, or a number the generator missed

            for c in inside:
                if c is None:
                    sentence = v.get("text", "")
                    raw, a, b = "", start, end
                    unit, agg = "", "UNKNOWN"
                else:
                    sent = sent_by_id[c.sentence_id]
                    sentence = sent.text.strip()
                    raw, a, b = c.raw, c.start_char, c.end_char
                    before = sent.text[: c.start_char - sent.start_char]
                    after  = sent.text[c.end_char - sent.start_char:]
                    unit = guess_unit(before, after)
                    agg  = guess_aggregation(before, after)
                rows.append({
                    "article_id":  task["id"],
                    "title":       task["data"].get("title", "")[:80],
                    "impact_type": labels[0],
                    "raw_value":   raw,
                    "aggregation": agg,
                    "unit":        unit,
                    "sentence":    " ".join(sentence.split()),
                    "start_char":  a,
                    "end_char":    b,
                    "review":      review_reason(labels[0], raw, unit),
                    "notes":       "",
                })

    with open(OUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["article_id"])
        w.writeheader()
        w.writerows(rows)

    print(f"Articles used             : {articles_used}")
    print(f"Draft claim rows          : {len(rows)}")
    print(f"Spans with no number found: {no_number_spans} (qualitative, or a number the generator missed)")
    print(f"Saved to                  : {OUT_FILE}")


if __name__ == "__main__":
    main()