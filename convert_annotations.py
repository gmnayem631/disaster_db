import json
import os
import random
from pathlib import Path
import spacy
from spacy.tokens import DocBin

# Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
EXPORT_FILE     = os.path.join(BASE_DIR, "data", "annotations", "annotations_export.json")
TRAIN_OUTPUT    = os.path.join(BASE_DIR, "data", "annotations", "train.spacy")
DEV_OUTPUT      = os.path.join(BASE_DIR, "data", "annotations", "dev.spacy")

# Labels to ignore during conversion
IGNORE_LABELS = {
    "HOUSES_DAMAGED",
    "SECONDARY_HAZARD",
    "INFRASTRUCTURE_DAMAGED",
    "EVENT_DATE",
    "INJURED",
    "CROPS_AFFECTED",
}

def convert_to_spacy(export_file, train_output, dev_output):
    with open(export_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    nlp = spacy.blank("en")
    all_examples = []
    skipped = 0

    for task in tasks:
        # Get article text
        text = task["data"].get("text", "")
        if not text:
            skipped += 1
            continue

        # Get annotations
        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            continue

        # Take first annotation per task
        result = annotations[0].get("result", [])

        entities = []
        for item in result:
            value = item.get("value", {})
            start  = value.get("start")
            end    = value.get("end")
            labels = value.get("labels", [])

            if not labels or start is None or end is None:
                continue

            label = labels[0]

            # Skip ignored labels
            if label in IGNORE_LABELS:
                continue

            entities.append((start, end, label))

        # Sort entities by start position
        entities.sort(key=lambda x: x[0])

        # Remove overlapping spans
        clean_entities = []
        last_end = 0
        for start, end, label in entities:
            if start >= last_end:
                clean_entities.append((start, end, label))
                last_end = end

        try:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in clean_entities:
                # span = doc.char_span(start, end, label=label)
                # if span is not None:
                #     ents.append(span)
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            all_examples.append(doc)
        except Exception as e:
            print(f"Error processing task {task.get('id')}: {e}")
            skipped += 1

    # Shuffle and split 70/15/15
    random.seed(42)
    random.shuffle(all_examples)

    n           = len(all_examples)
    train_docs  = all_examples[:int(0.70 * n)]
    dev_docs    = all_examples[int(0.70 * n):int(0.85 * n)]
    test_docs   = all_examples[int(0.85 * n):]

    # Save
    TEST_OUTPUT = os.path.join(BASE_DIR, "data", "annotations", "test.spacy")

    train_db = DocBin(docs=train_docs)
    train_db.to_disk(train_output)

    dev_db = DocBin(docs=dev_docs)
    dev_db.to_disk(dev_output)

    test_db = DocBin(docs=test_docs)
    test_db.to_disk(TEST_OUTPUT)

    print(f"Total articles processed : {len(all_examples)}")
    print(f"Skipped                  : {skipped}")
    print(f"Training set             : {len(train_docs)} articles")
    print(f"Dev set                  : {len(dev_docs)} articles")
    print(f"Test set                 : {len(test_docs)} articles")
    print(f"Train saved to           : {train_output}")
    print(f"Dev saved to             : {dev_output}")
    print(f"Test saved to            : {TEST_OUTPUT}")

if __name__ == "__main__":
    convert_to_spacy(EXPORT_FILE, TRAIN_OUTPUT, DEV_OUTPUT)