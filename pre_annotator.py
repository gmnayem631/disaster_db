import json
import os
import spacy

# Paths
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR      = os.path.join(BASE_DIR, "articles", "cleaned")
ANNOTATED_FILE   = os.path.join(BASE_DIR, "data", "annotations", "label_studio_import.json")
OUTPUT_FILE      = os.path.join(BASE_DIR, "data", "annotations", "pre_annotated_import.json")

# Load trained DistilBERT model
print("Loading model...")
nlp = spacy.load(os.path.join(BASE_DIR, "models", "bert_ner_model"))
print("Model loaded.")

def get_already_annotated_urls():
    """Get URLs of articles already imported into Label Studio."""
    if not os.path.exists(ANNOTATED_FILE):
        return set()
    with open(ANNOTATED_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)
    return {task["data"]["source_url"] for task in existing}

def pre_annotate_articles():
    already_annotated = get_already_annotated_urls()
    
    cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.json')]
    
    tasks = []
    skipped = 0
    processed = 0

    for filename in cleaned_files:
        filepath = os.path.join(CLEANED_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)

        # Skip already annotated articles
        if article.get("source_url") in already_annotated:
            skipped += 1
            continue

        text = article.get("text", "")
        if not text:
            continue

        # Run model
        doc = nlp(text)

        # Build predictions in Label Studio format
        predictions = []
        for ent in doc.ents:
            predictions.append({
                "value": {
                    "start":  ent.start_char,
                    "end":    ent.end_char,
                    "text":   ent.text,
                    "labels": [ent.label_]
                },
                "from_name": "label",
                "to_name":   "text",
                "type":      "labels"
            })

        task = {
            "data": {
                "text":       text,
                "title":      article.get("title", ""),
                "source_url": article.get("source_url", ""),
                "date":       article.get("publish_date", "")
            },
            "predictions": [{"result": predictions}]
        }
        tasks.append(task)
        processed += 1

        if processed % 20 == 0:
            print(f"Processed {processed} articles...")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"Pre-annotated : {processed}")
    print(f"Skipped       : {skipped} (already annotated)")
    print(f"Saved to      : {OUTPUT_FILE}")

if __name__ == "__main__":
    pre_annotate_articles()