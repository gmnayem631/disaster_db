import spacy
import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
GAZETTEER_FILE   = os.path.join(BASE_DIR, "data", "gazetteer", "bangladesh_gazetteer.json")
CLEANED_DIR      = os.path.join(BASE_DIR, "articles", "cleaned")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "gazetteer_model")

# ── Load spaCy and gazetteer ───────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

with open(GAZETTEER_FILE, "r", encoding="utf-8") as f:
    gazetteer = json.load(f)

# ── Build EntityRuler patterns ─────────────────────────────────────────────
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = []

for district in gazetteer["districts"]:
    patterns.append({"label": "BD_DISTRICT", "pattern": district})
    # Also add lowercase version
    patterns.append({"label": "BD_DISTRICT", "pattern": district.lower()})

for upazila in gazetteer["upazilas"]:
    patterns.append({"label": "BD_UPAZILA", "pattern": upazila})
    patterns.append({"label": "BD_UPAZILA", "pattern": upazila.lower()})

for union in gazetteer["unions"]:
    patterns.append({"label": "BD_UNION", "pattern": union})
    patterns.append({"label": "BD_UNION", "pattern": union.lower()})

ruler.add_patterns(patterns)

print(f"Total patterns added: {len(patterns)}")

# ── Test on cleaned articles ───────────────────────────────────────────────
def test_gazetteer_ner(num_articles=5):
    files = [f for f in os.listdir(CLEANED_DIR) 
             if f.endswith('.json')][:num_articles]

    for filename in files:
        filepath = os.path.join(CLEANED_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)

        print("\n" + "="*60)
        print(f"ARTICLE: {article['title']}")
        print("="*60)

        doc = nlp(article['text'])

        # Only show location-related entities
        location_labels = {"BD_DISTRICT", "BD_UPAZILA", "BD_UNION", "GPE", "LOC"}

        entities_by_label = {}
        for ent in doc.ents:
            if ent.label_ in location_labels:
                if ent.label_ not in entities_by_label:
                    entities_by_label[ent.label_] = []
                if ent.text not in entities_by_label[ent.label_]:
                    entities_by_label[ent.label_].append(ent.text)

        for label, entities in entities_by_label.items():
            print(f"\n{label}:")
            for e in entities:
                print(f"   - {e}")

# ── Save the pipeline with gazetteer ──────────────────────────────────────
def save_gazetteer_pipeline():
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    nlp.to_disk(OUTPUT_MODEL_DIR)
    print(f"\nPipeline saved to: {OUTPUT_MODEL_DIR}")

if __name__ == "__main__":
    test_gazetteer_ner(num_articles=5)
    save_gazetteer_pipeline()