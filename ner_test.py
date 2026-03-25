import spacy
import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR  = os.path.join(BASE_DIR, "articles", "cleaned")

# Load default English model
nlp = spacy.load("en_core_web_sm")

def test_ner_on_articles(num_articles=5):
    """Run spaCy NER on first N cleaned articles and print results."""
    
    files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.json')][:num_articles]
    
    for filename in files:
        filepath = os.path.join(CLEANED_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)
        
        print("\n" + "="*60)
        print(f"ARTICLE: {article['title']}")
        print("="*60)
        
        doc = nlp(article['text'])
        
        # Group entities by label
        entities_by_label = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities_by_label:
                entities_by_label[label] = []
            entities_by_label[label].append(ent.text)
        
        # Print grouped
        for label, entities in entities_by_label.items():
            print(f"\n{label}:")
            for e in entities:
                print(f"   - {e}")

if __name__ == "__main__":
    test_ner_on_articles(num_articles=5)