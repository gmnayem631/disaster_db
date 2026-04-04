import json
import os

# Paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR  = os.path.join(BASE_DIR, "articles", "cleaned")
OUTPUT_FILE  = os.path.join(BASE_DIR, "data", "annotations", "label_studio_import.json")

def convert_for_label_studio():
    files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.json')]
    
    tasks = []
    for filename in files:
        filepath = os.path.join(CLEANED_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)
        
        task = {
            "data": {
                "text":       article.get("text", ""),
                "title":      article.get("title", ""),
                "source_url": article.get("source_url", ""),
                "date":       article.get("publish_date", "")
            }
        }
        tasks.append(task)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(tasks)} articles")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_for_label_studio()