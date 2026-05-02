import json
import os
from entity_linker import extract_entities
from db_writer import insert_disaster_record

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "articles", "cleaned")

def process_single_article(filename):
    filepath = os.path.join(CLEANED_DIR, filename)
    
    with open(filepath, "r", encoding="utf-8") as f:
        article = json.load(f)
    
    print(f"Processing: {article.get('title', '')[:60]}")
    
    # Extract entities
    record = extract_entities(
        text         = article["text"],
        source_url   = article["source_url"],
        publish_date = article.get("publish_date")
    )
    
    # Add title
    record["title"] = article.get("title", "")
    
    # Insert to MongoDB
    result = insert_disaster_record(record)
    print(f"MongoDB result: {result['status']} — ID: {result['id']}")
    
    return record, result

if __name__ == "__main__":
    # Test on first 3 articles
    files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.json')][:3]
    
    for filename in files:
        print("\n" + "="*60)
        record, result = process_single_article(filename)
        print(f"Districts  : {record['locations']['districts']}")
        print(f"Disaster   : {record['disaster_type']}")
        print(f"Fatalities : {record['fatalities']}")
        print(f"Displaced  : {record['displaced']}")