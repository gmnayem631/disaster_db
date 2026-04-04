import json
import os
import re
from datetime import datetime

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "articles", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "articles", "cleaned")
LOG_FILE    = os.path.join(BASE_DIR, "logs", "preprocessor_log.txt")

os.makedirs(CLEANED_DIR, exist_ok=True)


# Helpers
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def clean_text(text):
    """Clean raw article text."""

    # Remove lines that are typical boilerplate
    boilerplate_patterns = [
        r'(?i)^also read[:\s].*$',
        r'(?i)^read more[:\s].*$',
        r'(?i)^follow us on.*$',
        r'(?i)^subscribe to.*$',
        r'(?i)^click here.*$',
        r'(?i)^photo[:\s].*$',
        r'(?i)^file photo.*$',
        r'(?i)^.*staff correspondent.*$',
        r'(?i)^.*our correspondent.*$',
        r'(?i)^.*special correspondent.*$',
    ]

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip lines matching boilerplate
        skip = False
        for pattern in boilerplate_patterns:
            if re.match(pattern, line):
                skip = True
                break

        if not skip:
            cleaned_lines.append(line)

    # Rejoin lines into paragraphs
    text = ' '.join(cleaned_lines)

    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # Remove weird unicode characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()


def is_flood_related(text, title):
    """Check if article is flood related."""
    keywords = [
        'flood', 'flooding', 'flooded', 'flash flood', 'floods','flash flood',
        'inundated', 'inundation', 'waterlogged', 'submerged',
        'deluge', 'overflowed', 'embankment breach'
    ]
    combined = (title + ' ' + text).lower()
    return any(keyword in combined for keyword in keywords)


# Core processor
def process_article(raw_data):
    """Clean and validate a single article. Returns cleaned dict or None."""

    title = raw_data.get('title', '').strip()
    text  = raw_data.get('text', '').strip()

    if not text or not title:
        return None

    cleaned_text = clean_text(text)

    # Skip if text too short after cleaning
    if len(cleaned_text.split()) < 80:
        return None

    # Skip if not flood related
    if not is_flood_related(cleaned_text, title):
        return None

    return {
        "title":               title,
        "text":                cleaned_text,
        "publish_date":        raw_data.get('publish_date'),
        "source_url":          raw_data.get('source_url'),
        "scraped_at":          raw_data.get('scraped_at'),
        "preprocessed_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# Main loop
def run_preprocessor():
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
    log(f"Starting preprocessor. {len(raw_files)} raw files found.")

    success  = 0
    skipped  = 0
    filtered = 0

    for i, filename in enumerate(raw_files, 1):
        raw_path     = os.path.join(RAW_DIR, filename)
        cleaned_path = os.path.join(CLEANED_DIR, filename)

        # Skip if already processed
        if os.path.exists(cleaned_path):
            log(f"[{i}/{len(raw_files)}] ALREADY EXISTS, skipping.")
            skipped += 1
            continue

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        result = process_article(raw_data)

        if result:
            with open(cleaned_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log(f"[{i}/{len(raw_files)}] CLEANED: {raw_data.get('title', '')[:60]}")
            success += 1
        else:
            log(f"[{i}/{len(raw_files)}] FILTERED OUT: {raw_data.get('title', '')[:60]}")
            filtered += 1

    log(f"Done. Cleaned: {success} | Filtered out: {filtered} | Already existed: {skipped}")


if __name__ == "__main__":
    run_preprocessor()