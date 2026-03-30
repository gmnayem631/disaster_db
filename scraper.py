import json
import os
import time
import hashlib
from datetime import datetime
from newspaper import Article
import requests

# Paths 
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
URLS_FILE   = os.path.join(BASE_DIR, "data", "urls.txt")
OUTPUT_DIR  = os.path.join(BASE_DIR, "articles", "raw")
LOG_FILE    = os.path.join(BASE_DIR, "logs", "scraper_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)


# Helpers
def url_to_filename(url):
    """Create a unique filename from a URL using hashing."""
    return hashlib.md5(url.encode()).hexdigest() + ".json"


def log(message):
    """Write a timestamped message to log file and print it."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def extract_date_fallback(url):
    """Try to extract a date from the URL string itself as fallback."""
    import re
    match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', url)
    if match:
        return "-".join(match.groups())
    return None


# Core scraper
def scrape_article(url):
    """Download and parse a single article. Returns a dict or None on failure."""
    try:
        article = Article(url)
        article.download()
        article.parse()

        # Date: try newspaper3k first, then URL fallback, then None
        if article.publish_date:
            date = article.publish_date.strftime("%Y-%m-%d")
        else:
            date = extract_date_fallback(url)

        # Basic validation - skip if no meaningful text
        if not article.text or len(article.text.strip()) < 100:
            log(f"SKIPPED (too short): {url}")
            return None

        return {
            "title":               article.title,
            "text":                article.text,
            "publish_date":        date,
            "source_url":          url,
            "scraped_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        log(f"ERROR scraping {url}: {e}")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_scraper():
    # Load URLs
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    log(f"Starting scraper. {len(urls)} URLs loaded.")

    success = 0
    skipped = 0
    failed  = 0

    for i, url in enumerate(urls, 1):
        filename = url_to_filename(url)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Skip if already scraped
        if os.path.exists(output_path):
            log(f"[{i}/{len(urls)}] ALREADY EXISTS, skipping: {url}")
            skipped += 1
            continue

        log(f"[{i}/{len(urls)}] Scraping: {url}")
        data = scrape_article(url)

        if data:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            success += 1
        else:
            failed += 1

        # Polite delay between requests
        time.sleep(2)

    log(f"Done. Success: {success} | Skipped: {failed} | Already existed: {skipped}")


if __name__ == "__main__":
    run_scraper()