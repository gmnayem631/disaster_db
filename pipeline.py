import os
import json
import time
import random
import logging
import hashlib
import schedule
from datetime import datetime
from newspaper import Article
from entity_linker import extract_entities
from db_writer import insert_disaster_record

# ── Logging setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "logs", "pipeline_log.txt")
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Newspaper sources ──────────────────────────────────────────────────────
SOURCES = [
    {
        "name": "The Daily Star",
        "sections": [
            "https://www.thedailystar.net/environment/disaster",
            "https://www.thedailystar.net/flood",
        ]
    },
    {
        "name": "Dhaka Tribune",
        "sections": [
            "https://www.dhakatribune.com/bangladesh/disaster",
        ]
    },
    {
        "name": "New Age",
        "sections": [
            "https://www.newagebd.net/flood",
        ]
    },
    {
        "name": "bdnews24",
        "sections": [
            "https://bdnews24.com/bangladesh/disaster",
        ]
    }
]

# ── Flood keywords ─────────────────────────────────────────────────────────
FLOOD_KEYWORDS = [
    'flood', 'flooding', 'flooded', 'flash flood',
    'inundated', 'inundation', 'waterlogged', 'submerged',
    'deluge', 'overflowed', 'embankment breach', 'tidal surge'
]

# ── Seen URLs tracker ──────────────────────────────────────────────────────
SEEN_URLS_FILE = os.path.join(BASE_DIR, "data", "seen_urls.json")

def load_seen_urls():
    if os.path.exists(SEEN_URLS_FILE):
        with open(SEEN_URLS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_urls(seen_urls):
    with open(SEEN_URLS_FILE, "w") as f:
        json.dump(list(seen_urls), f)

# ── Article scraper ────────────────────────────────────────────────────────
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text or len(article.text.strip()) < 100:
            return None

        # Date extraction
        if article.publish_date:
            date = article.publish_date.strftime("%Y-%m-%d")
        else:
            date = extract_date_from_meta(url)

        return {
            "title":        article.title,
            "text":         article.text,
            "publish_date": date,
            "source_url":   url,
            "scraped_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        log.error(f"Failed to scrape {url}: {e}")
        return None

def extract_date_from_meta(url):
    try:
        import requests
        from bs4 import BeautifulSoup
        headers  = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup     = BeautifulSoup(response.text, 'html.parser')
        meta_tags = [
            {"property": "article:published_time"},
            {"name": "pubdate"},
            {"name": "publishdate"},
            {"name": "date"},
            {"itemprop": "datePublished"},
        ]
        for attrs in meta_tags:
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                return tag["content"][:10]
        return None
    except:
        return None

# ── Text cleaner ───────────────────────────────────────────────────────────
def clean_text(text):
    import re
    boilerplate = [
        r'(?i)^also read[:\s].*$',
        r'(?i)^read more[:\s].*$',
        r'(?i)^follow us on.*$',
        r'(?i)^.*staff correspondent.*$',
        r'(?i)^.*our correspondent.*$',
    ]
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        skip = any(re.match(p, line) for p in boilerplate)
        if not skip:
            cleaned.append(line)
    text = ' '.join(cleaned)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def is_flood_related(text, title):
    combined = (title + ' ' + text).lower()
    return any(kw in combined for kw in FLOOD_KEYWORDS)

# ── Section scraper ────────────────────────────────────────────────────────
def get_article_urls_from_section(section_url):
    """Extract article URLs from a newspaper section page."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers  = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(section_url, headers=headers, timeout=15)
        soup     = BeautifulSoup(response.text, 'html.parser')
        
        urls = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Make absolute URL
            if href.startswith('http'):
                urls.add(href)
            elif href.startswith('/'):
                from urllib.parse import urlparse
                base = urlparse(section_url)
                urls.add(f"{base.scheme}://{base.netloc}{href}")
        
        return list(urls)
    except Exception as e:
        log.error(f"Failed to get URLs from {section_url}: {e}")
        return []

# ── Main pipeline ──────────────────────────────────────────────────────────
def run_pipeline():
    log.info("=" * 50)
    log.info("Pipeline started")
    
    seen_urls = load_seen_urls()
    
    success   = 0
    duplicate = 0
    failed    = 0
    skipped   = 0

    for source in SOURCES:
        log.info(f"Processing source: {source['name']}")
        
        for section_url in source["sections"]:
            urls = get_article_urls_from_section(section_url)
            log.info(f"Found {len(urls)} URLs in {section_url}")
            
            for url in urls:
                # Skip if already seen
                if url in seen_urls:
                    skipped += 1
                    continue
                
                seen_urls.add(url)
                
                # Scrape
                article = scrape_article(url)
                if not article:
                    failed += 1
                    time.sleep(random.uniform(1.5, 4.5))
                    continue
                
                # Clean
                cleaned_text = clean_text(article["text"])
                
                # Filter
                if not is_flood_related(cleaned_text, article["title"]):
                    skipped += 1
                    time.sleep(random.uniform(1.5, 4.5))
                    continue
                
                if len(cleaned_text.split()) < 80:
                    skipped += 1
                    continue
                
                # Extract entities
                try:
                    record = extract_entities(
                        text         = cleaned_text,
                        source_url   = url,
                        publish_date = article.get("publish_date")
                    )
                    record["title"]      = article.get("title", "")
                    record["scraped_at"] = article.get("scraped_at", "")
                    
                    # Insert to MongoDB
                    result = insert_disaster_record(record)
                    
                    if result["status"] == "inserted":
                        log.info(f"INSERTED: {article['title'][:60]}")
                        success += 1
                    else:
                        log.info(f"DUPLICATE: {article['title'][:60]}")
                        duplicate += 1
                        
                except Exception as e:
                    log.error(f"Entity extraction failed for {url}: {e}")
                    failed += 1
                
                # Random delay
                time.sleep(random.uniform(1.5, 4.5))
    
    # Save seen URLs
    save_seen_urls(seen_urls)
    
    log.info(f"Pipeline complete — Inserted: {success} | Duplicate: {duplicate} | Skipped: {skipped} | Failed: {failed}")
    log.info("=" * 50)

# ── Scheduler ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Scheduler started. Pipeline will run daily at 09:00.")
    
    # Run once immediately on startup
    run_pipeline()
    
    # Then schedule daily
    schedule.every().day.at("09:00").do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(60)