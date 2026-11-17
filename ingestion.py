"""Ingestion: find new articles on newspaper sites and store the disaster-related ones.

For each newspaper in data/newspapers.txt:
  1. Discover: read its news sitemap or RSS feed and record every recent
     article address in the `articles` collection (status `discovered`).
  2. Download each waiting address politely (see PoliteFetcher).
  3. Pull out the article text, clean it, and check it against
     data/hazard_keywords.txt.
     - mentions a hazard   -> text stored, status `cleaned` (run_pending extracts it)
     - no hazard / too short -> status `filtered_out`, text NOT stored
     - download failed      -> retried on a later run, `dead` after 3 failures

Nothing here touches the old `disaster_events` collection.

Run:  venv\\Scripts\\python ingestion.py --db disaster_db_test --limit 20
"""
import argparse
import os
import random
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit

from database import schema
from database.documents import canonical_url, to_utc, utc_now

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SOURCES_FILE  = os.path.join(BASE_DIR, "data", "newspapers.txt")
KEYWORDS_FILE = os.path.join(BASE_DIR, "data", "hazard_keywords.txt")

# A plain, honest identity for the scraper. It does not pretend to be a browser
# and carries no personal or project details.
USER_AGENT = "Mozilla/5.0 (compatible; DisasterDB/0.1; research)"

MIN_DELAY, MAX_DELAY = 5.0, 15.0   # seconds between two requests to the SAME site, chosen at random
LOOKBACK_DAYS = 3                  # ignore sitemap/feed entries older than this
MIN_WORDS     = 80                 # same minimum as preprocessor.py
SITEMAP_CHILDREN = 2               # for a sitemap index: how many of the newest news sitemaps to read

# Address parts that are never disaster news. Saves downloading them.
SKIP_URL_PARTS = ("/bangla/", "/sport/", "/sports/", "/entertainment/", "/lifestyle/",
                  "/showbiz/", "/cricket/", "/football/")

METHODS = {"sitemap", "rss", "section"}


# ── Configuration files ────────────────────────────────────────────────────
@dataclass
class Source:
    name: str
    method: str
    url: str


def load_sources(path=SOURCES_FILE):
    """Read data/newspapers.txt. Raises ValueError with the line number on a bad line."""
    sources = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"{path} line {number}: expected 'name | method | url', got {line!r}")
            name, method, url = parts
            if method not in METHODS:
                raise ValueError(f"{path} line {number}: method must be one of {sorted(METHODS)}")
            canonical_url(url)                       # raises if it is not a web address
            sources.append(Source(name, method, url))
    return sources


def load_hazard_keywords(path=KEYWORDS_FILE):
    """Read data/hazard_keywords.txt into {hazard: [compiled pattern per keyword]}."""
    hazards = {}
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            hazard, sep, words = line.partition(":")
            keywords = [w.strip().lower() for w in words.split(",") if w.strip()]
            if not sep or not hazard.strip() or not keywords:
                raise ValueError(f"{path} line {number}: expected 'HAZARD: keyword, keyword'")
            hazards[hazard.strip().upper()] = [
                (k, re.compile(r"(?<!\w)" + re.escape(k).replace(r"\ ", r"\s+") + r"(?!\w)",
                               re.IGNORECASE))
                for k in keywords]
    return hazards


def match_hazards(text, hazard_keywords):
    """Which hazards and which keywords appear in the text, as two sorted lists."""
    hazards, hits = set(), set()
    for hazard, patterns in hazard_keywords.items():
        for keyword, pattern in patterns:
            if pattern.search(text):
                hazards.add(hazard)
                hits.add(keyword)
    return sorted(hazards), sorted(hits)


# ── Polite downloading ─────────────────────────────────────────────────────
class Blocked(Exception):
    """The site asked us to stop (429/503/403) or could not be reached.
    We leave it alone for the rest of this run. Not the article's fault."""


class Disallowed(Exception):
    """robots.txt does not allow this address."""


class FetchError(Exception):
    """This one page could not be downloaded (404, 500, timeout...)."""


class PoliteFetcher:
    """Downloads pages without overloading any site.

    - Before the first request to a site, reads its robots.txt and obeys it.
    - Between two requests to the same site, waits a random 5-15 s, or the
      site's own crawl-delay if that is longer. Different sites do not wait
      for each other, so alternating between sites keeps the run short.
    - If a site answers "too many requests", "unavailable" or "forbidden",
      it is left alone for the rest of the run.
    """

    def __init__(self, session=None, *, sleep=time.sleep, clock=time.monotonic,
                 rand=random.uniform, min_delay=MIN_DELAY, max_delay=MAX_DELAY,
                 user_agent=USER_AGENT, timeout=30):
        if session is None:
            import requests
            session = requests.Session()
        session.headers["User-Agent"] = user_agent
        self.session, self.sleep, self.clock, self.rand = session, sleep, clock, rand
        self.min_delay, self.max_delay = min_delay, max_delay
        self.user_agent, self.timeout = user_agent, timeout
        self.robots, self.last_request, self.blocked = {}, {}, {}
        self.requests_made = 0

    def get(self, url):
        """Return the response for `url`, or raise Blocked / Disallowed / FetchError."""
        parts = urlsplit(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site in self.blocked:
            raise Blocked(self.blocked[site])
        robots = self._robots(site)
        if not robots.can_fetch(self.user_agent, url):
            raise Disallowed(f"robots.txt of {site} does not allow {url}")
        response = self._request(site, url, robots.crawl_delay(self.user_agent))
        if response.status_code in (403, 429, 503):
            reason = f"{site} answered {response.status_code}; left alone for the rest of this run"
            self.blocked[site] = reason
            raise Blocked(reason)
        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code} for {url}")
        return response

    def _request(self, site, url, crawl_delay=None):
        delay = max(self.rand(self.min_delay, self.max_delay), crawl_delay or 0)
        if site in self.last_request:
            remaining = delay - (self.clock() - self.last_request[site])
            if remaining > 0:
                self.sleep(remaining)
        self.requests_made += 1
        try:
            return self.session.get(url, timeout=self.timeout)
        except Exception as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self.last_request[site] = self.clock()

    def _robots(self, site):
        if site not in self.robots:
            parser = urllib.robotparser.RobotFileParser()
            try:
                response = self._request(site, site + "/robots.txt")
            except FetchError as exc:
                self.blocked[site] = f"robots.txt of {site} could not be read: {exc}"
                raise Blocked(self.blocked[site])
            if response.status_code in (401, 403):
                parser.disallow_all = True       # the standard rule for a forbidden robots.txt
            elif response.status_code >= 400:
                parser.allow_all = True          # no robots.txt: everything allowed
            else:
                parser.parse(response.text.splitlines())
            self.robots[site] = parser
        return self.robots[site]


# ── Discovering new article addresses ──────────────────────────────────────
def _local(tag):
    return tag.rsplit("}", 1)[-1]           # "{namespace}loc" -> "loc"


def _date(text):
    try:
        return to_utc(text.strip()) if text and text.strip() else None
    except (ValueError, TypeError):
        return None


def _read_sitemap(xml_bytes):
    """Return ("index", [(child_url, lastmod)]) or ("urls", [(url, date, title)])."""
    from defusedxml import ElementTree        # safe against malicious XML
    root = ElementTree.fromstring(xml_bytes)
    if _local(root.tag) == "sitemapindex":
        children = []
        for item in root:
            fields = {_local(e.tag): (e.text or "").strip() for e in item}
            if fields.get("loc"):
                children.append((fields["loc"], _date(fields.get("lastmod"))))
        return "index", children
    entries = []
    for item in root:
        fields = {}
        for element in item.iter():
            fields.setdefault(_local(element.tag), (element.text or "").strip())
        if fields.get("loc"):
            date = _date(fields.get("publication_date")) or _date(fields.get("lastmod"))
            entries.append((fields["loc"], date, fields.get("title") or None))
    return "urls", entries


def _discover_sitemap(source, fetcher):
    kind, items = _read_sitemap(fetcher.get(source.url).content)
    if kind == "urls":
        return items
    # A sitemap index: read the newest news sitemaps it lists.
    news = [c for c in items if "news" in c[0].lower()] or items
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    news.sort(key=lambda c: c[1] or oldest, reverse=True)
    entries = []
    for child_url, _ in news[:SITEMAP_CHILDREN]:
        child_kind, child_items = _read_sitemap(fetcher.get(child_url).content)
        if child_kind == "urls":
            entries.extend(child_items)
    return entries


def _discover_rss(source, fetcher):
    import feedparser
    feed = feedparser.parse(fetcher.get(source.url).content)
    entries = []
    for entry in feed.entries:
        stamp = entry.get("published_parsed") or entry.get("updated_parsed")
        date = datetime(*stamp[:6], tzinfo=timezone.utc) if stamp else None
        if entry.get("link"):
            entries.append((entry["link"], date, entry.get("title")))
    return entries


def _looks_like_article(path):
    last = path.rstrip("/").rsplit("/", 1)[-1]
    return path.count("/") >= 2 and (last.count("-") >= 3 or re.search(r"\d{5,}", last) is not None)


def _discover_section(source, fetcher):
    from bs4 import BeautifulSoup
    page = fetcher.get(source.url)
    host = urlsplit(source.url).netloc.lower()
    entries = []
    for link in BeautifulSoup(page.text, "html.parser").find_all("a", href=True):
        url = urljoin(source.url, link["href"])
        parts = urlsplit(url)
        if parts.netloc.lower() == host and _looks_like_article(parts.path):
            entries.append((url, None, link.get_text(strip=True) or None))
    return entries


DISCOVER = {"sitemap": _discover_sitemap, "rss": _discover_rss, "section": _discover_section}


def discover(source, fetcher, now, lookback_days=LOOKBACK_DAYS):
    """Recent article addresses of one source: list of (url, publish_date, title).
    Entries older than the look-back window, and skipped sections, are left out."""
    cutoff = now - timedelta(days=lookback_days)
    found, seen = [], set()
    for url, date, title in DISCOVER[source.method](source, fetcher):
        try:
            key = canonical_url(url)
        except ValueError:
            continue
        if key in seen or any(part in key.lower() for part in SKIP_URL_PARTS):
            continue
        if date is not None and date < cutoff:
            continue
        seen.add(key)
        found.append((url, date, title))
    return found


# ── Reading one article page ───────────────────────────────────────────────
META_DATE_TAGS = [{"property": "article:published_time"}, {"itemprop": "datePublished"},
                  {"name": "pubdate"}, {"name": "publishdate"}, {"name": "date"}]

# Typographic characters that the cleaner would otherwise delete. "10–15 lakh"
# must stay a range, not become "10 15 lakh".
ASCII_EQUIVALENTS = str.maketrans({"–": "-", "—": " - ", "‘": "'", "’": "'",
                                   "“": '"', "”": '"', " ": " "})


def parse_article(url, html):
    """Title, text and publication date from a page, using newspaper3k.
    Date order: newspaper3k, then meta tags, then a date in the address."""
    from bs4 import BeautifulSoup
    from newspaper import Article

    article = Article(url)
    article.download(input_html=html)
    article.parse()

    date, date_source = None, None
    if article.publish_date:
        date, date_source = _date(article.publish_date.isoformat()), "newspaper3k"
    if date is None:
        soup = BeautifulSoup(html, "html.parser")
        for attrs in META_DATE_TAGS:
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                date = _date(tag["content"])
                if date:
                    date_source = "meta_tag"
                    break
    if date is None:
        m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", url)
        if m:
            date, date_source = _date("-".join(m.groups())), "url"
    return {"title": article.title, "text": article.text,
            "publish_date": date, "publish_date_source": date_source}


def clean(text):
    from preprocessor import clean_text       # the same cleaning the training data had
    return clean_text((text or "").translate(ASCII_EQUIVALENTS))


# ── One run ────────────────────────────────────────────────────────────────
@dataclass
class IngestReport:
    started_at: object = None
    seconds: float = 0.0
    sources: dict = field(default_factory=dict)    # name -> {"found", "new", "error"}
    attempted: int = 0
    cleaned: int = 0
    filtered: dict = field(default_factory=dict)   # reason -> count
    failed: int = 0
    dead: int = 0
    not_attempted: int = 0                         # left waiting (site blocked us, or limit)
    blocked_sites: dict = field(default_factory=dict)
    requests_made: int = 0
    hazards: dict = field(default_factory=dict)    # hazard -> articles kept
    problems: list = field(default_factory=list)

    def summary(self):
        lines = [f"Ingestion run: {self.attempted} downloaded, {self.cleaned} kept, "
                 f"{sum(self.filtered.values())} filtered out, {self.failed} failed, "
                 f"{self.dead} dead, {self.not_attempted} left waiting, "
                 f"{self.requests_made} requests, {self.seconds / 60:.1f} min"]
        for name, s in self.sources.items():
            note = f"  ERROR: {s['error']}" if s["error"] else ""
            warn = "  WARNING: nothing found" if not s["error"] and s["found"] == 0 else ""
            lines.append(f"  {name:<24} found {s['found']:>4}, new {s['new']:>4}{note}{warn}")
        if self.filtered:
            lines.append("  filtered out: " + ", ".join(f"{r} {n}" for r, n in self.filtered.items()))
        if self.hazards:
            lines.append("  kept, by keyword hazard: " +
                         ", ".join(f"{h} {n}" for h, n in sorted(self.hazards.items())))
        for site, reason in self.blocked_sites.items():
            lines.append(f"  BLOCKED {reason}")
        lines.extend(f"  {p}" for p in self.problems)
        return "\n".join(lines)


def _round_robin(articles):
    """Alternate between sites so no site gets many requests in a row."""
    by_site = {}
    for a in articles:
        by_site.setdefault(urlsplit(a["url"]).netloc, []).append(a)
    queues = list(by_site.values())
    ordered = []
    while queues:
        for queue in list(queues):
            ordered.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return ordered


def run_ingestion(store, *, sources, hazard_keywords, fetcher=None, parse=parse_article,
                  limit=1000, lookback_days=LOOKBACK_DAYS, min_words=MIN_WORDS,
                  now=None, progress=None):
    """Discover, download, clean, filter and store. Returns an IngestReport.

    store:    database.store.Store
    fetcher:  PoliteFetcher (tests pass one with a fake session and clock)
    parse:    function(url, html) -> {title, text, publish_date, publish_date_source}
    now:      only for tests
    progress: function(str) for live messages during a long run (e.g. print)
    """
    clock = time.perf_counter()
    report = IngestReport(started_at=utc_now())
    fetcher = fetcher or PoliteFetcher()
    say = progress or (lambda message: None)
    when = now or utc_now()

    # 1. Discover new addresses
    for source in sources:
        stats = {"found": 0, "new": 0, "error": None}
        report.sources[source.name] = stats
        try:
            entries = discover(source, fetcher, when, lookback_days)
        except Exception as exc:
            stats["error"] = f"{type(exc).__name__}: {exc}"
            continue
        stats["found"] = len(entries)
        for url, date, _title in entries:
            _, created = store.add_article(url, publish_date=date,
                                           publish_date_source=source.method if date else None,
                                           origin="daily", now=now)
            stats["new"] += created
        say(f"{source.name}: {stats['found']} found, {stats['new']} new")

    # 2. Download and sort out every waiting address (new ones and due retries)
    # Take every waiting address, alternate between sites, THEN apply the limit,
    # so a limited run gets a fair share from each newspaper.
    # (pymongo treats limit=0 as "no limit".)
    pending = _round_robin(store.get_pending("scrape", limit=0, now=now))[:limit]
    for position, article in enumerate(pending, 1):
        article_id, url = article["_id"], article["url"]
        site = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        if site in fetcher.blocked:
            report.not_attempted += 1
            continue
        report.attempted += 1

        try:
            html = fetcher.get(url).text
            page = parse(url, html)
        except Blocked:
            report.attempted -= 1
            report.not_attempted += 1          # the site's choice, not the article's: no retry used
            continue
        except Disallowed as exc:
            store.mark_filtered_out(article_id, str(exc), now=now)
            report.filtered["robots.txt"] = report.filtered.get("robots.txt", 0) + 1
            continue
        except Exception as exc:
            status = store.mark_failed(article_id, stage="scrape", kind=type(exc).__name__,
                                       message=str(exc)[:500], now=now)
            if status == schema.DEAD:
                report.dead += 1
                report.problems.append(f"DEAD {url}: {exc}")
            else:
                report.failed += 1
            continue

        title = (page.get("title") or "").strip()
        text = clean(page.get("text"))
        if len(text.split()) < min_words:
            store.mark_filtered_out(article_id, f"too short ({len(text.split())} words)", now=now)
            report.filtered["too short"] = report.filtered.get("too short", 0) + 1
            continue
        hazards, hits = match_hazards(f"{title}\n{text}", hazard_keywords)
        if not hazards:
            store.mark_filtered_out(article_id, "no hazard keyword", now=now)
            report.filtered["no hazard keyword"] = report.filtered.get("no hazard keyword", 0) + 1
            continue

        # The sitemap/feed date is preferred: it comes from the newspaper's own list.
        if article.get("publish_date"):
            date, date_source = article["publish_date"], article.get("publish_date_source")
        else:
            date, date_source = page.get("publish_date"), page.get("publish_date_source")
        store.set_text(article_id, title=title, text=text, publish_date=date,
                       publish_date_source=date_source, keyword_hazards=hazards,
                       keyword_hits=hits, now=now)
        report.cleaned += 1
        for hazard in hazards:
            report.hazards[hazard] = report.hazards.get(hazard, 0) + 1
        say(f"[{position}/{len(pending)}] KEPT {', '.join(hazards)}: {title[:70]}")

    report.blocked_sites = dict(fetcher.blocked)
    report.requests_made = fetcher.requests_made
    report.seconds = time.perf_counter() - clock
    return report


def main():
    parser = argparse.ArgumentParser(description="Find, download and store disaster articles.")
    parser.add_argument("--db", required=True,
                        help="database name, e.g. disaster_db_test (no default on purpose)")
    parser.add_argument("--limit", type=int, default=1000, help="most pages to download this run")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    from database.store import Store, get_database
    store = Store(get_database(args.db))
    store.setup()
    report = run_ingestion(store, sources=load_sources(), hazard_keywords=load_hazard_keywords(),
                           limit=args.limit, lookback_days=args.lookback_days, progress=print)
    print(report.summary())


if __name__ == "__main__":
    main()
