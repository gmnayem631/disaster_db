"""Tests for ingestion.py.

A fake website (FakeSession) and a fake clock stand in for the internet and
for real waiting, so these run in a second and never contact a newspaper.
The database is mongomock, or real `disaster_db_test` with RUN_MONGO=1.
"""
from datetime import datetime, timedelta, timezone

import pytest

from database import schema
from database.store import Store
from ingestion import (Blocked, Disallowed, FetchError, PoliteFetcher, Source, clean,
                       discover, load_hazard_keywords, load_sources, match_hazards,
                       parse_article, run_ingestion)

NOW = datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)
KEYWORDS = load_hazard_keywords()

FILLER = ("Officials held a meeting at the district office to discuss the situation "
          "and plan the next steps for the residents of the area. ") * 6


def story(first_sentence):
    """A page body long enough to pass the 80-word minimum."""
    return f"{first_sentence} {FILLER}"


# ── Fakes ──────────────────────────────────────────────────────────────────
class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body
        self.content = body.encode("utf-8")


class FakeSession:
    """A pretend internet: address -> (status, body) or an exception to raise.
    Unknown addresses answer 404, so a site without robots.txt allows everything."""

    def __init__(self, pages=None):
        self.pages = dict(pages or {})
        self.headers = {}
        self.requested = []

    def get(self, url, timeout=None):
        self.requested.append(url)
        reply = self.pages.get(url, (404, "not found"))
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            reply = (200, reply)
        return FakeResponse(*reply)


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.t += seconds


def fetcher_for(pages, rand=lambda low, high: low):
    clock = FakeClock()
    session = FakeSession(pages)
    return PoliteFetcher(session, sleep=clock.sleep, clock=clock, rand=rand), session, clock


def fake_parse(url, html):
    """Stands in for newspaper3k: the page body is the article text."""
    return {"title": html.split(".")[0], "text": html,
            "publish_date": None, "publish_date_source": None}


def sitemap(*entries):
    """entries: (url, iso date or None, title)"""
    items = []
    for url, date, title in entries:
        news = (f"<news:news><news:publication><news:name>Paper</news:name></news:publication>"
                f"{f'<news:publication_date>{date}</news:publication_date>' if date else ''}"
                f"<news:title>{title}</news:title></news:news>")
        items.append(f"<url><loc>{url}</loc>{news}</url>")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
            'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">'
            + "".join(items) + "</urlset>")


@pytest.fixture
def store(test_db):
    s = Store(test_db)
    s.setup()
    return s


# ── Configuration files ────────────────────────────────────────────────────
def test_the_real_newspaper_list_loads():
    sources = load_sources()
    names = [s.name for s in sources]
    assert len(sources) >= 7
    assert "The Daily Star" in names and "Daily Observer" in names
    assert "The Daily Sun" not in names                 # commented out: it blocks us


def test_bad_newspaper_line_names_the_line(tmp_path):
    path = tmp_path / "papers.txt"
    path.write_text("# comment\nGood | sitemap | https://a.com/s.xml\nBad line without bars\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        load_sources(path)


def test_unknown_method_is_refused(tmp_path):
    path = tmp_path / "papers.txt"
    path.write_text("Paper | scrape-everything | https://a.com\n", encoding="utf-8")
    with pytest.raises(ValueError, match="method"):
        load_sources(path)


@pytest.mark.parametrize("text, hazards", [
    ("Flash floods hit Sylhet again.", ["FLOOD"]),
    ("Many flood-hit families need help.", ["FLOOD"]),
    ("The ceasefire held overnight.", []),                 # "fire" inside a word
    ("Engineers were firing the kiln.", []),
    ("The fire's cause is unknown.", ["FIRE"]),
    ("A bus collided with a truck on the highway.", ["ROAD_ACCIDENT"]),
    ("Seven died in a road\naccident in Feni.", ["ROAD_ACCIDENT"]),   # phrase split by a line break
    ("A nor'wester damaged homes.", ["STORM"]),
    ("Dengue cases rose and a cyclone formed.", ["CYCLONE", "DISEASE_OUTBREAK"]),
    ("Police opened fire on the crowd.", ["FIRE"]),        # known false match; the LLM must remove it
    ("The cabinet approved the budget.", []),
])
def test_hazard_keywords(text, hazards):
    assert match_hazards(text, KEYWORDS)[0] == hazards


def test_typographic_dash_survives_cleaning():
    assert "10-15 lakh" in clean("Some 10–15 lakh people were affected.")


# ── Polite downloading ─────────────────────────────────────────────────────
PAGES = {f"https://{site}.com/{n}": "page" for site in ("a", "b") for n in (1, 2)}


def test_same_site_waits_other_sites_do_not():
    fetcher, session, clock = fetcher_for(PAGES)
    fetcher.get("https://a.com/1")      # a: robots.txt, wait 5 s, page
    fetcher.get("https://b.com/1")      # b: robots.txt at once (a's wait does not apply), wait 5 s, page
    fetcher.get("https://a.com/2")      # a again: its 5 s already passed while b was served
    assert clock.slept == [5.0, 5.0]
    assert session.requested == ["https://a.com/robots.txt", "https://a.com/1",
                                 "https://b.com/robots.txt", "https://b.com/1", "https://a.com/2"]


def test_delay_is_random_between_the_limits():
    chosen = []

    def rand(low, high):
        chosen.append((low, high))
        return 12.5

    fetcher, _, clock = fetcher_for(PAGES, rand=rand)
    fetcher.get("https://a.com/1")
    assert set(chosen) == {(5.0, 15.0)}
    assert clock.slept == [12.5]


def test_longer_crawl_delay_from_robots_is_obeyed():
    fetcher, _, clock = fetcher_for({**PAGES, "https://a.com/robots.txt": "User-agent: *\nCrawl-delay: 30\n"})
    fetcher.get("https://a.com/1")
    fetcher.get("https://a.com/2")
    assert clock.slept[-1] == 30.0


def test_robots_disallow_is_obeyed():
    fetcher, session, _ = fetcher_for({**PAGES, "https://a.com/robots.txt":
                                       "User-agent: *\nDisallow: /private/\n"})
    with pytest.raises(Disallowed):
        fetcher.get("https://a.com/private/page")
    assert "https://a.com/private/page" not in session.requested
    assert fetcher.get("https://a.com/1").status_code == 200        # other paths are fine


def test_forbidden_robots_txt_means_nothing_is_allowed():
    fetcher, _, _ = fetcher_for({"https://a.com/robots.txt": (403, "no")})
    with pytest.raises(Disallowed):
        fetcher.get("https://a.com/news/1")


def test_too_many_requests_leaves_the_site_alone():
    fetcher, session, _ = fetcher_for({"https://a.com/1": (429, "slow down")})
    with pytest.raises(Blocked):
        fetcher.get("https://a.com/1")
    count = len(session.requested)
    with pytest.raises(Blocked):
        fetcher.get("https://a.com/2")
    assert len(session.requested) == count      # no further request was sent


def test_missing_page_is_an_ordinary_failure():
    fetcher, _, _ = fetcher_for({})
    with pytest.raises(FetchError, match="404"):
        fetcher.get("https://a.com/missing")


def test_network_error_is_an_ordinary_failure():
    fetcher, _, _ = fetcher_for({"https://a.com/1": TimeoutError("timed out")})
    with pytest.raises(FetchError, match="timed out"):
        fetcher.get("https://a.com/1")


# ── Discovering addresses ──────────────────────────────────────────────────
def test_news_sitemap_gives_addresses_dates_and_titles():
    xml = sitemap(
        ("https://www.thedailystar.net/news/flood-1", "2026-11-17T18:40:53+06:00", "Flood in Feni"),
        ("https://www.thedailystar.net/news/flood-1?utm_source=x", "2026-11-17T18:40:53+06:00", "same"),
        ("https://www.thedailystar.net/sports/cricket-1", "2026-11-17T10:00:00+06:00", "Cricket"),
        ("https://www.thedailystar.net/news/old-1", "2026-11-01T10:00:00+06:00", "Old"),
    )
    fetcher, _, _ = fetcher_for({"https://www.thedailystar.net/googlenews.xml": xml})
    source = Source("The Daily Star", "sitemap", "https://www.thedailystar.net/googlenews.xml")
    found = discover(source, fetcher, NOW)
    assert found == [("https://www.thedailystar.net/news/flood-1",
                      datetime(2026, 11, 17, 12, 40, 53, tzinfo=timezone.utc), "Flood in Feni")]


def test_sitemap_index_reads_the_newest_news_sitemaps():
    index = ('<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
             '<sitemap><loc>https://n.com/page-sitemap.xml</loc><lastmod>2026-11-17</lastmod></sitemap>'
             '<sitemap><loc>https://n.com/news-sitemap/3.xml</loc><lastmod>2026-11-10</lastmod></sitemap>'
             '<sitemap><loc>https://n.com/news-sitemap/1.xml</loc><lastmod>2026-11-17</lastmod></sitemap>'
             '<sitemap><loc>https://n.com/news-sitemap/2.xml</loc><lastmod>2026-11-16</lastmod></sitemap>'
             '</sitemapindex>')
    pages = {"https://n.com/sitemap.xml": index,
             "https://n.com/news-sitemap/1.xml": sitemap(("https://n.com/a/1-flood-hits-town-now", "2026-11-17", "A")),
             "https://n.com/news-sitemap/2.xml": sitemap(("https://n.com/a/2-fire-in-market-today", "2026-11-16", "B"))}
    fetcher, session, _ = fetcher_for(pages)
    found = discover(Source("New Age", "sitemap", "https://n.com/sitemap.xml"), fetcher, NOW)
    assert [f[0] for f in found] == ["https://n.com/a/1-flood-hits-town-now",
                                     "https://n.com/a/2-fire-in-market-today"]
    assert "https://n.com/page-sitemap.xml" not in session.requested
    assert "https://n.com/news-sitemap/3.xml" not in session.requested


def test_rss_feed_gives_addresses_and_dates():
    rss = ('<?xml version="1.0"?><rss version="2.0"><channel><title>Observer</title>'
           '<link>https://observerbd.com</link>'
           '<item><title>Fire in Dhaka</title><link>https://observerbd.com/news/606822</link>'
           '<pubDate>Tue, 17 Nov 2026 00:00:00 GMT</pubDate></item></channel></rss>')
    fetcher, _, _ = fetcher_for({"https://www.observerbd.com/rss": rss})
    found = discover(Source("Daily Observer", "rss", "https://www.observerbd.com/rss"), fetcher, NOW)
    assert found == [("https://observerbd.com/news/606822",
                      datetime(2026, 11, 17, tzinfo=timezone.utc), "Fire in Dhaka")]


def test_section_page_keeps_only_article_links_of_that_site():
    html = ('<a href="/bangladesh/flood-situation-worsens-in-sylhet">story</a>'
            '<a href="/news/1605566">numbered story</a>'
            '<a href="/about">about</a><a href="/bangladesh">section</a>'
            '<a href="https://other.com/bangladesh/some-long-article-slug-here">elsewhere</a>')
    fetcher, _, _ = fetcher_for({"https://p.com/bangladesh": html})
    found = discover(Source("P", "section", "https://p.com/bangladesh"), fetcher, NOW)
    assert [f[0] for f in found] == ["https://p.com/bangladesh/flood-situation-worsens-in-sylhet",
                                     "https://p.com/news/1605566"]


# ── A whole run ────────────────────────────────────────────────────────────
STAR_MAP = "https://www.thedailystar.net/googlenews.xml"
STAR = "https://www.thedailystar.net/news/"
TRIB_MAP = "https://www.dhakatribune.com/news-sitemap.xml"
TRIB = "https://www.dhakatribune.com/bangladesh/"
SOURCES = [Source("The Daily Star", "sitemap", STAR_MAP),
           Source("Dhaka Tribune", "sitemap", TRIB_MAP)]


def site_pages(extra=None):
    pages = {
        STAR_MAP: sitemap((STAR + "flood-feni", "2026-11-17T18:40:53+06:00", "Flood"),
                          (STAR + "budget-talks", "2026-11-17T17:00:00+06:00", "Budget"),
                          (STAR + "short-note", "2026-11-17T16:00:00+06:00", "Short")),
        STAR + "flood-feni": story("Floods marooned 10,000 families in Feni district."),
        STAR + "budget-talks": story("The cabinet discussed the national budget."),
        STAR + "short-note": "Floods were mentioned briefly.",
        TRIB_MAP: sitemap((TRIB + "bus-crash", None, "Crash")),
        TRIB + "bus-crash": story("A bus collided with a truck, killing five people."),
    }
    pages.update(extra or {})
    return pages


def ingest(store, pages, now=NOW, sources=SOURCES, **options):
    fetcher, session, clock = fetcher_for(pages)
    report = run_ingestion(store, sources=sources, hazard_keywords=KEYWORDS, fetcher=fetcher,
                           parse=fake_parse, now=now, **options)
    return report, session, clock


def test_a_run_keeps_disaster_articles_and_drops_the_rest(store):
    report, _, _ = ingest(store, site_pages())

    flood = store.get_article_by_url(STAR + "flood-feni")
    assert flood["processing_status"] == schema.CLEANED
    assert flood["keyword_hazards"] == ["FLOOD"]
    assert flood["keyword_hits"] == ["floods"]
    assert "10,000 families" in flood["text"]
    assert flood["source"] == "The Daily Star"
    assert flood["origin"] == "daily"
    # the sitemap's date is kept even though the page gave none
    assert flood["publish_date"] == datetime(2026, 11, 17, 12, 40, 53, tzinfo=timezone.utc)
    assert flood["publish_date_source"] == "sitemap"

    crash = store.get_article_by_url(TRIB + "bus-crash")
    assert crash["keyword_hazards"] == ["ROAD_ACCIDENT"]
    assert crash["publish_date"] is None                # no date anywhere: stays unknown

    budget = store.get_article_by_url(STAR + "budget-talks")
    assert budget["processing_status"] == schema.FILTERED_OUT
    assert budget["text"] is None                        # text of unrelated articles is not stored
    assert budget["last_error"]["message"] == "no hazard keyword"

    short = store.get_article_by_url(STAR + "short-note")
    assert short["processing_status"] == schema.FILTERED_OUT
    assert "too short" in short["last_error"]["message"]

    assert (report.attempted, report.cleaned, report.failed) == (4, 2, 0)
    assert report.filtered == {"no hazard keyword": 1, "too short": 1}
    assert report.sources["The Daily Star"] == {"found": 3, "new": 3, "error": None}
    assert report.hazards == {"FLOOD": 1, "ROAD_ACCIDENT": 1}

    # kept articles are exactly what run_pending() will pick up
    waiting = {a["url"] for a in store.get_pending("extraction", now=NOW)}
    assert waiting == {STAR + "flood-feni", TRIB + "bus-crash"}


def test_page_date_is_used_when_the_list_has_none(store):
    def parse_with_date(url, html):
        page = fake_parse(url, html)
        page.update(publish_date=datetime(2026, 11, 16, tzinfo=timezone.utc),
                    publish_date_source="meta_tag")
        return page

    fetcher, _, _ = fetcher_for(site_pages())
    run_ingestion(store, sources=SOURCES, hazard_keywords=KEYWORDS, fetcher=fetcher,
                  parse=parse_with_date, now=NOW)
    crash = store.get_article_by_url(TRIB + "bus-crash")
    flood = store.get_article_by_url(STAR + "flood-feni")
    assert crash["publish_date_source"] == "meta_tag"
    assert flood["publish_date_source"] == "sitemap"      # the list's date still wins


def test_second_run_downloads_nothing_again(store):
    ingest(store, site_pages())
    report, session, _ = ingest(store, site_pages(), now=NOW + timedelta(hours=1))
    assert report.attempted == 0
    assert report.sources["The Daily Star"]["new"] == 0
    assert [u for u in session.requested if "robots" not in u] == [STAR_MAP, TRIB_MAP]
    assert store.articles.count_documents({}) == 4


def test_requests_alternate_between_sites(store):
    pages = site_pages({TRIB_MAP: sitemap((TRIB + "bus-crash", None, "Crash"),
                                          (TRIB + "fire-two", None, "Fire")),
                        TRIB + "fire-two": story("A fire gutted a market.")})
    _, session, _ = ingest(store, pages)
    downloads = [u for u in session.requested
                 if "robots" not in u and u not in (STAR_MAP, TRIB_MAP)]
    sites = ["star" if "dailystar" in u else "trib" for u in downloads]
    assert sites[:4] == ["star", "trib", "star", "trib"]


def test_failed_download_is_retried_later(store):
    pages = site_pages({TRIB + "bus-crash": (500, "server error")})
    report, _, _ = ingest(store, pages)
    crash = store.get_article_by_url(TRIB + "bus-crash")
    assert report.failed == 1
    assert crash["processing_status"] == schema.SCRAPE_FAILED
    assert crash["retry_count"] == 1

    # not retried an hour later; retried the next day, and now the page works
    assert ingest(store, pages, now=NOW + timedelta(hours=1))[0].attempted == 0
    report, _, _ = ingest(store, site_pages(), now=NOW + DAY)
    assert report.cleaned == 1
    assert store.get_article_by_url(TRIB + "bus-crash")["processing_status"] == schema.CLEANED


def test_site_that_blocks_us_costs_its_articles_no_retries(store):
    pages = site_pages({STAR + "flood-feni": (429, "too many requests")})
    report, session, _ = ingest(store, pages)

    assert "429" in report.blocked_sites["https://www.thedailystar.net"]
    assert report.not_attempted == 3                     # the blocked page and the two after it
    for url in (STAR + "flood-feni", STAR + "budget-talks", STAR + "short-note"):
        article = store.get_article_by_url(url)
        assert article["processing_status"] == schema.DISCOVERED
        assert article["retry_count"] == 0
    assert store.get_article_by_url(TRIB + "bus-crash")["processing_status"] == schema.CLEANED
    assert "BLOCKED" in report.summary()


def test_page_forbidden_by_robots_is_set_aside_for_good(store):
    pages = site_pages({"https://www.dhakatribune.com/robots.txt":
                        "User-agent: *\nDisallow: /bangladesh/\n"})
    report, session, _ = ingest(store, pages)
    crash = store.get_article_by_url(TRIB + "bus-crash")
    assert crash["processing_status"] == schema.FILTERED_OUT
    assert TRIB + "bus-crash" not in session.requested
    assert report.filtered.get("robots.txt") == 1


def test_broken_source_is_reported_and_the_others_still_run(store):
    pages = site_pages({STAR_MAP: (500, "broken")})
    report, _, _ = ingest(store, pages)
    assert "HTTP 500" in report.sources["The Daily Star"]["error"]
    assert report.cleaned == 1                           # Dhaka Tribune still processed
    assert "ERROR" in report.summary()


def test_source_with_nothing_new_is_flagged(store):
    pages = site_pages({TRIB_MAP: sitemap()})
    report, _, _ = ingest(store, pages)
    assert "WARNING: nothing found" in report.summary()


def test_limit_caps_downloads_and_shares_them_between_sites(store):
    # The Daily Star is discovered first and has 3 articles; with a limit of 2
    # one must still come from Dhaka Tribune (found in the live run on 2026-11-18).
    report, session, _ = ingest(store, site_pages(), limit=2)
    downloads = [u for u in session.requested
                 if "robots" not in u and u not in (STAR_MAP, TRIB_MAP)]
    assert report.attempted == 2
    assert downloads == [STAR + "flood-feni", TRIB + "bus-crash"]
    assert len(store.get_pending("scrape", now=NOW)) == 2


def test_the_old_collection_is_not_touched(store):
    legacy = store.db[schema.LEGACY]
    legacy.insert_one({"_id": 1, "source_url": STAR + "flood-feni"})
    before = list(legacy.find())
    ingest(store, site_pages())
    assert list(legacy.find()) == before


# ── The real page reader (newspaper3k), on a made-up page ──────────────────
def test_parse_article_reads_text_and_meta_date():
    paragraphs = "".join(
        f"<p>Floodwater entered {n} more villages in Sunamganj district on Monday as the "
        f"Surma river kept rising above the danger mark, officials said.</p>" for n in range(2, 9))
    html = ('<html><head><title>Flood worsens in Sunamganj</title>'
            '<meta property="article:published_time" content="2026-11-17T09:30:00+06:00"></head>'
            f'<body><article><h1>Flood worsens in Sunamganj</h1>{paragraphs}</article></body></html>')
    page = parse_article("https://www.example.com/news/flood-worsens-in-sunamganj", html)
    assert "Surma river" in page["text"]
    assert page["publish_date"] == datetime(2026, 11, 17, 3, 30, tzinfo=timezone.utc)
    assert page["publish_date_source"] in ("newspaper3k", "meta_tag")
