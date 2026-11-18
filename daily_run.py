"""The daily job: collect new articles, check them with Qwen, extract the numbers.

  1. Ingestion   (ingestion.py)        newspapers -> `articles` (cleaned / filtered_out)
  2. Extraction  (extraction_service)  hazard check, then Branch A + Qwen -> `claims`
  3. Matching    (event_matcher)       extracted articles -> `events`

Everything printed is also written to logs/daily_<date>_<time>.log.
Windows Task Scheduler starts it once a day (docs/runbook.md). It can also be
run by hand:

    venv\\Scripts\\python daily_run.py --db disaster_db_test

Exit code (Task Scheduler shows it as "Last Run Result"):
  0  everything ran
  1  a stage stopped early or crashed (for example Ollama was not running);
     nothing is lost, waiting articles are picked up by the next run
  2  another run was still going, so this one did nothing

Nothing here touches the old `disaster_events` collection.
"""
import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(BASE_DIR, "logs")
LOCK_FILE = os.path.join(LOG_DIR, "daily_run.lock")
STALE_LOCK_HOURS = 20      # a lock older than this is left over from a crashed run

# Defaults: a normal day finds a few hundred pages, and a few dozen pass the
# keyword filter. The time limit keeps one long day from running into the next.
SCRAPE_LIMIT    = 1000
EXTRACT_LIMIT   = 200
EXTRACT_MINUTES = 300


class Log:
    """Prints a line with the time, and writes the same line to the log file."""

    def __init__(self, path=None):
        self.path = path
        self.file = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, message=""):
        for line in str(message).splitlines() or [""]:
            line = f"{datetime.now():%H:%M:%S}  {line}"
            if sys.stdout is not None:
                try:
                    print(line, flush=True)
                except UnicodeEncodeError:
                    # The Windows console cannot show some characters; the file still gets them.
                    print(line.encode("ascii", "replace").decode(), flush=True)
            if self.file:
                self.file.write(line + "\n")
                self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


class AlreadyRunning(Exception):
    pass


def take_lock(path=None, now=None):
    """Make sure only one daily run works at a time. A lock older than
    STALE_LOCK_HOURS is from a run that crashed, and is replaced."""
    path = path or LOCK_FILE
    now = now or time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        age_hours = (now - os.path.getmtime(path)) / 3600
        if age_hours < STALE_LOCK_HOURS:
            with open(path, encoding="utf-8") as f:
                raise AlreadyRunning(f"another run holds {path} ({f.read().strip()}, "
                                     f"{age_hours:.1f} h old)")
        os.remove(path)
    with open(path, "x", encoding="utf-8") as f:
        f.write(f"pid {os.getpid()} started {datetime.now():%Y-%m-%d %H:%M}")


def release_lock(path=None):
    try:
        path = path or LOCK_FILE
        os.remove(path)
    except FileNotFoundError:
        pass


@dataclass
class DailyReport:
    ingestion: object = None        # ingestion.IngestReport
    extraction: object = None       # extraction_service.RunReport
    matching: object = None         # event_matcher.MatchReport
    problems: list = field(default_factory=list)

    @property
    def exit_code(self):
        stopped = self.extraction is not None and self.extraction.stopped_early
        match_failed = self.matching is not None and self.matching.failed
        return 1 if self.problems or stopped or match_failed else 0


def daily_run(store, *, backend, log, sources=None, hazard_keywords=None, fetcher=None,
              branch_a=None, scrape_limit=SCRAPE_LIMIT, extract_limit=EXTRACT_LIMIT,
              extract_minutes=EXTRACT_MINUTES, lookback_days=None, skip_ingestion=False,
              skip_extraction=False, skip_matching=False, now=None):
    """Run the three stages and return a DailyReport. A crash in one stage is logged
    and the next stage still runs (articles already collected are not lost)."""
    import ingestion
    from event_matcher import match_pending
    from extraction_service import run_pending

    report = DailyReport()

    if skip_ingestion:
        log("== 1. Ingestion: skipped")
    else:
        log("== 1. Ingestion: collecting new articles")
        try:
            report.ingestion = ingestion.run_ingestion(
                store,
                sources=sources if sources is not None else ingestion.load_sources(),
                hazard_keywords=hazard_keywords or ingestion.load_hazard_keywords(),
                fetcher=fetcher, limit=scrape_limit,
                lookback_days=lookback_days or ingestion.LOOKBACK_DAYS,
                now=now, progress=log)
            log(report.ingestion.summary())
        except Exception:
            report.problems.append("ingestion crashed")
            log("INGESTION CRASHED:\n" + traceback.format_exc())

    if skip_extraction:
        log("== 2. Extraction: skipped")
    else:
        log("== 2. Extraction: hazard check, then Branch A + Qwen")
        try:
            report.extraction = run_pending(
                store, backend, limit=extract_limit, branch_a=branch_a,
                stop_at=time.monotonic() + extract_minutes * 60, progress=log, now=now)
            log(report.extraction.summary())
        except Exception:
            report.problems.append("extraction crashed")
            log("EXTRACTION CRASHED:\n" + traceback.format_exc())

    if skip_matching:
        log("== 3. Event matching: skipped")
    else:
        log("== 3. Event matching: extracted articles -> events")
        try:
            report.matching = match_pending(store, progress=log, now=now)
            log(report.matching.summary())
        except Exception:
            report.problems.append("event matching crashed")
            log("EVENT MATCHING CRASHED:\n" + traceback.format_exc())

    waiting = store.articles.count_documents({"processing_status": {"$in": ["cleaned", "extraction_failed"]}})
    log(f"== Done. Articles waiting for extraction now: {waiting}")
    for problem in report.problems:
        log(f"PROBLEM: {problem}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily job: collect, check and extract articles.")
    parser.add_argument("--db", required=True,
                        help="database name, e.g. disaster_db_test (no default on purpose)")
    parser.add_argument("--scrape-limit", type=int, default=SCRAPE_LIMIT,
                        help="most pages to download")
    parser.add_argument("--extract-limit", type=int, default=EXTRACT_LIMIT,
                        help="most articles to check and extract")
    parser.add_argument("--extract-minutes", type=int, default=EXTRACT_MINUTES,
                        help="no new extraction is started after this many minutes")
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-matching", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(BASE_DIR)          # Task Scheduler starts programs in another folder
    os.makedirs(LOG_DIR, exist_ok=True)
    log = Log(os.path.join(LOG_DIR, f"daily_{datetime.now():%Y-%m-%d_%H%M}.log"))
    try:
        try:
            take_lock()
        except AlreadyRunning as exc:
            log(f"Not started: {exc}")
            return 2
        try:
            from database.store import Store, get_database
            from extraction_service import OllamaBackend
            log(f"Daily run, database '{args.db}'")
            store = Store(get_database(args.db))
            store.setup()
            report = daily_run(
                store, backend=OllamaBackend(), log=log, scrape_limit=args.scrape_limit,
                extract_limit=args.extract_limit, extract_minutes=args.extract_minutes,
                skip_ingestion=args.skip_ingestion, skip_extraction=args.skip_extraction,
                skip_matching=args.skip_matching)
            log(f"Exit code {report.exit_code}")
            return report.exit_code
        except Exception:
            log("DAILY RUN CRASHED:\n" + traceback.format_exc())
            return 1
        finally:
            release_lock()
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
