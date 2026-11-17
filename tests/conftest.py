# Put the project root on sys.path so tests can import extraction.branch_b.*
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The only real database tests may ever use. Never "disaster_db".
REAL_TEST_DB = "disaster_db_test"
PROTECTED_DB = "disaster_db"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "qwen: needs Ollama running with qwen3:8b; slow on CPU. "
                   "Skipped unless the environment variable RUN_QWEN is 1.")


def pytest_collection_modifyitems(config, items):
    """Tests marked `qwen` are skipped in a normal run."""
    if os.environ.get("RUN_QWEN") == "1":
        return
    skip = pytest.mark.skip(reason="needs real Qwen; set RUN_QWEN=1 to run")
    for item in items:
        if "qwen" in item.keywords:
            item.add_marker(skip)


def _empty(db):
    assert db.name == REAL_TEST_DB, f"refusing to empty database {db.name!r}"
    for name in db.list_collection_names():
        db.drop_collection(name)


def _counts(db):
    return {name: db[name].count_documents({}) for name in sorted(db.list_collection_names())}


@pytest.fixture(scope="session")
def real_mongo_watch():
    """Only with RUN_MONGO=1. Counts the documents in the real project database
    before and after the whole run (read only) and prints both, as evidence
    that the tests did not touch it."""
    from database.store import get_database
    protected = get_database(PROTECTED_DB)
    before = _counts(protected)
    print(f"\n[{PROTECTED_DB}] before the tests: {before}")
    yield
    after = _counts(protected)
    print(f"\n[{PROTECTED_DB}] after the tests:  {after}")
    if after != before:
        print(f"[{PROTECTED_DB}] CHANGED during the run. If pipeline.py was running at the "
              f"same time, that explains it; otherwise investigate.")


@pytest.fixture
def test_db(request):
    """An empty database for one test.

    Normal run: mongomock, an in-memory imitation of MongoDB.
    With RUN_MONGO=1: the real server from .env, database `disaster_db_test` only.
    """
    if os.environ.get("RUN_MONGO") != "1":
        import mongomock
        yield mongomock.MongoClient(tz_aware=True)[REAL_TEST_DB]
        return

    request.getfixturevalue("real_mongo_watch")
    from database.store import get_database
    db = get_database(REAL_TEST_DB)
    _empty(db)
    yield db
    _empty(db)
