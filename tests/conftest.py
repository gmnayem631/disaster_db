# Put the project root on sys.path so tests can import extraction.branch_b.*
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
