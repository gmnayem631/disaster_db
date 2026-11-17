"""Tests for extraction_service.OllamaBackend.

The first group uses a stand-in client, so it checks OUR code (what we send,
how we read the reply, how we classify errors) without Ollama.

The last group is marked `qwen` and talks to the real model. It is skipped
unless you ask for it:

    $env:RUN_QWEN = "1"; venv\\Scripts\\python -m pytest tests -m qwen -s
"""
from types import SimpleNamespace

import pytest

from extraction.branch_b.schemas import BranchBResult
from extraction_service import (BackendError, BackendUnavailable, OllamaBackend,
                                extract_article)


class StandInClient:
    """Records what it was asked and returns a prepared reply or raises."""

    def __init__(self, reply='{"claims": []}', error=None):
        self.reply, self.error, self.requests = reply, error, []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.reply),
                               prompt_eval_count=812, eval_count=340)

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="other:1b", digest="aaa"),
                                       SimpleNamespace(model="qwen3:8b", digest="500a1f067a9f")])


class HttpError(Exception):
    """Looks like ollama.ResponseError: an exception carrying a status code."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


SCHEMA = BranchBResult.model_json_schema()


# ── Without Ollama ─────────────────────────────────────────────────────────
def test_request_has_the_required_shape():
    client = StandInClient()
    OllamaBackend(client=client).complete("system text", "user text", SCHEMA)
    [request] = client.requests

    assert request["model"] == "qwen3:8b"
    assert request["think"] is False                 # top level ...
    assert "think" not in request["options"]         # ... not inside options
    assert request["options"]["temperature"] == 0
    assert request["options"]["num_gpu"] == 0        # CPU only
    assert request["options"]["num_ctx"] == 8192
    assert request["format"] == SCHEMA
    assert request["stream"] is False
    assert request["messages"] == [{"role": "system", "content": "system text"},
                                   {"role": "user", "content": "user text"}]


def test_reply_and_token_counts_are_passed_on():
    response = OllamaBackend(client=StandInClient('{"claims": []}')).complete("s", "u", SCHEMA)
    assert response.content == '{"claims": []}'
    assert response.prompt_tokens == 812
    assert response.output_tokens == 340
    assert response.seconds >= 0


def test_missing_content_becomes_empty_text():
    response = OllamaBackend(client=StandInClient(reply=None)).complete("s", "u", SCHEMA)
    assert response.content == ""                    # the service treats this as a failed attempt


def test_server_not_running_is_unavailable():
    backend = OllamaBackend(client=StandInClient(error=ConnectionError("Failed to connect")))
    with pytest.raises(BackendUnavailable):
        backend.complete("s", "u", SCHEMA)


def test_model_not_installed_is_unavailable():
    backend = OllamaBackend(client=StandInClient(error=HttpError("model not found", 404)))
    with pytest.raises(BackendUnavailable):
        backend.complete("s", "u", SCHEMA)


@pytest.mark.parametrize("error", [HttpError("internal error", 500), TimeoutError("timed out")])
def test_other_failures_are_retryable_errors(error):
    backend = OllamaBackend(client=StandInClient(error=error))
    with pytest.raises(BackendError) as caught:
        backend.complete("s", "u", SCHEMA)
    assert not isinstance(caught.value, BackendUnavailable)


def test_describe_records_model_digest_and_options():
    info = OllamaBackend(client=StandInClient(), num_ctx=4096).describe()
    assert info["backend"] == "ollama" and info["model"] == "qwen3:8b"
    assert info["model_digest"] == "500a1f067a9f"
    assert info["think"] is False
    assert info["options"] == {"temperature": 0, "num_gpu": 0, "num_ctx": 4096, "num_predict": 4096}


def test_changing_the_request_options_does_not_change_the_backend():
    client = StandInClient()
    backend = OllamaBackend(client=client)
    backend.complete("s", "u", SCHEMA)
    client.requests[0]["options"]["num_gpu"] = 99
    assert backend.options["num_gpu"] == 0


def test_a_timeout_inside_extract_article_is_retried_not_raised():
    client = StandInClient(error=TimeoutError("timed out"))
    result = extract_article({"_id": "a", "text": "Floods killed 31 people in Feni."},
                             OllamaBackend(client=client), branch_a=lambda *a: {})
    assert result.status == "partial"
    assert len(client.requests) == 2                 # tried, retried, then reported
    assert "TimeoutError" in result.batches[0].error["message"]


# ── With real Qwen (skipped unless RUN_QWEN=1) ─────────────────────────────
QWEN_TEXT = ("Floods have devastated several districts of Bangladesh. "
             "The nationwide death toll has risen to 31 after three more people died in Feni district. "
             "At least 58 lakh people have been affected across 11 districts. "
             "Around 10,000 families remained marooned in Companiganj upazila.")


def _qwen_branch_a(text, source_url, publish_date=None):
    return {"disaster_type": [{"value": "Floods", "source": "model"}],
            "locations": {"districts": [{"value": "Feni", "source": "model"}],
                          "upazilas": [{"value": "Companiganj", "source": "model"}], "unions": []},
            "agencies_involved": [], "raw_entities": []}


@pytest.mark.qwen
def test_real_qwen_answers_every_candidate():
    """Checks the connection and the reply format. It does NOT check that
    Qwen's meanings are right; that is the gold comparison."""
    backend = OllamaBackend()
    result = extract_article({"_id": "qwen-smoke", "text": QWEN_TEXT}, backend,
                             branch_a=_qwen_branch_a)

    print(f"\nstatus={result.status}  candidates={result.candidate_count}  "
          f"seconds={result.seconds:.0f}  model_digest={backend.describe()['model_digest']}")
    for b in result.batches:
        print(f"  batch {b.index}: {b.status}, attempts={b.attempts}, "
              f"prompt_tokens={b.prompt_tokens}, output_tokens={b.output_tokens}, "
              f"seconds={b.seconds:.0f}, flags={b.flags}, error={b.error}")
    for c in result.claims:
        print(f"  {c['candidate_id']:<4} {c['raw_value']!r:<16} {c['impact_type']:<16} "
              f"{c['aggregation']:<14} unit={c['unit']!s:<9} value={c['normalized_value']} "
              f"{c['validation_status']} {c['validation_flags']}")

    assert result.status == "complete", result.error
    assert result.unanswered == []
    assert all(b.prompt_tokens < backend.options["num_ctx"] for b in result.batches)
