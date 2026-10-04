"""CLI: every subcommand drives the same pipeline as the HTTP API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import build_parser, main
from app.config import reset_settings_cache


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at a scratch index and make sure settings are re-read.

    ``seed`` resolves ``data/sample_docs`` relative to the working directory, so
    the cwd is pinned to the repo root regardless of where pytest was invoked.
    """
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vectorstore"))
    monkeypatch.setenv("RAG_EMBED_DIM", "128")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LLM_PROVIDER", "echo")
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_accepts_every_documented_subcommand() -> None:
    parser = build_parser()
    assert parser.parse_args(["seed"]).command == "seed"
    assert parser.parse_args(["docs"]).command == "docs"
    assert parser.parse_args(["run"]).command == "run"
    assert parser.parse_args(["ask", "why?"]).question == "why?"
    assert parser.parse_args(["ask", "why?", "--top-k", "2"]).top_k == 2
    assert parser.parse_args(["search", "what"]).query == "what"


def test_docs_reports_an_empty_index(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["docs"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 0
    assert payload["chunks"] == 0
    assert payload["embedder"].startswith("hashing-")


def test_seed_indexes_sample_documents(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["seed"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seeded_documents"] == 2, "lab-results.md and medication-safety.md"
    assert payload["indexed_chunks"] > 0


def test_ask_returns_a_grounded_answer_with_citations(capsys: pytest.CaptureFixture[str]) -> None:
    main(["seed"])
    capsys.readouterr()

    assert main(["ask", "What might an out-of-range lab test result mean?"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["grounded"] is True
    assert payload["citations"]
    assert payload["citations"][0]["source"] == "lab-results.md"
    assert "outside the range" in payload["answer"] or "reference range" in payload["answer"]
    assert payload["latency_ms"] >= 0.0


def test_ask_on_an_empty_index_reports_no_grounding(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "What is the refund policy?"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["grounded"] is False
    assert payload["citations"] == []
    # The CLI does not seed on boot the way the server does, so an empty index
    # has to say so rather than look like a retrieval miss.
    assert "ai-app seed" in payload["hint"]


def test_search_on_an_empty_index_hints_at_seeding(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "anything"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"] == []
    assert "ai-app seed" in payload["hint"]


def test_search_returns_scored_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    main(["seed"])
    capsys.readouterr()

    assert (
        main(["search", "Why tell a pharmacist about medicines and supplements?", "--top-k", "3"])
        == 0
    )
    hits = json.loads(capsys.readouterr().out)

    assert hits
    assert hits[0]["source"] == "medication-safety.md"
    assert all(0.0 <= hit["score"] <= 1.0 for hit in hits)


def test_seed_then_docs_shows_the_persisted_index(capsys: pytest.CaptureFixture[str]) -> None:
    main(["seed"])
    capsys.readouterr()

    main(["docs"])
    payload = json.loads(capsys.readouterr().out)

    sources = sorted(doc["source"] for doc in payload["documents"])
    assert sources == ["lab-results.md", "medication-safety.md"]
    assert payload["chunks"] > 0


def test_stdout_stays_clean_json_when_logging_is_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression guard: at the default log level the CLI used to write log lines
    # to stdout, which breaks `ai-app ask ... | jq`.
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    reset_settings_cache()

    assert main(["ask", "anything"]) == 0
    captured = capsys.readouterr()

    json.loads(captured.out)  # stdout must be parseable JSON on its own
    assert captured.err, "log output belongs on stderr"


def test_unknown_command_exits_with_an_error() -> None:
    # argparse rejects unknown subcommands before dispatch, so this is a
    # SystemExit(2) with usage on stderr, not a traceback.
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2
