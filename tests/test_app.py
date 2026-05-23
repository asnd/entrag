"""Tests for the Gradio application wrapper."""

from collections.abc import Generator

from src.app import create_app


class _FakeEngine:
    def __init__(self, chunks: list | None = None):
        self._chunks = chunks or []

    def search(self, message: str) -> list:
        return self._chunks


class _FailingEngine:
    def __init__(self, exc: Exception):
        self.exc = exc

    def search(self, message: str) -> list:
        raise self.exc


class _FakeGenerator:
    def __init__(self, response: str):
        self.response = response

    def generate_stream(self, query: str, chunks: list) -> Generator[str, None, None]:
        yield self.response


def _collect(gen: Generator[str, None, None]) -> str:
    return "".join(gen)


def test_create_app_uses_retrieval_engine():
    """The chat interface should delegate responses to the retrieval engine."""
    app = create_app(lambda: _FakeEngine(), lambda: _FakeGenerator("matched answer"))

    result = _collect(app.fn("boot failure", []))
    assert result == "matched answer"


def test_create_app_handles_missing_index():
    """The chat interface should surface an actionable ingestion hint."""
    app = create_app(
        lambda: _FailingEngine(FileNotFoundError("missing")),
        lambda: _FakeGenerator(""),
    )

    result = _collect(app.fn("boot failure", []))
    assert "entrag-ingest" in result
    assert "No KB index" in result


def test_create_app_handles_value_error():
    """The chat interface should surface configuration errors."""
    app = create_app(
        lambda: _FailingEngine(ValueError("LITELLM_API_KEY is not configured")),
        lambda: _FakeGenerator(""),
    )

    result = _collect(app.fn("test query", []))
    assert "Configuration error" in result
    assert "LITELLM_API_KEY" in result


def test_create_app_handles_generic_error():
    """The chat interface should surface a generic server error."""
    app = create_app(
        lambda: _FailingEngine(RuntimeError("unexpected failure")),
        lambda: _FakeGenerator(""),
    )

    result = _collect(app.fn("test query", []))
    assert "unexpected server error" in result


def test_create_app_handles_empty_message():
    """The chat interface should handle empty user input."""
    app = create_app(lambda: _FakeEngine(), lambda: _FakeGenerator("no results"))

    result = _collect(app.fn("", []))
    assert result == "no results"
