"""Tests for the Gradio application wrapper."""

from src.app import create_app


class _FakeEngine:
    def __init__(self, answer_text: str):
        self.answer_text = answer_text

    def answer(self, message: str) -> str:
        return f"{self.answer_text}: {message}"


class _FailingEngine:
    def __init__(self, exc: Exception):
        self.exc = exc

    def answer(self, message: str) -> str:
        raise self.exc


def test_create_app_uses_retrieval_engine():
    """The chat interface should delegate responses to the retrieval engine."""
    app = create_app(lambda: _FakeEngine("matched"))

    assert app.fn("boot failure", []) == "matched: boot failure"


def test_create_app_handles_missing_index():
    """The chat interface should surface an actionable ingestion hint."""
    app = create_app(lambda: _FailingEngine(FileNotFoundError("missing")))

    message = app.fn("boot failure", [])
    assert "entrag-ingest" in message
    assert "No KB index" in message


def test_create_app_handles_value_error():
    """The chat interface should surface configuration errors."""
    app = create_app(lambda: _FailingEngine(ValueError("LITELLM_API_KEY is not configured")))

    message = app.fn("test query", [])
    assert "Configuration error" in message
    assert "LITELLM_API_KEY" in message


def test_create_app_handles_generic_error():
    """The chat interface should surface a generic server error."""
    app = create_app(lambda: _FailingEngine(RuntimeError("unexpected failure")))

    message = app.fn("test query", [])
    assert "unexpected server error" in message


def test_create_app_handles_empty_message():
    """The chat interface should handle empty user input."""
    app = create_app(lambda: _FakeEngine("matched"))

    result = app.fn("", [])
    assert result == "matched: "
