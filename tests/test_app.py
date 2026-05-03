"""Tests for the Gradio application wrapper."""

from src.app import create_app


class _FakeEngine:
    def __init__(self, answer_text: str):
        self.answer_text = answer_text

    def answer(self, message: str) -> str:
        return f"{self.answer_text}: {message}"


def test_create_app_uses_retrieval_engine():
    """The chat interface should delegate responses to the retrieval engine."""
    app = create_app(lambda: _FakeEngine("matched"))

    assert app.fn("boot failure", []) == "matched: boot failure"


def test_create_app_handles_missing_index():
    """The chat interface should surface an actionable ingestion hint."""
    app = create_app(lambda: (_ for _ in ()).throw(FileNotFoundError("missing")))

    message = app.fn("boot failure", [])
    assert "entrag-ingest" in message
    assert "No KB index" in message
