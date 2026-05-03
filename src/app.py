"""Gradio-based RAG chat interface with auto-exposed API.

Usage:
    python -m src.app
"""

import logging
from collections.abc import Callable

from src.config import get_settings
from src.retrieval import RetrievalEngine, get_retrieval_engine

logger = logging.getLogger(__name__)


def create_app(
    engine_factory: Callable[[], RetrievalEngine] = get_retrieval_engine,
):
    """Create the Gradio chat interface."""
    import gradio as gr

    def rag_chat(message: str, history: list) -> str:
        """Search the indexed KB and return ranked excerpts with citations."""
        del history
        try:
            return engine_factory().answer(message)
        except FileNotFoundError:
            return (
                "No KB index is available yet. Run `entrag-ingest --source ./data/raw --reset` "
                "and try again."
            )
        except ValueError as exc:
            return f"Configuration error: {exc}"
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            logger.exception("RAG query failed: %s", exc)
            return "The KB assistant could not complete that query. Check the application logs."

    demo = gr.ChatInterface(
        fn=rag_chat,
        title="EntRAG - VMware/Broadcom KB Assistant",
        description=(
            "Search the indexed VMware/Broadcom knowledge base and return the strongest "
            "matching KB excerpts with source citations."
        ),
        api_name="rag_query",
    )
    return demo


def main() -> None:
    """Launch the Gradio app."""
    settings = get_settings()
    demo = create_app()
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    main()
