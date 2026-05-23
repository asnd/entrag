"""Gradio-based RAG chat interface with auto-exposed API.

Usage:
    python -m src.app
"""

import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from src.config import get_settings
from src.generation import RAGGenerator, create_generator
from src.retrieval import RetrievalEngine, create_retrieval_engine

if TYPE_CHECKING:
    import gradio as gr

logger = logging.getLogger(__name__)


def create_app(
    engine_factory: Callable[[], RetrievalEngine] = create_retrieval_engine,
    generator_factory: Callable[[], RAGGenerator] = create_generator,
) -> "gr.ChatInterface":
    """Create the Gradio chat interface."""
    import gradio as gr

    engine = engine_factory()
    generator = generator_factory()

    def rag_chat(message: str, _history: list) -> Generator[str, None, None]:
        """Search the VMware/Broadcom KB and return an LLM-synthesised answer with citations.

        Performs hybrid vector + full-text search over indexed KB articles, reranks
        the results, then calls the configured LLM to synthesise a grounded answer.
        Every claim in the answer is cited back to a specific KB article number.

        Args:
            message: The user's question about a VMware or Broadcom product issue.

        Returns:
            A streamed, cited answer drawn from indexed KB articles.
        """
        try:
            chunks = engine.search(message)
            yield from generator.generate_stream(message, chunks)
        except FileNotFoundError:
            yield (
                "No KB index is available yet. Run `entrag-ingest --source ./data/raw --reset` "
                "and try again."
            )
        except ValueError as exc:
            yield f"Configuration error: {exc}"
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            logger.exception("RAG query failed: %s", exc)
            yield (
                "The KB assistant hit an unexpected server error while processing that query. "
                "Check the application logs."
            )

    demo = gr.ChatInterface(
        fn=rag_chat,
        title="EntRAG - VMware/Broadcom KB Assistant",
        description=(
            "Ask questions about VMware/Broadcom products. "
            "Answers are synthesised from indexed KB articles with citations."
        ),
        api_name="rag_query",
    )
    return demo


def main() -> None:
    """Launch the Gradio app."""
    demo = create_app()
    settings = get_settings()
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        mcp_server=True,
    )


if __name__ == "__main__":
    main()
