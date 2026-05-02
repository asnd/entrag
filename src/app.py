"""Gradio-based RAG chat interface with auto-exposed API.

Usage:
    python -m src.app
"""

from src.config import get_settings


def create_app():
    """Create the Gradio chat interface (Phase 5 - not yet implemented)."""
    import gradio as gr

    def rag_chat(message: str, history: list) -> str:
        """Placeholder RAG chat function."""
        return (
            f"[Phase 5 not yet implemented] You asked: {message}\n\n"
            "The retrieval engine and UI will be implemented in Phase 4-5."
        )

    demo = gr.ChatInterface(
        fn=rag_chat,
        title="EntRAG - VMware/Broadcom KB Assistant",
        description="Ask questions about VMware/Broadcom products. Powered by RAG.",
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
