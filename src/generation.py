"""LLM generation layer — synthesises an answer from retrieved KB chunks."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

import litellm

from src.config import Settings, get_settings

if TYPE_CHECKING:
    from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert VMware/Broadcom support engineer with deep knowledge of vSphere, \
vCenter, ESXi, NSX, vSAN, and related products.

Answer the user's question using ONLY the Knowledge Base excerpts provided below. \
Follow these rules strictly:

1. Base your answer entirely on the excerpts — do not draw on outside knowledge.
2. Prioritise Resolution and Workaround sections when present.
3. Be precise and actionable; include exact steps, commands, or version numbers from the excerpts.
4. Cite the KB article number (e.g. KB318867) for every claim you make.
5. If the excerpts do not contain enough information to answer, say so explicitly \
   and suggest which KB sections might help.\
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.section_type.replace("_", " ").title() if chunk.section_type else "General"
        heading = f" — {chunk.section_heading}" if chunk.section_heading else ""
        parts.append(
            f"[{i}] KB{chunk.article_number}: {chunk.title}\n"
            f"Section: {section}{heading}\n\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


class RAGGenerator:
    """Calls the configured LLM with retrieved KB context and streams the answer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate_stream(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> Generator[str, None, None]:
        """Yield answer tokens from the LLM.  Yields a single error string on failure."""
        if not chunks:
            yield (
                "No relevant KB articles were found for that query. "
                "Try rephrasing, or run `entrag-ingest` to add more content."
            )
            return

        context = _format_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nKB Excerpts:\n\n{context}",
            },
        ]

        self.settings.validate_litellm_api_key()
        response = litellm.completion(
            model=self.settings.litellm_model,
            messages=messages,
            api_base=self.settings.litellm_base_url,
            api_key=self.settings.litellm_api_key,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        for event in response:
            token: str = event.choices[0].delta.content or ""
            if token:
                yield token


def create_generator() -> RAGGenerator:
    return RAGGenerator()
