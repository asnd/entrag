"""Tests for hybrid retrieval and reranking."""

from dataclasses import dataclass

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQueryResult

from src.config import Settings
from src.retrieval import RetrievalEngine


@dataclass
class _FakeEmbedModel:
    def get_query_embedding(self, query: str) -> list[float]:
        return [0.1, 0.2, float(len(query))]


class _FakeVectorStore:
    def __init__(self, result: VectorStoreQueryResult):
        self.result = result
        self.seen_queries = []

    def query(self, query):
        self.seen_queries.append(query)
        return self.result


def test_search_uses_hybrid_query_and_reranks_results():
    """Resolution-heavy matches should outrank weaker symptom-only matches."""
    vector_store = _FakeVectorStore(
        VectorStoreQueryResult(
            nodes=[
                TextNode(
                    text="The resolution is to install the corrected ESXi boot patch.",
                    metadata={
                        "article_number": "10001",
                        "title": "Fix ESXi boot failure",
                        "url": "https://kb.example.com/10001",
                        "section_type": "resolution",
                        "section_heading": "Resolution",
                    },
                ),
                TextNode(
                    text="Symptoms include boot failure after the firmware upgrade.",
                    metadata={
                        "article_number": "10002",
                        "title": "Boot failure symptoms",
                        "url": "https://kb.example.com/10002",
                        "section_type": "symptom",
                        "section_heading": "Symptoms",
                    },
                ),
            ],
            similarities=[0.45, 0.9],
        )
    )
    engine = RetrievalEngine(
        settings=Settings(
            litellm_api_key="sk-live",
            reranker_top_n=2,
            retrieval_similarity_top_k=2,
            retrieval_hybrid_alpha=0.3,
        ),
        vector_store=vector_store,
        embed_model=_FakeEmbedModel(),
    )

    results = engine.search("how to fix ESXi boot failure", top_k=2)

    assert len(results) == 2
    assert results[0].article_number == "10001"
    assert vector_store.seen_queries[0].mode == VectorStoreQueryMode.HYBRID
    assert vector_store.seen_queries[0].alpha == 0.3
    assert vector_store.seen_queries[0].query_str == "how to fix ESXi boot failure"


def test_search_returns_empty_for_blank_query():
    """Blank user input should not call the vector store."""
    vector_store = _FakeVectorStore(VectorStoreQueryResult(nodes=[], similarities=[]))
    engine = RetrievalEngine(
        settings=Settings(litellm_api_key="sk-live"),
        vector_store=vector_store,
        embed_model=_FakeEmbedModel(),
    )

    assert engine.search("   ") == []
    assert vector_store.seen_queries == []


def test_answer_includes_sources_and_scores():
    """Formatted answers should include citations for the retrieved chunks."""
    vector_store = _FakeVectorStore(
        VectorStoreQueryResult(
            nodes=[
                TextNode(
                    text="Apply patch ESXi70U3c and reboot the host.",
                    metadata={
                        "article_number": "10003",
                        "title": "Apply the ESXi patch",
                        "url": "https://kb.example.com/10003",
                        "section_type": "resolution",
                        "section_heading": "Resolution",
                    },
                )
            ],
            similarities=[0.8],
        )
    )
    engine = RetrievalEngine(
        settings=Settings(litellm_api_key="sk-live", reranker_top_n=1),
        vector_store=vector_store,
        embed_model=_FakeEmbedModel(),
    )

    answer = engine.answer("apply ESXi patch")

    assert "Top KB matches for: apply ESXi patch" in answer
    assert "Source: https://kb.example.com/10003" in answer
    assert "score" in answer
