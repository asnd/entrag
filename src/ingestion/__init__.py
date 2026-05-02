"""Document ingestion pipeline: parse, chunk, embed, store."""

import logging
from pathlib import Path
from typing import List

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.litellm import LiteLLMEmbedding

from src.config import get_settings
from src.scraper.parser import ParsedKBArticle, parse_directory

logger = logging.getLogger(__name__)

# Chunking constants
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def _article_to_documents(article: ParsedKBArticle) -> List[Document]:
    """Convert a parsed KB article into LlamaIndex Documents (chunks)."""
    docs = []
    base_meta = {
        "article_number": article.article_number,
        "title": article.title,
        "url": article.url,
        "product": article.product,
        "version": article.version,
        "last_updated": article.last_updated,
    }

    if article.sections:
        for section in article.sections:
            doc = Document(
                text=f"## {section.heading}\n{section.content}",
                metadata={
                    **base_meta,
                    "section_type": section.section_type,
                    "section_heading": section.heading,
                },
            )
            docs.append(doc)
    else:
        doc = Document(
            text=article.full_text,
            metadata=base_meta,
        )
        docs.append(doc)

    return docs


def ingest_directory(
    source_dir: Path,
    reset: bool = False,
) -> int:
    """Ingest all parsed KB articles from a directory into LanceDB.

    Args:
        source_dir: Directory containing HTML files.
        reset: If True, wipe the vector store before ingestion.

    Returns:
        Number of document chunks ingested.
    """
    settings = get_settings()
    logger.info(f"Ingesting articles from {source_dir}...")

    articles = parse_directory(source_dir)
    if not articles:
        logger.warning(f"No articles found in {source_dir}")
        return 0

    logger.info(f"Parsed {len(articles)} articles. Creating documents...")

    all_docs: List[Document] = []
    for article in articles:
        docs = _article_to_documents(article)
        all_docs.extend(docs)

    logger.info(f"Total documents (chunks): {len(all_docs)}")

    # Always use LiteLLM (supports both remote and local models)
    logger.info(f"Using embedding via LiteLLM: {settings.litellm_embedding_model}")
    embed_model = LiteLLMEmbedding(
        model=settings.litellm_embedding_model,
        api_base=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )

    # Set up chunker
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Set up vector store
    lancedb_path = Path(settings.lancedb_path)
    if reset and lancedb_path.exists():
        logger.warning(f"Resetting vector store at {lancedb_path}")
        import shutil
        shutil.rmtree(lancedb_path)

    lancedb_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store = LanceDBVectorStore(uri=str(lancedb_path))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Building vector index (this may take a while with local embeddings)...")
    index = VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    logger.info(f"Successfully ingested {len(all_docs)} document chunks into LanceDB at {lancedb_path}")
    return len(all_docs)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/raw")
    reset = "--reset" in sys.argv
    count = ingest_directory(src, reset=reset)
    print(f"Ingested {count} documents.")
