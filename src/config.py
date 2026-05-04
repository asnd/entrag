"""Application configuration using pydantic-settings.

All settings are loaded from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LiteLLM
    litellm_base_url: str = Field(default="http://localhost:4000")
    litellm_api_key: str = Field(default="sk-placeholder")
    litellm_model: str = Field(default="gpt-4o")

    # Embedding
    embedding_provider: Literal["litellm", "local"] = Field(default="litellm")
    litellm_embedding_model: str = Field(default="text-embedding-3-small")
    local_embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")

    # LanceDB
    lancedb_path: Path = Field(default=Path("./data/lancedb"))

    # Scraper
    scraper_use_auth: bool = Field(default=False)
    broadcom_username: str = Field(default="")
    broadcom_password: SecretStr = Field(default=SecretStr(""))
    scraper_delay_seconds: float = Field(default=3.0, ge=0.0)
    scraper_max_articles: int = Field(default=100, ge=1)
    scraper_output_dir: Path = Field(default=Path("./data/raw"))

    # Gradio
    gradio_server_name: str = Field(default="0.0.0.0")
    gradio_server_port: int = Field(default=7860, ge=1, le=65535)

    # Reranker
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_n: int = Field(default=5, ge=1)

    # Retrieval
    retrieval_similarity_top_k: int = Field(default=10, ge=1)
    retrieval_hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    return Settings()
