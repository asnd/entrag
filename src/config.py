"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_API_KEYS = frozenset({"sk-placeholder", "sk-your-key-here"})


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
    scraper_use_auth: bool = Field(default=False)  # Public scraping by default; auth is opt-in
    broadcom_username: str = Field(default="")
    broadcom_password: str = Field(default="")
    scraper_delay_seconds: float = Field(default=3.0, ge=0.0)  # Must be non-negative
    scraper_max_articles: int = Field(default=100, ge=1)  # Must be positive
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

    def ensure_litellm_api_key_configured(self) -> None:
        """Fail fast when the LiteLLM proxy key was not configured."""
        normalized_key = self.litellm_api_key.strip()
        if not normalized_key:
            raise ValueError(
                "LITELLM_API_KEY is not set. "
                "Set it in .env before running ingestion or retrieval."
            )
        if normalized_key in PLACEHOLDER_API_KEYS:
            raise ValueError(
                "LITELLM_API_KEY is not configured. "
                "Set it in .env before running ingestion or retrieval."
            )

    def resolved_litellm_base_url(self, use_local_models: bool = False) -> str:
        """Resolve the LiteLLM endpoint for remote or local model serving."""
        if not use_local_models:
            return self.litellm_base_url

        replacements = {
            "http://localhost:4000": "http://localhost:4001",
            "http://127.0.0.1:4000": "http://127.0.0.1:4001",
            "http://litellm:4000": "http://litellm-local:4000",
        }
        return replacements.get(self.litellm_base_url, self.litellm_base_url)

    def resolved_embedding_model(self, use_local_models: bool = False) -> str:
        """Resolve the embedding model alias exposed by LiteLLM."""
        if use_local_models or self.embedding_provider == "local":
            return "local-embedding"
        return self.litellm_embedding_model

    @field_validator("scraper_delay_seconds", mode="before")
    @classmethod
    def validate_scraper_delay(cls, v):
        """Ensure scraper delay is non-negative."""
        # Convert to float if it's a string from env var
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError("scraper_delay_seconds must be a number")
        if val < 0:
            raise ValueError("scraper_delay_seconds must be non-negative")
        return val

    @field_validator("scraper_max_articles", mode="before")
    @classmethod
    def validate_scraper_max_articles(cls, v):
        """Ensure scraper max articles is positive."""
        # Convert to int if it's a string from env var
        try:
            val = int(v)
        except (ValueError, TypeError):
            raise ValueError("scraper_max_articles must be an integer")
        if val < 1:
            raise ValueError("scraper_max_articles must be positive")
        return val

    @field_validator("retrieval_hybrid_alpha", mode="before")
    @classmethod
    def validate_retrieval_alpha(cls, v):
        """Ensure retrieval alpha is between 0 and 1."""
        # Convert to float if it's a string from env var
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError("retrieval_hybrid_alpha must be a number")
        if not 0 <= val <= 1:
            raise ValueError("retrieval_hybrid_alpha must be between 0 and 1")
        return val


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    return Settings()
