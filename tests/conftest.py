"""Pytest configuration and shared fixtures."""

import os
import tempfile

import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.config import get_settings


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("TEST_SETTING=test_value\n")
        f.write("LITELLM_BASE_URL=http://test.local:4000\n")
        f.write("SCRAPER_USE_AUTH=false\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def clean_settings(temp_env_file: str, monkeypatch: MonkeyPatch):
    """Provide clean Settings instance for each test."""
    # Clear any cached settings
    get_settings.cache_clear()

    # Set ENV_FILE to use our temp file
    monkeypatch.setenv("ENV_FILE", temp_env_file)

    return get_settings()
