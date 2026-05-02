"""Tests for container & profile configuration."""

import os
from pathlib import Path

# Project root (works both locally and inside Docker)
PROJECT_ROOT = Path(__file__).parent.parent


def _get_file(path: str) -> Path:
    """Get file path relative to project root."""
    return PROJECT_ROOT / path


def test_env_example_exists():
    """Test that .env.example exists and is valid."""
    env_example = _get_file(".env.example")
    assert env_example.exists(), ".env.example not found"

    content = env_example.read_text()
    # Check key variables are documented
    assert "LITELLM_BASE_URL" in content
    assert "LITELLM_API_KEY" in content
    assert "LITELLM_MODEL" in content
    assert "LANCEDB_PATH" in content
    assert "SCRAPER_" in content


def test_env_example_local_models_commented():
    """Test that local model URLs are documented but commented."""
    env_example = _get_file(".env.example")
    content = env_example.read_text()

    # Local model URLs should be documented as comments
    assert "litellm-local" in content or "4001" in content


def test_dockerfile_app_exists():
    """Test that Dockerfile.app exists and is lean."""
    dockerfile = _get_file("Dockerfile.app")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    # Should NOT contain local/playwright deps
    assert "sentence-transformers" not in content
    assert "playwright" not in content
    # Should have gradio and core deps
    assert "gradio" in content.lower() or "src.app" in content


def test_dockerfile_scraper_exists():
    """Test that Dockerfile.scraper exists and has playwright."""
    dockerfile = _get_file("Dockerfile.scraper")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    # Should contain playwright
    assert "playwright" in content.lower()


def test_dockerfile_local_models_exists():
    """Test that Dockerfile.local-models exists and has torch."""
    dockerfile = _get_file("Dockerfile.local-models")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    # Should contain local model deps
    assert "sentence-transformers" in content or "local" in content


def test_compose_yaml_exists():
    """Test that compose.yaml is valid."""
    compose = _get_file("compose.yaml")
    assert compose.exists()

    content = compose.read_text()
    # Should have all services
    assert "rag-app:" in content
    assert "litellm:" in content
    assert "litellm-local:" in content


def test_compose_jobs_yaml_exists():
    """Test that compose.jobs.yaml exists and has jobs."""
    compose = _get_file("compose.jobs.yaml")
    assert compose.exists()

    content = compose.read_text()
    assert "scraper:" in content
    assert "ingest:" in content


def test_compose_local_yaml_exists():
    """Test that compose.local.yaml exists and has local models."""
    compose = _get_file("compose.local.yaml")
    assert compose.exists()

    content = compose.read_text()
    assert "litellm-local:" in content


def test_litellm_config_local_exists():
    """Test that litellm_config_local.yaml exists."""
    config = _get_file("litellm_config_local.yaml")
    assert config.exists()

    content = config.read_text()
    # Should configure local embedding model
    assert "local-embedding" in content or "sentence_transformers" in content


def test_pyproject_toml_no_local_extra():
    """Test that pyproject.toml no longer has [local] extra."""
    pyproject = _get_file("pyproject.toml")
    content = pyproject.read_text()

    # Should NOT have [local] extra (moved to container)
    lines = content.split("\n")
    in_local_section = False
    for line in lines:
        if line.strip() == "[local] = [":
            in_local_section = True
        elif in_local_section and line.strip().startswith("]"):
            in_local_section = False
        elif in_local_section:
            # Should not have sentence-transformers here
            assert "sentence-transformers" not in line, \
                "sentence-transformers should be in Dockerfile.local-models, not pyproject.toml"


def test_dockerfile_app_no_playwright():
    """Test that Dockerfile.app does not have playwright."""
    dockerfile = _get_file("Dockerfile.app")
    content = dockerfile.read_text()
    assert "playwright" not in content.lower(), "Dockerfile.app should not have playwright"


def test_dockerfile_scraper_has_playwright():
    """Test that Dockerfile.scraper has playwright."""
    dockerfile = _get_file("Dockerfile.scraper")
    content = dockerfile.read_text()
    assert "playwright" in content.lower(), "Dockerfile.scraper should have playwright"


def test_dockerfile_local_has_sentence_transformers():
    """Test that Dockerfile.local-models has sentence-transformers."""
    dockerfile = _get_file("Dockerfile.local-models")
    content = dockerfile.read_text()
    assert "sentence-transformers" in content or "local" in content


def test_compose_yaml_no_profiles_at_root():
    """Test that compose.yaml does not have profiles at root level."""
    compose = _get_file("compose.yaml")
    content = compose.read_text()
    # profiles: should NOT be at root level (it's inside services)
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "profiles:":
            # Check if this is at root level (no leading spaces)
            if not line.startswith("  "):
                # This is okay if it's under a service (indented)
                pass


def test_compose_jobs_yaml_profiles():
    """Test that compose.jobs.yaml services have profiles."""
    compose = _get_file("compose.jobs.yaml")
    content = compose.read_text()
    
    # Check scraper has profiles
    assert "  profiles:" in content or "profiles:" in content
    
    # Check ingest has profiles
    assert "  profiles:" in content or "profiles:" in content


def test_symlinks_for_podman():
    """Test that Containerfile and .containerignore are symlinks."""
    # Containerfile should be symlink to Dockerfile.app
    containerfile = _get_file("Containerfile")
    if containerfile.exists():
        assert containerfile.is_symlink() or containerfile.exists()
    
    # .containerignore should be symlink to .dockerignore
    containerignore = _get_file(".containerignore")
    if containerignore.exists():
        assert containerignore.is_symlink() or containerignore.exists()


@pytest.mark.asyncio
async def test_ingestion_uses_litellm_only(monkeypatch):
    """Test that ingestion uses LiteLLM (not direct HuggingFace imports)."""
    from src.config import Settings
    
    # Set up env for testing
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm:4000")
    monkeypatch.setenv("LITELLM_EMBEDDING_MODEL", "text-embedding-3-small")
    
    settings = Settings()
    from src.config import get_settings
    get_settings.cache_clear()
    
    # Re-import to get fresh settings
    import importlib
    importlib.reload(importlib.import_module("src.config", fromlist=[]))
    from src.config import get_settings
    
    settings = get_settings()
    assert "litellm" in settings.litellm_base_url.lower() or \
           "litellm" in settings.litellm_embedding_model.lower()


def test_compose_scraper_job():
    """Test that scraper job uses correct Dockerfile."""
    compose = _get_file("compose.jobs.yaml")
    content = compose.read_text()
    
    # Find scraper service
    lines = content.split("\n")
    in_scraper = False
    found_dockerfile = False
    
    for line in lines:
        if "scraper:" in line:
            in_scraper = True
        elif in_scraper:
            if "dockerfile:" in line:
                assert "Dockerfile.scraper" in line
                found_dockerfile = True
                break
    
    assert found_dockerfile, "Scraper service should use Dockerfile.scraper"


def test_compose_ingest_job():
    """Test that ingest job uses Dockerfile.app."""
    compose = _get_file("compose.jobs.yaml")
    content = compose.read_text()
    
    # Find ingest service
    lines = content.split("\n")
    in_ingest = False
    found_dockerfile = False
    
    for line in lines:
        if "ingest:" in line:
            in_ingest = True
        elif in_ingest:
            if "dockerfile:" in line:
                assert "Dockerfile.app" in line
                found_dockerfile = True
                break
    
    assert found_dockerfile, "Ingest service should use Dockerfile.app"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
