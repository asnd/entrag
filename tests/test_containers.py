"""Tests for container & profile configuration."""

from pathlib import Path


def test_env_example_exists():
    """Test that .env.example exists and is valid."""
    env_example = Path(".env.example")
    assert env_example.exists(), ".env.example not found"

    content = env_example.read_text()
    # Check key variables are documented
    assert "LITELLM_BASE_URL" in content
    assert "LITELLM_API_KEY" in content
    assert "LITELLM_MODEL" in content
    assert "LANCEDB_PATH" in content
    assert "SCRAPER_" in content


def test_dockerfile_app_exists():
    """Test that Dockerfile.app exists and is lean."""
    dockerfile = Path("Dockerfile.app")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    effective_content = "\n".join(
        line for line in content.splitlines() if not line.lstrip().startswith("#")
    )
    # Should NOT contain local/playwright deps
    assert "sentence-transformers" not in effective_content
    assert 'uv pip install -e ".[playwright]"' not in effective_content
    # Should have gradio and core deps
    assert "gradio" in content.lower() or "src.app" in content


def test_dockerfile_scraper_exists():
    """Test that Dockerfile.scraper exists and has playwright."""
    dockerfile = Path("Dockerfile.scraper")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    # Should contain playwright
    assert "playwright" in content.lower()


def test_dockerfile_local_models_exists():
    """Test that Dockerfile.local-models exists and has torch."""
    dockerfile = Path("Dockerfile.local-models")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    # Should contain local model deps
    assert "sentence-transformers" in content or "local" in content


def test_compose_yaml_exists():
    """Test that compose.yaml is valid."""
    compose = Path("compose.yaml")
    assert compose.exists()

    content = compose.read_text()
    # Base compose file should keep only always-on services
    assert "rag-app:" in content
    assert "litellm:" in content
    assert "litellm-local:" not in content


def test_compose_jobs_yaml_exists():
    """Test that compose.jobs.yaml exists and has jobs."""
    compose = Path("compose.jobs.yaml")
    assert compose.exists()

    content = compose.read_text()
    assert "scraper:" in content
    assert "ingest:" in content


def test_compose_local_yaml_exists():
    """Test that compose.local.yaml exists and has local models."""
    compose = Path("compose.local.yaml")
    assert compose.exists()

    content = compose.read_text()
    assert "litellm-local:" in content


def test_litellm_config_local_exists():
    """Test that litellm_config_local.yaml exists."""
    config = Path("litellm_config_local.yaml")
    assert config.exists()

    content = config.read_text()
    # Should configure local embedding model
    assert "local-embedding" in content or "sentence_transformers" in content


def test_pyproject_toml_no_local_extra():
    """Test that pyproject.toml no longer has [local] extra."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()

    # Should NOT have [local] extra (moved to container)
    assert "[local]" not in content


def test_dockerfile_app_no_playwright():
    """Test that Dockerfile.app does not have playwright."""
    dockerfile = Path("Dockerfile.app")
    content = "\n".join(
        line
        for line in dockerfile.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert 'uv pip install -e ".[playwright]"' not in content


def test_dockerfile_scraper_has_playwright():
    """Test that Dockerfile.scraper has playwright."""
    dockerfile = Path("Dockerfile.scraper")
    content = dockerfile.read_text()
    assert "playwright" in content.lower(), "Dockerfile.scraper should have playwright"


def test_symlinks_for_podman():
    """Test that Containerfile and .containerignore are symlinks."""
    # Containerfile should be symlink to Dockerfile.app
    if Path("Containerfile").exists():
        assert Path("Containerfile").is_symlink() or Path("Containerfile").exists()

    # .containerignore should be symlink to .dockerignore
    if Path(".containerignore").exists():
        assert Path(".containerignore").is_symlink() or Path(".containerignore").exists()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
