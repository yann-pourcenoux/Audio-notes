"""
Pytest configuration and fixtures for audio-notes tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import os
import sys

# Import the main modules for testing
from audio_notes.config import ConfigManager, AudioNotesConfig
from audio_notes.obsidian_writer import ObsidianWriter, OllamaClient


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_vault_path(temp_dir: Path) -> Path:
    """Provide a temporary Obsidian vault directory."""
    vault_path = temp_dir / "test_vault"
    vault_path.mkdir(parents=True, exist_ok=True)
    
    # Create the Audio Notes subdirectory
    audio_notes_dir = vault_path / "Audio Notes"
    audio_notes_dir.mkdir(parents=True, exist_ok=True)
    
    return vault_path


@pytest.fixture
def test_config_path(temp_dir: Path) -> Path:
    """Provide a temporary configuration file path."""
    return temp_dir / "test_config.json"


@pytest.fixture
def config_manager(test_config_path: Path) -> ConfigManager:
    """Provide a ConfigManager instance with a temporary config file."""
    return ConfigManager(str(test_config_path))


@pytest.fixture
def default_config() -> AudioNotesConfig:
    """Provide a default AudioNotesConfig instance."""
    return AudioNotesConfig()


@pytest.fixture
def sample_transcription() -> str:
    """Provide a sample transcription text for testing."""
    return """
    This is a sample transcription for testing purposes. 
    It contains multiple sentences and some technical terms like AI, machine learning, 
    and natural language processing. The transcription should be long enough 
    to test various features like title generation, summary creation, and content enhancement.
    """


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Provide sample metadata for testing."""
    return {
        "duration": 120.5,
        "language": "en",
        "model": "whisper-large-v3",
        "timestamp": "2024-01-15T10:30:00Z",
        "file_size": 2048000,
        "sample_rate": 16000
    }


@pytest.fixture
def obsidian_writer_with_ai(test_vault_path: Path) -> ObsidianWriter:
    """Provide an ObsidianWriter instance with AI enhancement enabled."""
    return ObsidianWriter(vault_path=str(test_vault_path), use_ai_enhancement=True)


@pytest.fixture
def obsidian_writer_no_ai(test_vault_path: Path) -> ObsidianWriter:
    """Provide an ObsidianWriter instance with AI enhancement disabled."""
    return ObsidianWriter(vault_path=str(test_vault_path), use_ai_enhancement=False)


@pytest.fixture
def ollama_client() -> OllamaClient:
    """Provide an OllamaClient instance for testing."""
    return OllamaClient(model="qwen3:0.6b")


@pytest.fixture
def sample_audio_file_path(temp_dir: Path) -> Path:
    """Provide a path for a sample audio file (for testing file operations)."""
    audio_file = temp_dir / "sample_audio.wav"
    # Create an empty file for testing purposes
    audio_file.touch()
    return audio_file


@pytest.fixture
def cli_args_sample() -> Dict[str, Any]:
    """Provide sample CLI arguments for testing."""
    return {
        "language": "en",
        "task": "transcribe",
        "timestamps": "sentence",
        "precision": "float32",
        "temperature": 0.0,
        "beam_size": 5,
        "processing_method": "sequential",
        "output_format": "text",
        "output_dir": None,
        "vault_path": None,
        "enhance_notes": True,
        "batch_size": 1,
        "max_length": 30,
        "verbose": False
    }


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide the test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Ensure the package is importable
    if "src" not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    # Set environment variables for testing
    os.environ["AUDIO_NOTES_TEST_MODE"] = "true"
    
    yield
    
    # Cleanup after test
    if "AUDIO_NOTES_TEST_MODE" in os.environ:
        del os.environ["AUDIO_NOTES_TEST_MODE"]


@pytest.fixture
def mock_ollama_unavailable(monkeypatch):
    """Mock Ollama as unavailable for testing fallback behavior."""
    def mock_is_available():
        return False
    
    monkeypatch.setattr("audio_notes.obsidian_writer.OllamaClient.is_available", mock_is_available)


@pytest.fixture
def mock_ollama_available(monkeypatch):
    """Mock Ollama as available for testing AI features."""
    def mock_is_available():
        return True
    
    def mock_generate_text(prompt, max_tokens=500, temperature=0.7):
        # Return a simple mock response based on the prompt
        if "title" in prompt.lower():
            return "Test Audio Note Title"
        elif "summary" in prompt.lower():
            return "This is a test summary of the audio content."
        elif "tags" in prompt.lower():
            return "test, audio, transcription"
        else:
            return "Mock AI response"
    
    monkeypatch.setattr("audio_notes.obsidian_writer.OllamaClient.is_available", mock_is_available)
    monkeypatch.setattr("audio_notes.obsidian_writer.OllamaClient.generate_text", mock_generate_text)


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "config: Configuration tests")
    config.addinivalue_line("markers", "ollama: Ollama/AI tests")
    config.addinivalue_line("markers", "error_handling: Error handling tests")
    config.addinivalue_line("markers", "workflow: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "network: Tests requiring network access")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on their file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "config" in str(item.fspath):
            item.add_marker(pytest.mark.config)
        elif "ollama" in str(item.fspath):
            item.add_marker(pytest.mark.ollama)
        elif "error_handling" in str(item.fspath):
            item.add_marker(pytest.mark.error_handling)
        elif "workflow" in str(item.fspath):
            item.add_marker(pytest.mark.workflow)
        
        # Mark tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["comprehensive", "full", "complete"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that require network access
        if any(keyword in item.name.lower() for keyword in ["network", "api", "ollama", "external"]):
            item.add_marker(pytest.mark.network) 