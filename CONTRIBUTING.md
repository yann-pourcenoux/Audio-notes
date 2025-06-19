# Contributing to Audio Notes

🎉 Thank you for your interest in contributing to Audio Notes! We welcome contributions from the community and are excited to see what you'll bring to the project.

## 🚀 Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** your changes
6. **Submit** a pull request

## 📋 Development Setup

### Prerequisites

- Python 3.12+
- Git
- uv (recommended) or pip
- Ollama (for AI features testing)

### Local Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/audio-notes.git
cd audio-notes

# 2. Set up the development environment
pip install uv
uv sync --dev

# 3. Install pre-commit hooks (optional but recommended)
uv run pre-commit install

# 4. Verify the setup
uv run python -m audio_notes.cli status
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=audio_notes --cov-report=html

# Run specific test file
uv run pytest tests/test_whisper_transcriber.py

# Run tests in verbose mode
uv run pytest -v
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint code
uv run flake8 src/ tests/

# Type checking
uv run mypy src/
```

## 🎯 How to Contribute

### 🐛 Reporting Bugs

Before creating a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Test with the latest version** to ensure the bug still exists
3. **Gather relevant information** about your environment

When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages** or logs (if any)
- **Audio file details** (format, duration, size) if relevant

**Bug Report Template:**
```markdown
## Bug Description
Brief description of the issue.

## Steps to Reproduce
1. Run command: `uv run python -m audio_notes.cli ...`
2. Observe behavior: ...
3. Expected: ...
4. Actual: ...

## Environment
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python: [e.g., 3.12.0]
- Audio Notes Version: [e.g., 0.1.0]
- Audio File: [format, duration, size]

## Additional Context
Any other relevant information.
```

### ✨ Suggesting Features

We love feature suggestions! Before submitting:

1. **Check existing feature requests** to avoid duplicates
2. **Consider the scope** - does it fit the project's goals?
3. **Think about implementation** - is it technically feasible?

**Feature Request Template:**
```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why would this feature be useful? What problem does it solve?

## Proposed Implementation
How do you envision this working? (optional)

## Alternatives Considered
What other approaches have you considered?
```

### 🔧 Code Contributions

#### Types of Contributions We Welcome

- **Bug fixes** - Fix existing issues
- **Feature implementations** - Add new functionality
- **Performance improvements** - Optimize existing code
- **Documentation** - Improve or add documentation
- **Tests** - Add or improve test coverage
- **Examples** - Add usage examples or tutorials

#### Development Workflow

1. **Create an Issue** (for significant changes)
   - Discuss the change before implementing
   - Get feedback on the approach
   - Ensure alignment with project goals

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Make Your Changes**
   - Follow our coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run the test suite
   uv run pytest
   
   # Test manually with real audio files
   uv run python -m audio_notes.cli quick-note test_audio.wav
   
   # Check code quality
   uv run black --check src/
   uv run flake8 src/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat(component): add new feature description"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Commit Message Convention

We follow conventional commits for clear history:

- `feat(scope): add new feature`
- `fix(scope): fix bug description`
- `docs(scope): update documentation`
- `test(scope): add or update tests`
- `refactor(scope): refactor code`
- `perf(scope): improve performance`
- `chore(scope): update dependencies`

**Examples:**
```bash
feat(cli): add support for batch processing multiple files
fix(transcriber): handle empty audio files gracefully
docs(readme): add troubleshooting section
test(obsidian): add comprehensive integration tests
```

#### Pull Request Guidelines

- **Clear title** describing the change
- **Detailed description** explaining what and why
- **Reference issues** if applicable (`Fixes #123`)
- **Include tests** for new functionality
- **Update documentation** if needed
- **Ensure CI passes** before requesting review

**Pull Request Template:**
```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 📏 Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific conventions:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort

### Code Structure

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import click
import numpy as np
from transformers import pipeline

# Local imports
from audio_notes.transcriber import WhisperTranscriber
from audio_notes.utils import validate_audio_file
```

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Include type hints for all functions
- **Comments**: Explain complex logic, not obvious code

```python
def transcribe_audio(
    audio_path: Path, 
    language: str = "auto",
    task: str = "transcribe"
) -> Dict[str, Any]:
    """Transcribe audio file using Whisper model.
    
    Args:
        audio_path: Path to the audio file
        language: Language code or 'auto' for detection
        task: Either 'transcribe' or 'translate'
        
    Returns:
        Dictionary containing transcription results with text,
        chunks, language info, and metadata.
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format is unsupported
    """
```

### Testing Guidelines

- **Test coverage**: Aim for >80% coverage
- **Test types**: Unit tests, integration tests, end-to-end tests
- **Test naming**: Descriptive names explaining what is tested

```python
def test_transcribe_short_audio_file_returns_correct_text():
    """Test that short audio files are transcribed correctly."""
    
def test_invalid_audio_format_raises_value_error():
    """Test that unsupported audio formats raise ValueError."""
    
def test_ollama_integration_creates_enhanced_notes():
    """Test that Ollama integration generates enhanced notes."""
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_transcriber.py
│   ├── test_processor.py
│   └── test_obsidian.py
├── integration/          # Integration tests
│   ├── test_cli.py
│   └── test_workflow.py
├── fixtures/            # Test data
│   ├── audio/
│   └── expected_outputs/
└── conftest.py          # Pytest configuration
```

### Writing Tests

```python
import pytest
from pathlib import Path
from audio_notes.transcriber import WhisperTranscriber

class TestWhisperTranscriber:
    @pytest.fixture
    def transcriber(self):
        return WhisperTranscriber()
    
    @pytest.fixture
    def sample_audio(self):
        return Path("tests/fixtures/audio/sample.wav")
    
    def test_transcribe_returns_text(self, transcriber, sample_audio):
        result = transcriber.transcribe(sample_audio)
        assert "text" in result
        assert len(result["text"]) > 0
    
    def test_transcribe_invalid_file_raises_error(self, transcriber):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(Path("nonexistent.wav"))
```

### Running Specific Tests

```bash
# Run tests for specific component
uv run pytest tests/unit/test_transcriber.py

# Run tests matching pattern
uv run pytest -k "test_transcribe"

# Run tests with specific markers
uv run pytest -m "slow"  # if you use pytest markers
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation**
   - Docstrings for all public functions/classes
   - Inline comments for complex logic
   - Type hints for better IDE support

2. **User Documentation**
   - README.md (setup and usage)
   - CLI help messages
   - Example scripts and tutorials

3. **Developer Documentation**
   - CONTRIBUTING.md (this file)
   - Architecture decisions
   - API documentation

### Documentation Standards

- **Clear and concise** language
- **Working examples** that can be copy-pasted
- **Up-to-date** information (update with code changes)
- **Proper formatting** using Markdown

## 🎨 Component Guidelines

### Audio Processing

- **Support multiple formats** (WAV, MP3, M4A, etc.)
- **Handle edge cases** (empty files, corrupted audio)
- **Provide clear error messages**
- **Optimize for performance**

### AI Integration

- **Graceful degradation** when AI services unavailable
- **Clear configuration** for different models
- **Robust error handling** for API failures
- **Respect rate limits** and quotas

### CLI Interface

- **Intuitive commands** and options
- **Helpful error messages** with suggestions
- **Progress indicators** for long operations
- **Consistent option naming** across commands

### Obsidian Integration

- **Valid Markdown** output
- **Proper metadata** formatting
- **Flexible vault structure** support
- **Error handling** for file system issues

## 🚀 Release Process

### Version Management

We use semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** and ensure all tests pass
4. **Test installation** from clean environment
5. **Create release tag** and GitHub release
6. **Update documentation** if needed

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Help others** learn and contribute
- **Give constructive feedback** in reviews
- **Assume good intentions** from contributors

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Request Reviews**: Code-specific feedback

### Recognition

We recognize contributors in:
- **README.md** contributor section
- **Release notes** for significant contributions
- **GitHub contributor graphs** and statistics

## 📞 Getting Help

### Before Asking for Help

1. **Check the documentation** (README, CLI help)
2. **Search existing issues** for similar problems
3. **Try the troubleshooting guide** in README

### Where to Get Help

- **GitHub Issues**: For bugs and specific problems
- **GitHub Discussions**: For general questions and ideas
- **Code Review**: For feedback on your contributions

### How to Ask Good Questions

1. **Be specific** about the problem
2. **Include relevant context** (OS, Python version, etc.)
3. **Show what you've tried** already
4. **Provide minimal reproducible examples**

---

Thank you for contributing to Audio Notes! 🎉

Together, we're building something amazing for the audio note-taking community. Every contribution, no matter how small, makes a difference.

**Happy coding!** 🚀 