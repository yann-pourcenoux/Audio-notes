# Audio Notes

🎤 **AI-Powered Audio Transcription and Note Generation**

Transform your audio recordings into intelligent, searchable notes with automatic transcription using Whisper Large-v3 and AI-enhanced content generation using Qwen 3.0.6B, seamlessly integrated with Obsidian.

## ✨ Features

- 🎯 **High-Accuracy Transcription** - Uses OpenAI's Whisper Large-v3 model for state-of-the-art speech recognition
- 🧠 **AI-Enhanced Notes** - Intelligent title generation, tagging, and summarization using Qwen 3.0.6B via Ollama
- 📝 **Obsidian Integration** - Automatic note creation in your Obsidian vault with proper formatting
- 🌍 **Multi-Language Support** - Supports 50+ languages with automatic detection
- ⏱️ **Timestamp Support** - Word-level and sentence-level timestamps for precise navigation
- 🔄 **Translation** - Automatic translation to English from any supported language
- 📊 **Multiple Output Formats** - Text, JSON, SRT, VTT, and TSV formats
- 🚀 **Fast Processing** - Optimized for both CPU and GPU processing
- 🎛️ **Flexible CLI** - Comprehensive command-line interface with extensive configuration options

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (Required)
- **Ollama** (For AI enhancement features)
- **Audio files** in supported formats: WAV, MP3, M4A, OGG, FLAC, WMA, AAC

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/audio-notes.git
cd audio-notes

# Install with uv (recommended)
pip install uv
uv sync

# Or install with pip
pip install -e .
```

### 2. Set Up Ollama (Optional but Recommended)

For AI-enhanced note generation, install and configure Ollama:

```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
# Then pull the Qwen model
ollama pull qwen3:0.6b
```

### 3. Quick Test

```bash
# Test the installation
uv run python -m audio_notes.cli status

# Quick note creation (replace with your audio file)
uv run python -m audio_notes.cli quick-note your_audio.wav
```

## 📖 Usage Guide

### Basic Commands

#### 1. Quick Note Creation
Perfect for rapid note-taking workflows:

```bash
# Create a quick note with AI enhancement
uv run python -m audio_notes.cli quick-note recording.wav

# Specify custom vault location
uv run python -m audio_notes.cli quick-note meeting.mp3 --vault-path ~/Documents/MyVault
```

#### 2. Advanced Processing
For more control over transcription and output:

```bash
# Basic transcription
uv run python -m audio_notes.cli process interview.wav

# With specific language and timestamps
uv run python -m audio_notes.cli process lecture.mp3 --language en --timestamps sentence

# Translation to English
uv run python -m audio_notes.cli process spanish_audio.wav --task translate

# Multiple files with AI enhancement
uv run python -m audio_notes.cli process *.wav --enhance-notes --output-format json
```

#### 3. System Status
Check if all components are working:

```bash
uv run python -m audio_notes.cli status
```

### Command Reference

#### `quick-note` Command
Simplified command for rapid note creation:

```bash
uv run python -m audio_notes.cli quick-note [OPTIONS] AUDIO_FILE

Options:
  --vault-path PATH    Obsidian vault path (default: ./obsidian_vault)
  --help              Show help message
```

#### `process` Command
Advanced processing with full configuration options:

```bash
uv run python -m audio_notes.cli process [OPTIONS] AUDIO_FILES...

Key Options:
  -l, --language      Language code (auto, en, es, fr, etc.)
  -t, --task          transcribe or translate
  --timestamps        none, sentence, or word
  --temperature       Decoding temperature (0.0-1.0)
  --output-format     text, json, srt, vtt, or tsv
  --vault-path        Obsidian vault location
  --enhance-notes     Enable AI enhancement
  --dry-run          Preview without processing
```

#### `status` Command
Verify system components:

```bash
uv run python -m audio_notes.cli status [OPTIONS]

Options:
  --check-ollama      Specifically test Ollama integration
  --verbose          Show detailed component information
```

### Supported Audio Formats

- **WAV** - Uncompressed audio (recommended for best quality)
- **MP3** - Compressed audio (widely supported)
- **M4A** - Apple audio format
- **OGG** - Open-source audio format
- **FLAC** - Lossless compression
- **WMA** - Windows Media Audio
- **AAC** - Advanced Audio Coding

### Supported Languages

The system supports 50+ languages including:
- **English** (en) - Default
- **Spanish** (es)
- **French** (fr)
- **German** (de)
- **Chinese** (zh)
- **Japanese** (ja)
- **And many more...**

Use `auto` for automatic language detection.

## 🔧 Configuration

### Environment Setup

The system works out-of-the-box with minimal configuration. For optimal performance:

1. **GPU Support** (Optional):
   ```bash
   # Install PyTorch with CUDA support for faster processing
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Ollama Configuration**:
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Verify Qwen model is available
   ollama list
   ```

### Obsidian Integration

The tool automatically creates Obsidian-compatible notes with:

- **Metadata** - Creation date, source file, AI enhancement status
- **Tags** - Automatically generated relevant tags
- **Summary** - AI-generated content summary
- **Transcription** - Full transcribed text
- **Timestamps** - When enabled, precise timing information

Example note structure:
```markdown
# Meeting Discussion on Project Alpha

## Metadata
- **Created**: 2025-06-19T22:45:00
- **Source**: audio-transcription
- **Original File**: meeting.wav
- **AI Enhanced**: True
- **Tags**: #meeting, #project-alpha, #discussion

## Summary
Discussion covering project timeline, resource allocation, and next steps.

## Transcription
[00:00:00] Welcome everyone to today's meeting...
[00:01:30] Let's discuss the project timeline...
```

## 🛠️ Advanced Usage

### Batch Processing

Process multiple files efficiently:

```bash
# Process all WAV files in current directory
uv run python -m audio_notes.cli process *.wav --enhance-notes

# Process files with specific settings
uv run python -m audio_notes.cli process audio1.mp3 audio2.wav \
  --language auto \
  --timestamps sentence \
  --output-format json \
  --vault-path ~/MyVault
```

### Custom Output Formats

#### JSON Output
Structured data with metadata:
```bash
uv run python -m audio_notes.cli process recording.wav --output-format json
```

#### Subtitle Formats
For video integration:
```bash
# SRT format
uv run python -m audio_notes.cli process video_audio.wav --output-format srt --timestamps word

# VTT format (WebVTT)
uv run python -m audio_notes.cli process lecture.mp3 --output-format vtt --timestamps sentence
```

### Performance Optimization

#### CPU Optimization
```bash
# Use float32 precision for CPU (default)
uv run python -m audio_notes.cli process audio.wav --precision float32
```

#### GPU Optimization
```bash
# Use float16 precision for GPU (faster, requires CUDA)
uv run python -m audio_notes.cli process audio.wav --precision float16
```

#### Chunked Processing
For very long audio files:
```bash
uv run python -m audio_notes.cli process long_recording.wav \
  --processing-method chunked \
  --max-length 1800  # 30 minutes per chunk
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Ollama service is not available"
**Solution:**
```bash
# Start Ollama service
ollama serve

# Pull the required model
ollama pull qwen3:0.6b

# Verify status
uv run python -m audio_notes.cli status
```

#### 2. "Failed to load audio file"
**Possible causes:**
- Unsupported audio format
- Corrupted audio file
- Insufficient permissions

**Solution:**
```bash
# Check file format
file your_audio.wav

# Convert to supported format if needed
ffmpeg -i input.format output.wav

# Check file permissions
ls -la your_audio.wav
```

#### 3. "CUDA out of memory" (GPU users)
**Solution:**
```bash
# Use CPU processing
uv run python -m audio_notes.cli process audio.wav --precision float32

# Or reduce batch size
uv run python -m audio_notes.cli process *.wav --batch-size 1
```

#### 4. "Permission denied" when creating notes
**Solution:**
```bash
# Check vault directory permissions
ls -la obsidian_vault/

# Create vault directory manually
mkdir -p obsidian_vault/Audio\ Notes

# Or specify different vault path
uv run python -m audio_notes.cli quick-note audio.wav --vault-path ~/Documents/MyVault
```

### Performance Issues

#### Slow Transcription
- **Use GPU**: Install CUDA-enabled PyTorch for 3-5x speedup
- **Reduce precision**: Use `--precision float16` on GPU
- **Chunked processing**: Use `--processing-method chunked` for long files

#### AI Enhancement Issues
- **Check Ollama**: Ensure Ollama service is running
- **Model availability**: Verify `qwen3:0.6b` is pulled
- **Fallback behavior**: System continues without AI enhancement if Ollama unavailable

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Verbose output
uv run python -m audio_notes.cli process audio.wav --verbose

# Dry run to preview operations
uv run python -m audio_notes.cli process *.wav --dry-run
```

## 📊 Output Examples

### Text Output
```
This is the transcribed text from your audio file.
It includes proper punctuation and formatting.
```

### JSON Output
```json
{
  "text": "This is the transcribed text...",
  "chunks": [
    {
      "timestamp": [0.0, 5.0],
      "text": "This is the transcribed text..."
    }
  ],
  "language": {
    "detected": "en",
    "specified": "auto",
    "name": "English"
  },
  "task": "transcribe",
  "metadata": {
    "processing_time": 45.2,
    "audio_duration": 120.0,
    "model": "openai/whisper-large-v3",
    "device": "cpu"
  }
}
```

### SRT Subtitle Output
```
1
00:00:00,000 --> 00:00:05,000
This is the first subtitle segment.

2
00:00:05,000 --> 00:00:10,000
This is the second subtitle segment.
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/your-username/audio-notes.git
cd audio-notes
uv sync --dev

# Run tests
uv run pytest

# Code formatting
uv run black src/
uv run isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** - State-of-the-art speech recognition
- **Qwen** - Advanced language model for content enhancement
- **Ollama** - Local LLM serving platform
- **Obsidian** - Knowledge management system

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/audio-notes/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/audio-notes/discussions)
- **Documentation**: This README and inline help (`--help`)

---

**Made with ❤️ for the audio note-taking community**
