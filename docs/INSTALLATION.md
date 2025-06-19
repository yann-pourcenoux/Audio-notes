# Installation Guide

This guide provides detailed installation instructions for Audio Notes across different platforms and use cases.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.12 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space for dependencies
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.12+
- **RAM**: 8GB+ (for better performance)
- **Storage**: 5GB+ (includes models and test files)
- **GPU**: CUDA-compatible (optional, for faster processing)
- **Ollama**: For AI enhancement features

## 🚀 Quick Installation

### Option 1: Using uv (Recommended)

```bash
# Install uv package manager
pip install uv

# Clone and install
git clone https://github.com/your-username/audio-notes.git
cd audio-notes
uv sync

# Test installation
uv run python -m audio_notes.cli status
```

### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/your-username/audio-notes.git
cd audio-notes

# Install in development mode
pip install -e .

# Test installation
python -m audio_notes.cli status
```

## 🔧 Platform-Specific Installation

### Linux (Ubuntu/Debian)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 if not available
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install system dependencies for audio processing
sudo apt install ffmpeg libsndfile1 portaudio19-dev

# Install uv
pip install uv

# Clone and install Audio Notes
git clone https://github.com/your-username/audio-notes.git
cd audio-notes
uv sync

# Verify installation
uv run python -m audio_notes.cli status
```

### macOS

```bash
# Install Homebrew if not available
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12

# Install audio processing dependencies
brew install ffmpeg portaudio

# Install uv
pip3 install uv

# Clone and install Audio Notes
git clone https://github.com/your-username/audio-notes.git
cd audio-notes
uv sync

# Verify installation
uv run python -m audio_notes.cli status
```

### Windows

```powershell
# Install Python 3.12 from python.org or Microsoft Store

# Install Git if not available
winget install Git.Git

# Clone repository
git clone https://github.com/your-username/audio-notes.git
cd audio-notes

# Install uv
pip install uv

# Install dependencies
uv sync

# Verify installation
uv run python -m audio_notes.cli status
```

## 🧠 AI Enhancement Setup (Ollama)

### Installing Ollama

#### Linux
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull the Qwen model (in another terminal)
ollama pull qwen3:0.6b
```

#### macOS
```bash
# Download and install from https://ollama.ai
# Or use Homebrew
brew install ollama

# Start Ollama
ollama serve

# Pull the Qwen model (in another terminal)
ollama pull qwen3:0.6b
```

#### Windows
```powershell
# Download installer from https://ollama.ai
# Follow installation wizard

# Start Ollama (usually starts automatically)
# Open new terminal and pull model
ollama pull qwen3:0.6b
```

### Verifying Ollama Setup

```bash
# Check if Ollama is running
ollama list

# Test the model
ollama run qwen3:0.6b "Hello, how are you?"

# Test with Audio Notes
uv run python -m audio_notes.cli status --check-ollama
```

## ⚡ GPU Acceleration Setup

### NVIDIA GPU (CUDA)

```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled PyTorch (replace cu118 with your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test with Audio Notes
uv run python -m audio_notes.cli process test_audio.wav --precision float16
```

### AMD GPU (ROCm) - Linux Only

```bash
# Install ROCm (Ubuntu/Debian)
sudo apt install rocm-dev

# Install ROCm-enabled PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Verify ROCm support
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

### Apple Silicon (MPS) - macOS

```bash
# MPS support is included with standard PyTorch on macOS
# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Audio Notes will automatically use MPS when available
```

## 🔍 Verification and Testing

### Basic Functionality Test

```bash
# Check system status
uv run python -m audio_notes.cli status

# Test with sample audio (create a test file or use your own)
uv run python -m audio_notes.cli quick-note your_audio.wav

# Test advanced processing
uv run python -m audio_notes.cli process your_audio.wav --language auto --timestamps sentence
```

### Component-Specific Tests

```bash
# Test Whisper transcription only
uv run python -c "
from audio_notes.whisper_transcriber import WhisperTranscriber
transcriber = WhisperTranscriber()
print('Whisper loaded successfully')
"

# Test Ollama integration
uv run python -c "
from audio_notes.obsidian_writer import OllamaClient
client = OllamaClient()
print(f'Ollama available: {client.is_available()}')
"

# Test audio processing
uv run python -c "
from audio_notes.audio_processor import AudioProcessor
processor = AudioProcessor()
print('Audio processor initialized')
"
```

## 🐛 Troubleshooting Installation

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version
python3 --version

# If Python 3.12+ not available, install from python.org
# Or use pyenv for version management
```

#### Permission Errors
```bash
# Linux/macOS: Use virtual environment
python -m venv audio-notes-env
source audio-notes-env/bin/activate  # Linux/macOS
# audio-notes-env\Scripts\activate   # Windows

# Then install normally
pip install -e .
```

#### Audio Processing Errors
```bash
# Install system audio libraries
# Ubuntu/Debian:
sudo apt install libsndfile1-dev portaudio19-dev

# macOS:
brew install portaudio

# Windows: Usually works out of the box
```

#### CUDA Issues
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# Visit: https://pytorch.org/get-started/locally/
```

#### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check firewall settings (Windows)
# Ensure port 11434 is not blocked
```

### Performance Issues

#### Slow Transcription
1. **Enable GPU acceleration** (see GPU setup above)
2. **Use float16 precision**: `--precision float16`
3. **Reduce batch size**: `--batch-size 1`
4. **Use chunked processing**: `--processing-method chunked`

#### Memory Issues
1. **Close other applications**
2. **Use CPU instead of GPU**: `--precision float32`
3. **Process files individually** instead of batch processing
4. **Restart Python process** between large files

## 📊 Installation Verification Checklist

- [ ] Python 3.12+ installed and accessible
- [ ] Audio Notes repository cloned
- [ ] Dependencies installed successfully (`uv sync` or `pip install -e .`)
- [ ] Basic CLI commands work (`audio-notes status`)
- [ ] Audio processing works (test with sample file)
- [ ] Ollama installed and running (if using AI features)
- [ ] Qwen model downloaded (`ollama pull qwen3:0.6b`)
- [ ] GPU acceleration configured (if available)
- [ ] Test audio file processed successfully

## 🚀 Next Steps

After successful installation:

1. **Read the Usage Guide** in the main README
2. **Try the Quick Start** examples
3. **Configure your Obsidian vault** path
4. **Test with your own audio files**
5. **Explore advanced features** and configuration options

## 🆘 Getting Help

If you encounter issues during installation:

1. **Check this troubleshooting section** first
2. **Search existing GitHub issues** for similar problems
3. **Create a new issue** with detailed error information
4. **Include system information** (OS, Python version, error messages)

---

**Installation complete!** 🎉 You're ready to start transcribing and enhancing your audio notes. 