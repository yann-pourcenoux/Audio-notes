# Usage Examples

This document provides practical examples and use cases for Audio Notes, demonstrating how to use the tool effectively in different scenarios.

## 🎯 Quick Start Examples

### Basic Note Creation

```bash
# Simple transcription and note creation
uv run python -m audio_notes.cli quick-note meeting.wav

# Specify custom vault location
uv run python -m audio_notes.cli quick-note interview.mp3 --vault-path ~/Documents/MyVault
```

### Advanced Processing

```bash
# Transcribe with specific language
uv run python -m audio_notes.cli process lecture.wav --language en

# Generate subtitles for video
uv run python -m audio_notes.cli process video_audio.wav --output-format srt --timestamps word

# Translate foreign language to English
uv run python -m audio_notes.cli process spanish_meeting.mp3 --task translate
```

## 📚 Real-World Use Cases

### 1. Academic Research & Lectures

#### Scenario: Recording and Processing University Lectures

```bash
# Process lecture with detailed timestamps and AI enhancement
uv run python -m audio_notes.cli process \
  lecture_quantum_physics.mp3 \
  --language en \
  --timestamps sentence \
  --enhance-notes \
  --vault-path ~/University/Physics/Lectures

# Generate both notes and subtitles
uv run python -m audio_notes.cli process \
  lecture_series/*.wav \
  --output-format json \
  --timestamps word \
  --vault-path ~/University/Lectures \
  --enhance-notes
```

**Expected Output Structure:**
```
~/University/Physics/Lectures/
└── Audio Notes/
    ├── 20250619_lecture_quantum_physics.md
    └── 20250619_lecture_thermodynamics.md
```

**Note Content Example:**
```markdown
# Quantum Physics Fundamentals Lecture

## Metadata
- **Created**: 2025-06-19T14:30:00
- **Source**: audio-transcription
- **Original File**: lecture_quantum_physics.mp3
- **AI Enhanced**: True
- **Tags**: #physics, #quantum-mechanics, #university, #lecture

## Summary
Comprehensive introduction to quantum physics covering wave-particle duality, 
uncertainty principle, and basic quantum mechanical operators.

## Transcription
[00:00:00] Welcome to today's lecture on quantum physics fundamentals...
[00:02:15] The wave-particle duality is one of the most fundamental concepts...
```

### 2. Business Meetings & Interviews

#### Scenario: Processing Team Meetings and Client Interviews

```bash
# Quick meeting notes with AI enhancement
uv run python -m audio_notes.cli quick-note team_standup.wav --vault-path ~/Work/Meetings

# Process client interview with translation
uv run python -m audio_notes.cli process \
  client_interview_spanish.mp3 \
  --task translate \
  --enhance-notes \
  --vault-path ~/Work/Client_Interviews

# Batch process weekly meetings
uv run python -m audio_notes.cli process \
  week_*.wav \
  --language auto \
  --timestamps sentence \
  --output-format json \
  --vault-path ~/Work/Weekly_Reviews
```

### 3. Podcast & Content Creation

#### Scenario: Processing Podcast Episodes for Show Notes

```bash
# Generate detailed show notes with timestamps
uv run python -m audio_notes.cli process \
  podcast_episode_42.wav \
  --timestamps sentence \
  --enhance-notes \
  --output-format json \
  --vault-path ~/Podcast/Show_Notes

# Create subtitles for video podcast
uv run python -m audio_notes.cli process \
  podcast_video_audio.wav \
  --output-format vtt \
  --timestamps word \
  --temperature 0.1  # Lower temperature for more consistent output
```

### 4. Language Learning & Translation

#### Scenario: Processing Foreign Language Content

```bash
# Transcribe and translate foreign language audio
uv run python -m audio_notes.cli process \
  french_conversation.mp3 \
  --language fr \
  --task translate \
  --enhance-notes \
  --vault-path ~/Language_Learning/French

# Process multiple language files
for lang in spanish french german; do
  uv run python -m audio_notes.cli process \
    ${lang}_lesson*.mp3 \
    --language ${lang:0:2} \
    --task translate \
    --vault-path ~/Language_Learning/${lang^}
done
```

### 5. Medical & Legal Documentation

#### Scenario: Processing Professional Recordings

```bash
# Medical consultation with high precision
uv run python -m audio_notes.cli process \
  patient_consultation.wav \
  --language en \
  --temperature 0.0 \  # Highest precision
  --timestamps sentence \
  --output-format json \
  --vault-path ~/Medical/Consultations

# Legal deposition with detailed timestamps
uv run python -m audio_notes.cli process \
  deposition_recording.wav \
  --timestamps word \
  --output-format srt \
  --precision float32 \  # CPU processing for reliability
  --vault-path ~/Legal/Depositions
```

## ⚙️ Configuration Examples

### Performance Optimization

#### CPU-Optimized Processing
```bash
# Optimize for CPU processing
uv run python -m audio_notes.cli process \
  large_file.wav \
  --precision float32 \
  --processing-method chunked \
  --max-length 1800 \  # 30-minute chunks
  --batch-size 1
```

#### GPU-Optimized Processing
```bash
# Optimize for GPU processing
uv run python -m audio_notes.cli process \
  *.wav \
  --precision float16 \
  --batch-size 4 \
  --processing-method sequential
```

### Quality vs Speed Trade-offs

#### Maximum Quality
```bash
# Highest quality transcription
uv run python -m audio_notes.cli process \
  important_recording.wav \
  --temperature 0.0 \
  --precision float32 \
  --processing-method sequential \
  --enhance-notes
```

#### Speed-Optimized
```bash
# Fastest processing
uv run python -m audio_notes.cli process \
  *.wav \
  --precision float16 \
  --temperature 0.3 \
  --processing-method chunked \
  --no-vault  # Skip Obsidian integration
```

## 🔄 Batch Processing Workflows

### Weekly Meeting Processing

```bash
#!/bin/bash
# weekly_meeting_processor.sh

VAULT_PATH="~/Work/Weekly_Meetings"
WEEK=$(date +%Y-W%U)

# Create weekly directory
mkdir -p "$VAULT_PATH/$WEEK"

# Process all meeting recordings
uv run python -m audio_notes.cli process \
  meeting_*.wav \
  --language auto \
  --timestamps sentence \
  --enhance-notes \
  --output-format json \
  --vault-path "$VAULT_PATH/$WEEK"

echo "Processed $(ls meeting_*.wav | wc -l) meetings for $WEEK"
```

### Multi-Language Content Processing

```bash
#!/bin/bash
# multi_language_processor.sh

declare -A LANGUAGES=(
  ["en"]="English"
  ["es"]="Spanish" 
  ["fr"]="French"
  ["de"]="German"
)

for lang in "${!LANGUAGES[@]}"; do
  echo "Processing ${LANGUAGES[$lang]} files..."
  
  uv run python -m audio_notes.cli process \
    ${lang}_*.wav \
    --language $lang \
    --task translate \
    --enhance-notes \
    --vault-path "~/Multilingual/${LANGUAGES[$lang]}"
done
```

## 📊 Output Format Examples

### JSON Output Structure

```bash
# Generate JSON with metadata
uv run python -m audio_notes.cli process recording.wav --output-format json
```

**Output Example:**
```json
{
  "text": "Welcome to today's meeting. Let's start with the quarterly review...",
  "chunks": [
    {
      "timestamp": [0.0, 5.2],
      "text": "Welcome to today's meeting."
    },
    {
      "timestamp": [5.2, 12.8],
      "text": "Let's start with the quarterly review."
    }
  ],
  "language": {
    "detected": "en",
    "specified": "auto",
    "name": "English"
  },
  "task": "transcribe",
  "metadata": {
    "processing_time": 45.3,
    "audio_duration": 1820.5,
    "model": "openai/whisper-large-v3",
    "device": "cuda:0",
    "precision": "float16"
  }
}
```

### SRT Subtitle Output

```bash
# Generate SRT subtitles
uv run python -m audio_notes.cli process video.wav --output-format srt --timestamps word
```

**Output Example:**
```srt
1
00:00:00,000 --> 00:00:02,500
Welcome to today's presentation

2
00:00:02,500 --> 00:00:05,800
on artificial intelligence and machine learning.

3
00:00:05,800 --> 00:00:09,200
We'll cover the fundamental concepts and applications.
```

### VTT (WebVTT) Output

```bash
# Generate WebVTT subtitles
uv run python -m audio_notes.cli process lecture.wav --output-format vtt --timestamps sentence
```

**Output Example:**
```vtt
WEBVTT

00:00:00.000 --> 00:00:05.200
Welcome to today's lecture on quantum physics.

00:00:05.200 --> 00:00:12.800
We'll explore wave-particle duality and its implications.

00:00:12.800 --> 00:00:18.500
The uncertainty principle is fundamental to quantum mechanics.
```

## 🔧 Troubleshooting Examples

### Handling Problematic Audio

#### Low Quality Audio
```bash
# Use higher temperature for noisy audio
uv run python -m audio_notes.cli process \
  noisy_recording.wav \
  --temperature 0.8 \
  --processing-method chunked \
  --language auto
```

#### Very Long Files
```bash
# Process very long recordings efficiently
uv run python -m audio_notes.cli process \
  3_hour_conference.wav \
  --processing-method chunked \
  --max-length 900 \  # 15-minute chunks
  --precision float16 \
  --batch-size 1
```

#### Multiple Speakers
```bash
# Optimize for multiple speakers
uv run python -m audio_notes.cli process \
  panel_discussion.wav \
  --timestamps word \
  --temperature 0.3 \
  --enhance-notes
```

### Error Recovery

#### Network Issues with Ollama
```bash
# Process without AI enhancement if Ollama unavailable
uv run python -m audio_notes.cli process \
  recording.wav \
  --no-vault  # Skip Obsidian integration
# Then manually create notes later when Ollama is available
```

#### Memory Constraints
```bash
# Reduce memory usage
uv run python -m audio_notes.cli process \
  large_file.wav \
  --precision float32 \
  --processing-method chunked \
  --max-length 600 \  # 10-minute chunks
  --batch-size 1
```

## 🎨 Custom Workflows

### Research Interview Pipeline

```bash
#!/bin/bash
# research_interview_pipeline.sh

INTERVIEW_FILE="$1"
PARTICIPANT="$2"
PROJECT="$3"

# Create project directory
mkdir -p "~/Research/$PROJECT/Interviews"

# Process interview with AI enhancement
uv run python -m audio_notes.cli process \
  "$INTERVIEW_FILE" \
  --language auto \
  --timestamps sentence \
  --enhance-notes \
  --vault-path "~/Research/$PROJECT/Interviews" \
  --output-format json

# Generate subtitle file for video analysis
uv run python -m audio_notes.cli process \
  "$INTERVIEW_FILE" \
  --output-format srt \
  --timestamps word \
  --output-dir "~/Research/$PROJECT/Subtitles"

echo "Interview with $PARTICIPANT processed for project $PROJECT"
```

### Podcast Production Workflow

```bash
#!/bin/bash
# podcast_production.sh

EPISODE_AUDIO="$1"
EPISODE_NUMBER="$2"

# Generate show notes
uv run python -m audio_notes.cli process \
  "$EPISODE_AUDIO" \
  --enhance-notes \
  --vault-path "~/Podcast/Show_Notes" \
  --output-format json

# Create subtitles for video version
uv run python -m audio_notes.cli process \
  "$EPISODE_AUDIO" \
  --output-format vtt \
  --timestamps sentence \
  --output-dir "~/Podcast/Subtitles"

# Generate transcript for website
uv run python -m audio_notes.cli process \
  "$EPISODE_AUDIO" \
  --output-format text \
  --output-dir "~/Podcast/Transcripts"

echo "Episode $EPISODE_NUMBER production files generated"
```

## 📈 Performance Benchmarks

### Typical Processing Times

| Audio Length | CPU (float32) | GPU (float16) | Enhancement |
|--------------|---------------|---------------|-------------|
| 5 minutes    | 45-60 seconds | 15-20 seconds | +5-10 seconds |
| 30 minutes   | 4-6 minutes   | 1.5-2 minutes | +15-30 seconds |
| 2 hours      | 15-20 minutes | 5-8 minutes   | +1-2 minutes |

### Memory Usage

| Processing Mode | RAM Usage | VRAM Usage (GPU) |
|-----------------|-----------|------------------|
| Sequential      | 2-4 GB    | 4-6 GB          |
| Chunked         | 1-2 GB    | 2-3 GB          |
| Batch (4 files)| 4-8 GB    | 8-12 GB         |

---

These examples should help you get started with Audio Notes and adapt the tool to your specific use cases. For more advanced usage, see the main documentation and CLI help (`--help`). 