"""
Audio Notes - AI-powered audio processing and transcription system.
"""

__version__ = "0.1.0"
__author__ = "Audio Notes Team"
__description__ = "AI-powered audio notes processing and transcription system"

# Import main classes for easy access
from .audio_processor import AudioProcessor
from .whisper_transcriber import WhisperTranscriber
from .obsidian_writer import ObsidianWriter
from .config import AudioNotesConfig, ConfigManager

# Define what gets imported with "from audio_notes import *"
__all__ = [
    "AudioProcessor",
    "WhisperTranscriber", 
    "ObsidianWriter",
    "AudioNotesConfig",
    "ConfigManager",
    "__version__",
    "__author__",
    "__description__"
] 