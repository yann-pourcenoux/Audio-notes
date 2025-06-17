#!/usr/bin/env python3
"""
Simple test script for WhisperTranscriber.

This script performs a basic test to ensure the WhisperTranscriber is working correctly.
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from loguru import logger

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from whisper_transcriber import WhisperTranscriber
from audio_processor import AudioProcessor


def test_basic_functionality():
    """Test basic WhisperTranscriber functionality."""
    logger.info("🧪 Testing WhisperTranscriber basic functionality...")
    
    try:
        # Initialize transcriber
        logger.info("🔧 Initializing WhisperTranscriber...")
        transcriber = WhisperTranscriber()
        
        # Get model info
        model_info = transcriber.get_model_info()
        logger.info(f"📋 Model Info: {model_info}")
        
        # Test with dummy audio
        logger.info("🎵 Creating test audio...")
        test_audio = np.random.randn(16000 * 5)  # 5 seconds of random noise
        
        # Test language detection
        logger.info("🔍 Testing language detection...")
        lang_result = transcriber.detect_language(test_audio)
        logger.info(f"Language detection result: {lang_result}")
        
        # Test basic transcription
        logger.info("📝 Testing basic transcription...")
        result = transcriber.transcribe(
            test_audio,
            return_timestamps=True,
            task="transcribe"
        )
        
        logger.info(f"✅ Transcription completed successfully!")
        logger.info(f"Text: '{result.get('text', '')}'")
        logger.info(f"Metadata: {result.get('metadata', {})}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_audio_file():
    """Test with actual audio file."""
    logger.info("🧪 Testing with audio file...")
    
    try:
        # Create a test audio file
        test_file = Path("test_audio.wav")
        test_audio = np.random.randn(16000 * 10)  # 10 seconds
        sf.write(test_file, test_audio, 16000)
        
        logger.info(f"📁 Created test file: {test_file}")
        
        # Initialize transcriber and audio processor
        transcriber = WhisperTranscriber()
        audio_processor = AudioProcessor()
        
        # Test audio processing
        processed = audio_processor.process_audio(test_file)
        logger.info(f"🔧 Audio processed: {processed.keys()}")
        
        # Test transcription
        result = transcriber.transcribe(
            test_file,
            return_timestamps=True,
            task="transcribe"
        )
        
        logger.info(f"✅ File transcription completed!")
        logger.info(f"Text: '{result.get('text', '')}'")
        logger.info(f"File info: {result.get('file_info', {})}")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()
            logger.info(f"🗑️ Cleaned up {test_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        test_file = Path("test_audio.wav")
        if test_file.exists():
            test_file.unlink()
        
        return False


def main():
    """Run tests."""
    logger.info("🚀 Starting WhisperTranscriber Tests")
    logger.info("=" * 50)
    
    # Test 1: Basic functionality
    test1_passed = test_basic_functionality()
    
    logger.info("-" * 50)
    
    # Test 2: File processing
    test2_passed = test_with_audio_file()
    
    logger.info("=" * 50)
    
    if test1_passed and test2_passed:
        logger.success("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("test_whisper.log", level="DEBUG")
    
    sys.exit(main()) 