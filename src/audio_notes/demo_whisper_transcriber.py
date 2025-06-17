"""
Demo script for WhisperTranscriber functionality.

This script demonstrates all the capabilities of the WhisperTranscriber class
including basic transcription, language detection, timestamp generation,
long-form audio processing, and batch processing.
"""

import time
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from .whisper_transcriber import WhisperTranscriber
from .audio_processor import AudioProcessor


def demo_basic_transcription():
    """Demonstrate basic transcription functionality."""
    logger.info("🎬 Demo: Basic Transcription")
    
    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Display model information
        model_info = transcriber.get_model_info()
        logger.info(f"📋 Model Info: {model_info}")
        
        # Display supported languages
        languages = transcriber.get_supported_languages()
        logger.info(f"🌍 Supported Languages: {len(languages)} languages")
        logger.info(f"📝 Sample languages: {dict(list(languages.items())[:10])}")
        
        logger.success("✅ Basic transcription demo completed successfully")
        return transcriber
        
    except Exception as e:
        logger.error(f"❌ Basic transcription demo failed: {e}")
        return None


def demo_language_detection(transcriber: WhisperTranscriber):
    """Demonstrate language detection functionality."""
    logger.info("🎬 Demo: Language Detection")
    
    try:
        # Create a sample audio array (silence for demo)
        import numpy as np
        sample_audio = np.random.randn(16000 * 5)  # 5 seconds of random noise
        
        # Test language detection
        detection_result = transcriber.detect_language(sample_audio)
        logger.info(f"🔍 Language Detection Result: {detection_result}")
        
        logger.success("✅ Language detection demo completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Language detection demo failed: {e}")


def demo_transcription_options(transcriber: WhisperTranscriber):
    """Demonstrate various transcription options."""
    logger.info("🎬 Demo: Transcription Options")
    
    try:
        import numpy as np
        
        # Create sample audio
        sample_audio = np.random.randn(16000 * 10)  # 10 seconds
        
        # Test different transcription options
        options_to_test = [
            {
                "name": "Basic Transcription",
                "kwargs": {"task": "transcribe", "return_timestamps": False}
            },
            {
                "name": "Transcription with Sentence Timestamps",
                "kwargs": {"task": "transcribe", "return_timestamps": True}
            },
            {
                "name": "Transcription with Word Timestamps",
                "kwargs": {"task": "transcribe", "return_timestamps": "word"}
            },
            {
                "name": "Translation to English",
                "kwargs": {"task": "translate", "return_timestamps": True}
            },
            {
                "name": "Spanish Transcription",
                "kwargs": {"task": "transcribe", "language": "es", "return_timestamps": True}
            },
            {
                "name": "High Temperature (Creative)",
                "kwargs": {"task": "transcribe", "temperature": 0.8, "return_timestamps": True}
            },
            {
                "name": "Temperature Fallback",
                "kwargs": {"task": "transcribe", "temperature": [0.0, 0.2, 0.4, 0.6, 0.8], "return_timestamps": True}
            }
        ]
        
        for option in options_to_test:
            logger.info(f"🧪 Testing: {option['name']}")
            try:
                result = transcriber.transcribe(sample_audio, **option['kwargs'])
                logger.info(f"  ✅ Success: {len(result.get('text', ''))} chars, "
                           f"{len(result.get('chunks', []))} chunks")
                logger.info(f"  📊 Metadata: {result.get('metadata', {})}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed: {e}")
        
        logger.success("✅ Transcription options demo completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Transcription options demo failed: {e}")


def demo_long_form_processing(transcriber: WhisperTranscriber):
    """Demonstrate long-form audio processing."""
    logger.info("🎬 Demo: Long-Form Audio Processing")
    
    try:
        import numpy as np
        
        # Create a longer sample audio (2 minutes)
        long_audio = np.random.randn(16000 * 120)  # 120 seconds
        
        # Create a temporary audio file for long-form processing
        temp_audio_path = Path("temp_long_audio.wav")
        
        # Save the audio using soundfile
        import soundfile as sf
        sf.write(temp_audio_path, long_audio, 16000)
        
        logger.info(f"📁 Created temporary audio file: {temp_audio_path}")
        logger.info(f"⏱️ Audio duration: {len(long_audio) / 16000:.1f} seconds")
        
        # Test long-form transcription
        logger.info("🔄 Starting long-form transcription...")
        start_time = time.time()
        
        result = transcriber.transcribe_long_form(
            temp_audio_path,
            chunk_length_s=30,
            overlap_s=2.0,
            return_timestamps=True,
            task="transcribe"
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Long-form transcription completed in {processing_time:.2f}s")
        logger.info(f"📝 Total text length: {len(result.get('text', ''))} characters")
        logger.info(f"📊 Chunks processed: {result.get('metadata', {}).get('num_chunks', 0)}")
        logger.info(f"🕒 Total chunks: {len(result.get('chunks', []))}")
        
        # Clean up
        if temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.info(f"🗑️ Cleaned up temporary file: {temp_audio_path}")
        
        logger.success("✅ Long-form processing demo completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Long-form processing demo failed: {e}")
        # Clean up on error
        temp_audio_path = Path("temp_long_audio.wav")
        if temp_audio_path.exists():
            temp_audio_path.unlink()


def demo_batch_processing(transcriber: WhisperTranscriber):
    """Demonstrate batch processing functionality."""
    logger.info("🎬 Demo: Batch Processing")
    
    try:
        import numpy as np
        import soundfile as sf
        
        # Create multiple temporary audio files
        temp_files = []
        num_files = 3
        
        for i in range(num_files):
            # Create different length audio files
            duration = 10 + (i * 15)  # 10s, 25s, 40s
            audio_data = np.random.randn(16000 * duration)
            
            temp_path = Path(f"temp_batch_audio_{i+1}.wav")
            sf.write(temp_path, audio_data, 16000)
            temp_files.append(temp_path)
            
            logger.info(f"📁 Created {temp_path} ({duration}s)")
        
        # Define progress callback
        def progress_callback(current: int, total: int, file_path: str, result: Dict[str, Any]):
            logger.info(f"🔄 Progress: {current}/{total} - {Path(file_path).name}")
            if not result.get('error'):
                logger.info(f"  ✅ Success: {len(result.get('text', ''))} chars")
            else:
                logger.warning(f"  ❌ Error: {result.get('error')}")
        
        # Test batch transcription
        logger.info(f"🔄 Starting batch transcription of {len(temp_files)} files...")
        start_time = time.time()
        
        results = transcriber.batch_transcribe(
            temp_files,
            progress_callback=progress_callback,
            return_timestamps=True,
            task="transcribe"
        )
        
        processing_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not r.get('error')]
        failed_results = [r for r in results if r.get('error')]
        
        logger.info(f"✅ Batch processing completed in {processing_time:.2f}s")
        logger.info(f"📊 Successful: {len(successful_results)}/{len(results)}")
        logger.info(f"❌ Failed: {len(failed_results)}")
        
        for i, result in enumerate(results):
            if not result.get('error'):
                logger.info(f"  File {i+1}: {len(result.get('text', ''))} chars, "
                           f"{len(result.get('chunks', []))} chunks")
            else:
                logger.warning(f"  File {i+1}: Error - {result.get('error')}")
        
        # Clean up
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"🗑️ Cleaned up {temp_file}")
        
        logger.success("✅ Batch processing demo completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Batch processing demo failed: {e}")
        # Clean up on error
        for i in range(5):  # Clean up any potential temp files
            temp_path = Path(f"temp_batch_audio_{i+1}.wav")
            if temp_path.exists():
                temp_path.unlink()


def main():
    """Run all demos."""
    logger.info("🚀 Starting WhisperTranscriber Demo Suite")
    logger.info("=" * 60)
    
    try:
        # Demo 1: Basic functionality
        transcriber = demo_basic_transcription()
        if not transcriber:
            logger.error("❌ Cannot continue demos without working transcriber")
            return
        
        logger.info("-" * 60)
        
        # Demo 2: Language detection
        demo_language_detection(transcriber)
        
        logger.info("-" * 60)
        
        # Demo 3: Transcription options
        demo_transcription_options(transcriber)
        
        logger.info("-" * 60)
        
        # Demo 4: Long-form processing
        demo_long_form_processing(transcriber)
        
        logger.info("-" * 60)
        
        # Demo 5: Batch processing
        demo_batch_processing(transcriber)
        
        logger.info("=" * 60)
        logger.success("🎉 All demos completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo suite failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.add("demo_whisper.log", level="DEBUG")
    
    main() 