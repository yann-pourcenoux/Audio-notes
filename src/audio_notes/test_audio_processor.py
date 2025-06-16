"""
Test script for AudioProcessor to verify functionality.
"""

import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from loguru import logger

from .audio_processor import AudioProcessor


def create_test_audio_file(sample_rate: int = 44100, 
                          duration: float = 5.0,
                          channels: int = 2,
                          format: str = 'wav') -> Path:
    """Create a test audio file for testing purposes."""
    
    # Generate test signal (sine wave)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440.0  # A4 note
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Make stereo if needed
    if channels == 2:
        audio_data = np.column_stack([signal, signal * 0.8])  # Slightly different channels
    else:
        audio_data = signal
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # Write audio file
    sf.write(str(temp_path), audio_data, sample_rate)
    
    logger.info(f"Created test audio file: {temp_path}")
    logger.info(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz, Channels: {channels}")
    
    return temp_path


def test_audio_processor():
    """Test the AudioProcessor functionality."""
    
    logger.info("Starting AudioProcessor tests...")
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Test 1: Basic processing with stereo 44.1kHz file
    logger.info("\n=== Test 1: Basic processing (stereo 44.1kHz) ===")
    test_file_1 = create_test_audio_file(sample_rate=44100, duration=3.0, channels=2)
    
    try:
        result = processor.process_audio(test_file_1)
        
        assert result['sampling_rate'] == 16000, f"Expected 16000Hz, got {result['sampling_rate']}"
        assert len(result['array'].shape) == 1, f"Expected mono audio, got shape {result['array'].shape}"
        assert result['array'].dtype == np.float32, f"Expected float32, got {result['array'].dtype}"
        assert abs(result['duration'] - 3.0) < 0.1, f"Expected ~3.0s, got {result['duration']}"
        
        logger.success("✅ Test 1 passed: Basic processing works correctly")
        
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
    finally:
        test_file_1.unlink()  # Clean up
    
    # Test 2: Processing with already correct format (16kHz mono)
    logger.info("\n=== Test 2: Correct format (16kHz mono) ===")
    test_file_2 = create_test_audio_file(sample_rate=16000, duration=2.0, channels=1)
    
    try:
        result = processor.process_audio(test_file_2)
        
        assert result['sampling_rate'] == 16000
        assert result['original_sr'] == 16000
        assert len(result['array'].shape) == 1
        
        logger.success("✅ Test 2 passed: Already correct format handled properly")
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
    finally:
        test_file_2.unlink()
    
    # Test 3: Error handling with non-existent file
    logger.info("\n=== Test 3: Error handling ===")
    try:
        processor.process_audio("non_existent_file.wav")
        logger.error("❌ Test 3 failed: Should have raised FileNotFoundError")
    except FileNotFoundError:
        logger.success("✅ Test 3 passed: Correctly handles non-existent files")
    except Exception as e:
        logger.error(f"❌ Test 3 failed: Unexpected error {e}")
    
    # Test 4: Supported format checking
    logger.info("\n=== Test 4: Format support ===")
    supported_formats = AudioProcessor.get_supported_formats()
    logger.info(f"Supported formats: {supported_formats}")
    
    assert '.wav' in supported_formats
    assert '.mp3' in supported_formats
    assert '.m4a' in supported_formats
    
    assert AudioProcessor.is_supported_format("test.wav") == True
    assert AudioProcessor.is_supported_format("test.txt") == False
    
    logger.success("✅ Test 4 passed: Format checking works correctly")
    
    logger.info("\n🎉 All AudioProcessor tests completed!")


if __name__ == "__main__":
    test_audio_processor() 