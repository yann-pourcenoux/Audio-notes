#!/usr/bin/env python3
"""
Demo script for AudioProcessor - showing real-world usage examples.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

from .audio_processor import AudioProcessor


def demo_basic_processing(audio_file: Path):
    """Demo basic audio processing functionality."""
    
    logger.info(f"🎵 Demo: Basic Audio Processing")
    logger.info(f"Input file: {audio_file}")
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Get file info first
    try:
        info = processor.get_audio_info(audio_file)
        logger.info("📋 Audio File Information:")
        logger.info(f"  📁 File: {info['file_path']}")
        logger.info(f"  ⏱️  Duration: {info['duration']:.2f} seconds")
        logger.info(f"  🔊 Sample Rate: {info['sample_rate']} Hz")
        logger.info(f"  📻 Channels: {info['channels']}")
        logger.info(f"  📏 Format: {info['format']} ({info['subtype']})")
        logger.info(f"  💾 File Size: {info['file_size']:,} bytes")
        logger.info(f"  🔄 Needs Resampling: {info['needs_resampling']}")
        logger.info(f"  🎚️  Needs Mono Conversion: {info['needs_mono_conversion']}")
        logger.info(f"  ✂️  Estimated Chunks: {info['estimated_chunks']}")
        
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return
    
    # Process the audio
    try:
        logger.info("\n🔄 Processing audio...")
        result = processor.process_audio(audio_file)
        
        logger.success("✅ Processing completed successfully!")
        logger.info("📤 Output Information:")
        logger.info(f"  🎵 Array Shape: {result['array'].shape}")
        logger.info(f"  🔊 Sample Rate: {result['sampling_rate']} Hz")
        logger.info(f"  ⏱️  Duration: {result['duration']:.2f} seconds")
        logger.info(f"  💾 Data Type: {result['array'].dtype}")
        logger.info(f"  📊 Peak Amplitude: {result['array'].max():.3f}")
        logger.info(f"  📈 RMS Level: {(result['array']**2).mean()**0.5:.3f}")
        
        # Show original vs processed comparison
        logger.info(f"\n📊 Before/After Comparison:")
        logger.info(f"  Sample Rate: {result['original_sr']} Hz → {result['sampling_rate']} Hz")
        logger.info(f"  Channels: {info['channels']} → 1 (mono)")
        
        # Show if this is ready for Whisper
        logger.success("🎯 Audio is now optimized for Whisper Large-v3!")
        logger.info("   ✅ 16kHz sampling rate")
        logger.info("   ✅ Mono channel")
        logger.info("   ✅ Float32 format")
        logger.info("   ✅ Normalized amplitude")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")


def demo_long_audio_processing(audio_file: Path):
    """Demo long audio file processing with chunking."""
    
    logger.info(f"\n🎵 Demo: Long Audio Processing")
    
    processor = AudioProcessor()
    
    try:
        # Get file info to see if it's actually long
        info = processor.get_audio_info(audio_file)
        
        if info['duration'] <= 30:
            logger.warning(f"⚠️  Audio is only {info['duration']:.1f}s (≤30s), but demonstrating chunking anyway...")
        
        logger.info(f"📄 Processing with chunking (30s chunks, 1s overlap)...")
        chunks = processor.process_long_audio(audio_file, chunk_length=30.0, overlap=1.0)
        
        logger.success(f"✅ Split into {len(chunks)} chunks!")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"  📄 Chunk {i+1}: "
                       f"offset={chunk['chunk_offset']:.1f}s, "
                       f"length={chunk['chunk_length']:.1f}s, "
                       f"shape={chunk['array'].shape}")
        
        total_processed_duration = sum(chunk['duration'] for chunk in chunks)
        logger.info(f"📊 Total processed duration: {total_processed_duration:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Long audio processing failed: {e}")


def demo_format_support():
    """Demo format support checking."""
    
    logger.info(f"\n🎵 Demo: Supported Format Information")
    
    supported = AudioProcessor.get_supported_formats()
    logger.info(f"📁 Supported audio formats: {', '.join(sorted(supported))}")
    
    # Test some common files
    test_files = [
        "music.wav", "speech.mp3", "audio.m4a", "sound.flac", 
        "recording.aac", "track.ogg", "document.txt", "video.mp4"
    ]
    
    logger.info(f"\n🔍 Format checking examples:")
    for test_file in test_files:
        is_supported = AudioProcessor.is_supported_format(test_file)
        status = "✅ Supported" if is_supported else "❌ Not supported"
        logger.info(f"  {test_file}: {status}")


def main():
    """Main demo function with command line interface."""
    
    parser = argparse.ArgumentParser(description="AudioProcessor Demo")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (optional for format demo)")
    parser.add_argument("--basic", action="store_true", help="Run basic processing demo")
    parser.add_argument("--long", action="store_true", help="Run long audio processing demo")
    parser.add_argument("--formats", action="store_true", help="Show supported formats")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    logger.info("🎵 AudioProcessor Demo Starting...")
    logger.info("=" * 50)
    
    # Run format demo (doesn't need file)
    if args.formats or args.all or not any([args.basic, args.long, args.formats]):
        demo_format_support()
    
    # Run file-based demos if file provided
    if args.audio_file:
        audio_file = Path(args.audio_file)
        
        if not audio_file.exists():
            logger.error(f"❌ Audio file not found: {audio_file}")
            return 1
        
        if not AudioProcessor.is_supported_format(audio_file):
            logger.error(f"❌ Unsupported file format: {audio_file.suffix}")
            return 1
        
        if args.basic or args.all:
            demo_basic_processing(audio_file)
        
        if args.long or args.all:
            demo_long_audio_processing(audio_file)
    
    elif any([args.basic, args.long]):
        logger.error("❌ Audio file required for processing demos")
        return 1
    
    logger.info("\n" + "=" * 50)
    logger.success("🎉 AudioProcessor demo completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 