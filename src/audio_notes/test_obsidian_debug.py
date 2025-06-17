#!/usr/bin/env python3
"""
Debug version of the comprehensive test to identify assertion failures.
"""

import sys
import os
import logging
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_notes.obsidian_writer import ObsidianWriter

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_basic_note_creation():
    """Debug basic note creation."""
    logger.info("=== Debugging Basic Note Creation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=False)
        
        test_transcription = """
        This is a test transcription for basic note creation.
        It should work without AI enhancement.
        The note should be created with a timestamp-based title.
        """
        
        note_path, title = writer.create_note(
            transcription=test_transcription.strip(),
            original_filename="test_basic.wav"
        )
        
        logger.info(f"Created note: {title}")
        logger.info(f"Path: {note_path}")
        
        # Check if file exists
        path_obj = Path(note_path)
        logger.info(f"File exists: {path_obj.exists()}")
        
        if path_obj.exists():
            # Read and examine content
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Content length: {len(content)}")
                logger.info("Content preview:")
                logger.info(content[:500] + "..." if len(content) > 500 else content)
                
                # Check specific assertions
                has_transcription = "This is a test transcription" in content
                has_ai_enhanced = "AI Enhanced: False" in content
                
                logger.info(f"Has transcription text: {has_transcription}")
                logger.info(f"Has AI Enhanced: False: {has_ai_enhanced}")
                
                if not has_transcription:
                    logger.error("Missing transcription text in content")
                if not has_ai_enhanced:
                    logger.error("Missing 'AI Enhanced: False' in content")
                    
        return True

def debug_ai_enhanced_note():
    """Debug AI-enhanced note creation."""
    logger.info("=== Debugging AI-Enhanced Note Creation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=True)
        
        test_transcription = """
        I'm discussing the latest developments in artificial intelligence and machine learning.
        Today we're covering neural networks, deep learning architectures, and their applications
        in natural language processing. The transformer architecture has revolutionized
        how we approach language understanding and generation tasks.
        """
        
        note_path, title = writer.create_note(
            transcription=test_transcription.strip(),
            original_filename="ai_discussion.wav"
        )
        
        logger.info(f"AI-generated title: {title}")
        logger.info(f"Path: {note_path}")
        
        # Check if file exists
        path_obj = Path(note_path)
        logger.info(f"File exists: {path_obj.exists()}")
        
        if path_obj.exists():
            # Read and examine content
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Content length: {len(content)}")
                logger.info("Content preview:")
                logger.info(content[:500] + "..." if len(content) > 500 else content)
                
                # Check specific assertions
                has_ai_enhanced = "AI Enhanced: True" in content
                has_ai_content = "artificial intelligence" in content.lower() or "ai" in content.lower()
                
                logger.info(f"Has AI Enhanced: True: {has_ai_enhanced}")
                logger.info(f"Has AI content: {has_ai_content}")
                
                if not has_ai_enhanced:
                    logger.error("Missing 'AI Enhanced: True' in content")
                if not has_ai_content:
                    logger.error("Missing AI-related content")
                    
        return True

if __name__ == "__main__":
    debug_basic_note_creation()
    print("\n" + "="*50 + "\n")
    debug_ai_enhanced_note() 