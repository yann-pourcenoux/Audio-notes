#!/usr/bin/env python3
"""
Comprehensive test suite for ObsidianWriter class.
Tests all functionality including AI enhancement, error handling, and edge cases.
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_notes.obsidian_writer import ObsidianWriter, OllamaClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_ollama_availability():
    """Test Ollama service availability."""
    logger.info("=== Testing Ollama Availability ===")
    client = OllamaClient()
    available = client.is_available()
    logger.info(f"Ollama available: {available}")
    return available

def test_basic_note_creation():
    """Test basic note creation without AI enhancement."""
    logger.info("=== Testing Basic Note Creation (No AI) ===")
    
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
        
        # Verify the note exists
        assert Path(note_path).exists(), "Note file should exist"
        
        # Check content
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "This is a test transcription" in content
            assert "AI Enhanced: False" in content
            
        logger.info("✅ Basic note creation test passed")
        return True

def test_ai_enhanced_note_creation():
    """Test AI-enhanced note creation."""
    logger.info("=== Testing AI-Enhanced Note Creation ===")
    
    if not test_ollama_availability():
        logger.warning("Skipping AI tests - Ollama not available")
        return False
    
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
        
        # Verify the note exists
        assert Path(note_path).exists(), "Note file should exist"
        
        # Check content
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "AI Enhanced: True" in content
            assert "artificial intelligence" in content.lower() or "ai" in content.lower()
            
        logger.info("✅ AI-enhanced note creation test passed")
        return True

def test_error_handling():
    """Test various error handling scenarios."""
    logger.info("=== Testing Error Handling ===")
    
    # Test empty transcription
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=False)
        
        try:
            writer.create_note("")
            assert False, "Should have raised ValueError for empty transcription"
        except ValueError as e:
            logger.info(f"✅ Correctly caught empty transcription error: {e}")
        
        # Test invalid vault path
        try:
            invalid_writer = ObsidianWriter("", use_ai_enhancement=False)
            assert False, "Should have raised ValueError for empty vault path"
        except ValueError as e:
            logger.info(f"✅ Correctly caught empty vault path error: {e}")
    
    logger.info("✅ Error handling tests passed")
    return True

def test_filename_sanitization():
    """Test filename sanitization with problematic characters."""
    logger.info("=== Testing Filename Sanitization ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=False)
        
        # Test transcription that might generate problematic title
        problematic_transcription = """
        This is a test with "quotes" and <brackets> and /slashes/ 
        and other *special* characters that shouldn't be in filenames.
        """
        
        note_path, title = writer.create_note(
            transcription=problematic_transcription.strip(),
            original_filename="problematic_chars.wav"
        )
        
        logger.info(f"Sanitized title: {title}")
        logger.info(f"Path: {note_path}")
        
        # Verify the note exists and path is valid
        assert Path(note_path).exists(), "Note file should exist"
        
        # Check that filename doesn't contain problematic characters
        filename = Path(note_path).name
        problematic_chars = '<>:"/\\|?*'
        for char in problematic_chars:
            assert char not in filename, f"Filename should not contain '{char}'"
            
        logger.info("✅ Filename sanitization test passed")
        return True

def test_file_conflict_resolution():
    """Test file conflict resolution when duplicate names occur."""
    logger.info("=== Testing File Conflict Resolution ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=False)
        
        test_transcription = "This is a test for file conflict resolution."
        
        # Create first note
        note_path1, title1 = writer.create_note(
            transcription=test_transcription,
            original_filename="conflict_test.wav"
        )
        
        # Create second note with same content (should get unique name)
        note_path2, title2 = writer.create_note(
            transcription=test_transcription,
            original_filename="conflict_test.wav"
        )
        
        logger.info(f"First note: {note_path1}")
        logger.info(f"Second note: {note_path2}")
        
        # Verify both notes exist and have different paths
        assert Path(note_path1).exists(), "First note should exist"
        assert Path(note_path2).exists(), "Second note should exist"
        assert note_path1 != note_path2, "Notes should have different paths"
        
        logger.info("✅ File conflict resolution test passed")
        return True

def test_note_appending():
    """Test appending content to existing notes."""
    logger.info("=== Testing Note Appending ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=False)
        
        # Create initial note
        initial_content = "This is the initial content of the note."
        note_path, title = writer.create_note(
            transcription=initial_content,
            original_filename="append_test.wav"
        )
        
        # Append additional content
        additional_content = "This is additional content appended later."
        writer.append_to_note(note_path, additional_content, "Follow-up")
        
        # Verify both contents are in the file
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert initial_content in content
            assert additional_content in content
            assert "Follow-up" in content
            
        logger.info("✅ Note appending test passed")
        return True

def test_vault_organization():
    """Test vault organization and directory structure."""
    logger.info("=== Testing Vault Organization ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=False)
        
        # Create multiple notes
        for i in range(3):
            transcription = f"This is test note number {i+1}."
            writer.create_note(
                transcription=transcription,
                original_filename=f"test_{i+1}.wav"
            )
        
        # Check vault structure
        vault_path = Path(temp_dir)
        audio_notes_dir = vault_path / "Audio Notes"
        
        assert vault_path.exists(), "Vault directory should exist"
        assert audio_notes_dir.exists(), "Audio Notes subdirectory should exist"
        
        # Check that notes were created
        notes = list(audio_notes_dir.glob("*.md"))
        assert len(notes) == 3, f"Should have 3 notes, found {len(notes)}"
        
        # List notes using the writer method
        note_list = writer.list_notes()
        assert len(note_list) == 3, f"list_notes should return 3 notes, got {len(note_list)}"
        
        logger.info(f"Created {len(notes)} notes in organized vault structure")
        logger.info("✅ Vault organization test passed")
        return True

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("🚀 Starting Comprehensive Obsidian Writer Tests")
    
    tests = [
        test_basic_note_creation,
        test_ai_enhanced_note_creation,
        test_error_handling,
        test_filename_sanitization,
        test_file_conflict_resolution,
        test_note_appending,
        test_vault_organization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Obsidian Writer is working correctly.")
        return True
    else:
        logger.error(f"❌ {failed} tests failed. Please review the issues.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 