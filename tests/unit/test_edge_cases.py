#!/usr/bin/env python3

from audio_notes.obsidian_writer import ObsidianWriter
import tempfile
import os

def test_edge_cases():
    """Test various edge cases for Obsidian note creation."""
    
    print("🧪 Testing Edge Cases for Obsidian Note Creation")
    print("=" * 50)
    
    # Test 1: Empty transcription
    print("\n1. Testing empty transcription...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ObsidianWriter(vault_path=temp_dir, use_ai_enhancement=True)
            note_path, note_title = writer.create_note(
                transcription="",
                original_filename="empty_test.wav"
            )
            print("❌ Should have failed with empty transcription")
    except ValueError as e:
        print(f"✅ Correctly handled empty transcription: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
    
    # Test 2: Very long transcription
    print("\n2. Testing very long transcription...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ObsidianWriter(vault_path=temp_dir, use_ai_enhancement=True)
            long_text = "This is a very long transcription. " * 1000  # ~35,000 chars
            note_path, note_title = writer.create_note(
                transcription=long_text,
                original_filename="long_test.wav"
            )
            print(f"✅ Successfully created note with long transcription: {note_path}")
    except Exception as e:
        print(f"❌ Failed with long transcription: {e}")
    
    # Test 3: Special characters in transcription
    print("\n3. Testing special characters...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ObsidianWriter(vault_path=temp_dir, use_ai_enhancement=True)
            special_text = "Test with émojis 🎵🎤 and spëcial chars: <>:\"/\\|?*"
            note_path, note_title = writer.create_note(
                transcription=special_text,
                original_filename="special_test.wav"
            )
            print(f"✅ Successfully handled special characters: {note_path}")
    except Exception as e:
        print(f"❌ Failed with special characters: {e}")
    
    # Test 4: Unicode content
    print("\n4. Testing Unicode content...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ObsidianWriter(vault_path=temp_dir, use_ai_enhancement=True)
            unicode_text = "Testing 中文, العربية, русский, 日本語, and 한국어 text"
            note_path, note_title = writer.create_note(
                transcription=unicode_text,
                original_filename="unicode_test.wav"
            )
            print(f"✅ Successfully handled Unicode: {note_path}")
    except Exception as e:
        print(f"❌ Failed with Unicode: {e}")
    
    # Test 5: AI enhancement disabled
    print("\n5. Testing with AI enhancement disabled...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ObsidianWriter(vault_path=temp_dir, use_ai_enhancement=False)
            note_path, note_title = writer.create_note(
                transcription="Simple test without AI enhancement",
                original_filename="no_ai_test.wav"
            )
            print(f"✅ Successfully created note without AI: {note_path}")
    except Exception as e:
        print(f"❌ Failed without AI enhancement: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Edge case testing completed!")

if __name__ == "__main__":
    test_edge_cases() 