#!/usr/bin/env python3

import sys
sys.path.append('src')
from audio_notes.obsidian_writer import ObsidianWriter
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)

def test_obsidian_note_creation():
    """Test Obsidian note creation with the actual transcription content."""
    
    # Use the actual transcription content
    transcription = " BELL RINGS"
    vault_path = "test_end_to_end_vault"
    
    print(f"Testing with transcription: '{transcription}'")
    print(f"Vault path: {vault_path}")
    
    try:
        # Initialize ObsidianWriter
        writer = ObsidianWriter(vault_path=vault_path, use_ai_enhancement=True)
        print(f"✅ ObsidianWriter initialized")
        
        # Test create_note
        note_path, note_title = writer.create_note(
            transcription=transcription,
            original_filename="test_audio.wav",
            metadata={
                'language': 'unknown',
                'task': 'transcribe',
                'test_mode': True
            }
        )
        
        print(f"✅ Note created successfully!")
        print(f"   Path: {note_path}")
        print(f"   Title: {note_title}")
        
        # Verify the file exists
        import os
        if os.path.exists(note_path):
            print(f"✅ File exists on disk: {note_path}")
            with open(note_path, 'r') as f:
                content = f.read()
                print(f"✅ File content ({len(content)} chars):")
                print("=" * 50)
                print(content)
                print("=" * 50)
        else:
            print(f"❌ File does not exist: {note_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obsidian_note_creation() 