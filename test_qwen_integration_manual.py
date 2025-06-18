#!/usr/bin/env python3
"""
Manual test for Qwen integration via Ollama
"""

import sys
import os
sys.path.append('src')

from audio_notes.obsidian_writer import ObsidianWriter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_integration():
    """Test Qwen integration with substantial content."""
    
    # Test content - more substantial for better AI enhancement
    test_transcription = """
    Hello, this is a comprehensive test of the audio notes system. 
    I'm testing the integration between OpenAI's Whisper Large-v3 model for transcription 
    and Qwen 3.0.6B via Ollama for intelligent content enhancement.
    
    The system should be able to process this audio transcription and enhance it with:
    - Intelligent title generation based on the content
    - Relevant tags for organization
    - Content summarization for quick reference
    - Proper markdown formatting for Obsidian
    
    This is particularly important for productivity workflows where users need to quickly
    process and organize their audio notes, meeting recordings, lectures, or personal voice memos.
    The AI enhancement should make the notes more searchable and better organized.
    
    Today we're discussing the implementation of machine learning models in production environments,
    focusing on the challenges of model deployment, monitoring, and maintenance. Key topics include
    containerization with Docker, orchestration with Kubernetes, and monitoring with MLflow.
    """
    
    # Test vault path
    vault_path = "./test_qwen_manual_vault"
    
    print("=== Testing Qwen Integration ===")
    print(f"Vault path: {vault_path}")
    print(f"Content length: {len(test_transcription)} characters")
    
    try:
        # Create ObsidianWriter with AI enhancement
        writer = ObsidianWriter(vault_path, use_ai_enhancement=True)
        
        # Test if Ollama is available
        if not writer.use_ai_enhancement:
            print("❌ AI enhancement not available - Ollama may not be running")
            return False
            
        print("✅ AI enhancement is available")
        
        # Create note with AI enhancement
        note_path, note_content = writer.create_note(
            transcription=test_transcription,
            original_filename="test_manual_audio.wav"
        )
        
        print(f"✅ Note created successfully at: {note_path}")
        print("\n=== Generated Note Content ===")
        print(note_content)
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen_integration()
    if success:
        print("\n🎉 Qwen integration test completed successfully!")
    else:
        print("\n💥 Qwen integration test failed!")
        sys.exit(1) 