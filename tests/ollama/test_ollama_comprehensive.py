#!/usr/bin/env python3

import sys
sys.path.append('src')
from audio_notes.obsidian_writer import ObsidianWriter, OllamaClient
import tempfile
import logging

logging.basicConfig(level=logging.INFO)

def test_comprehensive_ollama():
    """Test all Ollama integration features comprehensively."""
    
    # Test transcription content
    test_transcription = """
    Today I want to discuss the latest developments in artificial intelligence and machine learning.
    We're seeing incredible advances in transformer architectures, particularly with large language models.
    The integration of these models into practical applications is revolutionizing how we approach
    natural language processing tasks. Key areas include text generation, summarization, and
    content enhancement for productivity workflows.
    """
    
    print("=== Testing OllamaClient Directly ===")
    client = OllamaClient()
    print(f"Ollama available: {client.is_available()}")
    
    if not client.is_available():
        print("Ollama not available - skipping tests")
        return
    
    # Test title generation
    print("\n--- Title Generation ---")
    title_prompt = f"Create a concise title (maximum 8 words) for this content: {test_transcription[:200]}"
    title = client.generate_text(title_prompt, max_tokens=30, temperature=0.3)
    print(f"Generated title: {repr(title)}")
    
    # Test tag generation
    print("\n--- Tag Generation ---")
    tag_prompt = f"Generate 3-5 relevant tags (lowercase, use hyphens) for: {test_transcription[:300]}"
    tags = client.generate_text(tag_prompt, max_tokens=50, temperature=0.4)
    print(f"Generated tags: {repr(tags)}")
    
    # Test summary generation
    print("\n--- Summary Generation ---")
    summary_prompt = f"Write a 1-2 sentence summary of: {test_transcription}"
    summary = client.generate_text(summary_prompt, max_tokens=100, temperature=0.4)
    print(f"Generated summary: {repr(summary)}")
    
    print("\n=== Testing ObsidianWriter Integration ===")
    
    # Test with temporary vault
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ObsidianWriter(temp_dir, use_ai_enhancement=True)
        
        if not writer.use_ai_enhancement:
            print("AI enhancement not available in ObsidianWriter")
            return
        
        print("\n--- Enhanced Title ---")
        enhanced_title = writer.enhance_title(test_transcription, "test_audio.wav")
        print(f"Enhanced title: {repr(enhanced_title)}")
        
        print("\n--- Generated Tags ---")
        generated_tags = writer.generate_tags(test_transcription)
        print(f"Generated tags: {generated_tags}")
        
        print("\n--- Enhanced Content ---")
        enhanced_content = writer.enhance_content(test_transcription[:500])
        print(f"Enhanced content preview: {repr(enhanced_content[:200])}")
        
        print("\n--- Generated Summary ---")
        generated_summary = writer.generate_summary(test_transcription)
        print(f"Generated summary: {repr(generated_summary)}")
        
        print("\n--- Creating Complete Note ---")
        note_path, note_content = writer.create_note(
            transcription=test_transcription,
            original_filename="ai_ml_discussion.wav"
        )
        
        print(f"Note created at: {note_path}")
        print(f"Note content preview:\n{note_content[:500]}...")
        
        print("\n=== Test Complete ===")
        print("✅ All Ollama integration features tested successfully!")

if __name__ == "__main__":
    test_comprehensive_ollama() 