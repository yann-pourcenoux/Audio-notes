#!/usr/bin/env python3

import sys
sys.path.append('src')
from audio_notes.obsidian_writer import OllamaClient
import logging

logging.basicConfig(level=logging.INFO)

def test_ollama_responses():
    client = OllamaClient()
    print('Ollama available:', client.is_available())

    if not client.is_available():
        print("Ollama not available - skipping tests")
        return

    # Test simple title generation
    test_text = 'This is a test transcription about machine learning and AI development.'
    
    print('\n=== Testing Title Generation ===')
    title = client.generate_text('Create a short title (max 8 words) for: ' + test_text)
    print(f'Generated title: {repr(title)}')
    
    print('\n=== Testing Direct Response ===')
    direct = client.generate_text('What is 2+2? Answer with just the number.')
    print(f'Direct response: {repr(direct)}')
    
    print('\n=== Testing Raw Response ===')
    # Test without the "Give a direct answer" prefix
    import ollama
    try:
        raw_response = ollama.generate(
            model="qwen3:0.6b",
            prompt="Create a title for: machine learning discussion",
            options={"temperature": 0.3, "num_predict": 50}
        )
        print(f'Raw ollama response: {repr(raw_response.get("response", ""))}')
    except Exception as e:
        print(f'Raw ollama error: {e}')

if __name__ == "__main__":
    test_ollama_responses() 