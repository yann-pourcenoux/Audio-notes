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
    
    print('\n=== Testing Modern Chat API Directly ===')
    # Updated to use modern ollama.chat() API for consistency
    import ollama
    try:
        raw_response = ollama.chat(
            model="qwen3:0.6b",
            messages=[
                {'role': 'user', 'content': 'Create a title for: machine learning discussion'}
            ],
            options={"temperature": 0.3, "num_predict": 50}
        )
        response_content = raw_response.get('message', {}).get('content', '')
        print(f'Raw ollama response: {repr(response_content)}')
    except Exception as e:
        print(f'Raw ollama error: {e}')

    print('\n=== Testing Structured Response Generation ===')
    # Test structured response using Pydantic models
    from audio_notes.obsidian_writer import TitleResponse
    
    structured_title = client.generate_structured_response(
        'Generate a title for a discussion about artificial intelligence',
        TitleResponse,
        max_tokens=100,
        temperature=0.3
    )
    
    if structured_title:
        print(f'Structured title: {structured_title.title}')
        if hasattr(structured_title, 'reasoning') and structured_title.reasoning:
            print(f'Reasoning: {structured_title.reasoning}')
    else:
        print('Structured response failed - using fallback')

    print('\n=== Testing Enhanced Error Handling ===')
    # Test retry mechanism by temporarily causing failures
    original_max_retries = client.max_retries
    client.max_retries = 1  # Reduce retries for faster testing
    
    # This should work normally
    normal_response = client.generate_text("Say 'Hello'", max_tokens=10)
    print(f'Normal response: {repr(normal_response)}')
    
    # Restore original settings
    client.max_retries = original_max_retries
    
    print('\n=== Testing Availability Caching ===')
    import time
    
    # First availability check
    start_time = time.time()
    available1 = client.is_available()
    first_duration = time.time() - start_time
    
    # Second check should use cache
    start_time = time.time()
    available2 = client.is_available()
    second_duration = time.time() - start_time
    
    print(f'First check: {available1} ({first_duration:.3f}s)')
    print(f'Second check: {available2} ({second_duration:.3f}s)')
    print(f'Caching working: {second_duration < first_duration}')

if __name__ == "__main__":
    test_ollama_responses() 