#!/usr/bin/env python3

import requests
import json

def test_raw_ollama():
    base_url = "http://localhost:11434"
    api_url = f"{base_url}/api/generate"
    model = "qwen3:0.6b"
    
    # Test simple prompt without system message
    simple_prompt = "Generate a title for this: Hello, this is a test transcription."
    
    payload = {
        "model": model,
        "prompt": simple_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 50
        }
    }
    
    print("Testing simple prompt...")
    print(f"Prompt: {simple_prompt}")
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print(f"Full response: {json.dumps(result, indent=2)}")
        print(f"Response text: {repr(result.get('response', ''))}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_raw_ollama() 