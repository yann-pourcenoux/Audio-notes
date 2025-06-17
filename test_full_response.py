#!/usr/bin/env python3

import sys
sys.path.append('src')

import requests
import json

def test_full_response():
    base_url = "http://localhost:11434"
    api_url = f"{base_url}/api/generate"
    model = "qwen3:0.6b"
    
    prompt = """Generate a concise title (maximum 8 words) for this audio transcription:

"Hello, this is a test transcription of an audio file. I'm testing the Obsidian writer integration with Qwen 3.0.6B model."

Title:"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 200  # More tokens to see full response
        }
    }
    
    print("Testing full response...")
    print(f"Prompt: {prompt}")
    print("\n" + "="*50)
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        full_response = result.get('response', '')
        
        print(f"FULL RAW RESPONSE:")
        print(repr(full_response))
        print("\n" + "="*50)
        print(f"FULL RAW RESPONSE (formatted):")
        print(full_response)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_full_response() 