#!/usr/bin/env python3

import requests
import json

def test_hidethinking_options():
    base_url = "http://localhost:11434"
    api_url = f"{base_url}/api/generate"
    model = "qwen3:0.6b"
    
    prompt = "Generate a title for this: Hello, this is a test transcription."
    
    # Test different ways to pass hidethinking
    test_configs = [
        {
            "name": "hidethinking in options",
            "payload": {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50,
                    "hidethinking": True
                }
            }
        },
        {
            "name": "hidethinking as top-level",
            "payload": {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "hidethinking": True,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50
                }
            }
        },
        {
            "name": "think false in options",
            "payload": {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50,
                    "think": False
                }
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n=== Testing: {config['name']} ===")
        try:
            response = requests.post(api_url, json=config['payload'], timeout=30)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '')
            print(f"Response: {repr(response_text[:200])}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_hidethinking_options() 