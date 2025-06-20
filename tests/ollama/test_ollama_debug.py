#!/usr/bin/env python3

import sys
sys.path.append('src')

from audio_notes.obsidian_writer import OllamaClient
import logging

logging.basicConfig(level=logging.DEBUG)

def test_ollama():
    client = OllamaClient()
    
    print("Testing Ollama availability...")
    print(f"Available: {client.is_available()}")
    
    test_text = """Hello, this is a test transcription of an audio file. 
    I'm testing the Obsidian writer integration with Qwen 3.0.6B model.
    The system should be able to enhance this content, generate appropriate tags,
    and create a well-structured markdown note for Obsidian."""
    
    print("\n=== Testing Title Generation ===")
    title_prompt = f"""Generate a concise title (maximum 8 words) for this audio transcription:

"{test_text[:300]}"

Title:"""
    
    print(f"Prompt: {title_prompt}")
    title_result = client.generate_text(title_prompt, max_tokens=50, temperature=0.3)
    print(f"Raw result: {repr(title_result)}")
    
    print("\n=== Testing Tag Generation ===")
    tag_prompt = f"""Generate 3-5 tags for organizing this transcription. Use lowercase and hyphens instead of spaces.

"{test_text[:500]}"

Tags:"""
    
    print(f"Prompt: {tag_prompt}")
    tag_result = client.generate_text(tag_prompt, max_tokens=100, temperature=0.4)
    print(f"Raw result: {repr(tag_result)}")
    
    print("\n=== Testing Summary Generation ===")
    summary_prompt = f"""Summarize this transcription in 1-2 sentences:

"{test_text}"

Summary:"""
    
    print(f"Prompt: {summary_prompt}")
    summary_result = client.generate_text(summary_prompt, max_tokens=100, temperature=0.4)
    print(f"Raw result: {repr(summary_result)}")

if __name__ == "__main__":
    test_ollama() 