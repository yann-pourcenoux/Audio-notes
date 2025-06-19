#!/usr/bin/env python3
"""
Comprehensive End-to-End Workflow Test for Audio Notes CLI
Tests all major functionality and configurations
"""

import subprocess
import os
import json
import tempfile
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {result.stderr}")
        return False, result.stderr
    return True, result.stdout

def test_comprehensive_workflow():
    """Test the complete end-to-end workflow with various configurations."""
    
    print("🧪 COMPREHENSIVE END-TO-END WORKFLOW TEST")
    print("=" * 60)
    
    # Test 1: Quick Note Creation
    print("\n1️⃣ Testing Quick Note Creation...")
    success, output = run_command(
        "uv run python -m audio_notes.cli quick-note test_audio.wav --vault-path test_final_workflow_vault"
    )
    if success:
        print("✅ Quick note creation: PASSED")
        # Check if note was created
        vault_path = Path("test_final_workflow_vault/Audio Notes")
        if vault_path.exists() and list(vault_path.glob("*.md")):
            print("✅ Obsidian note created: PASSED")
        else:
            print("❌ Obsidian note creation: FAILED")
    else:
        print("❌ Quick note creation: FAILED")
    
    # Test 2: Process Command with Various Options
    print("\n2️⃣ Testing Process Command with Various Options...")
    
    # Test 2a: Basic processing
    success, output = run_command(
        "uv run python -m audio_notes.cli process test_audio.wav --output-format text"
    )
    if success:
        print("✅ Basic processing: PASSED")
    else:
        print("❌ Basic processing: FAILED")
    
    # Test 2b: JSON output
    success, output = run_command(
        "uv run python -m audio_notes.cli process test_audio.wav --output-format json --language auto --task transcribe"
    )
    if success:
        print("✅ JSON output processing: PASSED")
        # Check if JSON file exists
        if os.path.exists("test_audio_transcription.json"):
            print("✅ JSON file created: PASSED")
            try:
                with open("test_audio_transcription.json", 'r') as f:
                    data = json.load(f)
                    if 'text' in data and 'metadata' in data:
                        print("✅ JSON structure valid: PASSED")
                    else:
                        print("❌ JSON structure invalid: FAILED")
            except:
                print("❌ JSON parsing: FAILED")
        else:
            print("❌ JSON file creation: FAILED")
    else:
        print("❌ JSON output processing: FAILED")
    
    # Test 3: Status Command
    print("\n3️⃣ Testing Status Command...")
    success, output = run_command("uv run python -m audio_notes.cli status")
    if success and "Whisper available" in output and "Ollama available" in output:
        print("✅ Status command: PASSED")
        print("✅ All components available: PASSED")
    else:
        print("❌ Status command: FAILED")
    
    # Test 4: Help Commands
    print("\n4️⃣ Testing Help Commands...")
    success, output = run_command("uv run python -m audio_notes.cli --help")
    if success and "process" in output and "quick-note" in output:
        print("✅ Main help: PASSED")
    else:
        print("❌ Main help: FAILED")
    
    success, output = run_command("uv run python -m audio_notes.cli process --help")
    if success and "audio_files" in output:
        print("✅ Process help: PASSED")
    else:
        print("❌ Process help: FAILED")
    
    # Test 5: Error Handling
    print("\n5️⃣ Testing Error Handling...")
    success, output = run_command("uv run python -m audio_notes.cli quick-note non_existent_file.wav")
    if not success and "does not exist" in output:
        print("✅ Non-existent file error: PASSED")
    else:
        print("❌ Non-existent file error: FAILED")
    
    # Test 6: Configuration Testing
    print("\n6️⃣ Testing Different Configurations...")
    
    # Test with timestamps
    success, output = run_command(
        "uv run python -m audio_notes.cli process test_audio.wav --timestamps word --temperature 0.3"
    )
    if success:
        print("✅ Timestamp configuration: PASSED")
    else:
        print("❌ Timestamp configuration: FAILED")
    
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE WORKFLOW TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_comprehensive_workflow() 