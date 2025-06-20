#!/usr/bin/env python3
"""
Test Suite for Audio Notes Configuration System
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_notes.config import ConfigManager, AudioNotesConfig, get_config_manager

def test_config_basic():
    """Test basic configuration functionality"""
    print("🧪 Testing Configuration System...")
    
    # Test default configuration
    config = AudioNotesConfig()
    assert config.language == "auto"
    assert config.task == "transcribe"
    assert config.enhance_notes == True
    print("  ✅ Default configuration works")
    
    # Test config manager
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        manager = ConfigManager(str(config_path))
        
        # Test file creation
        assert config_path.exists()
        print("  ✅ Configuration file created")
        
        # Test get/set operations
        assert manager.get('language') == 'auto'
        assert manager.set('language', 'en') == True
        assert manager.get('language') == 'en'
        print("  ✅ Get/set operations work")
        
        # Test CLI merging
        cli_args = {'language': None, 'temperature': 0.5}
        merged = manager.merge_with_cli_args(cli_args)
        assert merged['language'] == 'en'  # From config
        assert merged['temperature'] == 0.5  # From CLI
        print("  ✅ CLI argument merging works")
        
        # Test reset
        manager.reset('language')
        assert manager.get('language') == 'auto'
        print("  ✅ Reset functionality works")
        
    print("  🎯 Configuration system: PASSED\n")

def test_cli_integration():
    """Test CLI integration patterns"""
    print("🧪 Testing CLI Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        manager = ConfigManager(str(config_path))
        
        # Set user preferences
        manager.set('vault_path', '/test/vault')
        manager.set('language', 'en')
        manager.set('enhance_notes', True)
        
        # Simulate CLI command
        cli_args = {
            'language': None,        # Use config
            'vault_path': '/override/vault',  # Override config
            'enhance_notes': None,   # Use config
            'temperature': 0.3       # New value
        }
        
        merged = manager.merge_with_cli_args(cli_args)
        
        assert merged['language'] == 'en'
        assert merged['vault_path'] == '/override/vault'
        assert merged['enhance_notes'] == True
        assert merged['temperature'] == 0.3
        
        print("  ✅ CLI integration works correctly")
        
    print("  🎯 CLI integration: PASSED\n")

def run_tests():
    """Run all tests"""
    print("🚀 Audio Notes Configuration System Tests")
    print("=" * 50)
    
    try:
        test_config_basic()
        test_cli_integration()
        
        print("🎉 ALL TESTS PASSED!")
        print("✅ Configuration system is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 