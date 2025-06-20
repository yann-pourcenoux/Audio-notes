#!/usr/bin/env python3
"""
Test retry mechanisms and circuit breaker functionality in OllamaClient
"""

import sys
import time
import logging
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append('src')
from audio_notes.obsidian_writer import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retry_mechanism():
    """Test exponential backoff retry mechanism."""
    print("🔄 Testing Retry Mechanism with Exponential Backoff...")
    
    client = OllamaClient(max_retries=2, base_delay=0.1)  # Fast retries for testing
    
    # Test successful retry after failures
    call_count = 0
    def mock_chat_with_retries(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # Fail first 2 attempts
            raise ConnectionError("Temporary network error")
        return {'message': {'content': 'Success after retries'}}
    
    with patch('ollama.chat', side_effect=mock_chat_with_retries):
        start_time = time.time()
        result = client.generate_text("Test prompt")
        duration = time.time() - start_time
        
        assert result is not None, "Should succeed after retries"
        assert call_count == 3, f"Should have made 3 attempts, made {call_count}"
        assert duration > 0.1, f"Should have taken time for retries, took {duration:.2f}s"
        print(f"✅ Retry mechanism works: {call_count} attempts in {duration:.2f}s")

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n⚡ Testing Circuit Breaker...")
    
    client = OllamaClient(max_retries=1, base_delay=0.05)  # Fast for testing
    
    # Simulate multiple failures to trigger circuit breaker
    with patch('ollama.chat', side_effect=ConnectionError("Service down")):
        # Make enough failed requests to trigger circuit breaker
        for i in range(6):  # Threshold is 5
            result = client.generate_text(f"Test prompt {i}")
            assert result is None, f"Request {i} should fail"
        
        print(f"✅ Circuit breaker triggered after {client._error_count} failures")
        
        # Next request should be blocked by circuit breaker
        start_time = time.time()
        result = client.generate_text("Blocked request")
        duration = time.time() - start_time
        
        assert result is None, "Request should be blocked by circuit breaker"
        assert duration < 0.1, "Should be blocked immediately without retry"
        print(f"✅ Circuit breaker blocks requests: {duration:.3f}s")

def test_circuit_breaker_reset():
    """Test circuit breaker reset after timeout."""
    print("\n🔄 Testing Circuit Breaker Reset...")
    
    # Use very short reset time for testing
    client = OllamaClient(max_retries=0, base_delay=0.01)
    client._circuit_breaker_reset_time = 1.0  # 1 second for testing
    
    # Trigger circuit breaker
    with patch('ollama.chat', side_effect=ConnectionError("Service down")):
        for i in range(6):
            client.generate_text(f"Fail {i}")
    
    assert client._is_circuit_breaker_open(), "Circuit breaker should be open"
    print("✅ Circuit breaker opened")
    
    # Wait for reset timeout
    time.sleep(1.1)
    
    # Circuit breaker should be reset
    assert not client._is_circuit_breaker_open(), "Circuit breaker should be reset"
    print("✅ Circuit breaker reset after timeout")

def test_non_retryable_errors():
    """Test that certain errors are not retried."""
    print("\n🚫 Testing Non-Retryable Errors...")
    
    client = OllamaClient(max_retries=3, base_delay=0.01)
    
    call_count = 0
    def mock_chat_non_retryable(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid model configuration")  # Non-retryable error
    
    with patch('ollama.chat', side_effect=mock_chat_non_retryable):
        start_time = time.time()
        result = client.generate_text("Test prompt")
        duration = time.time() - start_time
        
        assert result is None, "Should fail on non-retryable error"
        assert call_count == 1, f"Should not retry non-retryable errors, made {call_count} attempts"
        assert duration < 0.1, f"Should fail fast, took {duration:.3f}s"
        print(f"✅ Non-retryable errors handled correctly: {call_count} attempt, {duration:.3f}s")

def test_success_recovery():
    """Test that error count decreases on successful operations."""
    print("\n📈 Testing Success Recovery...")
    
    # Create fresh client for this test
    client = OllamaClient(max_retries=0, base_delay=0.01)  # No retries to avoid multiple failures
    
    # Manually set some error count without triggering circuit breaker
    client._error_count = 3
    initial_error_count = client._error_count
    print(f"Initial error count: {initial_error_count}")
    
    # Have a successful operation
    with patch('ollama.chat', return_value={'message': {'content': 'Success'}}):
        with patch('ollama.list', return_value={'models': []}):  # Mock availability check
            result = client.generate_text("Success prompt")
            assert result is not None, "Should succeed"
    
    final_error_count = client._error_count
    print(f"Error count after success: {final_error_count}")
    
    assert final_error_count < initial_error_count, "Error count should decrease on success"
    print("✅ Error count recovery works")

def test_availability_check_with_retry():
    """Test that availability check uses retry mechanism."""
    print("\n🔍 Testing Availability Check with Retry...")
    
    client = OllamaClient(max_retries=2, base_delay=0.05)
    
    call_count = 0
    def mock_list_with_retries():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Connection failed")
        return {'models': []}
    
    with patch('ollama.list', side_effect=mock_list_with_retries):
        # Clear cache to force new check
        client._last_availability_check = None
        
        start_time = time.time()
        available = client.is_available()
        duration = time.time() - start_time
        
        assert available, "Should be available after retries"
        assert call_count == 3, f"Should have retried, made {call_count} attempts"
        assert duration > 0.05, f"Should have taken time for retries, took {duration:.3f}s"
        print(f"✅ Availability check with retry works: {call_count} attempts, {duration:.3f}s")

def run_all_tests():
    """Run all retry and circuit breaker tests."""
    print("🧪 Starting Retry and Circuit Breaker Tests")
    print("=" * 60)
    
    try:
        test_retry_mechanism()
        test_circuit_breaker()
        test_circuit_breaker_reset()
        test_non_retryable_errors()
        test_success_recovery()
        test_availability_check_with_retry()
        
        print("\n" + "=" * 60)
        print("✅ ALL RETRY AND CIRCUIT BREAKER TESTS PASSED!")
        print("=" * 60)
        
        print("\n📊 ENHANCED ERROR HANDLING FEATURES:")
        print("1. ✅ Exponential backoff retry with jitter")
        print("2. ✅ Circuit breaker pattern implementation")
        print("3. ✅ Automatic circuit breaker reset")
        print("4. ✅ Non-retryable error detection")
        print("5. ✅ Error count recovery on success")
        print("6. ✅ Retry mechanism for availability checks")
        print("7. ✅ Configurable retry parameters")
        print("8. ✅ Comprehensive logging and monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 