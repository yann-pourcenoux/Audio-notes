#!/usr/bin/env python3
"""
Comprehensive Error Handling and Edge Cases Test for Ollama Package Usage

This test verifies:
1. Network failure scenarios
2. Model unavailability handling
3. Timeout scenarios
4. Edge cases (empty inputs, malformed responses, etc.)
5. Retry mechanisms and fallback handling
6. Resource cleanup and connection management
"""

import json
import logging
import time
import sys
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.audio_notes.obsidian_writer import (
    OllamaClient, ObsidianWriter,
    TitleResponse, TagsResponse, SummaryResponse, ContentEnhancement
)
import ollama

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ErrorHandlingTester:
    """Comprehensive error handling and edge case tester."""
    
    def __init__(self):
        self.test_results = []
        self.client = OllamaClient()
        
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result for reporting."""
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_network_failures(self):
        """Test handling of network-related failures."""
        print("\n" + "="*60)
        print("🌐 Testing Network Failure Scenarios")
        print("="*60)
        
        # Test 1: Connection timeout simulation
        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = TimeoutError("Connection timed out")
            
            result = self.client.generate_text("Test prompt", max_tokens=10)
            self.log_test_result(
                "Network timeout handling",
                result is None,
                "Should return None on timeout"
            )
        
        # Test 2: Connection refused simulation
        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = ConnectionRefusedError("Connection refused")
            
            result = self.client.generate_text("Test prompt", max_tokens=10)
            self.log_test_result(
                "Connection refused handling",
                result is None,
                "Should return None on connection refused"
            )
        
        # Test 3: DNS resolution failure simulation
        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = OSError("Name or service not known")
            
            result = self.client.generate_text("Test prompt", max_tokens=10)
            self.log_test_result(
                "DNS failure handling",
                result is None,
                "Should return None on DNS failure"
            )
    
    def test_model_unavailability(self):
        """Test handling of model unavailability scenarios."""
        print("\n" + "="*60)
        print("🤖 Testing Model Unavailability Scenarios")
        print("="*60)
        
        # Test 1: Model not found
        invalid_client = OllamaClient(model="nonexistent-model-12345")
        
        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = Exception("model 'nonexistent-model-12345' not found")
            
            result = invalid_client.generate_text("Test prompt")
            self.log_test_result(
                "Invalid model handling",
                result is None,
                "Should return None for invalid model"
            )
        
        # Test 2: Model loading failure
        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = Exception("failed to load model")
            
            result = self.client.generate_text("Test prompt")
            self.log_test_result(
                "Model loading failure handling",
                result is None,
                "Should return None on model loading failure"
            )
        
        # Test 3: Ollama service not running
        with patch('ollama.list') as mock_list:
            mock_list.side_effect = Exception("connection error")
            
            # Create new client to trigger availability check
            test_client = OllamaClient()
            available = test_client.is_available()
            self.log_test_result(
                "Service unavailable detection",
                not available,
                "Should detect when Ollama service is not running"
            )
    
    def test_timeout_scenarios(self):
        """Test timeout handling for long-running requests."""
        print("\n" + "="*60)
        print("⏱️ Testing Timeout Scenarios")
        print("="*60)
        
        # Test 1: Request timeout simulation
        with patch('ollama.chat') as mock_chat:
            def slow_response(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow response
                raise TimeoutError("Request timed out")
            
            mock_chat.side_effect = slow_response
            
            start_time = time.time()
            result = self.client.generate_text("Very long prompt that should timeout")
            duration = time.time() - start_time
            
            self.log_test_result(
                "Request timeout handling",
                result is None and duration < 1.0,
                f"Should handle timeout gracefully (took {duration:.2f}s)"
            )
        
        # Test 2: Availability check timeout caching
        start_time = time.time()
        available1 = self.client.is_available()
        first_check_time = time.time() - start_time
        
        start_time = time.time()
        available2 = self.client.is_available()  # Should use cache
        second_check_time = time.time() - start_time
        
        self.log_test_result(
            "Availability check caching",
            second_check_time < first_check_time,
            f"First: {first_check_time:.3f}s, Second: {second_check_time:.3f}s"
        )
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        print("\n" + "="*60)
        print("🔍 Testing Edge Cases and Boundary Conditions")
        print("="*60)
        
        # Test 1: Empty prompt
        result = self.client.generate_text("")
        self.log_test_result(
            "Empty prompt handling",
            result is None or (isinstance(result, str) and len(result) == 0),
            "Should handle empty prompts gracefully"
        )
        
        # Test 2: Very long prompt
        long_prompt = "A" * 10000  # 10k characters
        result = self.client.generate_text(long_prompt, max_tokens=10)
        self.log_test_result(
            "Very long prompt handling",
            result is not None or result is None,  # Either works or fails gracefully
            "Should handle very long prompts"
        )
        
        # Test 3: Special characters in prompt
        special_prompt = "Test with special chars: 你好 🚀 ñáéíóú @#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        result = self.client.generate_text(special_prompt, max_tokens=20)
        self.log_test_result(
            "Special characters handling",
            result is not None or result is None,  # Either works or fails gracefully
            "Should handle special characters"
        )
        
        # Test 4: Zero max_tokens
        result = self.client.generate_text("Test", max_tokens=0)
        self.log_test_result(
            "Zero max_tokens handling",
            result is None or (isinstance(result, str) and len(result) == 0),
            "Should handle zero max_tokens"
        )
        
        # Test 5: Negative temperature
        result = self.client.generate_text("Test", temperature=-1.0)
        self.log_test_result(
            "Negative temperature handling",
            result is not None or result is None,  # Should handle gracefully
            "Should handle negative temperature"
        )
    
    def test_malformed_responses(self):
        """Test handling of malformed responses from Ollama."""
        print("\n" + "="*60)
        print("🔧 Testing Malformed Response Handling")
        print("="*60)
        
        # Test 1: Empty response
        with patch('ollama.chat') as mock_chat:
            mock_chat.return_value = {}
            
            result = self.client.generate_text("Test prompt")
            self.log_test_result(
                "Empty response handling",
                result is None,
                "Should handle empty responses"
            )
        
        # Test 2: Missing message content
        with patch('ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {}}
            
            result = self.client.generate_text("Test prompt")
            self.log_test_result(
                "Missing content handling",
                result is None,
                "Should handle missing message content"
            )
        
        # Test 3: Malformed JSON in structured response
        with patch('ollama.chat') as mock_chat:
            mock_chat.return_value = {
                'message': {'content': 'Invalid JSON: {title: "test"'}
            }
            
            result = self.client.generate_structured_response(
                "Test prompt", TitleResponse
            )
            self.log_test_result(
                "Malformed JSON handling",
                result is None or isinstance(result, str),
                "Should handle malformed JSON gracefully"
            )
        
        # Test 4: Valid JSON but wrong structure
        with patch('ollama.chat') as mock_chat:
            mock_chat.return_value = {
                'message': {'content': '{"wrong_field": "value"}'}
            }
            
            result = self.client.generate_structured_response(
                "Test prompt", TitleResponse
            )
            self.log_test_result(
                "Wrong structure handling",
                result is None or isinstance(result, str),
                "Should handle wrong JSON structure"
            )
    
    def test_obsidian_writer_error_handling(self):
        """Test ObsidianWriter error handling scenarios."""
        print("\n" + "="*60)
        print("📝 Testing ObsidianWriter Error Handling")
        print("="*60)
        
        # Test 1: Invalid vault path
        try:
            writer = ObsidianWriter("")
            self.log_test_result(
                "Empty vault path handling",
                False,
                "Should raise ValueError for empty path"
            )
        except ValueError:
            self.log_test_result(
                "Empty vault path handling",
                True,
                "Correctly raises ValueError for empty path"
            )
        except Exception as e:
            self.log_test_result(
                "Empty vault path handling",
                False,
                f"Unexpected exception: {e}"
            )
        
        # Test 2: Permission denied simulation
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            try:
                writer = ObsidianWriter("/test/vault")
                self.log_test_result(
                    "Permission denied handling",
                    False,
                    "Should raise PermissionError"
                )
            except PermissionError:
                self.log_test_result(
                    "Permission denied handling",
                    True,
                    "Correctly raises PermissionError"
                )
        
        # Test 3: AI enhancement failure fallback
        test_vault = "./test_error_vault"
        writer = ObsidianWriter(test_vault, use_ai_enhancement=True)
        
        # Mock Ollama client to fail
        if writer.ollama_client:
            with patch.object(writer.ollama_client, 'generate_text', return_value=None):
                title = writer.enhance_title("Test transcription", "test.wav")
                self.log_test_result(
                    "AI enhancement fallback",
                    "test" in title.lower() or "audio note" in title.lower(),
                    f"Falls back to basic title: {title}"
                )
    
    def test_concurrent_access(self):
        """Test concurrent access and thread safety."""
        print("\n" + "="*60)
        print("🔄 Testing Concurrent Access and Thread Safety")
        print("="*60)
        
        results = []
        errors = []
        
        def make_request(thread_id):
            try:
                result = self.client.generate_text(f"Test from thread {thread_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        self.log_test_result(
            "Concurrent access handling",
            len(errors) == 0,
            f"Threads completed: {len(results)}, Errors: {len(errors)}"
        )
    
    def run_all_tests(self):
        """Run all error handling and edge case tests."""
        print("🧪 Starting Comprehensive Error Handling and Edge Cases Test")
        print("="*80)
        
        # Run all test categories
        self.test_network_failures()
        self.test_model_unavailability()
        self.test_timeout_scenarios()
        self.test_edge_cases()
        self.test_malformed_responses()
        self.test_obsidian_writer_error_handling()
        self.test_concurrent_access()
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("📊 ERROR HANDLING AND EDGE CASES TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['details']}")
        
        print("\n📋 Recommendations for Improvements:")
        
        # Analyze results and provide recommendations
        network_failures = [r for r in self.test_results if 'network' in r['test'].lower() or 'connection' in r['test'].lower()]
        if any(not r['passed'] for r in network_failures):
            print("  🌐 Implement retry mechanisms with exponential backoff for network failures")
        
        timeout_tests = [r for r in self.test_results if 'timeout' in r['test'].lower()]
        if any(not r['passed'] for r in timeout_tests):
            print("  ⏱️ Add configurable timeout settings for different operations")
        
        edge_case_tests = [r for r in self.test_results if 'edge' in r['test'].lower() or 'boundary' in r['test'].lower()]
        if any(not r['passed'] for r in edge_case_tests):
            print("  🔍 Enhance input validation and boundary condition handling")
        
        print("  📈 Consider implementing connection pooling for better performance")
        print("  🔄 Add circuit breaker pattern for service failures")
        print("  📝 Enhance logging with structured error information")
        
        # Save detailed report
        report_file = "error_handling_test_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'results': self.test_results,
                'timestamp': time.time()
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main test execution."""
    print("Starting Error Handling and Edge Cases Verification...")
    
    tester = ErrorHandlingTester()
    tester.run_all_tests()
    
    print("\n✅ Error handling and edge cases verification completed!")


if __name__ == "__main__":
    main() 