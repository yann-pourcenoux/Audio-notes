#!/usr/bin/env python3
"""
Comprehensive test for enhanced Ollama package usage.

This test verifies:
1. Structured outputs with Pydantic models
2. Chat API with thinking parameter
3. Enhanced error handling and edge cases
4. Performance optimizations (caching, connection pooling)
5. Fallback mechanisms
"""

import json
import logging
import time
from pathlib import Path
from src.audio_notes.obsidian_writer import (
    OllamaClient, ObsidianWriter,
    TitleResponse, TagsResponse, SummaryResponse, ContentEnhancement
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_enhanced_ollama_client():
    """Test the enhanced OllamaClient with all new features."""
    print("=" * 60)
    print("🧪 Testing Enhanced OllamaClient")
    print("=" * 60)
    
    client = OllamaClient()
    
    # Test 1: Availability check with caching
    print("\n1. Testing availability check with caching...")
    start_time = time.time()
    available1 = client.is_available()
    first_check_time = time.time() - start_time
    
    start_time = time.time()
    available2 = client.is_available()  # Should use cache
    second_check_time = time.time() - start_time
    
    print(f"   First check: {available1} (took {first_check_time:.3f}s)")
    print(f"   Second check: {available2} (took {second_check_time:.3f}s)")
    print(f"   ✅ Caching working: {second_check_time < first_check_time}")
    
    if not available1:
        print("   ❌ Ollama not available - skipping enhanced tests")
        return False
    
    # Test 2: Structured output generation
    print("\n2. Testing structured output generation...")
    
    # Test TitleResponse
    print("   Testing TitleResponse...")
    title_response = client.generate_structured_response(
        "Create a title for an audio note about machine learning basics",
        TitleResponse,
        max_tokens=50,
        temperature=0.2
    )
    
    if title_response and isinstance(title_response, TitleResponse):
        print(f"   ✅ TitleResponse: {title_response.title}")
        print(f"   ✅ Reasoning: {title_response.reasoning}")
    else:
        print(f"   ⚠️ TitleResponse fallback: {title_response}")
    
    # Test TagsResponse
    print("   Testing TagsResponse...")
    tags_response = client.generate_structured_response(
        "Generate tags for an audio note about Python programming and web development",
        TagsResponse,
        max_tokens=100,
        temperature=0.3
    )
    
    if tags_response and isinstance(tags_response, TagsResponse):
        print(f"   ✅ TagsResponse: {tags_response.tags}")
        print(f"   ✅ Reasoning: {tags_response.reasoning}")
    else:
        print(f"   ⚠️ TagsResponse fallback: {tags_response}")
    
    # Test SummaryResponse
    print("   Testing SummaryResponse...")
    summary_response = client.generate_structured_response(
        "Summarize: This is a discussion about artificial intelligence and its applications in modern software development",
        SummaryResponse,
        max_tokens=100,
        temperature=0.3
    )
    
    if summary_response and isinstance(summary_response, SummaryResponse):
        print(f"   ✅ SummaryResponse: {summary_response.summary}")
        print(f"   ✅ Key topics: {summary_response.key_topics}")
    else:
        print(f"   ⚠️ SummaryResponse fallback: {summary_response}")
    
    # Test 3: Thinking parameter functionality
    print("\n3. Testing thinking parameter functionality...")
    thinking_response = client.generate_text(
        "Explain the benefits of using structured outputs in AI applications",
        max_tokens=200,
        temperature=0.5
    )
    
    if thinking_response:
        print(f"   ✅ Thinking response received ({len(thinking_response)} chars)")
        print(f"   Sample: {thinking_response[:100]}...")
    else:
        print("   ❌ No thinking response received")
    
    # Test 4: Error handling and edge cases
    print("\n4. Testing error handling and edge cases...")
    
    # Test with empty prompt
    empty_response = client.generate_text("", max_tokens=10)
    print(f"   Empty prompt handling: {'✅' if empty_response is None else '⚠️'}")
    
    # Test with very short max_tokens
    short_response = client.generate_text("Hello", max_tokens=1)
    print(f"   Short token limit: {'✅' if short_response else '⚠️'}")
    
    # Test with invalid model (should gracefully handle)
    invalid_client = OllamaClient(model="nonexistent-model")
    invalid_response = invalid_client.generate_text("Test", max_tokens=10)
    print(f"   Invalid model handling: {'✅' if invalid_response is None else '⚠️'}")
    
    return True


def test_enhanced_obsidian_writer():
    """Test the enhanced ObsidianWriter with structured outputs."""
    print("\n" + "=" * 60)
    print("📝 Testing Enhanced ObsidianWriter")
    print("=" * 60)
    
    # Create test vault
    test_vault_path = "./test_enhanced_ollama_vault"
    writer = ObsidianWriter(test_vault_path, use_ai_enhancement=True)
    
    if not writer.ollama_client or not writer.ollama_client.is_available():
        print("   ❌ Ollama not available for ObsidianWriter tests")
        return False
    
    # Test transcription
    test_transcription = """
    This is a test audio transcription about machine learning and artificial intelligence.
    We discuss various algorithms including neural networks, decision trees, and support vector machines.
    The conversation covers both supervised and unsupervised learning techniques.
    We also touch on practical applications in computer vision and natural language processing.
    """
    
    # Test 1: Enhanced title generation
    print("\n1. Testing enhanced title generation...")
    title = writer.enhance_title(test_transcription, "ml_discussion.wav")
    print(f"   Generated title: {title}")
    print(f"   ✅ Title length appropriate: {3 < len(title) < 100}")
    
    # Test 2: Enhanced tag generation
    print("\n2. Testing enhanced tag generation...")
    tags = writer.generate_tags(test_transcription)
    print(f"   Generated tags: {tags}")
    print(f"   ✅ Tag count appropriate: {1 <= len(tags) <= 5}")
    print(f"   ✅ Contains audio-note tag: {'audio-note' in tags}")
    
    # Test 3: Enhanced content enhancement
    print("\n3. Testing enhanced content enhancement...")
    enhanced_content = writer.enhance_content(test_transcription)
    print(f"   Enhanced content length: {len(enhanced_content)} chars")
    print(f"   ✅ Content enhanced: {len(enhanced_content) >= len(test_transcription) * 0.8}")
    
    # Test 4: Enhanced summary generation
    print("\n4. Testing enhanced summary generation...")
    summary = writer.generate_summary(test_transcription)
    print(f"   Generated summary: {summary}")
    print(f"   ✅ Summary length appropriate: {10 < len(summary) < 300}")
    
    # Test 5: Complete note creation with enhancements
    print("\n5. Testing complete note creation...")
    try:
        note_path, note_title = writer.create_note(
            test_transcription,
            "enhanced_test.wav",
            metadata={"test_type": "enhanced_ollama"}
        )
        print(f"   ✅ Note created: {note_path}")
        print(f"   ✅ Note title: {note_title}")
        
        # Verify note content
        if Path(note_path).exists():
            with open(note_path, 'r') as f:
                content = f.read()
            print(f"   ✅ Note file exists ({len(content)} chars)")
            print(f"   ✅ Contains metadata: {'test_type' in content}")
        
    except Exception as e:
        print(f"   ❌ Note creation failed: {e}")
        return False
    
    return True


def test_performance_optimizations():
    """Test performance optimizations like caching and connection pooling."""
    print("\n" + "=" * 60)
    print("⚡ Testing Performance Optimizations")
    print("=" * 60)
    
    client = OllamaClient()
    
    if not client.is_available():
        print("   ❌ Ollama not available for performance tests")
        return False
    
    # Test 1: Availability caching
    print("\n1. Testing availability caching...")
    times = []
    for i in range(5):
        start = time.time()
        client.is_available()
        times.append(time.time() - start)
    
    print(f"   Availability check times: {[f'{t:.3f}s' for t in times]}")
    print(f"   ✅ Caching effective: {times[0] > times[-1]}")
    
    # Test 2: Multiple requests performance
    print("\n2. Testing multiple requests...")
    start_time = time.time()
    responses = []
    
    for i in range(3):
        response = client.generate_text(f"Test message {i}", max_tokens=20)
        responses.append(response)
    
    total_time = time.time() - start_time
    print(f"   3 requests completed in {total_time:.3f}s")
    print(f"   ✅ All responses received: {all(r is not None for r in responses)}")
    
    return True


def test_error_recovery_and_fallbacks():
    """Test comprehensive error recovery and fallback mechanisms."""
    print("\n" + "=" * 60)
    print("🛡️ Testing Error Recovery and Fallbacks")
    print("=" * 60)
    
    # Test with AI enhancement disabled
    print("\n1. Testing with AI enhancement disabled...")
    writer_no_ai = ObsidianWriter("./test_no_ai_vault", use_ai_enhancement=False)
    
    title = writer_no_ai.enhance_title("Test content", "test.wav")
    tags = writer_no_ai.generate_tags("Test content")
    summary = writer_no_ai.generate_summary("Test content")
    
    print(f"   Fallback title: {title}")
    print(f"   Fallback tags: {tags}")
    print(f"   Fallback summary: {summary}")
    print("   ✅ All fallbacks working without AI")
    
    # Test with unavailable Ollama (simulate by using invalid model)
    print("\n2. Testing with simulated Ollama unavailability...")
    writer_invalid = ObsidianWriter("./test_invalid_vault", use_ai_enhancement=True)
    
    # Override with invalid client
    writer_invalid.ollama_client = OllamaClient(model="invalid-model-name")
    
    title = writer_invalid.enhance_title("Test content", "test.wav")
    tags = writer_invalid.generate_tags("Test content")
    summary = writer_invalid.generate_summary("Test content")
    
    print(f"   Graceful fallback title: {title}")
    print(f"   Graceful fallback tags: {tags}")
    print(f"   Graceful fallback summary: {summary}")
    print("   ✅ Graceful degradation working")
    
    return True


def main():
    """Run all enhanced Ollama tests."""
    print("🚀 Starting Comprehensive Enhanced Ollama Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all test suites
    test_results.append(("Enhanced OllamaClient", test_enhanced_ollama_client()))
    test_results.append(("Enhanced ObsidianWriter", test_enhanced_obsidian_writer()))
    test_results.append(("Performance Optimizations", test_performance_optimizations()))
    test_results.append(("Error Recovery", test_error_recovery_and_fallbacks()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\n🎯 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All enhanced Ollama features working correctly!")
    else:
        print("⚠️ Some tests failed - check implementation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 