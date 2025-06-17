"""
Obsidian Markdown Writer Module

This module provides functionality to write transcriptions as Markdown notes in an Obsidian vault,
integrating Qwen 3:0.6B via Ollama for intelligent content enhancement.
"""

import os
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API to use Qwen 3:0.6B model."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:0.6b"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate text using Qwen 3:0.6B model via Ollama."""
        if not self.is_available():
            logger.warning("Ollama service is not available")
            return None
            
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens * 3  # Give more room for thinking + answer
                },
                "format": "",
                "raw": False,
                "keep_alive": "5m"
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get("response", "").strip()
            
            # Parse response to extract final answer from thinking process
            import re
            
            # Look for quoted suggestions in the thinking process
            quotes_match = re.findall(r'"([^"]+)"', raw_response)
            if quotes_match:
                # Filter out the original content and keep suggestions
                suggestions = [q for q in quotes_match if not q.startswith("Hello") and len(q) < 100]
                if suggestions:
                    return max(suggestions, key=len).strip()
            
            # Look for explicit suggestions like "Maybe X" or "Perhaps Y"
            suggestion_patterns = [
                r'(?:maybe|perhaps|could be|might be)\s+["\']?([^"\'\n.!?]+)["\']?',
                r'(?:i suggest|i recommend|how about)\s+["\']?([^"\'\n.!?]+)["\']?',
                r'(?:title could be|title should be|title would be)\s+["\']?([^"\'\n.!?]+)["\']?'
            ]
            
            for pattern in suggestion_patterns:
                match = re.search(pattern, raw_response, re.IGNORECASE)
                if match:
                    suggestion = match.group(1).strip()
                    if len(suggestion) > 3 and len(suggestion) < 50:  # Reasonable title length
                        return suggestion
            
            # Look for the last coherent phrase before the text cuts off
            # Split by sentences and find the last complete thought
            sentences = re.split(r'[.!?]+', raw_response)
            for sentence in reversed(sentences):
                sentence = sentence.strip()
                # Look for title-like phrases
                if (len(sentence) > 5 and len(sentence) < 50 and 
                    not sentence.lower().startswith(('okay', 'let me', 'first', 'i need', 'i should', 'the user', 'maybe', 'wait'))):
                    # Clean up any remaining artifacts
                    clean_sentence = re.sub(r'^[^a-zA-Z]*', '', sentence)
                    if clean_sentence:
                        return clean_sentence.strip()
            
            # Final fallback - return raw response
            return raw_response.strip()
            
        except requests.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Ollama response: {e}")
            return None


class ObsidianWriter:
    """Writer for creating and managing Obsidian markdown notes with AI enhancement."""
    
    def __init__(self, vault_path: str, use_ai_enhancement: bool = True):
        self.vault_path = Path(vault_path)
        self.use_ai_enhancement = use_ai_enhancement
        self.ollama_client = None
        
        # Validate vault path
        if not vault_path or not vault_path.strip():
            raise ValueError("Vault path cannot be empty")
        
        # Ensure vault directory exists with proper error handling
        try:
            self.vault_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized Obsidian vault at: {self.vault_path}")
        except PermissionError as e:
            logger.error(f"Permission denied creating vault directory {self.vault_path}: {e}")
            raise
        except OSError as e:
            logger.error(f"OS error creating vault directory {self.vault_path}: {e}")
            raise
        
        # Create subdirectories for organization with error handling
        try:
            self.audio_notes_dir = self.vault_path / "Audio Notes"
            self.audio_notes_dir.mkdir(exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.error(f"Error creating Audio Notes directory: {e}")
            raise
        
        # Initialize Ollama client with error handling and resource management
        if use_ai_enhancement:
            try:
                self.ollama_client = OllamaClient()
                if not self.ollama_client.is_available():
                    logger.warning("Ollama service not available. AI enhancement will be disabled.")
                    self.use_ai_enhancement = False
                    self.ollama_client = None
                else:
                    logger.info("Ollama client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}. AI enhancement disabled.")
                self.use_ai_enhancement = False
                self.ollama_client = None
        
    def enhance_title(self, transcription: str, original_filename: str = "") -> str:
        """Generate an intelligent title for the note using Qwen 3:0.6B."""
        if not self.use_ai_enhancement or not self.ollama_client:
            # Fallback: use filename or timestamp
            if original_filename:
                return f"Audio Note - {Path(original_filename).stem}"
            return f"Audio Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Extract first few sentences for context
        preview = transcription[:300] + "..." if len(transcription) > 300 else transcription
        
        prompt = f"""Generate a concise, descriptive title (maximum 8 words) for this audio transcription:

"{preview}"

Title:"""
        
        enhanced_title = self.ollama_client.generate_text(prompt, max_tokens=50, temperature=0.3)
        
        if enhanced_title:
            # Clean and validate the title
            enhanced_title = re.sub(r'[<>:"/\\|?*]', '', enhanced_title)  # Remove invalid filename chars
            enhanced_title = enhanced_title.strip().strip('"').strip("'")
            if len(enhanced_title) > 100:  # Reasonable length limit
                enhanced_title = enhanced_title[:100]
            return enhanced_title
        
        # Fallback if AI enhancement fails
        return f"Audio Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def generate_tags(self, transcription: str) -> List[str]:
        """Generate relevant tags for the note using Qwen 3:0.6B."""
        if not self.use_ai_enhancement or not self.ollama_client:
            return ["audio-note", "transcription"]
        
        # Use first 500 characters for tag generation
        content_preview = transcription[:500] + "..." if len(transcription) > 500 else transcription
        
        prompt = f"""Generate 3-5 relevant tags for organizing this transcription. Use lowercase and hyphens instead of spaces:

"{content_preview}"

Tags:"""
        
        tags_response = self.ollama_client.generate_text(prompt, max_tokens=100, temperature=0.4)
        
        if tags_response:
            # Parse and clean tags
            tags = [tag.strip().lower() for tag in tags_response.split(',')]
            tags = [re.sub(r'[^a-z0-9\-]', '', tag) for tag in tags if tag.strip()]
            tags = [tag for tag in tags if len(tag) > 1 and len(tag) < 30]  # Reasonable length
            return tags[:5] if tags else ["audio-note", "transcription"]
        
        return ["audio-note", "transcription"]
    
    def enhance_content(self, transcription: str) -> str:
        """Enhance the transcription content with better formatting and structure."""
        if not self.use_ai_enhancement or not self.ollama_client:
            return transcription
        
        prompt = f"""Improve this transcription with proper formatting, fix grammar, and add paragraph breaks:

"{transcription}"

Improved version:"""
        
        enhanced_content = self.ollama_client.generate_text(
            prompt, 
            max_tokens=len(transcription.split()) * 2,  # Allow for expansion
            temperature=0.3
        )
        
        return enhanced_content if enhanced_content else transcription
    
    def generate_summary(self, transcription: str) -> str:
        """Generate a brief summary of the transcription."""
        if not self.use_ai_enhancement or not self.ollama_client:
            # Simple fallback summary
            words = transcription.split()
            if len(words) <= 50:
                return "Short audio note."
            return f"Audio transcription with approximately {len(words)} words."
        
        prompt = f"""Summarize this transcription in 1-2 sentences:

"{transcription}"

Summary:"""
        
        summary = self.ollama_client.generate_text(prompt, max_tokens=100, temperature=0.4)
        return summary if summary else "Audio transcription note."
    
    def create_note(self, 
                   transcription: str, 
                   original_filename: str = "",
                   metadata: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Create a new Obsidian note with the transcription.
        
        Returns:
            Tuple of (note_path, note_title)
        """
        # Input validation
        if not transcription or not transcription.strip():
            raise ValueError("Transcription content cannot be empty")
        
        try:
            # Generate enhanced content with fallback handling
            logger.debug("Generating enhanced content...")
            title = self.enhance_title(transcription, original_filename)
            tags = self.generate_tags(transcription)
            enhanced_content = self.enhance_content(transcription)
            summary = self.generate_summary(transcription)
            
            # Create safe filename with comprehensive sanitization
            safe_filename = self._sanitize_filename(title)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{safe_filename}.md"
            
            # Ensure filename isn't too long for filesystem
            if len(filename) > 255:
                filename = f"{timestamp}_{safe_filename[:200]}.md"
            
            note_path = self.audio_notes_dir / filename
            
            # Check if file already exists and handle conflicts
            counter = 1
            original_note_path = note_path
            while note_path.exists():
                stem = original_note_path.stem
                note_path = self.audio_notes_dir / f"{stem}_{counter}.md"
                counter += 1
                if counter > 100:  # Prevent infinite loop
                    raise RuntimeError("Unable to create unique filename after 100 attempts")
            
            # Prepare metadata
            note_metadata = {
                "created": datetime.now().isoformat(),
                "source": "audio-transcription",
                "original_file": original_filename,
                "ai_enhanced": self.use_ai_enhancement,
                "tags": tags
            }
            
            if metadata:
                note_metadata.update(metadata)
            
            # Create note content
            content_lines = [
                f"# {title}",
                "",
                "## Metadata",
                f"- **Created**: {note_metadata['created']}",
                f"- **Source**: {note_metadata['source']}",
                f"- **Original File**: {note_metadata.get('original_file', 'N/A')}",
                f"- **AI Enhanced**: {note_metadata['ai_enhanced']}",
                f"- **Tags**: {', '.join([f'#{tag}' for tag in tags])}",
                "",
                "## Summary",
                summary,
                "",
                "## Transcription",
                enhanced_content,
                "",
                "---",
                f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*"
            ]
            
            # Write the note with proper error handling
            try:
                with open(note_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(content_lines))
                logger.info(f"Created Obsidian note: {note_path}")
            except PermissionError as e:
                logger.error(f"Permission denied writing to {note_path}: {e}")
                raise
            except OSError as e:
                logger.error(f"OS error writing to {note_path}: {e}")
                raise
            
            return str(note_path), title
            
        except ValueError as e:
            logger.error(f"Validation error creating note: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating Obsidian note: {e}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for all operating systems."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        
        # Replace spaces and multiple underscores
        filename = re.sub(r'\s+', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        
        # Remove leading/trailing underscores and dots
        filename = filename.strip('_.')
        
        # Ensure filename isn't empty
        if not filename:
            filename = "untitled"
        
        # Ensure filename doesn't start with a dot (hidden file)
        if filename.startswith('.'):
            filename = 'file' + filename
        
        return filename
    
    def append_to_note(self, note_path: str, content: str, section_title: str = "Additional Content"):
        """Append content to an existing note."""
        try:
            note_path = Path(note_path)
            if not note_path.exists():
                raise FileNotFoundError(f"Note not found: {note_path}")
            
            # Read existing content
            with open(note_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Append new content
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_section = f"\n\n## {section_title} ({timestamp})\n{content}"
            
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(existing_content + new_section)
            
            logger.info(f"Appended content to note: {note_path}")
            
        except Exception as e:
            logger.error(f"Error appending to note: {e}")
            raise
    
    def list_notes(self) -> List[Dict[str, str]]:
        """List all audio notes in the vault."""
        notes = []
        try:
            for note_file in self.audio_notes_dir.glob("*.md"):
                notes.append({
                    "path": str(note_file),
                    "name": note_file.stem,
                    "modified": datetime.fromtimestamp(note_file.stat().st_mtime).isoformat()
                })
        except Exception as e:
            logger.error(f"Error listing notes: {e}")
        
        return sorted(notes, key=lambda x: x["modified"], reverse=True)


def create_test_vault(test_path: str = "./test_obsidian_vault") -> str:
    """Create a test Obsidian vault for testing purposes."""
    vault_path = Path(test_path)
    vault_path.mkdir(parents=True, exist_ok=True)
    
    # Create .obsidian directory to make it a proper vault
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir(exist_ok=True)
    
    return str(vault_path)


if __name__ == "__main__":
    # Test the ObsidianWriter
    logging.basicConfig(level=logging.INFO)
    
    # Create test vault
    test_vault = create_test_vault()
    
    # Test transcription
    test_transcription = """
    Hello, this is a test transcription of an audio file. 
    I'm testing the Obsidian writer integration with Qwen 3.0.6B model.
    The system should be able to enhance this content, generate appropriate tags,
    and create a well-structured markdown note for Obsidian.
    """
    
    # Create writer and test
    writer = ObsidianWriter(test_vault, use_ai_enhancement=True)
    
    try:
        note_path, title = writer.create_note(
            test_transcription, 
            "test_audio.wav",
            {"language": "en", "duration": "30s"}
        )
        print(f"Created test note: {title}")
        print(f"Path: {note_path}")
        
        # List notes
        notes = writer.list_notes()
        print(f"Total notes in vault: {len(notes)}")
        
    except Exception as e:
        print(f"Test failed: {e}")