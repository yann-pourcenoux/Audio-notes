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
from typing import Dict, List, Optional, Tuple, Union
import ollama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class TitleResponse(BaseModel):
    """Structured response for title generation."""
    title: str = Field(..., description="A concise title (max 8 words)", max_length=100)
    reasoning: Optional[str] = Field(None, description="Brief reasoning for the title choice")


class TagsResponse(BaseModel):
    """Structured response for tag generation."""
    tags: List[str] = Field(..., description="List of 3-5 relevant tags", min_items=1, max_items=5)
    reasoning: Optional[str] = Field(None, description="Brief reasoning for tag choices")


class SummaryResponse(BaseModel):
    """Structured response for summary generation."""
    summary: str = Field(..., description="A concise 1-2 sentence summary", max_length=200)
    key_topics: Optional[List[str]] = Field(None, description="Key topics discussed")


class ContentEnhancement(BaseModel):
    """Structured response for content enhancement."""
    enhanced_content: str = Field(..., description="Enhanced and formatted content")
    improvements_made: Optional[List[str]] = Field(None, description="List of improvements applied")


class OllamaClient:
    """Enhanced client for interacting with Ollama API using latest features."""
    
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model
        self._connection_pool = {}  # Simple connection pooling
        self._last_availability_check = None
        self._availability_cache_timeout = 30  # seconds
        
    def is_available(self) -> bool:
        """Check if Ollama service is available with caching."""
        current_time = datetime.now().timestamp()
        
        # Use cached result if available and recent
        if (self._last_availability_check and 
            current_time - self._last_availability_check < self._availability_cache_timeout):
            return hasattr(self, '_cached_availability') and self._cached_availability
        
        try:
            # Try to list models to check if Ollama is running
            models = ollama.list()
            self._cached_availability = True
            self._last_availability_check = current_time
            return True
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            self._cached_availability = False
            self._last_availability_check = current_time
            return False
    
    def _chat_with_thinking(self, messages: List[Dict], 
                           response_format: Optional[BaseModel] = None,
                           max_tokens: int = 500, 
                           temperature: float = 0.7) -> Optional[Union[str, BaseModel]]:
        """Enhanced chat method using thinking parameter and structured outputs."""
        if not self.is_available():
            logger.warning("Ollama service is not available")
            return None
            
        try:
            # Add thinking control message for better reasoning
            enhanced_messages = [
                {'role': 'control', 'content': 'thinking'},
                *messages
            ]
            
            # Prepare chat options
            chat_options = {
                'model': self.model,
                'messages': enhanced_messages,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            }
            
            # Add structured output format if provided
            if response_format:
                chat_options['format'] = response_format.model_json_schema()
            
            # Make the chat request
            response = ollama.chat(**chat_options)
            
            if not response or not response.get('message', {}).get('content'):
                logger.warning("Empty response from Ollama")
                return None
            
            content = response['message']['content'].strip()
            
            # Parse structured response if format was specified
            if response_format:
                try:
                    return response_format.model_validate_json(content)
                except Exception as e:
                    logger.warning(f"Failed to parse structured response: {e}")
                    # Fallback to extracting useful content from raw response
                    return self._extract_from_thinking_response(content)
            
            # For unstructured responses, extract useful content
            return self._extract_from_thinking_response(content)
            
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None
    
    def _extract_from_thinking_response(self, content: str) -> Optional[str]:
        """Extract useful content from thinking-enabled responses."""
        if not content:
            return None
        
        # Look for structured thinking patterns
        patterns = [
            r"Here is my response:\s*(.+)",
            r"My answer:\s*(.+)",
            r"Response:\s*(.+)",
            r"Final answer:\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 3:
                    return extracted
        
        # Fallback: look for content after thinking section
        thinking_end_patterns = [
            r"Here is my thought process:.*?(?=Here is my response:|My answer:|Response:|$)",
            r"Let me think.*?(?=Here is my response:|My answer:|Response:|$)",
        ]
        
        for pattern in thinking_end_patterns:
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up and return
        content = content.strip()
        if len(content) > 3:
            return content
        
        return None
    
    def generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate text using enhanced chat API with thinking."""
        messages = [{'role': 'user', 'content': prompt}]
        return self._chat_with_thinking(messages, None, max_tokens, temperature)
    
    def generate_structured_response(self, prompt: str, 
                                   response_format: BaseModel,
                                   max_tokens: int = 500, 
                                   temperature: float = 0.7) -> Optional[BaseModel]:
        """Generate structured response using Pydantic models."""
        messages = [{'role': 'user', 'content': prompt}]
        return self._chat_with_thinking(messages, response_format, max_tokens, temperature)


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
        
        # Initialize Ollama client with enhanced error handling and resource management
        if use_ai_enhancement:
            try:
                self.ollama_client = OllamaClient()
                if not self.ollama_client.is_available():
                    logger.warning("Ollama service not available. AI enhancement will be disabled.")
                    self.use_ai_enhancement = False
                    self.ollama_client = None
                else:
                    logger.info("Enhanced Ollama client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}. AI enhancement disabled.")
                self.use_ai_enhancement = False
                self.ollama_client = None
        
    def enhance_title(self, transcription: str, original_filename: str = "") -> str:
        """Generate an intelligent title using structured outputs."""
        if not self.use_ai_enhancement or not self.ollama_client:
            # Fallback: use filename or timestamp
            if original_filename:
                return f"Audio Note - {Path(original_filename).stem}"
            return f"Audio Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Extract first few sentences for context
        preview = transcription[:200] + "..." if len(transcription) > 200 else transcription
        
        # Enhanced prompt for structured output
        prompt = f"""Analyze this audio transcription and create a concise, descriptive title.

Content: {preview}

Requirements:
- Maximum 8 words
- Descriptive and specific to the content
- Suitable as a filename (no special characters)
- Focus on the main topic or purpose"""
        
        try:
            # Try structured response first
            response = self.ollama_client.generate_structured_response(
                prompt, TitleResponse, max_tokens=100, temperature=0.2
            )
            
            if response and isinstance(response, TitleResponse):
                title = response.title.strip()
                # Clean and validate the title
                title = re.sub(r'[<>:"/\\|?*]', '', title)
                title = title.strip().strip('"').strip("'")
                
                if len(title) > 3 and len(title) <= 100:
                    logger.debug(f"Generated structured title: {title}")
                    return title
            
            # Fallback to unstructured generation
            enhanced_title = self.ollama_client.generate_text(
                f"Create a short title (max 8 words) for: {preview}", 
                max_tokens=20, temperature=0.2
            )
            
            if enhanced_title and len(enhanced_title.strip()) > 2:
                # Clean and validate the title
                enhanced_title = re.sub(r'[<>:"/\\|?*]', '', enhanced_title)
                enhanced_title = enhanced_title.strip().strip('"').strip("'")
                enhanced_title = re.sub(r'^(title[:\s]*|answer[:\s]*)', '', enhanced_title, flags=re.IGNORECASE)
                
                if len(enhanced_title.strip()) > 2:
                    return enhanced_title.strip()
                    
        except Exception as e:
            logger.debug(f"Title generation failed: {e}")
        
        # Fallback if AI enhancement fails
        return f"Audio Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def generate_tags(self, transcription: str) -> List[str]:
        """Generate relevant tags using structured outputs."""
        if not self.use_ai_enhancement or not self.ollama_client:
            return ["audio-note", "transcription"]
        
        # Use first 300 characters for tag generation
        content_preview = transcription[:300] + "..." if len(transcription) > 300 else transcription
        
        prompt = f"""Analyze this audio transcription and generate relevant tags.

Content: {content_preview}

Requirements:
- 3-5 tags maximum
- Lowercase with hyphens only
- Relevant to the content topics
- Useful for organization and search"""
        
        try:
            # Try structured response first
            response = self.ollama_client.generate_structured_response(
                prompt, TagsResponse, max_tokens=100, temperature=0.3
            )
            
            if response and isinstance(response, TagsResponse):
                tags = []
                for tag in response.tags:
                    # Clean and validate tags
                    clean_tag = re.sub(r'[^a-z0-9\-]', '', tag.lower().strip())
                    if 1 < len(clean_tag) < 30:
                        tags.append(clean_tag)
                
                if tags:
                    # Ensure we have basic tags
                    if "audio-note" not in tags:
                        tags.append("audio-note")
                    logger.debug(f"Generated structured tags: {tags}")
                    return tags[:5]  # Limit to 5 tags
            
            # Fallback to unstructured generation
            tags_response = self.ollama_client.generate_text(
                f"List 3-5 relevant tags (lowercase, hyphens only) for: {content_preview}",
                max_tokens=30, temperature=0.3
            )
            
            if tags_response:
                # Parse and clean tags
                tags_text = tags_response.lower().strip()
                tags_text = re.sub(r'^(tags[:\s]*|answer[:\s]*)', '', tags_text, flags=re.IGNORECASE)
                tags = re.split(r'[,\s\n]+', tags_text)
                tags = [re.sub(r'[^a-z0-9\-]', '', tag.strip()) for tag in tags if tag.strip()]
                tags = [tag for tag in tags if 1 < len(tag) < 30]
                
                if tags and len(tags) <= 5:
                    return tags + ["audio-note"] if "audio-note" not in tags else tags
                    
        except Exception as e:
            logger.debug(f"Tag generation failed: {e}")
        
        return ["audio-note", "transcription"]
    
    def enhance_content(self, transcription: str) -> str:
        """Enhance content using structured outputs."""
        if not self.use_ai_enhancement or not self.ollama_client:
            return transcription
        
        # For content enhancement, use a smaller chunk to avoid truncation
        content_chunk = transcription[:800] if len(transcription) > 800 else transcription
        
        prompt = f"""Improve this audio transcription by fixing grammar, formatting, and structure.

Original text: {content_chunk}

Requirements:
- Fix grammar and spelling errors
- Add proper paragraph breaks
- Improve readability
- Maintain original meaning and content
- Keep the same length approximately"""
        
        try:
            # Try structured response first
            response = self.ollama_client.generate_structured_response(
                prompt, ContentEnhancement, 
                max_tokens=min(len(content_chunk.split()) * 2, 1000), 
                temperature=0.2
            )
            
            if response and isinstance(response, ContentEnhancement):
                enhanced = response.enhanced_content.strip()
                if len(enhanced) > len(content_chunk) * 0.3:  # Reasonable length check
                    logger.debug(f"Enhanced content using structured output")
                    return enhanced
            
            # Fallback to unstructured generation
            enhanced_content = self.ollama_client.generate_text(
                f"Fix grammar and format this text with proper paragraphs:\n\n{content_chunk}\n\nImproved text:",
                max_tokens=min(len(content_chunk.split()) * 2, 1000),
                temperature=0.2
            )
            
            if enhanced_content and len(enhanced_content.strip()) > len(content_chunk) * 0.3:
                return enhanced_content
                
        except Exception as e:
            logger.debug(f"Content enhancement failed: {e}")
        
        return transcription
    
    def generate_summary(self, transcription: str) -> str:
        """Generate a summary using structured outputs."""
        if not self.use_ai_enhancement or not self.ollama_client:
            # Simple fallback summary
            words = transcription.split()
            if len(words) <= 50:
                return "Short audio note."
            return f"Audio transcription with approximately {len(words)} words."
        
        # Use first 500 chars for summary
        content_preview = transcription[:500] + "..." if len(transcription) > 500 else transcription
        
        prompt = f"""Create a concise summary of this audio transcription.

Content: {content_preview}

Requirements:
- 1-2 sentences maximum
- Capture the main topic and key points
- Be specific and informative"""
        
        try:
            # Try structured response first
            response = self.ollama_client.generate_structured_response(
                prompt, SummaryResponse, max_tokens=100, temperature=0.3
            )
            
            if response and isinstance(response, SummaryResponse):
                summary = response.summary.strip()
                if len(summary) > 10:
                    logger.debug(f"Generated structured summary")
                    return summary
            
            # Fallback to unstructured generation
            summary = self.ollama_client.generate_text(
                f"Write a 1-sentence summary of: {content_preview}",
                max_tokens=50, temperature=0.3
            )
            
            if summary and len(summary.strip()) > 10:
                # Clean the summary
                summary = re.sub(r'^(summary[:\s]*|answer[:\s]*)', '', summary, flags=re.IGNORECASE)
                summary = summary.strip().strip('"').strip("'")
                
                if len(summary.strip()) > 10:
                    return summary.strip()
                    
        except Exception as e:
            logger.debug(f"Summary generation failed: {e}")
        
        # Fallback
        words = transcription.split()
        return f"Audio discussion covering {len(words)} words on various topics."
    
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