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
import ollama

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API to use Qwen 3:0.6B model."""
    
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model
        
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            # Try to list models to check if Ollama is running
            models = ollama.list()
            return True
        except Exception:
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate text using Qwen 3:0.6B model via Ollama."""
        if not self.is_available():
            logger.warning("Ollama service is not available")
            return None
            
        try:
            # Use more direct prompting to reduce thinking patterns
            # Add clear instruction to be concise
            direct_prompt = f"Answer concisely and directly. {prompt}"
            
            response = ollama.generate(
                model=self.model,
                prompt=direct_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["</think>", "\n\n"]  # Stop at end of thinking or double newline
                }
            )
            
            raw_response = response.get("response", "").strip()
            
            if not raw_response:
                return None
                
            # Handle various response patterns
            cleaned_response = self._extract_useful_content(raw_response)
            
            if cleaned_response and len(cleaned_response.strip()) > 2:
                return cleaned_response.strip()
            
            # If extraction failed, try fallback approach
            return self._fallback_extraction(raw_response)
            
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None
    
    def _extract_useful_content(self, raw_response: str) -> Optional[str]:
        """Extract useful content from Qwen's response, handling various patterns."""
        
        # Pattern 1: Complete <think>...</think> blocks
        if '<think>' in raw_response and '</think>' in raw_response:
            # Extract content after </think>
            after_think = raw_response.split('</think>', 1)[-1].strip()
            if after_think and len(after_think) > 3:
                return self._clean_response_line(after_think)
        
        # Pattern 2: Incomplete <think> blocks (truncated responses)
        if raw_response.startswith('<think>'):
            # Look for quoted content within the thinking
            quotes = re.findall(r'"([^"]{3,100})"', raw_response)
            if quotes:
                # Return the longest reasonable quote
                best_quote = max(quotes, key=len) if quotes else None
                if best_quote and len(best_quote) > 3:
                    return best_quote.strip()
            
            # Look for title-like content after colons
            title_patterns = [
                r'title[:\s]+([^\n.]{5,50})',
                r'answer[:\s]+([^\n.]{3,50})',
                r'result[:\s]+([^\n.]{3,50})',
                r'summary[:\s]+([^\n.]{10,100})',
                r'tags?[:\s]+([^\n.]{5,80})'
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, raw_response, re.IGNORECASE)
                if match:
                    result = match.group(1).strip()
                    if result and not result.lower().startswith(('the ', 'this ', 'that ', 'i ', 'let me')):
                        return result
            
            # Look for any direct statements (not reasoning)
            lines = raw_response.split('\n')
            for line in lines:
                line = line.strip()
                # Skip reasoning phrases and look for direct answers
                if (line and len(line) > 5 and 
                    not any(phrase in line.lower() for phrase in [
                        'let me', 'i need', 'the user', 'okay', 'hmm', 'so i',
                        'i should', 'i think', 'maybe', 'perhaps', 'considering',
                        'first', 'second', 'then', 'next', 'also', 'however'
                    ])):
                    
                    # Look for lines that seem like answers
                    if ('"' in line or 
                        line.endswith('.') or line.endswith('!') or
                        any(word in line.lower() for word in ['ai', 'ml', 'machine', 'learning', 'discussion', 'note'])):
                        
                        # Extract quoted content if present
                        quote_match = re.search(r'"([^"]+)"', line)
                        if quote_match:
                            return quote_match.group(1).strip()
                        
                        # Or return the line if it looks like a direct answer
                        cleaned = self._clean_response_line(line)
                        if cleaned and len(cleaned) > 3:
                            return cleaned
        
        # Pattern 3: Direct responses without thinking tags
        else:
            return self._clean_response_line(raw_response)
        
        return None
    
    def _clean_response_line(self, line: str) -> Optional[str]:
        """Clean a response line to extract useful content."""
        if not line:
            return None
            
        # Remove common prefixes
        line = re.sub(r'^(answer[:\s]*|result[:\s]*|title[:\s]*)', '', line, flags=re.IGNORECASE)
        
        # Remove leading/trailing quotes and whitespace
        line = line.strip().strip('"').strip("'").strip()
        
        # Remove invalid filename characters for titles
        line = re.sub(r'[<>:"/\\|?*]', '', line)
        
        # Take first sentence if multiple
        if '. ' in line:
            line = line.split('. ')[0] + '.'
        
        return line if len(line) > 3 else None
    
    def _fallback_extraction(self, raw_response: str) -> Optional[str]:
        """Fallback extraction method for difficult responses."""
        
        # Try to find any meaningful content
        lines = raw_response.split('\n')
        
        # Look for the shortest meaningful line (likely to be a title/answer)
        candidates = []
        for line in lines:
            line = line.strip()
            if (line and 3 <= len(line) <= 100 and 
                not line.startswith(('<', '>', '#', '```', '---')) and
                not any(phrase in line.lower() for phrase in [
                    'think', 'okay', 'let me', 'the user', 'i need', 'hmm'
                ])):
                candidates.append(line)
        
        if candidates:
            # Return the shortest candidate (likely to be most direct)
            best = min(candidates, key=len)
            return self._clean_response_line(best)
        
        # Absolute fallback - return first 50 chars
        cleaned = re.sub(r'^[<>]*\s*', '', raw_response)
        return cleaned[:50].strip() if cleaned else None


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
        preview = transcription[:200] + "..." if len(transcription) > 200 else transcription
        
        # More direct prompt
        prompt = f"""Content: {preview}

Task: Create a title (max 8 words)
Title:"""
        
        enhanced_title = self.ollama_client.generate_text(prompt, max_tokens=20, temperature=0.2)
        
        if enhanced_title and len(enhanced_title.strip()) > 2:
            # Clean and validate the title
            enhanced_title = re.sub(r'[<>:"/\\|?*]', '', enhanced_title)  # Remove invalid filename chars
            enhanced_title = enhanced_title.strip().strip('"').strip("'")
            
            # Remove common prefixes
            enhanced_title = re.sub(r'^(title[:\s]*|answer[:\s]*)', '', enhanced_title, flags=re.IGNORECASE)
            
            if len(enhanced_title) > 100:  # Reasonable length limit
                enhanced_title = enhanced_title[:100]
            
            # Ensure we have a meaningful title
            if len(enhanced_title.strip()) > 2:
                return enhanced_title.strip()
        
        # Fallback if AI enhancement fails
        return f"Audio Note - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def generate_tags(self, transcription: str) -> List[str]:
        """Generate relevant tags for the note using Qwen 3:0.6B."""
        if not self.use_ai_enhancement or not self.ollama_client:
            return ["audio-note", "transcription"]
        
        # Use first 300 characters for tag generation
        content_preview = transcription[:300] + "..." if len(transcription) > 300 else transcription
        
        # More direct prompt
        prompt = f"""Content: {content_preview}

Task: List 3-5 tags (lowercase, hyphens only)
Tags:"""
        
        tags_response = self.ollama_client.generate_text(prompt, max_tokens=30, temperature=0.3)
        
        if tags_response:
            # Parse and clean tags - handle various formats
            tags_text = tags_response.lower().strip()
            
            # Remove common prefixes
            tags_text = re.sub(r'^(tags[:\s]*|answer[:\s]*)', '', tags_text, flags=re.IGNORECASE)
            
            # Split by various delimiters
            tags = re.split(r'[,\s\n]+', tags_text)
            tags = [re.sub(r'[^a-z0-9\-]', '', tag.strip()) for tag in tags if tag.strip()]
            tags = [tag for tag in tags if len(tag) > 1 and len(tag) < 30]  # Reasonable length
            
            # Ensure we have at least some basic tags
            if tags and len(tags) <= 5:
                return tags + ["audio-note"] if "audio-note" not in tags else tags
        
        return ["audio-note", "transcription"]
    
    def enhance_content(self, transcription: str) -> str:
        """Enhance the transcription content with better formatting and structure."""
        if not self.use_ai_enhancement or not self.ollama_client:
            return transcription
        
        # For content enhancement, use a smaller chunk to avoid truncation
        content_chunk = transcription[:800] if len(transcription) > 800 else transcription
        
        prompt = f"""Fix grammar and format this text with proper paragraphs:

{content_chunk}

Improved text:"""
        
        enhanced_content = self.ollama_client.generate_text(
            prompt, 
            max_tokens=min(len(content_chunk.split()) * 2, 1000),  # Allow for expansion but limit
            temperature=0.2
        )
        
        if enhanced_content and len(enhanced_content.strip()) > len(content_chunk) * 0.3:
            # Only use enhanced content if it seems reasonable (not truncated too much)
            return enhanced_content
        
        return transcription
    
    def generate_summary(self, transcription: str) -> str:
        """Generate a brief summary of the transcription."""
        if not self.use_ai_enhancement or not self.ollama_client:
            # Simple fallback summary
            words = transcription.split()
            if len(words) <= 50:
                return "Short audio note."
            return f"Audio transcription with approximately {len(words)} words."
        
        # Use first 500 chars for summary
        content_preview = transcription[:500] + "..." if len(transcription) > 500 else transcription
        
        prompt = f"""Content: {content_preview}

Task: Write a 1-sentence summary
Summary:"""
        
        summary = self.ollama_client.generate_text(prompt, max_tokens=50, temperature=0.3)
        
        if summary and len(summary.strip()) > 10:
            # Clean the summary
            summary = re.sub(r'^(summary[:\s]*|answer[:\s]*)', '', summary, flags=re.IGNORECASE)
            summary = summary.strip().strip('"').strip("'")
            
            if len(summary.strip()) > 10:
                return summary.strip()
        
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