#!/usr/bin/env python3
"""
Audio Notes CLI - Command Line Interface for Audio Transcription and Note Generation

This CLI provides a comprehensive interface for processing audio files using:
- OpenAI Whisper Large-v3 for transcription
- Qwen 3.0.6B via Ollama for intelligent content enhancement
- Obsidian vault integration for organized note storage
"""

import click
import os
import sys
import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Import our modules
from .audio_processor import AudioProcessor
from .whisper_transcriber import WhisperTranscriber
from .obsidian_writer import ObsidianWriter, OllamaClient
from .config import get_config_manager, get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Whisper supported languages (key subset)
WHISPER_LANGUAGES = {
    'af': 'afrikaans', 'ar': 'arabic', 'bg': 'bulgarian', 'bn': 'bengali', 'ca': 'catalan',
    'cs': 'czech', 'da': 'danish', 'de': 'german', 'el': 'greek', 'en': 'english',
    'es': 'spanish', 'et': 'estonian', 'fi': 'finnish', 'fr': 'french', 'he': 'hebrew',
    'hi': 'hindi', 'hr': 'croatian', 'hu': 'hungarian', 'id': 'indonesian', 'it': 'italian',
    'ja': 'japanese', 'ko': 'korean', 'lt': 'lithuanian', 'lv': 'latvian', 'nl': 'dutch',
    'no': 'norwegian', 'pl': 'polish', 'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian',
    'sk': 'slovak', 'sl': 'slovenian', 'sv': 'swedish', 'th': 'thai', 'tr': 'turkish',
    'uk': 'ukrainian', 'vi': 'vietnamese', 'zh': 'chinese'
}

# Output formats
OUTPUT_FORMATS = ['text', 'json', 'srt', 'vtt', 'tsv']

# Timestamp levels
TIMESTAMP_LEVELS = ['none', 'sentence', 'word']

# Model precision options
PRECISION_OPTIONS = ['float16', 'float32']

# Processing methods for long-form audio
PROCESSING_METHODS = ['sequential', 'chunked']

class AudioNotesCLI:
    """Main CLI class with intelligent features powered by Qwen 3.0.6B"""
    
    def __init__(self):
        self.ollama_client = None
        self.setup_ollama_client()
    
    def setup_ollama_client(self):
        """Initialize Ollama client for intelligent CLI features"""
        try:
            self.ollama_client = OllamaClient()
            if not self.ollama_client.is_available():
                logger.warning("Ollama not available - intelligent features disabled")
                self.ollama_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            self.ollama_client = None
    
    def get_intelligent_suggestion(self, context: str, query: str) -> Optional[str]:
        """Get intelligent suggestions from Qwen 3.0.6B"""
        if not self.ollama_client:
            return None
        
        try:
            prompt = f"""Context: {context}
Query: {query}

Provide a brief, helpful suggestion (1-2 sentences max):"""
            
            response = self.ollama_client.generate_text(prompt)
            return response.strip() if response else None
        except Exception as e:
            logger.debug(f"Failed to get intelligent suggestion: {e}")
            return None
    
    def suggest_language(self, audio_path: str) -> Optional[str]:
        """Suggest language based on file path and intelligent analysis"""
        if not self.ollama_client:
            return None
        
        try:
            filename = os.path.basename(audio_path)
            context = f"Audio file: {filename}"
            query = "What language might this audio file contain based on the filename?"
            
            suggestion = self.get_intelligent_suggestion(context, query)
            if suggestion:
                # Try to extract language code from suggestion
                for code, lang in WHISPER_LANGUAGES.items():
                    if lang.lower() in suggestion.lower():
                        return code
        except Exception as e:
            logger.debug(f"Failed to suggest language: {e}")
        
        return None

# Initialize CLI instance
cli_instance = AudioNotesCLI()

@click.group()
@click.version_option(version='1.0.0', prog_name='audio-notes')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Audio Notes - AI-Powered Audio Transcription and Note Generation
    
    Process audio files using Whisper Large-v3 for transcription and Qwen 3.0.6B
    for intelligent content enhancement, with automatic Obsidian vault integration.
    """
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj['verbose'] = True

@cli.command()
@click.argument('audio_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--language', '-l', 
              type=click.Choice(list(WHISPER_LANGUAGES.keys()) + ['auto']),
              help='Language of the audio (auto for automatic detection)')
@click.option('--task', '-t',
              type=click.Choice(['transcribe', 'translate']),
              help='Task to perform: transcribe or translate to English')
@click.option('--timestamps', '--ts',
              type=click.Choice(TIMESTAMP_LEVELS),
              help='Timestamp level: none, sentence, or word')
@click.option('--precision', '-p',
              type=click.Choice(PRECISION_OPTIONS),
              help='Model precision: float16 (GPU) or float32 (CPU)')
@click.option('--temperature', '--temp',
              type=float,
              help='Temperature for decoding (0.0-1.0)')
@click.option('--beam-size', '--beams',
              type=int,
              help='Number of beams for beam search decoding')
@click.option('--processing-method', '--method',
              type=click.Choice(PROCESSING_METHODS),
              help='Processing method for long-form audio')
@click.option('--output-format', '--format',
              type=click.Choice(OUTPUT_FORMATS),
              help='Output format')
@click.option('--output-dir', '--out',
              type=click.Path(),
              help='Output directory (default: current directory)')
@click.option('--vault-path', '--vault',
              type=click.Path(),
              help='Obsidian vault path (default: ./obsidian_vault)')
@click.option('--no-vault', is_flag=True,
              help='Skip Obsidian vault integration')
@click.option('--batch-size', '--batch',
              type=int,
              help='Batch size for processing multiple files')
@click.option('--max-length', '--max',
              type=int,
              help='Maximum length in seconds for chunked processing')
@click.option('--enhance-notes', '--enhance', is_flag=True,
              help='Use AI to enhance notes with titles and summaries')
@click.option('--dry-run', is_flag=True,
              help='Show what would be processed without actually processing')
@click.pass_context
def process(ctx, audio_files, language, task, timestamps, precision, temperature, 
           beam_size, processing_method, output_format, output_dir, vault_path, 
           no_vault, batch_size, max_length, enhance_notes, dry_run):
    """Process audio files for transcription or translation.
    
    AUDIO_FILES: One or more audio file paths to process
    
    Examples:
        audio-notes process recording.wav
        audio-notes process *.mp3 --language en --task transcribe
        audio-notes process interview.m4a --timestamps sentence --enhance
    """
    verbose = ctx.obj.get('verbose', False)
    
    # Load configuration and merge with CLI arguments
    config_manager = get_config_manager()
    cli_args = {
        'language': language,
        'task': task,
        'timestamps': timestamps,
        'precision': precision,
        'temperature': temperature,
        'beam_size': beam_size,
        'processing_method': processing_method,
        'output_format': output_format,
        'output_dir': output_dir,
        'vault_path': vault_path,
        'batch_size': batch_size,
        'max_length': max_length,
        'enhance_notes': enhance_notes,
        'verbose': verbose
    }
    
    # Merge configuration with CLI args (CLI args take precedence)
    merged_config = config_manager.merge_with_cli_args(cli_args)
    
    # Extract values from merged config
    language = merged_config['language']
    task = merged_config['task']
    timestamps = merged_config['timestamps']
    precision = merged_config['precision']
    temperature = merged_config['temperature']
    beam_size = merged_config['beam_size']
    processing_method = merged_config['processing_method']
    output_format = merged_config['output_format']
    output_dir = merged_config['output_dir']
    vault_path = merged_config['vault_path']
    batch_size = merged_config['batch_size']
    max_length = merged_config['max_length']
    enhance_notes = merged_config['enhance_notes']
    verbose = merged_config['verbose']
    
    # Convert audio_files tuple to list
    audio_file_list = list(audio_files)
    
    # Intelligent suggestions
    if len(audio_file_list) == 1 and language == 'auto':
        suggested_lang = cli_instance.suggest_language(audio_file_list[0])
        if suggested_lang:
            click.echo(f"💡 Intelligent suggestion: Language might be '{WHISPER_LANGUAGES[suggested_lang]}' ({suggested_lang})")
            if click.confirm(f"Use suggested language '{suggested_lang}'?"):
                language = suggested_lang
    
    # Validate parameters
    if temperature < 0.0 or temperature > 1.0:
        raise click.BadParameter("Temperature must be between 0.0 and 1.0")
    
    if beam_size < 1:
        raise click.BadParameter("Beam size must be at least 1")
    
    # Set up paths
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()
    
    if not no_vault:
        if vault_path:
            vault_path = Path(vault_path)
        else:
            vault_path = Path.cwd() / 'obsidian_vault'
    
    # Show processing plan
    click.echo(f"🎵 Processing {len(audio_file_list)} audio file(s)")
    click.echo(f"📝 Task: {task}")
    click.echo(f"🌍 Language: {language}")
    click.echo(f"⏱️  Timestamps: {timestamps}")
    click.echo(f"🎯 Precision: {precision}")
    click.echo(f"📊 Output format: {output_format}")
    
    if not no_vault:
        click.echo(f"📚 Obsidian vault: {vault_path}")
    
    if dry_run:
        click.echo("🔍 Dry run mode - no actual processing")
        return
    
    # Initialize components
    try:
        click.echo("🔧 Initializing components...")
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(
            torch_dtype=torch.float32 if precision == 'float32' else torch.float16
        )
        
        # Initialize audio processor
        audio_processor = AudioProcessor()
        
        # Initialize Obsidian writer if needed
        obsidian_writer = None
        if not no_vault:
            obsidian_writer = ObsidianWriter(vault_path=str(vault_path))
        
        click.echo("✅ Components initialized successfully")
        
    except Exception as e:
        click.echo(f"❌ Failed to initialize components: {e}", err=True)
        sys.exit(1)
    
    # Process files
    results = []
    failed_files = []
    
    with click.progressbar(audio_file_list, label='Processing files') as files:
        for audio_file in files:
            try:
                result = process_single_file(
                    audio_file=audio_file,
                    transcriber=transcriber,
                    audio_processor=audio_processor,
                    obsidian_writer=obsidian_writer,
                    language=language,
                    task=task,
                    timestamps=timestamps,
                    temperature=temperature,
                    processing_method=processing_method,
                    output_format=output_format,
                    output_path=output_path,
                    enhance_notes=enhance_notes,
                    verbose=verbose
                )
                results.append(result)
                
            except Exception as e:
                error_msg = f"Failed to process {audio_file}: {e}"
                if verbose:
                    logger.exception(error_msg)
                else:
                    logger.error(error_msg)
                failed_files.append((audio_file, str(e)))
    
    # Show results
    click.echo(f"\n📊 Processing complete!")
    click.echo(f"✅ Successfully processed: {len(results)} files")
    
    if failed_files:
        click.echo(f"❌ Failed to process: {len(failed_files)} files")
        for file_path, error in failed_files:
            click.echo(f"   - {file_path}: {error}")
    
    # Show output locations
    if results:
        click.echo(f"\n📁 Output saved to: {output_path}")
        if not no_vault:
            click.echo(f"📚 Notes added to Obsidian vault: {vault_path}")

def process_single_file(audio_file: str, transcriber: WhisperTranscriber, 
                       audio_processor: AudioProcessor, obsidian_writer: Optional[ObsidianWriter],
                       language: str, task: str, timestamps: str, temperature: float,
                       processing_method: str, output_format: str,
                       output_path: Path, enhance_notes: bool, verbose: bool) -> Dict:
    """Process a single audio file"""
    
    # Load and validate audio
    audio_data, sample_rate = audio_processor.load_audio(audio_file)
    
    # Prepare transcription options
    transcribe_options = {
        'task': task,
        'temperature': temperature,
        'word_timestamps': timestamps == 'word',
        'return_timestamps': timestamps != 'none'
    }
    
    if language != 'auto':
        transcribe_options['language'] = language
    
    # Perform transcription
    if verbose:
        click.echo(f"🎤 Transcribing {os.path.basename(audio_file)}...")
    
    result = transcriber.transcribe(audio_data, **transcribe_options)
    
    # Format output based on requested format
    formatted_output = format_transcription_output(result, output_format, timestamps)
    
    # Save output file
    output_file = save_output_file(audio_file, formatted_output, output_format, output_path)
    
    # Add to Obsidian vault if enabled
    if obsidian_writer and enhance_notes:
        try:
            note_path, note_title = obsidian_writer.create_note(
                transcription=result['text'],
                original_filename=audio_file,
                metadata={
                    'language': result.get('language', 'unknown'),
                    'task': task,
                    'timestamp_level': timestamps,
                    'processing_method': processing_method,
                    'model_precision': transcribe_options.get('compute_type', 'float32')
                }
            )
            if verbose:
                click.echo(f"📝 Created Obsidian note: {note_path}")
        except Exception as e:
            logger.warning(f"Failed to create Obsidian note: {e}")
    
    return {
        'audio_file': audio_file,
        'output_file': output_file,
        'transcription_length': len(result['text']),
        'language': result.get('language', 'unknown'),
        'duration': result.get('duration', 0)
    }

def format_transcription_output(result: Dict, output_format: str, timestamps: str) -> str:
    """Format transcription result based on requested output format"""
    
    if output_format == 'json':
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    elif output_format == 'srt':
        return format_srt(result, timestamps)
    
    elif output_format == 'vtt':
        return format_vtt(result, timestamps)
    
    elif output_format == 'tsv':
        return format_tsv(result, timestamps)
    
    else:  # text format
        if timestamps == 'sentence' and 'segments' in result:
            lines = []
            for segment in result['segments']:
                start_time = format_timestamp(segment['start'])
                lines.append(f"[{start_time}] {segment['text'].strip()}")
            return '\n'.join(lines)
        else:
            return result['text']

def format_srt(result: Dict, timestamps: str) -> str:
    """Format as SRT subtitle file"""
    if 'segments' not in result:
        return result['text']
    
    srt_lines = []
    for i, segment in enumerate(result['segments'], 1):
        start_time = format_timestamp_srt(segment['start'])
        end_time = format_timestamp_srt(segment['end'])
        
        srt_lines.extend([
            str(i),
            f"{start_time} --> {end_time}",
            segment['text'].strip(),
            ""
        ])
    
    return '\n'.join(srt_lines)

def format_vtt(result: Dict, timestamps: str) -> str:
    """Format as WebVTT file"""
    if 'segments' not in result:
        return f"WEBVTT\n\n{result['text']}"
    
    vtt_lines = ["WEBVTT", ""]
    
    for segment in result['segments']:
        start_time = format_timestamp_vtt(segment['start'])
        end_time = format_timestamp_vtt(segment['end'])
        
        vtt_lines.extend([
            f"{start_time} --> {end_time}",
            segment['text'].strip(),
            ""
        ])
    
    return '\n'.join(vtt_lines)

def format_tsv(result: Dict, timestamps: str) -> str:
    """Format as TSV (Tab-Separated Values)"""
    if 'segments' not in result:
        return f"start\tend\ttext\n0.0\t{result.get('duration', 0)}\t{result['text']}"
    
    tsv_lines = ["start\tend\ttext"]
    
    for segment in result['segments']:
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip().replace('\t', ' ')
        tsv_lines.append(f"{start:.3f}\t{end:.3f}\t{text}")
    
    return '\n'.join(tsv_lines)

def format_timestamp(seconds: float) -> str:
    """Format timestamp as MM:SS"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

def save_output_file(audio_file: str, content: str, output_format: str, output_path: Path) -> str:
    """Save formatted output to file"""
    
    # Determine output filename
    audio_path = Path(audio_file)
    base_name = audio_path.stem
    
    # Extension mapping
    extensions = {
        'text': '.txt',
        'json': '.json',
        'srt': '.srt',
        'vtt': '.vtt',
        'tsv': '.tsv'
    }
    
    extension = extensions.get(output_format, '.txt')
    output_file = output_path / f"{base_name}_transcription{extension}"
    
    # Ensure unique filename
    counter = 1
    while output_file.exists():
        output_file = output_path / f"{base_name}_transcription_{counter}{extension}"
        counter += 1
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(output_file)

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--vault-path', '--vault',
              type=click.Path(),
              help='Obsidian vault path (default: ./obsidian_vault)')
def quick_note(audio_file, vault_path):
    """Quick transcription and note creation for a single audio file.
    
    This is a simplified command for rapid note-taking workflows.
    """
    
    click.echo(f"🚀 Quick processing: {os.path.basename(audio_file)}")
    
    # Load configuration and merge with CLI arguments
    config_manager = get_config_manager()
    cli_args = {'vault_path': vault_path}
    merged_config = config_manager.merge_with_cli_args(cli_args)
    
    # Set up vault path
    vault_path = merged_config.get('vault_path')
    if vault_path:
        vault_path = Path(vault_path)
    else:
        vault_path = Path.cwd() / 'obsidian_vault'
    
    try:
        # Initialize components with defaults optimized for speed
        transcriber = WhisperTranscriber(torch_dtype=torch.float32)
        audio_processor = AudioProcessor()
        obsidian_writer = ObsidianWriter(vault_path=str(vault_path))
        
        # Process with speed-optimized settings
        audio_data, sample_rate = audio_processor.load_audio(audio_file)
        
        result = transcriber.transcribe(
            audio_data,
            task='transcribe',
            temperature=0.0,
            word_timestamps=False
        )
        
        # Create enhanced note
        note_path, note_title = obsidian_writer.create_note(
            transcription=result['text'],
            original_filename=audio_file,
            metadata={
                'language': result.get('language', 'unknown'),
                'quick_mode': True,
                'processing_time': datetime.now().isoformat()
            }
        )
        
        click.echo(f"✅ Quick note created: {note_path}")
        click.echo(f"📝 Transcription length: {len(result['text'])} characters")
        click.echo(f"🌍 Detected language: {result.get('language', 'unknown')}")
        
    except Exception as e:
        click.echo(f"❌ Quick note failed: {e}", err=True)
        sys.exit(1)

@cli.group()
def config():
    """Manage configuration settings"""
    pass

@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """Set a configuration value
    
    KEY: Configuration key (e.g., language, vault_path, precision)
    VALUE: Value to set
    
    Examples:
        audio-notes config set language en
        audio-notes config set vault_path ~/Documents/MyVault
        audio-notes config set enhance_notes true
    """
    config_manager = get_config_manager()
    
    # Convert string values to appropriate types
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    elif value.isdigit():
        value = int(value)
    elif value.replace('.', '').isdigit():
        value = float(value)
    
    success = config_manager.set(key, value)
    if success:
        click.echo(f"✅ Set {key} = {value}")
    else:
        click.echo(f"❌ Failed to set {key}. Check the key name and value format.")

@config.command('get')
@click.argument('key', required=False)
def config_get(key):
    """Get configuration value(s)
    
    KEY: Configuration key to get (optional, shows all if not provided)
    
    Examples:
        audio-notes config get language
        audio-notes config get
    """
    config_manager = get_config_manager()
    
    if key:
        value = config_manager.get(key)
        if value is not None:
            click.echo(f"{key}: {value}")
        else:
            click.echo(f"❌ Configuration key '{key}' not found")
    else:
        # Show all configuration
        config_dict = config_manager.list_all()
        click.echo("📋 Current Configuration:")
        click.echo("=" * 40)
        for k, v in config_dict.items():
            click.echo(f"{k}: {v}")

@config.command('reset')
@click.argument('key', required=False)
@click.option('--all', is_flag=True, help='Reset all configuration to defaults')
def config_reset(key, all):
    """Reset configuration to defaults
    
    KEY: Specific key to reset (optional)
    
    Examples:
        audio-notes config reset language
        audio-notes config reset --all
    """
    config_manager = get_config_manager()
    
    if all:
        success = config_manager.reset()
        if success:
            click.echo("✅ Reset all configuration to defaults")
        else:
            click.echo("❌ Failed to reset configuration")
    elif key:
        success = config_manager.reset(key)
        if success:
            click.echo(f"✅ Reset {key} to default")
        else:
            click.echo(f"❌ Failed to reset {key}")
    else:
        click.echo("❌ Please specify a key to reset or use --all flag")

@config.command('path')
def config_path():
    """Show configuration file path"""
    config_manager = get_config_manager()
    click.echo(f"📁 Configuration file: {config_manager.config_path}")

@cli.command()
@click.option('--check-whisper', is_flag=True, help='Check Whisper model availability')
@click.option('--check-ollama', is_flag=True, help='Check Ollama availability')
@click.option('--check-all', is_flag=True, help='Check all components')
def status(check_whisper, check_ollama, check_all):
    """Check the status of audio processing components."""
    
    if check_all:
        check_whisper = check_ollama = True
    
    if not any([check_whisper, check_ollama]):
        check_whisper = check_ollama = True  # Default to checking all
    
    click.echo("🔍 Component Status Check\n")
    
    if check_whisper:
        click.echo("🎤 Whisper Transcriber:")
        try:
            transcriber = WhisperTranscriber()
            click.echo("   ✅ Whisper available")
            click.echo(f"   📦 Model: {transcriber.model_id}")
            click.echo(f"   🖥️  Device: {transcriber.device}")
        except Exception as e:
            click.echo(f"   ❌ Whisper unavailable: {e}")
        click.echo()
    
    if check_ollama:
        click.echo("🤖 Ollama (Qwen 3.0.6B):")
        try:
            ollama_client = OllamaClient()
            if ollama_client.is_available():
                click.echo("   ✅ Ollama available")
                click.echo(f"   🧠 Model: qwen3:0.6b")
                
                # Test response
                test_response = ollama_client.generate_text("Hello, are you working?")
                if test_response:
                    click.echo("   ✅ Model responding correctly")
                else:
                    click.echo("   ⚠️  Model not responding")
            else:
                click.echo("   ❌ Ollama not available")
        except Exception as e:
            click.echo(f"   ❌ Ollama error: {e}")
        click.echo()
    
    click.echo("💡 Use 'audio-notes process --help' for usage information")

if __name__ == '__main__':
    cli() 