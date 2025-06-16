"""
Main entry point for the Audio Notes application.
"""

import logging
from typing import Optional

import click
import torch
import transformers
from loguru import logger

from . import __version__, __description__


def hello() -> str:
    """Simple hello function for testing."""
    return "Hello from Audio Notes!"


@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.version_option(version=__version__)
def main(verbose: bool) -> None:
    """
    Audio Notes - AI-powered audio processing and transcription system.
    """
    if verbose:
        logger.add("audio_notes.log", level="DEBUG")
    else:
        logger.add("audio_notes.log", level="INFO")
    
    logger.info(f"Starting {__description__} v{__version__}")
    
    # Environment verification
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.get_device_name()}")
    
    click.echo("✅ Audio Notes environment is ready!")
    click.echo(f"📦 Version: {__version__}")
    click.echo(f"🔧 PyTorch: {torch.__version__}")
    click.echo(f"🤗 Transformers: {transformers.__version__}")
    
    if torch.cuda.is_available():
        click.echo(f"🚀 CUDA: Available ({torch.cuda.device_count()} device(s))")
    else:
        click.echo("💻 CUDA: Not available (CPU only)")


if __name__ == "__main__":
    main()
