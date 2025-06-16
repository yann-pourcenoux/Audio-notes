"""
Audio File Loader and Format Normalizer for Whisper Large-v3.

This module provides functionality to load, normalize, and preprocess audio files
for optimal compatibility with OpenAI's Whisper Large-v3 model through the
HuggingFace Transformers pipeline.

Key features:
- Support for multiple audio formats (WAV, MP3, M4A, FLAC, etc.)
- Automatic resampling to 16kHz (Whisper's optimal sample rate)
- Mono channel conversion (Whisper requirement)
- Amplitude normalization to prevent clipping
- Long-form audio handling (30-second chunk processing)
- Batch processing capabilities
- Error handling and logging
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings

import librosa
import numpy as np
import soundfile as sf
from loguru import logger
import pandas as pd


# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class AudioProcessor:
    """
    Audio processor optimized for Whisper Large-v3 model.
    
    This class handles loading, normalizing, and preprocessing audio files
    to ensure optimal compatibility with the Whisper model requirements.
    """
    
    # Whisper Large-v3 optimal settings
    WHISPER_SAMPLE_RATE = 16000  # 16kHz optimal for Whisper
    WHISPER_CHUNK_LENGTH = 30.0  # 30-second receptive field
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    
    def __init__(self, 
                 target_sample_rate: int = WHISPER_SAMPLE_RATE,
                 normalize_audio: bool = True,
                 chunk_length: Optional[float] = None):
        """
        Initialize the AudioProcessor.
        
        Args:
            target_sample_rate: Target sample rate for audio (default: 16000 for Whisper)
            normalize_audio: Whether to normalize audio amplitude (default: True)
            chunk_length: Maximum chunk length in seconds for long audio (default: None for full audio)
        """
        self.target_sample_rate = target_sample_rate
        self.normalize_audio = normalize_audio
        self.chunk_length = chunk_length
        
        logger.info(f"AudioProcessor initialized with sample_rate={target_sample_rate}Hz, "
                   f"normalize={normalize_audio}, chunk_length={chunk_length}s")
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to numpy array.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_array, original_sample_rate)
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio format is not supported
            RuntimeError: If the audio file cannot be loaded
        """
        file_path = Path(file_path)
        
        # Validate file existence
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Validate file format
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}. "
                           f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
        
        try:
            # Load audio with librosa (automatically handles format conversion)
            logger.debug(f"Loading audio file: {file_path}")
            audio_data, original_sr = librosa.load(str(file_path), sr=None, mono=False)
            
            logger.debug(f"Loaded audio: shape={audio_data.shape}, "
                        f"original_sr={original_sr}Hz, dtype={audio_data.dtype}")
            
            return audio_data, original_sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")
    
    def convert_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert audio to mono channel.
        
        Whisper models expect single-channel (mono) input. If the audio has
        multiple channels, this method averages them to create a mono signal.
        
        Args:
            audio_data: Audio array (can be mono or multi-channel)
            
        Returns:
            Mono audio array
        """
        if audio_data.ndim == 1:
            # Already mono
            return audio_data
        elif audio_data.ndim == 2:
            # Multi-channel: average across channels
            logger.debug(f"Converting {audio_data.shape[0]}-channel audio to mono")
            return np.mean(audio_data, axis=0)
        else:
            raise ValueError(f"Unexpected audio dimensions: {audio_data.ndim}")
    
    def resample_audio(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio_data: Input audio array
            original_sr: Original sample rate
            
        Returns:
            Resampled audio array
        """
        if original_sr == self.target_sample_rate:
            logger.debug(f"Audio already at target sample rate ({self.target_sample_rate}Hz)")
            return audio_data
        
        logger.debug(f"Resampling audio from {original_sr}Hz to {self.target_sample_rate}Hz")
        resampled_audio = librosa.resample(
            audio_data, 
            orig_sr=original_sr, 
            target_sr=self.target_sample_rate,
            res_type='soxr_hq'  # High-quality resampling
        )
        
        return resampled_audio
    
    def normalize_amplitude(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to prevent clipping while maintaining signal quality.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Normalized audio array
        """
        if not self.normalize_audio:
            return audio_data
        
        # Find current peak
        current_peak = np.max(np.abs(audio_data))
        
        if current_peak == 0:
            logger.warning("Audio contains only silence")
            return audio_data
        
        # Calculate normalization factor
        target_peak = 0.95
        normalization_factor = target_peak / current_peak
        
        # Apply normalization
        normalized_audio = audio_data * normalization_factor
        
        logger.debug(f"Normalized audio: peak {current_peak:.3f} -> {target_peak}")
        
        return normalized_audio
    
    def process_audio(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Complete audio processing pipeline for Whisper Large-v3.
        
        This method performs the full preprocessing pipeline:
        1. Load audio file
        2. Convert to mono
        3. Resample to 16kHz
        4. Normalize amplitude
        5. Prepare for HuggingFace pipeline
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing:
            - 'array': Processed audio array ready for Whisper
            - 'sampling_rate': Sample rate (16000)
            - 'duration': Duration in seconds
            - 'original_sr': Original sample rate
            - 'file_path': Path to the original file
        """
        file_path = Path(file_path)
        logger.info(f"Processing audio file: {file_path.name}")
        
        # Load audio
        audio_data, original_sr = self.load_audio(file_path)
        
        # Convert to mono
        audio_data = self.convert_to_mono(audio_data)
        
        # Resample to target rate
        audio_data = self.resample_audio(audio_data, original_sr)
        
        # Normalize amplitude
        audio_data = self.normalize_amplitude(audio_data)
        
        # Calculate duration
        audio_duration = len(audio_data) / self.target_sample_rate
        
        result = {
            'array': audio_data.astype(np.float32),  # Ensure float32 for HuggingFace
            'sampling_rate': self.target_sample_rate,
            'duration': audio_duration,
            'original_sr': original_sr,
            'file_path': str(file_path)
        }
        
        logger.info(f"Audio processing complete: duration={audio_duration:.2f}s, "
                   f"shape={audio_data.shape}, sample_rate={self.target_sample_rate}Hz")
        
        return result
    
    def process_long_audio(self, 
                          file_path: Union[str, Path],
                          chunk_length: Optional[float] = None,
                          overlap: float = 1.0) -> List[Dict[str, Any]]:
        """
        Process long audio files by splitting into chunks.
        
        For audio longer than Whisper's 30-second receptive field, this method
        splits the audio into overlapping chunks for optimal transcription.
        
        Args:
            file_path: Path to the audio file
            chunk_length: Length of each chunk in seconds (default: 30.0)
            overlap: Overlap between chunks in seconds (default: 1.0)
            
        Returns:
            List of processed audio chunks, each with metadata
        """
        if chunk_length is None:
            chunk_length = self.WHISPER_CHUNK_LENGTH
        
        file_path = Path(file_path)
        logger.info(f"Processing long audio file: {file_path.name} "
                   f"(chunk_length={chunk_length}s, overlap={overlap}s)")
        
        # Get audio duration first
        info = sf.info(str(file_path))
        total_duration = info.duration
        
        if total_duration <= chunk_length:
            logger.info(f"Audio duration ({total_duration:.2f}s) <= chunk_length ({chunk_length}s), "
                       "processing as single chunk")
            return [self.process_audio(file_path)]
        
        chunks = []
        current_offset = 0.0
        chunk_index = 0
        
        while current_offset < total_duration:
            # Calculate chunk duration
            remaining_duration = total_duration - current_offset
            current_chunk_length = min(chunk_length, remaining_duration)
            
            logger.debug(f"Processing chunk {chunk_index}: "
                        f"offset={current_offset:.2f}s, duration={current_chunk_length:.2f}s")
            
            # Process chunk
            chunk_data = self.process_audio(
                file_path, 
                offset=current_offset, 
                duration=current_chunk_length
            )
            
            # Add chunk metadata
            chunk_data.update({
                'chunk_index': chunk_index,
                'chunk_offset': current_offset,
                'chunk_length': current_chunk_length,
                'total_duration': total_duration
            })
            
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            current_offset += chunk_length - overlap
            chunk_index += 1
        
        logger.info(f"Split long audio into {len(chunks)} chunks")
        return chunks
    
    def batch_process(self, 
                     file_paths: List[Union[str, Path]],
                     progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process multiple audio files in batch.
        
        Args:
            file_paths: List of audio file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        results = []
        errors = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Process file
                result = self.process_audio(file_path)
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), file_path)
                    
            except Exception as e:
                error_info = {
                    'file_path': str(file_path),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                errors.append(error_info)
                logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Batch processing complete: {len(results)} successful, {len(errors)} failed")
        
        if errors:
            logger.warning(f"Errors encountered: {errors}")
        
        return results
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported audio formats."""
        return list(AudioProcessor.SUPPORTED_FORMATS)
    
    @staticmethod
    def is_supported_format(file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in AudioProcessor.SUPPORTED_FORMATS
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get audio file information without loading the full file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            info = sf.info(str(file_path))
            
            return {
                'file_path': str(file_path),
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype,
                'file_size': file_path.stat().st_size,
                'needs_resampling': info.samplerate != self.target_sample_rate,
                'needs_mono_conversion': info.channels > 1,
                'estimated_chunks': max(1, int(np.ceil(info.duration / self.WHISPER_CHUNK_LENGTH)))
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get audio info for {file_path}: {str(e)}") 