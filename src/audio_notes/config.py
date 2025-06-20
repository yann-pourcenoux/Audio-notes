#!/usr/bin/env python3
"""
Configuration Management for Audio Notes

This module provides a comprehensive configuration system that allows users to:
- Store persistent settings in a JSON configuration file
- Override config values via command-line arguments
- Manage configuration through CLI commands
- Validate configuration values using Pydantic models
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

class AudioNotesConfig(BaseModel):
    """Configuration model for audio-notes"""
    
    # Core settings
    language: str = Field(default="auto")
    task: str = Field(default="transcribe")
    timestamps: str = Field(default="none")
    precision: str = Field(default="float32")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    beam_size: int = Field(default=5, ge=1, le=20)
    processing_method: str = Field(default="sequential")
    
    # Output settings
    output_format: str = Field(default="text")
    output_dir: Optional[str] = Field(default=None)
    vault_path: Optional[str] = Field(default=None)
    
    # AI settings
    enhance_notes: bool = Field(default=True)
    
    # Processing settings
    batch_size: int = Field(default=1, ge=1, le=10)
    max_length: int = Field(default=30, ge=5, le=300)
    
    # Other settings
    verbose: bool = Field(default=False)
    
    @validator('language')
    def validate_language(cls, v):
        valid_languages = {
            'auto', 'af', 'ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en',
            'es', 'et', 'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja',
            'ko', 'lt', 'lv', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl',
            'sv', 'th', 'tr', 'uk', 'vi', 'zh'
        }
        if v not in valid_languages:
            raise ValueError(f"Invalid language code: {v}")
        return v
    
    @validator('task')
    def validate_task(cls, v):
        if v not in ['transcribe', 'translate']:
            raise ValueError(f"Invalid task: {v}. Must be 'transcribe' or 'translate'")
        return v
    
    @validator('timestamps')
    def validate_timestamps(cls, v):
        if v not in ['none', 'sentence', 'word']:
            raise ValueError(f"Invalid timestamps level: {v}")
        return v
    
    @validator('precision')
    def validate_precision(cls, v):
        if v not in ['float16', 'float32']:
            raise ValueError(f"Invalid precision: {v}")
        return v
    
    @validator('processing_method')
    def validate_processing_method(cls, v):
        if v not in ['sequential', 'chunked']:
            raise ValueError(f"Invalid processing method: {v}")
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        if v not in ['text', 'json', 'srt', 'vtt', 'tsv']:
            raise ValueError(f"Invalid output format: {v}")
        return v

class ConfigManager:
    """Configuration file manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".audio-notes" / "config.json"
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()
    
    def _load_config(self) -> AudioNotesConfig:
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return AudioNotesConfig(**config_data)
            except Exception as e:
                logger.warning(f"Error loading config: {e}")
                return AudioNotesConfig()
        else:
            config = AudioNotesConfig()
            self._save_config(config)
            return config
    
    def _save_config(self, config: AudioNotesConfig) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config.dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def get(self, key: str) -> Any:
        """Get configuration value"""
        try:
            return getattr(self._config, key)
        except AttributeError:
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            config_dict = self._config.dict()
            config_dict[key] = value
            new_config = AudioNotesConfig(**config_dict)
            self._save_config(new_config)
            self._config = new_config
            return True
        except Exception as e:
            logger.error(f"Error setting {key}: {e}")
            return False
    
    def reset(self, key: Optional[str] = None) -> bool:
        """Reset configuration"""
        try:
            if key is None:
                self._config = AudioNotesConfig()
                self._save_config(self._config)
            else:
                default_config = AudioNotesConfig()
                default_value = getattr(default_config, key)
                return self.set(key, default_value)
            return True
        except Exception as e:
            logger.error(f"Error resetting config: {e}")
            return False
    
    def list_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self._config.dict()
    
    def get_config_object(self) -> AudioNotesConfig:
        """Get configuration object"""
        return self._config
    
    def merge_with_cli_args(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge config with CLI args"""
        merged = self._config.dict()
        for key, value in cli_args.items():
            if value is not None:
                merged[key] = value
        return merged

# Global config manager
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global config manager"""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> AudioNotesConfig:
    """Get current configuration"""
    return get_config_manager().get_config_object()

def set_config_value(key: str, value: Any) -> bool:
    """Set a configuration value using the global manager"""
    return get_config_manager().set(key, value)

def get_config_value(key: str) -> Any:
    """Get a configuration value using the global manager"""
    return get_config_manager().get(key)

def reset_config(key: Optional[str] = None) -> bool:
    """Reset configuration using the global manager"""
    return get_config_manager().reset(key)

def list_config() -> Dict[str, Any]:
    """List all configuration values using the global manager"""
    return get_config_manager().list_all() 