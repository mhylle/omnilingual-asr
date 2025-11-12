"""
API Configuration

Configuration settings for the Omnilingual ASR API.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """API configuration settings."""

    # API Settings
    app_name: str = "Omnilingual ASR API"
    app_version: str = "1.0.0"
    app_description: str = "Speech recognition API supporting 1,600+ languages"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1

    # Model Settings
    default_model: str = "ctc_1b"
    max_audio_duration: float = 40.0  # seconds
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    # Processing Settings
    default_batch_size: int = 1
    max_batch_size: int = 10
    allowed_audio_formats: list = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]

    # Upload Settings
    upload_dir: str = "./uploads"
    cleanup_uploads: bool = True

    # CORS Settings
    cors_enabled: bool = True
    cors_origins: list = [
        "http://localhost:4202",
        "http://127.0.0.1:4202",
        "http://localhost:4200",  # Backward compatibility
        "http://127.0.0.1:4200"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
