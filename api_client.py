#!/usr/bin/env python3
"""
API Client Example

Example client for the Omnilingual ASR API.
"""

import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


class ASRClient:
    """Client for Omnilingual ASR API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.

        Args:
            base_url: API base URL
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def list_languages(self) -> Dict[str, Any]:
        """List supported languages."""
        response = self.session.get(f"{self.base_url}/languages")
        response.raise_for_status()
        return response.json()

    def get_info(self) -> Dict[str, Any]:
        """Get API information."""
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()

    def transcribe(
        self,
        audio_file: Path,
        model: str = "ctc_1b",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file.

        Args:
            audio_file: Path to audio file
            model: Model name
            language: Language code or name

        Returns:
            Transcription result
        """
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Prepare request
        files = {"file": open(audio_file, "rb")}
        params = {"model": model}
        if language:
            params["language"] = language

        # Send request
        response = self.session.post(
            f"{self.base_url}/transcribe",
            files=files,
            params=params
        )
        response.raise_for_status()

        return response.json()

    def transcribe_batch(
        self,
        audio_files: List[Path],
        model: str = "ctc_1b",
        language: Optional[str] = None,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Transcribe multiple audio files.

        Args:
            audio_files: List of audio file paths
            model: Model name
            language: Language code or name
            batch_size: Processing batch size

        Returns:
            Batch transcription results
        """
        # Validate files
        for audio_file in audio_files:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Prepare request
        files = [("files", open(f, "rb")) for f in audio_files]
        params = {"model": model, "batch_size": batch_size}
        if language:
            params["language"] = language

        # Send request
        response = self.session.post(
            f"{self.base_url}/transcribe/batch",
            files=files,
            params=params
        )
        response.raise_for_status()

        return response.json()


def main():
    """Example usage."""
    print("=" * 60)
    print("Omnilingual ASR API Client - Examples")
    print("=" * 60)

    # Initialize client
    client = ASRClient(base_url="http://localhost:8000")

    try:
        # 1. Health check
        print("\n1. Health Check:")
        print("-" * 60)
        health = client.health()
        print(json.dumps(health, indent=2))

        # 2. List models
        print("\n2. Available Models:")
        print("-" * 60)
        models = client.list_models()
        print(f"Default model: {models['default']}")
        print(f"Available: {', '.join(models['models'][:5])}...")

        # 3. List languages
        print("\n3. Supported Languages:")
        print("-" * 60)
        languages = client.list_languages()
        print(f"Total supported: {languages['total_supported']}")
        print("Common languages:")
        for name, code in list(languages['common_languages'].items())[:5]:
            print(f"  {name:12} → {code}")

        # 4. Get API info
        print("\n4. API Information:")
        print("-" * 60)
        info = client.get_info()
        print(f"API: {info['api']['name']} v{info['api']['version']}")
        print(f"Max duration: {info['capabilities']['max_audio_duration']}")
        print(f"Supported formats: {', '.join(info['capabilities']['supported_formats'])}")

        # 5. Transcribe example (if file exists)
        print("\n5. Transcription Example:")
        print("-" * 60)

        test_file = Path("recording.wav")
        if test_file.exists():
            print(f"Transcribing: {test_file}")
            result = client.transcribe(
                test_file,
                model="ctc_1b",
                language="english"
            )

            print(f"\n✅ Success!")
            print(f"Transcription: {result['transcription']}")
            print(f"Processing time: {result['metadata']['processing_time']}")
        else:
            print(f"⚠️  No test file found ({test_file})")
            print("   Create one with: python main.py record --duration 5")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("   Make sure the server is running:")
        print("   python api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
