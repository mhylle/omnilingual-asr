"""
Audio Recording Module

Simple audio recorder using sounddevice for microphone input.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional


class AudioRecorder:
    """
    Record audio from microphone and save to file.

    Audio is automatically configured for Omnilingual ASR:
    - Sample rate: 16000 Hz
    - Channels: 1 (mono)
    - Format: WAV
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Sample rate in Hz (default: 16000 for ASR)
        """
        self.sample_rate = sample_rate
        self.channels = 1  # Mono
        self.recording = None

    def record(
        self,
        duration: float,
        output_file: Optional[Path] = None,
        show_progress: bool = True
    ) -> Path:
        """
        Record audio from microphone.

        Args:
            duration: Recording duration in seconds (max 40 for ASR)
            output_file: Output file path (default: recording.wav)
            show_progress: Show recording progress

        Returns:
            Path to saved audio file

        Raises:
            ValueError: If duration > 40 seconds (ASR limitation)
        """
        if duration > 40:
            raise ValueError(
                "⚠️  Recording duration must be ≤40 seconds (ASR limitation)"
            )

        if output_file is None:
            output_file = Path("recording.wav")

        print(f"\n🎤 Recording {duration} seconds...")
        print("   Speak now!\n")

        # Record audio
        self.recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )

        # Show progress
        if show_progress:
            sd.wait()  # Wait until recording is finished
            print("✅ Recording complete!")
        else:
            sd.wait()

        # Save to file
        sf.write(output_file, self.recording, self.sample_rate)
        print(f"💾 Saved to: {output_file}")

        return output_file

    def list_devices(self) -> None:
        """List available audio input devices."""
        print("\n🎙️  Available Audio Input Devices:\n")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']}")
                print(f"      Channels: {device['max_input_channels']}")
                print(f"      Sample Rate: {device['default_samplerate']} Hz")
                print()

    def set_device(self, device_id: int) -> None:
        """
        Set audio input device.

        Args:
            device_id: Device ID from list_devices()
        """
        sd.default.device = device_id
        device_info = sd.query_devices(device_id)
        print(f"✅ Using device: {device_info['name']}")
