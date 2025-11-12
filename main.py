#!/usr/bin/env python3
"""
Meta Omnilingual ASR - Speech Recognition Application

A simple application for speech recognition using Meta's Omnilingual ASR system.
Supports 1,600+ languages with multiple model sizes.

Usage:
    python main.py record --duration 10 --language english
    python main.py transcribe audio.wav --language spanish
    python main.py models
    python main.py languages
"""

import argparse
from pathlib import Path
import sys

from transcriber import Transcriber
from audio_recorder import AudioRecorder


def record_command(args):
    """Record audio from microphone and transcribe."""
    try:
        # Initialize recorder
        recorder = AudioRecorder()

        # Record audio
        audio_file = recorder.record(
            duration=args.duration,
            output_file=Path(args.output) if args.output else None
        )

        # Transcribe if requested
        if args.transcribe:
            print("\n" + "="*50)
            transcriber = Transcriber(model=args.model)
            transcriptions = transcriber.transcribe(
                audio_file,
                language=args.language
            )

            print("\n📝 Transcription:")
            print("-" * 50)
            print(transcriptions[0])
            print("-" * 50)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def transcribe_command(args):
    """Transcribe existing audio file(s)."""
    try:
        # Initialize transcriber
        transcriber = Transcriber(model=args.model)

        # Transcribe files
        transcriptions = transcriber.transcribe(
            args.files,
            language=args.language,
            batch_size=args.batch_size
        )

        # Display results
        print("\n" + "="*50)
        print("📝 Transcription Results:")
        print("="*50)

        for file, text in zip(args.files, transcriptions):
            print(f"\n🎵 File: {file}")
            print("-" * 50)
            print(text)
            print()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def models_command(args):
    """List available models."""
    Transcriber.list_models()


def languages_command(args):
    """List supported languages."""
    Transcriber.list_languages()


def devices_command(args):
    """List audio input devices."""
    recorder = AudioRecorder()
    recorder.list_devices()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Meta Omnilingual ASR - Speech Recognition (1,600+ languages)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 10 seconds and transcribe in English
  python main.py record --duration 10 --language english

  # Transcribe existing audio file
  python main.py transcribe audio.wav --language spanish

  # Transcribe multiple files with larger model
  python main.py transcribe file1.wav file2.wav --model llm_1b

  # List available models
  python main.py models

  # List supported languages
  python main.py languages
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record audio from microphone")
    record_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (max 40, default: 10)"
    )
    record_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: recording.wav)"
    )
    record_parser.add_argument(
        "--language",
        type=str,
        help="Language for transcription (e.g., 'english', 'eng_Latn')"
    )
    record_parser.add_argument(
        "--model",
        type=str,
        default="ctc_1b",
        help="Model to use (default: ctc_1b)"
    )
    record_parser.add_argument(
        "--transcribe",
        action="store_true",
        default=True,
        help="Transcribe after recording (default: True)"
    )
    record_parser.set_defaults(func=record_command)

    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file(s)")
    transcribe_parser.add_argument(
        "files",
        nargs="+",
        help="Audio file(s) to transcribe"
    )
    transcribe_parser.add_argument(
        "--language",
        type=str,
        help="Language code (e.g., 'english', 'eng_Latn')"
    )
    transcribe_parser.add_argument(
        "--model",
        type=str,
        default="ctc_1b",
        help="Model to use (default: ctc_1b)"
    )
    transcribe_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    transcribe_parser.set_defaults(func=transcribe_command)

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.set_defaults(func=models_command)

    # Languages command
    languages_parser = subparsers.add_parser("languages", help="List supported languages")
    languages_parser.set_defaults(func=languages_command)

    # Devices command
    devices_parser = subparsers.add_parser("devices", help="List audio input devices")
    devices_parser.set_defaults(func=devices_command)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
