#!/usr/bin/env python3
"""
Example usage of the Omnilingual ASR application.

This demonstrates how to use the transcriber and recorder modules directly.
"""

from pathlib import Path
from transcriber import Transcriber
from audio_recorder import AudioRecorder


def example_list_info():
    """Example: List available models and languages."""
    print("=" * 60)
    print("EXAMPLE 1: List Available Models and Languages")
    print("=" * 60)

    Transcriber.list_models()
    Transcriber.list_languages()


def example_record_and_transcribe():
    """Example: Record audio and transcribe."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Record Audio and Transcribe")
    print("=" * 60)

    # Initialize recorder
    recorder = AudioRecorder()

    # Record 5 seconds
    print("\n🎤 Recording 5 seconds of audio...")
    audio_file = recorder.record(
        duration=5.0,
        output_file=Path("example_recording.wav")
    )

    # Initialize transcriber with fast model
    print("\n🚀 Loading transcriber...")
    transcriber = Transcriber(model="ctc_1b")

    # Transcribe
    print("\n📝 Transcribing...")
    transcriptions = transcriber.transcribe(
        audio_file,
        language="english"
    )

    print("\n✅ Result:")
    print("-" * 60)
    print(transcriptions[0])
    print("-" * 60)


def example_transcribe_file():
    """Example: Transcribe an existing audio file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Transcribe Existing File")
    print("=" * 60)

    # Check if example file exists
    audio_file = Path("example_recording.wav")
    if not audio_file.exists():
        print("\n⚠️  No audio file found. Run example 2 first or provide your own .wav file")
        return

    # Initialize transcriber
    print("\n🚀 Loading transcriber with language-aware model...")
    transcriber = Transcriber(model="llm_1b")

    # Transcribe with language hint
    print("\n📝 Transcribing with language hint (English)...")
    transcriptions = transcriber.transcribe(
        audio_file,
        language="eng_Latn"  # Using ISO language code
    )

    print("\n✅ Result:")
    print("-" * 60)
    print(transcriptions[0])
    print("-" * 60)


def example_batch_transcribe():
    """Example: Batch transcribe multiple files."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Transcribe Multiple Files")
    print("=" * 60)

    # Find all .wav files in current directory
    audio_files = list(Path(".").glob("*.wav"))

    if not audio_files:
        print("\n⚠️  No .wav files found in current directory")
        return

    print(f"\n📁 Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"   - {f}")

    # Initialize transcriber
    print("\n🚀 Loading transcriber...")
    transcriber = Transcriber(model="ctc_1b")

    # Batch transcribe
    print(f"\n📝 Batch transcribing {len(audio_files)} file(s)...")
    transcriptions = transcriber.transcribe(
        audio_files,
        batch_size=2  # Process 2 files at a time
    )

    # Display results
    print("\n✅ Results:")
    print("=" * 60)
    for file, text in zip(audio_files, transcriptions):
        print(f"\n🎵 {file.name}")
        print("-" * 60)
        print(text)

    print("=" * 60)


def example_multilingual():
    """Example: Transcribe in different languages."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multilingual Transcription")
    print("=" * 60)

    print("\n📚 This example shows how to transcribe audio in different languages.")
    print("   You'll need audio files in different languages to test this.")

    # Example language codes
    languages = {
        "english_audio.wav": "english",
        "spanish_audio.wav": "spanish",
        "french_audio.wav": "french",
        "chinese_audio.wav": "cmn_Hans",
        "arabic_audio.wav": "arb_Arab"
    }

    print("\n💡 Example usage:")
    print("-" * 60)

    for file, lang in languages.items():
        print(f"""
# Transcribe {file} in {lang}
transcriber = Transcriber(model="llm_1b")
text = transcriber.transcribe("{file}", language="{lang}")
        """)


def main():
    """Run examples."""
    print("\n" + "=" * 60)
    print("Meta Omnilingual ASR - Usage Examples")
    print("=" * 60)

    print("\nChoose an example to run:")
    print("  1. List available models and languages")
    print("  2. Record audio and transcribe")
    print("  3. Transcribe existing file")
    print("  4. Batch transcribe multiple files")
    print("  5. Multilingual transcription examples")
    print("  0. Exit")

    choice = input("\nEnter choice (0-5): ").strip()

    if choice == "1":
        example_list_info()
    elif choice == "2":
        example_record_and_transcribe()
    elif choice == "3":
        example_transcribe_file()
    elif choice == "4":
        example_batch_transcribe()
    elif choice == "5":
        example_multilingual()
    elif choice == "0":
        print("\n👋 Goodbye!")
        return
    else:
        print("\n❌ Invalid choice")

    print("\n✨ Example complete!\n")


if __name__ == "__main__":
    main()
