"""
Omnilingual ASR Transcriber Module

Wrapper for Meta's Omnilingual ASR system for easy speech recognition.
"""

from typing import List, Optional, Union
from pathlib import Path
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


class Transcriber:
    """
    Speech recognition transcriber using Meta's Omnilingual ASR.

    Supports 1,600+ languages with multiple model sizes.
    """

    AVAILABLE_MODELS = {
        "ctc_300m": "omniASR_CTC_300M",
        "ctc_1b": "omniASR_CTC_1B",
        "ctc_3b": "omniASR_CTC_3B",
        "ctc_7b": "omniASR_CTC_7B",
        "llm_300m": "omniASR_LLM_300M",
        "llm_1b": "omniASR_LLM_1B",
        "llm_3b": "omniASR_LLM_3B",
        "llm_7b": "omniASR_LLM_7B",
        "llm_7b_zs": "omniASR_LLM_7B_ZS",
    }

    COMMON_LANGUAGES = {
        "english": "eng_Latn",
        "spanish": "spa_Latn",
        "french": "fra_Latn",
        "german": "deu_Latn",
        "italian": "ita_Latn",
        "portuguese": "por_Latn",
        "danish": "dan_Latn",
        "russian": "rus_Cyrl",
        "chinese": "cmn_Hans",
        "japanese": "jpn_Jpan",
        "korean": "kor_Hang",
        "arabic": "arb_Arab",
        "hindi": "hin_Deva",
    }

    def __init__(self, model: str = "ctc_1b", device: str = "cuda"):
        """
        Initialize transcriber with specified model.

        Args:
            model: Model size (ctc_300m, ctc_1b, llm_1b, etc.)
            device: Device to use (cuda or cpu)
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model}' not found. Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model
        self.model_card = self.AVAILABLE_MODELS[model]
        self.device = device

        print(f"Loading model: {self.model_card}...")
        print("⚠️  First run will download the model (may take several minutes)")

        self.pipeline = ASRInferencePipeline(model_card=self.model_card)

        print(f"✅ Model loaded successfully!")

    def transcribe(
        self,
        audio_files: Union[str, Path, List[Union[str, Path]]],
        language: Optional[str] = None,
        batch_size: int = 1
    ) -> List[str]:
        """
        Transcribe audio file(s) to text.

        Args:
            audio_files: Single file path or list of file paths
            language: Language code (e.g., 'eng_Latn') or common name (e.g., 'english')
            batch_size: Number of files to process simultaneously

        Returns:
            List of transcriptions

        Note:
            - Audio files must be <40 seconds
            - Output is lowercase without punctuation
        """
        # Convert single file to list
        if isinstance(audio_files, (str, Path)):
            audio_files = [audio_files]

        # Convert string paths to Path objects
        audio_files = [Path(f) for f in audio_files]

        # Validate files exist
        for audio_file in audio_files:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Convert language name to code if needed
        lang_code = None
        if language:
            lang_code = self._resolve_language(language)
            print(f"Using language: {lang_code}")

        # Transcribe
        print(f"Transcribing {len(audio_files)} file(s)...")

        if lang_code and "LLM" in self.model_card:
            # Language conditioning only works with LLM models
            transcriptions = self.pipeline.transcribe(
                [str(f) for f in audio_files],
                lang=[lang_code] * len(audio_files),
                batch_size=batch_size
            )
        else:
            transcriptions = self.pipeline.transcribe(
                [str(f) for f in audio_files],
                batch_size=batch_size
            )

        return transcriptions

    def _resolve_language(self, language: str) -> str:
        """
        Convert common language name to language code.

        Args:
            language: Language name or code

        Returns:
            Language code (e.g., 'eng_Latn')
        """
        # Check if already a valid code (contains underscore)
        if "_" in language:
            return language

        # Check common languages
        lang_lower = language.lower()
        if lang_lower in self.COMMON_LANGUAGES:
            return self.COMMON_LANGUAGES[lang_lower]

        # Return as-is and let the model handle it
        print(f"⚠️  Unknown language '{language}', using as-is")
        return language

    @classmethod
    def list_models(cls) -> None:
        """Print available models with descriptions."""
        print("\n📦 Available Models:\n")
        print("CTC Models (faster, parallel generation):")
        print("  - ctc_300m: 300M parameters")
        print("  - ctc_1b:   1B parameters (recommended)")
        print("  - ctc_3b:   3B parameters")
        print("  - ctc_7b:   7B parameters")
        print("\nLLM Models (language-aware, autoregressive):")
        print("  - llm_300m: 300M parameters")
        print("  - llm_1b:   1B parameters")
        print("  - llm_3b:   3B parameters")
        print("  - llm_7b:   7B parameters (requires ~17GB VRAM)")
        print("  - llm_7b_zs: 7B zero-shot model")
        print()

    @classmethod
    def list_languages(cls) -> None:
        """Print common supported languages."""
        print("\n🌍 Common Supported Languages:\n")
        for name, code in sorted(cls.COMMON_LANGUAGES.items()):
            print(f"  {name:12} → {code}")
        print("\n💡 Supports 1,600+ languages total!")
        print("   Use format: {language_code}_{script}")
        print("   Examples: eng_Latn, cmn_Hans, arb_Arab\n")
