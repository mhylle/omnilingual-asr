# Meta Omnilingual ASR - Speech Recognition Application

A simple Python application for speech recognition using Meta's Omnilingual ASR system. Supports **1,600+ languages** with state-of-the-art accuracy.

## Features

- 🌍 **1,600+ Languages** - Including many low-resource languages
- 🎤 **Record from Microphone** - Simple audio recording interface
- 📁 **Transcribe Files** - Process existing audio files
- 🚀 **Multiple Models** - Choose from 300M to 7B parameter models
- ⚡ **Fast Inference** - CTC models for real-time transcription
- 🎯 **Language-Aware** - LLM models with language conditioning

## Installation

### 1. System Dependencies

Install system audio libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 portaudio19-dev

# macOS
brew install libsndfile portaudio

# Fedora/RHEL
sudo dnf install libsndfile portaudio-devel
```

### 2. Python Environment

Create virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run will download the selected model (may take several minutes). The 7B model requires ~30GB download and ~17GB VRAM.

## Quick Start

### List Available Models

```bash
python main.py models
```

Output:
```
📦 Available Models:

CTC Models (faster, parallel generation):
  - ctc_300m: 300M parameters
  - ctc_1b:   1B parameters (recommended)
  - ctc_3b:   3B parameters
  - ctc_7b:   7B parameters

LLM Models (language-aware, autoregressive):
  - llm_300m: 300M parameters
  - llm_1b:   1B parameters
  - llm_3b:   3B parameters
  - llm_7b:   7B parameters (requires ~17GB VRAM)
  - llm_7b_zs: 7B zero-shot model
```

### List Supported Languages

```bash
python main.py languages
```

### Record and Transcribe

Record 10 seconds of audio and transcribe in English:

```bash
python main.py record --duration 10 --language english
```

Record in Spanish with a larger model:

```bash
python main.py record --duration 15 --language spanish --model llm_1b
```

### Transcribe Existing Files

Transcribe a single audio file:

```bash
python main.py transcribe audio.wav --language english
```

Transcribe multiple files:

```bash
python main.py transcribe file1.wav file2.wav file3.wav --language french
```

Use a specific language code:

```bash
python main.py transcribe audio.wav --language cmn_Hans
```

## Usage Guide

### Commands

#### `record` - Record from Microphone

```bash
python main.py record [OPTIONS]

Options:
  --duration SECONDS    Recording duration (max 40 seconds, default: 10)
  --output FILE        Output file path (default: recording.wav)
  --language LANG      Language for transcription
  --model MODEL        Model to use (default: ctc_1b)
  --transcribe         Transcribe after recording (default: True)
```

#### `transcribe` - Transcribe Audio Files

```bash
python main.py transcribe FILES... [OPTIONS]

Options:
  --language LANG      Language code or name
  --model MODEL        Model to use (default: ctc_1b)
  --batch-size N       Batch size for processing (default: 1)
```

#### `models` - List Available Models

```bash
python main.py models
```

#### `languages` - List Supported Languages

```bash
python main.py languages
```

#### `devices` - List Audio Input Devices

```bash
python main.py devices
```

### Language Codes

Use common language names or ISO codes:

**Common Names:**
- `english`, `spanish`, `french`, `german`, `italian`
- `portuguese`, `russian`, `chinese`, `japanese`, `korean`
- `arabic`, `hindi`

**ISO Format:** `{language_code}_{script}`
- `eng_Latn` - English (Latin script)
- `cmn_Hans` - Mandarin Chinese (Simplified)
- `cmn_Hant` - Mandarin Chinese (Traditional)
- `arb_Arab` - Arabic
- `rus_Cyrl` - Russian (Cyrillic)
- `jpn_Jpan` - Japanese

See full list: [lang_ids.py](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/lang_ids.py)

### Model Selection

Choose based on your needs:

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| `ctc_300m` | Fastest | Good | ~2GB | Real-time, low-resource |
| `ctc_1b` | Fast | Better | ~4GB | **Recommended default** |
| `ctc_3b` | Medium | Great | ~8GB | High accuracy |
| `ctc_7b` | Slower | Best | ~17GB | Maximum accuracy |
| `llm_1b` | Fast | Better | ~4GB | Language-specific tuning |
| `llm_7b` | Slower | Best | ~17GB | Best with language hints |

**CTC Models:** Faster, parallel generation, no language conditioning
**LLM Models:** Language-aware, better with `--language` specified

## Examples

### Basic Recording

```bash
# Record 5 seconds in English
python main.py record --duration 5 --language english

# Record 20 seconds in Spanish with better model
python main.py record --duration 20 --language spanish --model llm_1b

# Record without auto-transcription
python main.py record --duration 10 --output my_audio.wav --transcribe false
```

### Transcribing Files

```bash
# Single file
python main.py transcribe interview.wav --language english

# Multiple files with batch processing
python main.py transcribe *.wav --batch-size 4 --model ctc_3b

# Specific language code
python main.py transcribe chinese_audio.wav --language cmn_Hans --model llm_1b
```

### Advanced Usage

```python
# Python API usage
from transcriber import Transcriber

# Initialize with specific model
transcriber = Transcriber(model="llm_1b")

# Transcribe files
texts = transcriber.transcribe(
    ["audio1.wav", "audio2.wav"],
    language="eng_Latn",
    batch_size=2
)

for text in texts:
    print(text)
```

## Limitations

⚠️ **Important Constraints:**

1. **Audio Length:** Maximum 40 seconds per file
2. **Output Format:** Lowercase text without punctuation
3. **Model Download:** First run downloads model (large files)
4. **VRAM Requirements:** Larger models need significant GPU memory
5. **Audio Format:** Automatically converts to 16kHz mono WAV

## Troubleshooting

### Model Download Issues

If model download fails:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python main.py transcribe test.wav
```

### Audio Recording Issues

Check available devices:
```bash
python main.py devices
```

Test microphone:
```python
import sounddevice as sd
sd.query_devices()  # List all devices
sd.rec(16000, samplerate=16000, channels=1)  # Test recording
```

### CUDA/GPU Issues

Force CPU usage:
```python
# In transcriber.py, change:
transcriber = Transcriber(model="ctc_1b", device="cpu")
```

### Audio Format Issues

Convert audio to supported format:
```bash
# Using ffmpeg
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Project Structure

```
meta_speech/
├── main.py              # CLI interface
├── transcriber.py       # ASR wrapper
├── audio_recorder.py    # Microphone recording
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Performance Tips

1. **Use CTC models** for faster transcription
2. **Batch multiple files** to improve throughput
3. **Use GPU** for better performance with large models
4. **Keep audio <40s** to avoid errors
5. **Specify language** with LLM models for better accuracy

## Contributing

Based on Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) research.

## License

This application is a simple wrapper around Meta's Omnilingual ASR system. Please refer to the [original repository](https://github.com/facebookresearch/omnilingual-asr) for licensing information.

## Resources

- [Omnilingual ASR GitHub](https://github.com/facebookresearch/omnilingual-asr)
- [Model Inference Guide](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/README.md)
- [Supported Languages](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/lang_ids.py)

## Support

For issues with:
- **This application:** Open an issue in this repository
- **Omnilingual ASR:** See [upstream repository](https://github.com/facebookresearch/omnilingual-asr/issues)
