# Omnilingual ASR REST API

REST API for speech recognition supporting 1,600+ languages using Meta's Omnilingual ASR system.

## Features

- 🌐 **RESTful API** - Standard HTTP endpoints
- 📁 **File Upload** - Single and batch transcription
- 🚀 **Fast** - Async processing with FastAPI
- 📚 **Auto Documentation** - OpenAPI/Swagger docs
- 🔧 **Configurable** - Multiple models and languages
- 🐳 **Production Ready** - CORS, error handling, logging

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r api_requirements.txt

# Or use the start script
./start_api.sh
```

### 2. Start Server

```bash
# Development mode
python api.py

# Production mode with Uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# Using the start script
./start_api.sh
```

### 3. Access Documentation

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Info**: http://localhost:8000/info

## API Endpoints

### Health & Info

#### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": true,
  "current_model": "ctc_1b"
}
```

#### `GET /models`
List available ASR models.

**Response:**
```json
{
  "models": ["ctc_300m", "ctc_1b", "llm_1b", ...],
  "details": { ... },
  "default": "ctc_1b"
}
```

#### `GET /languages`
List supported languages.

**Response:**
```json
{
  "total_supported": "1600+",
  "common_languages": {
    "english": "eng_Latn",
    "spanish": "spa_Latn",
    ...
  }
}
```

### Transcription

#### `POST /transcribe`
Transcribe a single audio file.

**Parameters:**
- `file` (form-data): Audio file (WAV, FLAC, MP3, etc.)
- `model` (query, optional): Model name (default: `ctc_1b`)
- `language` (query, optional): Language code or name

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/transcribe?model=ctc_1b&language=english" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "success": true,
  "transcription": "hello world this is a test",
  "metadata": {
    "filename": "audio.wav",
    "model": "ctc_1b",
    "language": "english",
    "processing_time": "0.45s",
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

#### `POST /transcribe/batch`
Transcribe multiple audio files.

**Parameters:**
- `files` (form-data): Multiple audio files
- `model` (query, optional): Model name
- `language` (query, optional): Language code
- `batch_size` (query, optional): Processing batch size (1-10)

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/transcribe/batch?batch_size=2" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "results": [
    {
      "filename": "audio1.wav",
      "transcription": "first audio file"
    },
    {
      "filename": "audio2.wav",
      "transcription": "second audio file"
    },
    {
      "filename": "audio3.wav",
      "transcription": "third audio file"
    }
  ],
  "metadata": {
    "model": "ctc_1b",
    "batch_size": 2,
    "processing_time": "1.23s",
    "avg_time_per_file": "0.41s"
  }
}
```

## Python Client

### Installation

```python
pip install requests
```

### Usage Example

```python
from api_client import ASRClient
from pathlib import Path

# Initialize client
client = ASRClient(base_url="http://localhost:8000")

# Check health
health = client.health()
print(health)

# List models
models = client.list_models()
print(models['models'])

# Transcribe single file
result = client.transcribe(
    Path("audio.wav"),
    model="ctc_1b",
    language="english"
)
print(result['transcription'])

# Batch transcribe
results = client.transcribe_batch(
    [Path("audio1.wav"), Path("audio2.wav")],
    model="llm_1b",
    language="spanish",
    batch_size=2
)
for r in results['results']:
    print(f"{r['filename']}: {r['transcription']}")
```

### Run Example Client

```bash
# Make sure API server is running first
python api_client.py
```

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model
DEFAULT_MODEL=ctc_1b
MAX_AUDIO_DURATION=40.0

# Processing
MAX_BATCH_SIZE=10
CLEANUP_UPLOADS=true
```

### Configuration Options

See `config.py` for all available settings:

- **Server**: host, port, workers, reload
- **Model**: default model, max duration, file size limits
- **Processing**: batch size, upload directory
- **CORS**: enabled origins

## Examples

### cURL Examples

**Health check:**
```bash
curl http://localhost:8000/health
```

**List models:**
```bash
curl http://localhost:8000/models
```

**Transcribe English audio:**
```bash
curl -X POST "http://localhost:8000/transcribe?language=english" \
  -F "file=@recording.wav"
```

**Transcribe with specific model:**
```bash
curl -X POST "http://localhost:8000/transcribe?model=llm_1b&language=spanish" \
  -F "file=@spanish_audio.wav"
```

**Batch transcribe:**
```bash
curl -X POST "http://localhost:8000/transcribe/batch?batch_size=3" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

### JavaScript/Fetch Example

```javascript
// Transcribe audio file
async function transcribe(audioFile) {
  const formData = new FormData();
  formData.append('file', audioFile);

  const response = await fetch(
    'http://localhost:8000/transcribe?language=english',
    {
      method: 'POST',
      body: formData
    }
  );

  const result = await response.json();
  console.log(result.transcription);
}

// Usage with file input
const fileInput = document.getElementById('audio-file');
transcribe(fileInput.files[0]);
```

## Production Deployment

### Using Uvicorn

```bash
# Single worker
uvicorn api:app --host 0.0.0.0 --port 8000

# Multiple workers (CPU-based scaling)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# With auto-reload (development)
uvicorn api:app --reload
```

### Using Gunicorn

```bash
pip install gunicorn

gunicorn api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables

```bash
export DEFAULT_MODEL=ctc_1b
export MAX_BATCH_SIZE=10
export WORKERS=4

python api.py
```

## Performance Considerations

### Model Selection

- **`ctc_1b`** - Recommended default (fast, good accuracy)
- **`ctc_3b`** - Better accuracy, slower
- **`llm_1b`** - Language-aware, use with `language` parameter
- **`ctc_7b`** - Best accuracy, requires ~17GB VRAM

### Optimization Tips

1. **Use GPU** - Significant performance improvement
2. **Batch Processing** - Process multiple files together
3. **Model Caching** - API reuses loaded models
4. **Worker Count** - Scale with available CPU/GPU
5. **File Format** - WAV files process faster than MP3

### Resource Requirements

| Model | VRAM | Speed | Use Case |
|-------|------|-------|----------|
| ctc_300m | ~2GB | Fastest | Real-time, low-resource |
| ctc_1b | ~4GB | Fast | **Recommended** |
| ctc_3b | ~8GB | Medium | High accuracy |
| ctc_7b | ~17GB | Slower | Maximum accuracy |

## Limitations

⚠️ **Important:**

1. **Audio Length**: Maximum 40 seconds per file
2. **File Size**: Maximum 50MB per file
3. **Output Format**: Lowercase, no punctuation
4. **Model Loading**: First request downloads model (large files)
5. **Concurrent Requests**: Limited by available GPU memory

## Troubleshooting

### Server won't start

```bash
# Check port availability
lsof -i :8000

# Use different port
PORT=8080 python api.py
```

### Model download fails

```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Restart API
python api.py
```

### Out of memory

```bash
# Use smaller model
DEFAULT_MODEL=ctc_300m python api.py

# Or use CPU
# (modify transcriber initialization in api.py)
```

### Upload fails

```bash
# Check file format
file audio.wav

# Convert if needed
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## API Testing

### Test with Interactive Docs

1. Start server: `python api.py`
2. Open browser: http://localhost:8000/docs
3. Try endpoints interactively

### Test with Python Client

```bash
python api_client.py
```

### Test with cURL

```bash
# Create test recording
python main.py record --duration 5

# Transcribe via API
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@recording.wav"
```

## Development

### Project Structure

```
meta_speech/
├── api.py              # FastAPI application
├── config.py           # Configuration
├── api_client.py       # Python client
├── transcriber.py      # ASR wrapper
├── api_requirements.txt # Dependencies
└── start_api.sh       # Start script
```

### Adding New Endpoints

```python
# In api.py
@app.post("/custom-endpoint")
async def custom_endpoint(param: str):
    """Your custom endpoint."""
    return {"result": "success"}
```

### Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests (create test file first)
pytest test_api.py
```

## License

Based on Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr).

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Omnilingual ASR GitHub](https://github.com/facebookresearch/omnilingual-asr)
- [OpenAPI Specification](http://localhost:8000/openapi.json)
