# API Quick Start Guide

Get the Omnilingual ASR API running in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r api_requirements.txt
```

## 2. Start the Server

**Option A - Simple:**
```bash
python api.py
```

**Option B - Using script:**
```bash
./start_api.sh
```

**Option C - Production mode:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 3. Test the API

Open browser to: **http://localhost:8000/docs**

Or test with command line:

```bash
# Quick test
python test_api_simple.py

# Or use cURL
curl http://localhost:8000/health
```

## 4. Transcribe Audio

### Using cURL

```bash
# Create a test recording first
python main.py record --duration 5

# Transcribe it
curl -X POST "http://localhost:8000/transcribe?language=english" \
  -F "file=@recording.wav"
```

### Using Python Client

```python
from api_client import ASRClient
from pathlib import Path

client = ASRClient()
result = client.transcribe(Path("recording.wav"), language="english")
print(result['transcription'])
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:8000/transcribe?language=english', {
  method: 'POST',
  body: formData
})
.then(r => r.json())
.then(data => console.log(data.transcription));
```

## 5. Explore API

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Endpoints:**
- `GET /health` - Health check
- `GET /models` - List models
- `GET /languages` - List languages
- `POST /transcribe` - Transcribe single file
- `POST /transcribe/batch` - Transcribe multiple files

## Docker Deployment

```bash
# Build image
docker build -t omnilingual-asr-api .

# Run container
docker run -p 8000:8000 omnilingual-asr-api

# Or use docker-compose
docker-compose up
```

## Configuration

Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Common Issues

**Port already in use:**
```bash
PORT=8080 python api.py
```

**Cannot connect:**
- Check firewall settings
- Ensure server is running
- Try http://127.0.0.1:8000 instead of localhost

**Model download slow:**
- First run downloads models (large files)
- Subsequent runs will be faster
- Models cached in ~/.cache/huggingface/

## Next Steps

- Read [API_README.md](API_README.md) for complete documentation
- Check [examples](api_client.py) for Python client usage
- Explore [interactive docs](http://localhost:8000/docs)
