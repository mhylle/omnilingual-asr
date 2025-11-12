# 🚀 Quick Start Guide - Omnilingual ASR

Complete guide to get the full stack running with CORS properly configured.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Microphone access

## Step-by-Step Setup

### 1. Start the Backend API

**Terminal 1:**
```bash
# From project root
./start_api.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r api_requirements.txt
python api.py
```

**Verify backend is running:**
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": false,
  "current_model": null
}
```

### 2. Start the Frontend

**Terminal 2:**
```bash
cd frontend
npm install  # First time only
npm start
```

**Frontend will be available at:**
- 🌐 **http://localhost:4202**

### 3. Test CORS Configuration

**Terminal 3:**
```bash
./test_cors.sh
```

This will verify:
- ✅ Backend is running
- ✅ CORS headers are correct
- ✅ Port 4202 is allowed
- ✅ All API endpoints work

### 4. Use the Application

1. **Open browser**: http://localhost:4202
2. **Allow microphone** when prompted
3. **Click "Start Recording"**
4. **Speak** (max 40 seconds)
5. **Click "Stop"**
6. **Click "Transcribe"**
7. **View result!**

## Ports Configuration

| Service | Port | URL |
|---------|------|-----|
| Backend API | 8000 | http://localhost:8000 |
| API Docs | 8000 | http://localhost:8000/docs |
| Frontend | 4202 | http://localhost:4202 |

## CORS Configuration

The backend is configured to allow requests from:
- ✅ `http://localhost:4202` (primary)
- ✅ `http://127.0.0.1:4202`
- ✅ `http://localhost:4200` (backward compatibility)
- ✅ `http://127.0.0.1:4200`

### Changing CORS Settings

Edit `config.py`:
```python
cors_origins: list = [
    "http://localhost:4202",
    "http://127.0.0.1:4202",
    # Add your custom origins here
]
```

Or use `.env` file:
```bash
CORS_ORIGINS=["http://localhost:4202", "http://127.0.0.1:4202"]
```

## Troubleshooting

### CORS Errors in Browser Console

**Error:** `Access to fetch at 'http://localhost:8000/transcribe' from origin 'http://localhost:4202' has been blocked by CORS policy`

**Solution:**
1. Check backend is running: `curl http://localhost:8000/health`
2. Run CORS test: `./test_cors.sh`
3. Verify `config.py` includes your frontend URL
4. Restart backend API

### Frontend Can't Connect to Backend

**Symptoms:**
- Network errors in browser console
- API calls timing out

**Solutions:**
1. **Check backend is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check frontend API URL:**
   - File: `frontend/src/app/services/asr-api.service.ts`
   - Should be: `private readonly baseUrl = 'http://localhost:8000';`

3. **Check browser console:**
   - F12 → Console tab
   - Look for error messages

### Microphone Not Working

**Solutions:**
1. Check browser permissions (click lock icon in address bar)
2. Use HTTPS or localhost (required for microphone access)
3. Try different browser
4. Check system microphone settings

### Port Already in Use

**Backend (port 8000):**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
PORT=8080 python api.py
```

**Frontend (port 4202):**
```bash
# Find process using port
lsof -i :4202

# Kill process
kill -9 <PID>

# Or edit angular.json to use different port
```

## Testing the Full Stack

### 1. Test Backend Only

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# List languages
curl http://localhost:8000/languages
```

### 2. Test with Audio File

```bash
# Record test audio
python main.py record --duration 5

# Transcribe via API
curl -X POST "http://localhost:8000/transcribe?language=english" \
  -F "file=@recording.wav"
```

### 3. Test Frontend

1. Open: http://localhost:4202
2. Check browser console (F12)
3. Click "Start Recording"
4. Allow microphone
5. Speak for a few seconds
6. Click "Stop"
7. Click "Transcribe"
8. Check browser Network tab for API calls

## Verification Checklist

Before reporting issues, verify:

- [ ] Backend running on port 8000
- [ ] Frontend running on port 4202
- [ ] CORS test passes (`./test_cors.sh`)
- [ ] Browser console shows no errors
- [ ] Microphone permissions granted
- [ ] Network tab shows successful API calls
- [ ] `/health` endpoint returns 200 OK

## Development Tips

### Watch Logs

**Backend:**
```bash
# See API logs in terminal where you started python api.py
```

**Frontend:**
```bash
# Angular dev server logs in terminal where you ran npm start
# Browser console (F12) for client-side errors
```

### Hot Reload

Both services support hot reload:
- **Backend**: Restart required for config changes
- **Frontend**: Auto-reloads on file changes

### Testing Different Languages

```javascript
// In browser console
// Test Spanish
fetch('http://localhost:8000/transcribe?language=spanish&model=llm_1b', {
  method: 'POST',
  body: formData
});
```

## Production Deployment

For production, you'll need:
1. HTTPS for microphone access
2. Proper CORS configuration
3. Environment variables
4. Process manager (PM2, systemd)

See [API_README.md](API_README.md:1) for production deployment details.

## Next Steps

- 📖 Read [API Documentation](API_README.md:1)
- 🎨 Read [Frontend Documentation](frontend/README.md:1)
- 🔧 Explore [Configuration Options](config.py:1)
- 🧪 Run [API Tests](test_api_simple.py:1)

## Quick Command Reference

```bash
# Start everything
./start_api.sh                    # Terminal 1
cd frontend && npm start           # Terminal 2
./test_cors.sh                     # Terminal 3 (optional)

# Stop everything
Ctrl+C                             # In each terminal

# Restart backend
Ctrl+C && python api.py

# Restart frontend
Ctrl+C && npm start

# Test CORS
./test_cors.sh

# Test API
python test_api_simple.py
```

## Getting Help

1. Check browser console (F12)
2. Check backend terminal logs
3. Run `./test_cors.sh`
4. Review [Troubleshooting](#troubleshooting) section
5. Check [API_README.md](API_README.md:1) for API issues
6. Check [frontend/README.md](frontend/README.md:1) for frontend issues

---

**Ready?** Open two terminals and let's go! 🚀
