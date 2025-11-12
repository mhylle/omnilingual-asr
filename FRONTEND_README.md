# Omnilingual ASR - Frontend Quick Start

Angular frontend for speech recognition with 1,600+ language support.

## 🚀 Quick Start

### 1. Start the Backend API

```bash
# In the project root
./start_api.sh
# or
python api.py
```

API should be running at: http://localhost:8000

### 2. Start the Frontend

```bash
cd frontend
npm install  # First time only
npm start
```

Frontend will be available at: http://localhost:4200

### 3. Use the App

1. **Click "Start Recording"** - Allow microphone access
2. **Speak** - Watch the duration timer (max 40 seconds)
3. **Click "Stop"** - When finished speaking
4. **Click "Transcribe"** - Send to API for transcription
5. **View Result** - Your speech converted to text!

## ✨ Key Features

### 🎤 Audio Recording
- Direct microphone access
- Pause/Resume functionality
- Real-time duration tracking
- Visual warnings at 35s and 40s

### 🌍 Multi-language Support
- English, Spanish, French, German, Italian
- Portuguese, Russian, Chinese, Japanese
- Korean, Arabic, Hindi
- Plus 1,588 more via API!

### 🤖 Multiple Models
- **CTC 1B** - Fast & Recommended (default)
- **CTC 300M** - Fastest option
- **CTC 3B** - Better accuracy
- **LLM 1B** - Language-aware
- **LLM 3B** - Best accuracy

### 📝 Smart UI
- Duration warnings prevent 40s timeout
- Real-time transcription processing
- Copy to clipboard
- Responsive design

## 📁 Project Structure

```
frontend/
├── src/app/
│   ├── services/
│   │   ├── asr-api.service.ts       # API communication
│   │   └── audio-recorder.service.ts # Recording logic
│   ├── app.ts                        # Main component
│   ├── app.html                      # Template
│   ├── app.scss                      # Styles
│   └── app.config.ts                 # Configuration
```

## ⚙️ Configuration

API URL is set in `src/app/services/asr-api.service.ts`:

```typescript
private readonly baseUrl = 'http://localhost:8000';
```

Change this if your API runs elsewhere.

## 🛠️ Development Commands

```bash
# Start dev server
npm start

# Build for production
npm run build

# Run tests
npm test

# Generate component
ng generate component my-component
```

## 🔧 Troubleshooting

### Microphone Not Working
- Check browser permissions
- Use HTTPS or localhost
- Try a different browser

### Cannot Connect to API
- Ensure API is running: `http://localhost:8000/health`
- Check CORS settings
- Verify firewall rules

### Build Errors
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Angular cache
ng cache clean
```

## 📱 Browser Compatibility

Requires modern browser with:
- MediaRecorder API
- getUserMedia API
- ES2022+

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14.1+
- Edge 90+

## 🎯 Usage Tips

1. **Stay Under 40s** - Recording auto-warns you
2. **Choose Right Model** - CTC 1B for speed, LLM 3B for accuracy
3. **Select Language** - Better results with correct language
4. **Good Microphone** - Quality matters for accuracy
5. **Quiet Environment** - Less background noise = better results

## 📊 Performance

- **Initial Load**: ~2-3 seconds
- **Recording Start**: <100ms
- **Transcription**:
  - CTC 1B: 0.5-2s (10s audio)
  - LLM 3B: 2-5s (10s audio)

## 🌐 Full Stack Setup

```bash
# Terminal 1: Start API
./start_api.sh

# Terminal 2: Start Frontend
cd frontend && npm start

# Open browser
# Frontend: http://localhost:4200
# API Docs: http://localhost:8000/docs
```

## 📚 Resources

- **Frontend Details**: [frontend/README.md](frontend/README.md:1)
- **API Documentation**: [API_README.md](API_README.md:1)
- **Quick Start**: [QUICKSTART_API.md](QUICKSTART_API.md:1)
- **Main README**: [README.md](README.md:1)

## 🎨 Tech Stack

- **Framework**: Angular 19+ (standalone components)
- **State**: Signals (zoneless change detection)
- **HTTP**: Angular HttpClient
- **Audio**: MediaRecorder API
- **Styling**: SCSS with CSS custom properties

## 🔐 Security Notes

- Microphone access requires user permission
- Audio data sent to backend only on user action
- No automatic recording or transmission
- Recordings not saved unless explicitly requested

---

**Ready to try it?** Start the API and frontend, then speak into your microphone! 🎤
