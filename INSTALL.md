# Installation Guide - Omnilingual ASR

Complete installation guide for Ubuntu/Debian systems.

## System Requirements

- **OS**: Ubuntu 20.04+ / Debian 11+ (or WSL2)
- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **RAM**: 8GB minimum (16GB recommended for larger models)
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

## Quick Install (Ubuntu/Debian/WSL)

### 1. Install System Dependencies

```bash
# Update package list
sudo apt update

# Install Python 3 and development tools
sudo apt install -y python3 python3-pip python3-venv

# Install audio libraries
sudo apt install -y libsndfile1 portaudio19-dev

# Install Node.js (if not already installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3 --version  # Should be 3.10+
node --version     # Should be 18+
npm --version
```

### 2. Clone or Navigate to Project

```bash
cd ~/projects/meta_speech
```

### 3. Install Backend

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r api_requirements.txt
```

### 4. Install Frontend

```bash
cd frontend
npm install
cd ..
```

### 5. Test Installation

```bash
# Test backend
python3 api.py &
API_PID=$!
sleep 5
curl http://localhost:8000/health
kill $API_PID

# Test frontend
cd frontend
npm run build
cd ..
```

## Running the Application

### Option 1: Using Helper Scripts (Recommended)

```bash
# Terminal 1: Start Backend
./start_api.sh

# Terminal 2: Start Frontend
cd frontend && ./start_frontend.sh
```

### Option 2: Manual Start

```bash
# Terminal 1: Backend
source venv/bin/activate
python3 api.py

# Terminal 2: Frontend
cd frontend
npm start
```

### Option 3: Quick Run (if already installed)

```bash
# Backend only
./run_api.sh

# Frontend only
cd frontend && npm start
```

## Verification

### 1. Test Backend

```bash
# Health check
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","timestamp":"...","model_loaded":false,"current_model":null}
```

### 2. Test Frontend

Open browser: http://localhost:4202

Should see: "🎙️ Omnilingual ASR" interface

### 3. Test CORS

```bash
./test_cors.sh
```

Should show all green checkmarks.

## Common Issues

### Python Command Not Found

**Error:**
```
python: command not found
```

**Solution:**
```bash
# Use python3 instead
python3 api.py

# Or create alias (optional)
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc
```

### Permission Denied

**Error:**
```
Permission denied: ./start_api.sh
```

**Solution:**
```bash
chmod +x start_api.sh run_api.sh test_cors.sh
chmod +x frontend/start_frontend.sh
```

### Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find what's using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
PORT=8080 python3 api.py
```

### Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r api_requirements.txt
```

### Audio Libraries Missing

**Error:**
```
OSError: cannot load library 'libsndfile.so'
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install libsndfile1 portaudio19-dev

# macOS
brew install libsndfile portaudio

# Then reinstall Python packages
pip install --force-reinstall soundfile sounddevice
```

### NPM Install Fails

**Error:**
```
npm ERR! code EACCES
```

**Solution:**
```bash
# Fix npm permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Or use nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

## Platform-Specific Instructions

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 node libsndfile portaudio

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Continue with step 3 above
```

### Windows (WSL2)

```bash
# Install WSL2 and Ubuntu
wsl --install -d Ubuntu

# Inside WSL, follow Ubuntu instructions above

# Access from Windows:
# Backend: http://localhost:8000
# Frontend: http://localhost:4202
```

### Windows (Native)

```powershell
# Install Python 3.10+ from python.org
# Install Node.js from nodejs.org

# In PowerShell:
python -m venv venv
venv\Scripts\activate
pip install -r api_requirements.txt

# Frontend:
cd frontend
npm install
npm start
```

## GPU Support (Optional)

For faster transcription with NVIDIA GPU:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Production Setup

For production deployment:

```bash
# Install additional dependencies
pip install gunicorn

# Set environment variables
export WORKERS=4
export RELOAD=false

# Run with gunicorn
gunicorn api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Uninstall

```bash
# Remove virtual environment
rm -rf venv

# Remove node modules
rm -rf frontend/node_modules

# Remove generated files
rm -rf uploads
rm -rf frontend/dist
```

## Next Steps

After installation:

1. ✅ Read [START_GUIDE.md](START_GUIDE.md:1) for usage instructions
2. ✅ Read [PORTS_SUMMARY.md](PORTS_SUMMARY.md:1) for port configuration
3. ✅ Run `./test_cors.sh` to verify setup
4. ✅ Start transcribing! 🎤

## Support

If you encounter issues:

1. Check [Troubleshooting](#common-issues) section above
2. Review [START_GUIDE.md](START_GUIDE.md:1) troubleshooting
3. Check logs in terminal output
4. Verify all dependencies are installed

## Environment Variables

Optional configuration via `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit configuration
nano .env
```

Available variables: see [.env.example](.env.example:1)

---

**Installation complete?** Run `./test_cors.sh` to verify everything works! ✅
