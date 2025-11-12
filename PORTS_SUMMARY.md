# Ports Summary

Quick reference for all ports used in the Omnilingual ASR project.

## Active Ports

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Backend API** | 8000 | http://localhost:8000 | FastAPI REST API |
| **API Documentation** | 8000 | http://localhost:8000/docs | Swagger UI |
| **API Redoc** | 8000 | http://localhost:8000/redoc | ReDoc API docs |
| **Frontend** | 4202 | http://localhost:4202 | Angular application |

## CORS Configuration

The backend API allows requests from:
- `http://localhost:4202` ✅ (primary frontend port)
- `http://127.0.0.1:4202` ✅
- `http://localhost:4200` ✅ (backward compatibility)
- `http://127.0.0.1:4200` ✅

## Quick Start

```bash
# Terminal 1: Start Backend
./start_api.sh

# Terminal 2: Start Frontend
cd frontend && ./start_frontend.sh

# Terminal 3: Test CORS
./test_cors.sh
```

## Port Configuration Files

| Service | Configuration File | Port Setting |
|---------|-------------------|--------------|
| Backend | `config.py` | `port: int = 8000` |
| Backend | `.env` | `PORT=8000` |
| Frontend | `angular.json` | `"port": 4202` |
| CORS | `config.py` | `cors_origins: list = [...]` |

## Changing Ports

### Change Backend Port

**Option 1 - Environment Variable:**
```bash
PORT=8080 python api.py
```

**Option 2 - Edit config.py:**
```python
port: int = 8080
```

**Option 3 - Edit .env:**
```
PORT=8080
```

### Change Frontend Port

**Edit angular.json:**
```json
"serve": {
  "builder": "@angular/build:dev-server",
  "options": {
    "port": 4203
  },
  ...
}
```

**Don't forget to update CORS in config.py!**

## Firewall Rules

If you're exposing the API externally:

```bash
# Allow port 8000
sudo ufw allow 8000/tcp

# Allow port 4202
sudo ufw allow 4202/tcp
```

## Docker Port Mapping

If using Docker:

```bash
# Backend
docker run -p 8000:8000 omnilingual-asr-api

# With custom port
docker run -p 8080:8000 omnilingual-asr-api
```

See `docker-compose.yml` for container networking.

## Production Ports

For production deployment, consider:
- Backend: Port 80/443 (HTTPS)
- Frontend: Served via Nginx/Apache
- Use reverse proxy for SSL/TLS

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8000
lsof -i :4202

# Kill the process
kill -9 <PID>

# Or change the port (see above)
```

### CORS Issues

1. Ensure frontend port matches CORS config
2. Run `./test_cors.sh` to verify
3. Check `config.py` cors_origins list
4. Restart backend after changes

### Cannot Access from Another Machine

1. Check firewall rules
2. Ensure backend uses `0.0.0.0` not `127.0.0.1`
3. Add remote IP to CORS origins
4. Consider security implications

## Security Notes

- Only expose ports you need
- Use HTTPS in production
- Restrict CORS to known origins
- Keep ports behind firewall when possible
- Use environment variables for sensitive config

---

**Quick Test:** `./test_cors.sh` verifies all ports and CORS are working correctly.
