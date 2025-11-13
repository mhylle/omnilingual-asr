import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import io

from api import app

client = TestClient(app)

def test_upload_chunk_creates_session():
    # Create fake audio file
    audio_data = b"RIFF" + b"\x00" * 40  # Minimal WAV header
    files = {"file": ("test.wav", io.BytesIO(audio_data), "audio/wav")}
    data = {
        "chunk_index": 0,
        "timestamp": 1699123456.789
    }

    response = client.post("/api/transcribe/chunk", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert "session_id" in result
    assert "chunk_id" in result
    assert result["chunk_id"] == 0
    assert result["status"] == "pending"

def test_upload_chunk_to_existing_session():
    # First chunk
    audio_data = b"RIFF" + b"\x00" * 40
    files = {"file": ("test1.wav", io.BytesIO(audio_data), "audio/wav")}
    data = {"chunk_index": 0, "timestamp": 1699123456.789}
    response1 = client.post("/api/transcribe/chunk", files=files, data=data)
    session_id = response1.json()["session_id"]

    # Second chunk to same session
    files = {"file": ("test2.wav", io.BytesIO(audio_data), "audio/wav")}
    data = {"session_id": session_id, "chunk_index": 1, "timestamp": 1699123457.789}
    response2 = client.post("/api/transcribe/chunk", files=files, data=data)

    assert response2.status_code == 200
    result = response2.json()
    assert result["session_id"] == session_id
    assert result["chunk_id"] == 1

def test_upload_chunk_saves_file():
    audio_data = b"RIFF" + b"\x00" * 40
    files = {"file": ("test.wav", io.BytesIO(audio_data), "audio/wav")}
    data = {"chunk_index": 0, "timestamp": 1699123456.789}

    response = client.post("/api/transcribe/chunk", files=files, data=data)
    result = response.json()

    # Verify file was saved
    audio_path = Path(result["audio_path"])
    assert audio_path.exists()

    # Cleanup
    audio_path.unlink()
