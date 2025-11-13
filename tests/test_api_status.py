import pytest
from fastapi.testclient import TestClient

from api import app, session_manager

client = TestClient(app)

def test_get_status_for_session():
    # Create session and add chunks directly
    session_id = session_manager.create_session()
    session_manager.add_chunk(session_id, 0, "uploads/test1.wav", 1699123456.789)
    session_manager.add_chunk(session_id, 1, "uploads/test2.wav", 1699123457.789)
    session_manager.update_chunk_draft(session_id, 0, "Hello world")

    response = client.get(f"/api/transcribe/status/{session_id}")

    assert response.status_code == 200
    result = response.json()
    assert "session_id" in result
    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["chunk_index"] == 0
    assert result["chunks"][0]["draft_text"] == "Hello world"
    assert result["chunks"][0]["status"] == "draft"
    assert result["chunks"][1]["status"] == "pending"

def test_get_status_nonexistent_session():
    response = client.get("/api/transcribe/status/nonexistent-id")

    assert response.status_code == 404

def test_status_includes_all_chunk_fields():
    session_id = session_manager.create_session()
    session_manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)
    session_manager.update_chunk_draft(session_id, 0, "Draft text")
    session_manager.update_chunk_final(session_id, 0, "Final text")

    response = client.get(f"/api/transcribe/status/{session_id}")
    result = response.json()

    chunk = result["chunks"][0]
    assert "chunk_index" in chunk
    assert "draft_text" in chunk
    assert "final_text" in chunk
    assert "status" in chunk
    assert "timestamp" in chunk
    assert chunk["draft_text"] == "Draft text"
    assert chunk["final_text"] == "Final text"
    assert chunk["status"] == "final"
