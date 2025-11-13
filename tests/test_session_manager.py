import pytest
from session_manager import SessionManager, ChunkStatus

def test_create_new_session():
    manager = SessionManager()
    session_id = manager.create_session()

    assert session_id is not None
    assert len(session_id) == 36  # UUID format
    assert session_id in manager.sessions

def test_add_chunk_to_session():
    manager = SessionManager()
    session_id = manager.create_session()

    chunk_id = manager.add_chunk(
        session_id=session_id,
        chunk_index=0,
        audio_path="uploads/test.wav",
        timestamp=1699123456.789
    )

    assert chunk_id == 0
    chunk = manager.get_chunk(session_id, chunk_id)
    assert chunk["audio_path"] == "uploads/test.wav"
    assert chunk["status"] == ChunkStatus.PENDING
    assert chunk["draft_text"] is None
    assert chunk["final_text"] is None

def test_update_chunk_with_draft():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)

    manager.update_chunk_draft(session_id, 0, "Hello world")

    chunk = manager.get_chunk(session_id, 0)
    assert chunk["draft_text"] == "Hello world"
    assert chunk["status"] == ChunkStatus.DRAFT

def test_update_chunk_with_final():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)
    manager.update_chunk_draft(session_id, 0, "Hello world")

    manager.update_chunk_final(session_id, 0, "Hello, world!")

    chunk = manager.get_chunk(session_id, 0)
    assert chunk["final_text"] == "Hello, world!"
    assert chunk["status"] == ChunkStatus.FINAL

def test_get_all_chunks_for_session():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test1.wav", 1699123456.789)
    manager.add_chunk(session_id, 1, "uploads/test2.wav", 1699123457.789)

    chunks = manager.get_all_chunks(session_id)

    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1

def test_get_pending_chunks_for_finalization():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test1.wav", 1699123456.789)
    manager.update_chunk_draft(session_id, 0, "chunk 1")
    manager.add_chunk(session_id, 1, "uploads/test2.wav", 1699123457.789)
    manager.update_chunk_draft(session_id, 1, "chunk 2")

    pending = manager.get_chunks_for_finalization(session_id)

    assert len(pending) == 2
    assert all(c["status"] == ChunkStatus.DRAFT for c in pending)
