import pytest
import threading
import time
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


# Error case tests
def test_add_chunk_with_negative_index():
    manager = SessionManager()
    session_id = manager.create_session()

    with pytest.raises(ValueError, match="chunk_index must be non-negative"):
        manager.add_chunk(session_id, -1, "uploads/test.wav", 1699123456.789)


def test_add_chunk_with_empty_path():
    manager = SessionManager()
    session_id = manager.create_session()

    with pytest.raises(ValueError, match="audio_path cannot be empty"):
        manager.add_chunk(session_id, 0, "", 1699123456.789)

    with pytest.raises(ValueError, match="audio_path cannot be empty"):
        manager.add_chunk(session_id, 0, "   ", 1699123456.789)


def test_add_chunk_with_negative_timestamp():
    manager = SessionManager()
    session_id = manager.create_session()

    with pytest.raises(ValueError, match="timestamp must be positive"):
        manager.add_chunk(session_id, 0, "uploads/test.wav", -1.0)


def test_add_duplicate_chunk():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test1.wav", 1699123456.789)

    with pytest.raises(ValueError, match="Chunk 0 already exists"):
        manager.add_chunk(session_id, 0, "uploads/test2.wav", 1699123457.789)


def test_add_chunk_to_nonexistent_session():
    manager = SessionManager()

    with pytest.raises(ValueError, match="Session .* not found"):
        manager.add_chunk("nonexistent", 0, "uploads/test.wav", 1699123456.789)


def test_get_chunk_from_nonexistent_session():
    manager = SessionManager()

    with pytest.raises(ValueError, match="Session .* not found"):
        manager.get_chunk("nonexistent", 0)


def test_get_nonexistent_chunk():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)

    with pytest.raises(ValueError, match="Chunk 99 not found"):
        manager.get_chunk(session_id, 99)


def test_update_draft_on_nonexistent_chunk():
    manager = SessionManager()
    session_id = manager.create_session()

    with pytest.raises(ValueError, match="Chunk 0 not found"):
        manager.update_chunk_draft(session_id, 0, "test")


def test_update_final_on_nonexistent_chunk():
    manager = SessionManager()
    session_id = manager.create_session()

    with pytest.raises(ValueError, match="Chunk 0 not found"):
        manager.update_chunk_final(session_id, 0, "test")


def test_delete_session():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)

    manager.delete_session(session_id)

    assert session_id not in manager.sessions
    with pytest.raises(ValueError, match="Session .* not found"):
        manager.get_all_chunks(session_id)


def test_delete_nonexistent_session():
    manager = SessionManager()

    with pytest.raises(ValueError, match="Session .* not found"):
        manager.delete_session("nonexistent")


# Thread safety tests
def test_concurrent_chunk_additions():
    """Test that concurrent chunk additions are thread-safe."""
    manager = SessionManager()
    session_id = manager.create_session()
    errors = []

    def add_chunks(start_idx, count):
        try:
            for i in range(start_idx, start_idx + count):
                manager.add_chunk(session_id, i, f"uploads/test{i}.wav", float(i))
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        t = threading.Thread(target=add_chunks, args=(i * 10, 10))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors occurred: {errors}"
    chunks = manager.get_all_chunks(session_id)
    assert len(chunks) == 50


def test_concurrent_draft_updates():
    """Test that concurrent draft updates are thread-safe."""
    manager = SessionManager()
    session_id = manager.create_session()

    # Add chunks first
    for i in range(10):
        manager.add_chunk(session_id, i, f"uploads/test{i}.wav", float(i))

    errors = []

    def update_drafts(indices):
        try:
            for idx in indices:
                manager.update_chunk_draft(session_id, idx, f"draft_{idx}")
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        indices = list(range(i * 2, i * 2 + 2))
        t = threading.Thread(target=update_drafts, args=(indices,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify all drafts were updated
    chunks = manager.get_all_chunks(session_id)
    for chunk in chunks:
        assert chunk["draft_text"] == f"draft_{chunk['chunk_index']}"
        assert chunk["status"] == ChunkStatus.DRAFT


def test_chunk_ordering_preserved():
    """Test that chunks are returned in correct order regardless of insertion order."""
    manager = SessionManager()
    session_id = manager.create_session()

    # Add chunks in non-sequential order
    for idx in [5, 2, 8, 1, 9, 0, 3, 7, 4, 6]:
        manager.add_chunk(session_id, idx, f"uploads/test{idx}.wav", float(idx))

    chunks = manager.get_all_chunks(session_id)
    indices = [c["chunk_index"] for c in chunks]

    assert indices == list(range(10)), "Chunks should be sorted by chunk_index"
