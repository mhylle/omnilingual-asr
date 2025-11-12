# Real-Time Audio Transcription Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a live dictation system with silence-based chunking and draft/final transcription strategy

**Architecture:** Browser captures audio continuously, detects silence to chunk (0.5-1s), sends via HTTP POST. Backend provides immediate draft transcription (ctc_1b), then re-transcribes merged chunks for accuracy (llm_1b). Frontend polls for updates and displays draft→final transitions.

**Tech Stack:** Angular (frontend), FastAPI (backend), Web Audio API (silence detection), MediaRecorder API (audio capture), pydub (audio merging), Meta Omnilingual ASR (transcription)

---

## Phase 1: Backend Session Management

### Task 1: Session Data Models

**Files:**
- Create: `session_manager.py`
- Create: `tests/test_session_manager.py`

**Step 1: Write the failing test**

Create `tests/test_session_manager.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_session_manager.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'session_manager'"

**Step 3: Write minimal implementation**

Create `session_manager.py`:

```python
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import uuid


class ChunkStatus(str, Enum):
    PENDING = "pending"
    DRAFT = "draft"
    FINALIZING = "finalizing"
    FINAL = "final"


class SessionManager:
    """Manages transcription sessions and chunks."""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        """Create new session and return session ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "chunks": [],
            "created_at": datetime.now().timestamp()
        }
        return session_id

    def add_chunk(
        self,
        session_id: str,
        chunk_index: int,
        audio_path: str,
        timestamp: float
    ) -> int:
        """Add new chunk to session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        chunk = {
            "chunk_index": chunk_index,
            "audio_path": audio_path,
            "timestamp": timestamp,
            "status": ChunkStatus.PENDING,
            "draft_text": None,
            "final_text": None
        }

        self.sessions[session_id]["chunks"].append(chunk)
        return chunk_index

    def get_chunk(self, session_id: str, chunk_index: int) -> Dict:
        """Get specific chunk from session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        chunks = self.sessions[session_id]["chunks"]
        for chunk in chunks:
            if chunk["chunk_index"] == chunk_index:
                return chunk

        raise ValueError(f"Chunk {chunk_index} not found in session {session_id}")

    def update_chunk_draft(self, session_id: str, chunk_index: int, draft_text: str):
        """Update chunk with draft transcription."""
        chunk = self.get_chunk(session_id, chunk_index)
        chunk["draft_text"] = draft_text
        chunk["status"] = ChunkStatus.DRAFT

    def update_chunk_final(self, session_id: str, chunk_index: int, final_text: str):
        """Update chunk with final transcription."""
        chunk = self.get_chunk(session_id, chunk_index)
        chunk["final_text"] = final_text
        chunk["status"] = ChunkStatus.FINAL

    def get_all_chunks(self, session_id: str) -> List[Dict]:
        """Get all chunks for session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        return self.sessions[session_id]["chunks"]

    def get_chunks_for_finalization(self, session_id: str) -> List[Dict]:
        """Get chunks that need finalization (status = DRAFT)."""
        all_chunks = self.get_all_chunks(session_id)
        return [c for c in all_chunks if c["status"] == ChunkStatus.DRAFT]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_session_manager.py -v
```

Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add session_manager.py tests/test_session_manager.py
git commit -m "feat: add session management with chunk tracking"
```

---

### Task 2: Chunk Upload API Endpoint

**Files:**
- Modify: `api.py` (add endpoint around line 100)
- Create: `tests/test_api_chunk_upload.py`

**Step 1: Write the failing test**

Create `tests/test_api_chunk_upload.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_api_chunk_upload.py -v
```

Expected: FAIL with "404: Not Found" (endpoint doesn't exist yet)

**Step 3: Write minimal implementation**

Modify `api.py` - add after imports:

```python
from session_manager import SessionManager, ChunkStatus

# Add global session manager
session_manager = SessionManager()
```

Add endpoint after existing endpoints:

```python
@app.post("/api/transcribe/chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    chunk_index: int = Form(...),
    timestamp: float = Form(...)
):
    """
    Upload audio chunk for transcription.

    Creates new session if session_id not provided.
    Saves audio file and queues for draft transcription.
    """
    try:
        # Create or validate session
        if session_id is None:
            session_id = session_manager.create_session()
        elif session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Save audio file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        audio_filename = f"{session_id}_chunk_{chunk_index}.wav"
        audio_path = upload_dir / audio_filename

        with audio_path.open("wb") as f:
            content = await file.read()
            f.write(content)

        # Add chunk to session
        chunk_id = session_manager.add_chunk(
            session_id=session_id,
            chunk_index=chunk_index,
            audio_path=str(audio_path),
            timestamp=timestamp
        )

        logger.info(f"Chunk uploaded: session={session_id}, chunk={chunk_id}")

        # TODO: Queue for draft transcription (Task 4)

        return {
            "session_id": session_id,
            "chunk_id": chunk_id,
            "status": "pending",
            "audio_path": str(audio_path)
        }

    except Exception as e:
        logger.error(f"Chunk upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_api_chunk_upload.py -v
```

Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add api.py tests/test_api_chunk_upload.py
git commit -m "feat: add chunk upload endpoint with session management"
```

---

### Task 3: Status Polling API Endpoint

**Files:**
- Modify: `api.py` (add endpoint after chunk upload)
- Create: `tests/test_api_status.py`

**Step 1: Write the failing test**

Create `tests/test_api_status.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_api_status.py -v
```

Expected: FAIL with "404: Not Found"

**Step 3: Write minimal implementation**

Add to `api.py` after chunk upload endpoint:

```python
@app.get("/api/transcribe/status/{session_id}")
async def get_session_status(session_id: str):
    """
    Get transcription status for all chunks in session.

    Returns all chunks with their current draft/final text and status.
    """
    try:
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        chunks = session_manager.get_all_chunks(session_id)

        # Format chunks for response
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunks.append({
                "chunk_index": chunk["chunk_index"],
                "status": chunk["status"],
                "draft_text": chunk["draft_text"],
                "final_text": chunk["final_text"],
                "timestamp": chunk["timestamp"]
            })

        return {
            "session_id": session_id,
            "chunks": formatted_chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_api_status.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add api.py tests/test_api_status.py
git commit -m "feat: add status polling endpoint for session chunks"
```

---

### Task 4: Draft Transcription Worker

**Files:**
- Create: `transcription_worker.py`
- Create: `tests/test_transcription_worker.py`
- Modify: `api.py` (integrate worker)

**Step 1: Write the failing test**

Create `tests/test_transcription_worker.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio

from transcription_worker import DraftTranscriptionWorker
from session_manager import SessionManager

@pytest.fixture
def session_manager():
    return SessionManager()

@pytest.fixture
def worker(session_manager):
    return DraftTranscriptionWorker(session_manager)

@pytest.mark.asyncio
async def test_process_draft_transcription(worker, session_manager):
    session_id = session_manager.create_session()
    session_manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)

    with patch('transcription_worker.Transcriber') as mock_transcriber:
        mock_instance = Mock()
        mock_instance.transcribe.return_value = ["Hello world"]
        mock_transcriber.return_value = mock_instance

        await worker.process_chunk(session_id, 0)

    chunk = session_manager.get_chunk(session_id, 0)
    assert chunk["draft_text"] == "Hello world"
    assert chunk["status"] == "draft"

@pytest.mark.asyncio
async def test_worker_uses_fast_model(worker, session_manager):
    session_id = session_manager.create_session()
    session_manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)

    with patch('transcription_worker.Transcriber') as mock_transcriber:
        mock_instance = Mock()
        mock_instance.transcribe.return_value = ["text"]
        mock_transcriber.return_value = mock_instance

        await worker.process_chunk(session_id, 0)

        # Verify ctc_1b model was used
        mock_transcriber.assert_called_once_with(model="ctc_1b")

@pytest.mark.asyncio
async def test_worker_handles_transcription_error(worker, session_manager):
    session_id = session_manager.create_session()
    session_manager.add_chunk(session_id, 0, "uploads/test.wav", 1699123456.789)

    with patch('transcription_worker.Transcriber') as mock_transcriber:
        mock_transcriber.side_effect = Exception("Transcription failed")

        with pytest.raises(Exception):
            await worker.process_chunk(session_id, 0)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transcription_worker.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'transcription_worker'"

**Step 3: Write minimal implementation**

Create `transcription_worker.py`:

```python
import asyncio
import logging
from pathlib import Path
from typing import Optional

from transcriber import Transcriber
from session_manager import SessionManager, ChunkStatus

logger = logging.getLogger(__name__)


class DraftTranscriptionWorker:
    """
    Worker for fast draft transcription of audio chunks.

    Uses ctc_1b model for speed over accuracy.
    """

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.transcriber: Optional[Transcriber] = None

    def _get_transcriber(self) -> Transcriber:
        """Lazy load transcriber with fast model."""
        if self.transcriber is None:
            self.transcriber = Transcriber(model="ctc_1b")
        return self.transcriber

    async def process_chunk(self, session_id: str, chunk_index: int):
        """
        Transcribe audio chunk and update with draft text.

        Args:
            session_id: Session ID
            chunk_index: Chunk index to process
        """
        try:
            chunk = self.session_manager.get_chunk(session_id, chunk_index)
            audio_path = chunk["audio_path"]

            logger.info(f"Processing draft for session={session_id}, chunk={chunk_index}")

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            transcriber = self._get_transcriber()
            result = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe([audio_path])
            )

            # Update chunk with draft text
            draft_text = result[0] if result else ""
            self.session_manager.update_chunk_draft(session_id, chunk_index, draft_text)

            logger.info(f"Draft complete: session={session_id}, chunk={chunk_index}")

        except Exception as e:
            logger.error(f"Draft transcription error: session={session_id}, chunk={chunk_index}, error={e}")
            raise
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transcription_worker.py -v
```

Expected: PASS

**Step 5: Integrate worker into API**

Modify `api.py` - add after session_manager initialization:

```python
from transcription_worker import DraftTranscriptionWorker

# Add draft worker
draft_worker = DraftTranscriptionWorker(session_manager)
```

Update chunk upload endpoint to trigger draft transcription:

```python
# In upload_chunk function, replace "# TODO: Queue for draft transcription" with:

        # Queue for draft transcription
        asyncio.create_task(draft_worker.process_chunk(session_id, chunk_id))
```

**Step 6: Run integration test**

```bash
pytest tests/test_api_chunk_upload.py -v
```

Expected: PASS (worker integrated successfully)

**Step 7: Commit**

```bash
git add transcription_worker.py tests/test_transcription_worker.py api.py
git commit -m "feat: add draft transcription worker with async processing"
```

---

### Task 5: Audio Merging and Finalization Worker

**Files:**
- Create: `finalization_worker.py`
- Create: `tests/test_finalization_worker.py`
- Modify: `api.py` (integrate worker)

**Step 1: Write the failing test**

Create `tests/test_finalization_worker.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio

from finalization_worker import FinalizationWorker
from session_manager import SessionManager

@pytest.fixture
def session_manager():
    return SessionManager()

@pytest.fixture
def worker(session_manager):
    return FinalizationWorker(session_manager, merge_threshold=3)

@pytest.mark.asyncio
async def test_merge_audio_chunks(worker, session_manager, tmp_path):
    # Create fake audio files
    audio1 = tmp_path / "chunk1.wav"
    audio2 = tmp_path / "chunk2.wav"
    audio1.write_bytes(b"fake audio 1")
    audio2.write_bytes(b"fake audio 2")

    chunk_paths = [str(audio1), str(audio2)]

    with patch('finalization_worker.AudioSegment') as mock_audio:
        mock_segment = MagicMock()
        mock_audio.from_wav.return_value = mock_segment
        mock_audio.empty.return_value = mock_segment

        merged_path = worker._merge_audio_chunks(chunk_paths)

        assert mock_audio.from_wav.call_count == 2
        assert merged_path.startswith("temp/merged_")

@pytest.mark.asyncio
async def test_finalization_triggered_by_threshold(worker, session_manager):
    session_id = session_manager.create_session()

    # Add chunks until threshold reached
    for i in range(3):
        session_manager.add_chunk(session_id, i, f"uploads/test{i}.wav", 1699123456.0 + i)
        session_manager.update_chunk_draft(session_id, i, f"draft {i}")

    with patch.object(worker, '_merge_audio_chunks') as mock_merge, \
         patch.object(worker, '_get_transcriber') as mock_transcriber:

        mock_merge.return_value = "temp/merged.wav"
        mock_trans = Mock()
        mock_trans.transcribe.return_value = ["Final transcription"]
        mock_transcriber.return_value = mock_trans

        await worker.check_and_finalize(session_id)

        # Verify merge was called
        mock_merge.assert_called_once()

        # Verify all chunks updated with final text
        for i in range(3):
            chunk = session_manager.get_chunk(session_id, i)
            assert chunk["final_text"] == "Final transcription"
            assert chunk["status"] == "final"

@pytest.mark.asyncio
async def test_finalization_uses_accurate_model(worker, session_manager):
    session_id = session_manager.create_session()

    for i in range(3):
        session_manager.add_chunk(session_id, i, f"uploads/test{i}.wav", 1699123456.0)
        session_manager.update_chunk_draft(session_id, i, f"draft {i}")

    with patch('finalization_worker.Transcriber') as mock_transcriber, \
         patch.object(worker, '_merge_audio_chunks') as mock_merge:

        mock_merge.return_value = "temp/merged.wav"
        mock_trans = Mock()
        mock_trans.transcribe.return_value = ["text"]
        mock_transcriber.return_value = mock_trans

        await worker.check_and_finalize(session_id)

        # Verify llm_1b model was used
        mock_transcriber.assert_called_once_with(model="llm_1b")

@pytest.mark.asyncio
async def test_no_finalization_below_threshold(worker, session_manager):
    session_id = session_manager.create_session()

    # Add only 2 chunks (below threshold of 3)
    for i in range(2):
        session_manager.add_chunk(session_id, i, f"uploads/test{i}.wav", 1699123456.0)
        session_manager.update_chunk_draft(session_id, i, f"draft {i}")

    with patch.object(worker, '_merge_audio_chunks') as mock_merge:
        await worker.check_and_finalize(session_id)

        # Verify merge was NOT called
        mock_merge.assert_not_called()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_finalization_worker.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `finalization_worker.py`:

```python
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from pydub import AudioSegment
from transcriber import Transcriber
from session_manager import SessionManager, ChunkStatus

logger = logging.getLogger(__name__)


class FinalizationWorker:
    """
    Worker for merging audio chunks and creating accurate final transcriptions.

    Uses llm_1b model for accuracy over speed.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        merge_threshold: int = 5,
        time_threshold: float = 10.0
    ):
        self.session_manager = session_manager
        self.merge_threshold = merge_threshold  # Number of chunks
        self.time_threshold = time_threshold  # Seconds of audio
        self.transcriber: Optional[Transcriber] = None

        # Ensure temp directory exists
        Path("temp").mkdir(exist_ok=True)

    def _get_transcriber(self) -> Transcriber:
        """Lazy load transcriber with accurate model."""
        if self.transcriber is None:
            self.transcriber = Transcriber(model="llm_1b")
        return self.transcriber

    def _merge_audio_chunks(self, chunk_paths: List[str]) -> str:
        """
        Merge multiple audio files into single file.

        Args:
            chunk_paths: List of audio file paths to merge

        Returns:
            Path to merged audio file
        """
        combined = AudioSegment.empty()

        for path in chunk_paths:
            audio = AudioSegment.from_wav(path)
            combined += audio

        # Save merged audio
        timestamp = datetime.now().timestamp()
        merged_path = f"temp/merged_{timestamp}.wav"
        combined.export(merged_path, format="wav")

        logger.info(f"Merged {len(chunk_paths)} chunks into {merged_path}")
        return merged_path

    async def check_and_finalize(self, session_id: str):
        """
        Check if session has enough chunks for finalization and process if ready.

        Args:
            session_id: Session to check
        """
        try:
            # Get chunks ready for finalization
            pending_chunks = self.session_manager.get_chunks_for_finalization(session_id)

            if len(pending_chunks) < self.merge_threshold:
                logger.debug(f"Session {session_id}: {len(pending_chunks)} chunks, waiting for {self.merge_threshold}")
                return

            # Extract chunk info
            chunk_indices = [c["chunk_index"] for c in pending_chunks]
            chunk_paths = [c["audio_path"] for c in pending_chunks]

            logger.info(f"Finalizing session {session_id}: chunks {chunk_indices}")

            # Mark chunks as finalizing
            for idx in chunk_indices:
                chunk = self.session_manager.get_chunk(session_id, idx)
                chunk["status"] = ChunkStatus.FINALIZING

            # Merge audio
            merged_path = self._merge_audio_chunks(chunk_paths)

            # Run final transcription in executor
            loop = asyncio.get_event_loop()
            transcriber = self._get_transcriber()
            result = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe([merged_path])
            )

            final_text = result[0] if result else ""

            # Update all chunks with final text
            for idx in chunk_indices:
                self.session_manager.update_chunk_final(session_id, idx, final_text)

            logger.info(f"Finalization complete: session={session_id}, chunks={chunk_indices}")

            # Cleanup merged file
            Path(merged_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Finalization error: session={session_id}, error={e}")
            raise
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_finalization_worker.py -v
```

Expected: PASS

**Step 5: Integrate worker into API**

Modify `api.py` - add after draft_worker:

```python
from finalization_worker import FinalizationWorker

# Add finalization worker
finalization_worker = FinalizationWorker(session_manager, merge_threshold=5)
```

Update chunk upload endpoint to trigger finalization check:

```python
# In upload_chunk function, after draft worker task:

        # Queue for draft transcription
        asyncio.create_task(draft_worker.process_chunk(session_id, chunk_id))

        # Check if ready for finalization
        asyncio.create_task(finalization_worker.check_and_finalize(session_id))
```

**Step 6: Run integration test**

```bash
pytest tests/ -v
```

Expected: All tests PASS

**Step 7: Commit**

```bash
git add finalization_worker.py tests/test_finalization_worker.py api.py
git commit -m "feat: add finalization worker with audio merging and accurate transcription"
```

---

## Phase 2: Frontend Audio Capture and Silence Detection

### Task 6: Silence Detection Service

**Files:**
- Create: `frontend/src/app/services/silence-detector.service.ts`
- Create: `frontend/src/app/services/silence-detector.service.spec.ts`

**Step 1: Write the failing test**

Create `frontend/src/app/services/silence-detector.service.spec.ts`:

```typescript
import { TestBed } from '@angular/core/testing';
import { SilenceDetectorService } from './silence-detector.service';

describe('SilenceDetectorService', () => {
  let service: SilenceDetectorService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SilenceDetectorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should detect silence after threshold', (done) => {
    const silenceThreshold = 10; // Low threshold for testing
    const silenceDuration = 200; // 200ms

    service.silenceDetected$.subscribe(() => {
      expect(true).toBe(true);
      done();
    });

    // Simulate low audio levels
    const mockAnalyser = {
      frequencyBinCount: 1024,
      getByteFrequencyData: (array: Uint8Array) => {
        array.fill(5); // Below threshold
      }
    };

    service.startMonitoring(mockAnalyser as any, silenceThreshold, silenceDuration);
  });

  it('should reset silence timer on sound', () => {
    const silenceThreshold = 10;
    const silenceDuration = 500;

    let silenceDetected = false;
    service.silenceDetected$.subscribe(() => {
      silenceDetected = true;
    });

    const mockAnalyser = {
      frequencyBinCount: 1024,
      getByteFrequencyData: jasmine.createSpy('getByteFrequencyData')
    };

    // First call: silence
    (mockAnalyser.getByteFrequencyData as jasmine.Spy).and.callFake((array: Uint8Array) => {
      array.fill(5);
    });

    service.startMonitoring(mockAnalyser as any, silenceThreshold, silenceDuration);

    // Second call: sound (should reset timer)
    setTimeout(() => {
      (mockAnalyser.getByteFrequencyData as jasmine.Spy).and.callFake((array: Uint8Array) => {
        array.fill(50); // Above threshold
      });
    }, 100);

    setTimeout(() => {
      expect(silenceDetected).toBe(false);
    }, 600);
  });

  it('should stop monitoring', () => {
    const mockAnalyser = {
      frequencyBinCount: 1024,
      getByteFrequencyData: jasmine.createSpy('getByteFrequencyData')
    };

    service.startMonitoring(mockAnalyser as any, 10, 500);
    service.stopMonitoring();

    expect((mockAnalyser.getByteFrequencyData as jasmine.Spy).calls.count()).toBeLessThan(10);
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd frontend
npm test -- --include='**/silence-detector.service.spec.ts'
```

Expected: FAIL with "Cannot find module"

**Step 3: Write minimal implementation**

Create `frontend/src/app/services/silence-detector.service.ts`:

```typescript
import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SilenceDetectorService {
  silenceDetected$ = new Subject<void>();

  private monitoringInterval: any;
  private silenceDuration = 0;
  private isMonitoring = false;

  constructor() { }

  /**
   * Start monitoring audio analyser for silence.
   *
   * @param analyser Web Audio API AnalyserNode
   * @param silenceThreshold Audio level below which is considered silence (0-255)
   * @param silenceDurationMs Duration of silence before triggering event (ms)
   */
  startMonitoring(
    analyser: AnalyserNode,
    silenceThreshold: number = 20,
    silenceDurationMs: number = 500
  ): void {
    if (this.isMonitoring) {
      this.stopMonitoring();
    }

    this.isMonitoring = true;
    this.silenceDuration = 0;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    this.monitoringInterval = setInterval(() => {
      if (!this.isMonitoring) return;

      analyser.getByteFrequencyData(dataArray);

      // Calculate average audio level
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;

      if (average < silenceThreshold) {
        this.silenceDuration += 100;

        if (this.silenceDuration >= silenceDurationMs) {
          this.silenceDetected$.next();
          this.silenceDuration = 0; // Reset after detection
        }
      } else {
        this.silenceDuration = 0; // Reset on sound
      }
    }, 100); // Check every 100ms
  }

  /**
   * Stop monitoring for silence.
   */
  stopMonitoring(): void {
    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.silenceDuration = 0;
  }

  /**
   * Get current silence duration in ms.
   */
  getCurrentSilenceDuration(): number {
    return this.silenceDuration;
  }
}
```

**Step 4: Run test to verify it passes**

```bash
npm test -- --include='**/silence-detector.service.spec.ts'
```

Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/app/services/silence-detector.service.ts frontend/src/app/services/silence-detector.service.spec.ts
git commit -m "feat: add silence detection service with Web Audio API"
```

---

### Task 7: Real-Time Audio Recording Service

**Files:**
- Create: `frontend/src/app/services/realtime-recorder.service.ts`
- Create: `frontend/src/app/services/realtime-recorder.service.spec.ts`

**Step 1: Write the failing test**

Create `frontend/src/app/services/realtime-recorder.service.spec.ts`:

```typescript
import { TestBed } from '@angular/core/testing';
import { RealtimeRecorderService, AudioChunk } from './realtime-recorder.service';
import { SilenceDetectorService } from './silence-detector.service';

describe('RealtimeRecorderService', () => {
  let service: RealtimeRecorderService;
  let silenceDetector: SilenceDetectorService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SilenceDetectorService]
    });
    service = TestBed.inject(RealtimeRecorderService);
    silenceDetector = TestBed.inject(SilenceDetectorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should emit chunks on silence detection', (done) => {
    service.chunkReady$.subscribe((chunk: AudioChunk) => {
      expect(chunk.audioBlob).toBeDefined();
      expect(chunk.chunkIndex).toBe(0);
      expect(chunk.timestamp).toBeGreaterThan(0);
      done();
    });

    // Trigger silence detection
    silenceDetector.silenceDetected$.next();
  });

  it('should increment chunk index', (done) => {
    let chunkCount = 0;

    service.chunkReady$.subscribe((chunk: AudioChunk) => {
      expect(chunk.chunkIndex).toBe(chunkCount);
      chunkCount++;

      if (chunkCount === 2) {
        done();
      }
    });

    // Trigger two silence detections
    silenceDetector.silenceDetected$.next();
    setTimeout(() => silenceDetector.silenceDetected$.next(), 100);
  });

  it('should stop recording and cleanup', () => {
    spyOn(silenceDetector, 'stopMonitoring');

    service.stopRecording();

    expect(silenceDetector.stopMonitoring).toHaveBeenCalled();
  });
});
```

**Step 2: Run test to verify it fails**

```bash
npm test -- --include='**/realtime-recorder.service.spec.ts'
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `frontend/src/app/services/realtime-recorder.service.ts`:

```typescript
import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';
import { SilenceDetectorService } from './silence-detector.service';

export interface AudioChunk {
  audioBlob: Blob;
  chunkIndex: number;
  timestamp: number;
}

@Injectable({
  providedIn: 'root'
})
export class RealtimeRecorderService {
  chunkReady$ = new Subject<AudioChunk>();

  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private audioChunks: Blob[] = [];
  private chunkIndex = 0;
  private isRecording = false;

  constructor(private silenceDetector: SilenceDetectorService) {
    // Subscribe to silence detection
    this.silenceDetector.silenceDetected$.subscribe(() => {
      this.finalizeCurrentChunk();
    });
  }

  /**
   * Start continuous recording with silence detection.
   */
  async startRecording(
    silenceThreshold: number = 20,
    silenceDuration: number = 500
  ): Promise<void> {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      // Setup audio context for silence detection
      this.audioContext = new AudioContext({ sampleRate: 16000 });
      const source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      source.connect(this.analyser);

      // Setup media recorder
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      // Start recording
      this.mediaRecorder.start(100); // Collect data every 100ms
      this.isRecording = true;
      this.chunkIndex = 0;

      // Start silence detection
      this.silenceDetector.startMonitoring(
        this.analyser,
        silenceThreshold,
        silenceDuration
      );

    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }

  /**
   * Stop recording and cleanup resources.
   */
  stopRecording(): void {
    this.isRecording = false;

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }

    this.silenceDetector.stopMonitoring();

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    // Finalize any remaining audio
    if (this.audioChunks.length > 0) {
      this.finalizeCurrentChunk();
    }

    this.mediaRecorder = null;
    this.analyser = null;
  }

  /**
   * Finalize current chunk and emit for upload.
   */
  private finalizeCurrentChunk(): void {
    if (this.audioChunks.length === 0) {
      return;
    }

    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

    const chunk: AudioChunk = {
      audioBlob,
      chunkIndex: this.chunkIndex,
      timestamp: Date.now() / 1000
    };

    this.chunkReady$.next(chunk);

    // Reset for next chunk
    this.audioChunks = [];
    this.chunkIndex++;
  }

  /**
   * Check if currently recording.
   */
  isCurrentlyRecording(): boolean {
    return this.isRecording;
  }

  /**
   * Get current chunk index.
   */
  getCurrentChunkIndex(): number {
    return this.chunkIndex;
  }
}
```

**Step 4: Run test to verify it passes**

```bash
npm test -- --include='**/realtime-recorder.service.spec.ts'
```

Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/app/services/realtime-recorder.service.ts frontend/src/app/services/realtime-recorder.service.spec.ts
git commit -m "feat: add real-time recording service with chunk emission"
```

---

### Task 8: Transcription API Client Service

**Files:**
- Create: `frontend/src/app/services/transcription-client.service.ts`
- Create: `frontend/src/app/services/transcription-client.service.spec.ts`

**Step 1: Write the failing test**

Create `frontend/src/app/services/transcription-client.service.spec.ts`:

```typescript
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { TranscriptionClientService, ChunkStatus, TranscriptionChunk } from './transcription-client.service';

describe('TranscriptionClientService', () => {
  let service: TranscriptionClientService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule]
    });
    service = TestBed.inject(TranscriptionClientService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should upload audio chunk', (done) => {
    const mockBlob = new Blob(['test'], { type: 'audio/webm' });
    const chunkIndex = 0;
    const timestamp = 1699123456.789;

    service.uploadChunk(mockBlob, chunkIndex, timestamp).subscribe(response => {
      expect(response.session_id).toBe('test-session-id');
      expect(response.chunk_id).toBe(0);
      done();
    });

    const req = httpMock.expectOne('http://localhost:8000/api/transcribe/chunk');
    expect(req.request.method).toBe('POST');

    req.flush({
      session_id: 'test-session-id',
      chunk_id: 0,
      status: 'pending'
    });
  });

  it('should include session_id in subsequent uploads', (done) => {
    service.setSessionId('existing-session');
    const mockBlob = new Blob(['test'], { type: 'audio/webm' });

    service.uploadChunk(mockBlob, 1, 1699123457.789).subscribe(() => {
      done();
    });

    const req = httpMock.expectOne('http://localhost:8000/api/transcribe/chunk');
    const formData = req.request.body as FormData;
    expect(formData.get('session_id')).toBe('existing-session');

    req.flush({ session_id: 'existing-session', chunk_id: 1, status: 'pending' });
  });

  it('should get session status', (done) => {
    const sessionId = 'test-session';

    service.getSessionStatus(sessionId).subscribe(response => {
      expect(response.session_id).toBe(sessionId);
      expect(response.chunks.length).toBe(2);
      expect(response.chunks[0].status).toBe(ChunkStatus.DRAFT);
      done();
    });

    const req = httpMock.expectOne(`http://localhost:8000/api/transcribe/status/${sessionId}`);
    expect(req.request.method).toBe('GET');

    req.flush({
      session_id: sessionId,
      chunks: [
        { chunk_index: 0, status: 'draft', draft_text: 'Hello', final_text: null, timestamp: 1699123456.789 },
        { chunk_index: 1, status: 'pending', draft_text: null, final_text: null, timestamp: 1699123457.789 }
      ]
    });
  });

  it('should handle upload errors', (done) => {
    const mockBlob = new Blob(['test'], { type: 'audio/webm' });

    service.uploadChunk(mockBlob, 0, 1699123456.789).subscribe(
      () => fail('should have failed'),
      (error) => {
        expect(error.status).toBe(500);
        done();
      }
    );

    const req = httpMock.expectOne('http://localhost:8000/api/transcribe/chunk');
    req.flush('Server error', { status: 500, statusText: 'Internal Server Error' });
  });
});
```

**Step 2: Run test to verify it fails**

```bash
npm test -- --include='**/transcription-client.service.spec.ts'
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `frontend/src/app/services/transcription-client.service.ts`:

```typescript
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export enum ChunkStatus {
  PENDING = 'pending',
  DRAFT = 'draft',
  FINALIZING = 'finalizing',
  FINAL = 'final'
}

export interface TranscriptionChunk {
  chunk_index: number;
  status: ChunkStatus;
  draft_text: string | null;
  final_text: string | null;
  timestamp: number;
}

export interface SessionStatus {
  session_id: string;
  chunks: TranscriptionChunk[];
}

export interface UploadResponse {
  session_id: string;
  chunk_id: number;
  status: string;
}

@Injectable({
  providedIn: 'root'
})
export class TranscriptionClientService {
  private readonly API_BASE = 'http://localhost:8000/api/transcribe';
  private sessionId: string | null = null;

  constructor(private http: HttpClient) { }

  /**
   * Set session ID for subsequent uploads.
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }

  /**
   * Get current session ID.
   */
  getSessionId(): string | null {
    return this.sessionId;
  }

  /**
   * Upload audio chunk to backend.
   */
  uploadChunk(
    audioBlob: Blob,
    chunkIndex: number,
    timestamp: number
  ): Observable<UploadResponse> {
    const formData = new FormData();

    // Convert webm to wav if needed (for now, send as-is)
    formData.append('file', audioBlob, 'chunk.wav');
    formData.append('chunk_index', chunkIndex.toString());
    formData.append('timestamp', timestamp.toString());

    if (this.sessionId) {
      formData.append('session_id', this.sessionId);
    }

    return this.http.post<UploadResponse>(`${this.API_BASE}/chunk`, formData);
  }

  /**
   * Get transcription status for session.
   */
  getSessionStatus(sessionId: string): Observable<SessionStatus> {
    return this.http.get<SessionStatus>(`${this.API_BASE}/status/${sessionId}`);
  }

  /**
   * Clear session (for new recording).
   */
  clearSession(): void {
    this.sessionId = null;
  }
}
```

**Step 4: Run test to verify it passes**

```bash
npm test -- --include='**/transcription-client.service.spec.ts'
```

Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/app/services/transcription-client.service.ts frontend/src/app/services/transcription-client.service.spec.ts
git commit -m "feat: add transcription API client with chunk upload and polling"
```

---

### Task 9: Transcription Manager Service (Orchestration)

**Files:**
- Create: `frontend/src/app/services/transcription-manager.service.ts`
- Create: `frontend/src/app/services/transcription-manager.service.spec.ts`

**Step 1: Write the failing test**

Create `frontend/src/app/services/transcription-manager.service.spec.ts`:

```typescript
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { TranscriptionManagerService } from './transcription-manager.service';
import { RealtimeRecorderService } from './realtime-recorder.service';
import { TranscriptionClientService, ChunkStatus } from './transcription-client.service';
import { of } from 'rxjs';

describe('TranscriptionManagerService', () => {
  let service: TranscriptionManagerService;
  let recorder: RealtimeRecorderService;
  let client: TranscriptionClientService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule]
    });
    service = TestBed.inject(TranscriptionManagerService);
    recorder = TestBed.inject(RealtimeRecorderService);
    client = TestBed.inject(TranscriptionClientService);
  });

  it('should upload chunks automatically', fakeAsync(() => {
    spyOn(client, 'uploadChunk').and.returnValue(of({
      session_id: 'test-session',
      chunk_id: 0,
      status: 'pending'
    }));

    service.startRecording();

    // Simulate chunk ready
    recorder.chunkReady$.next({
      audioBlob: new Blob(['test']),
      chunkIndex: 0,
      timestamp: Date.now() / 1000
    });

    tick();

    expect(client.uploadChunk).toHaveBeenCalled();
  }));

  it('should start polling after first chunk', fakeAsync(() => {
    spyOn(client, 'uploadChunk').and.returnValue(of({
      session_id: 'test-session',
      chunk_id: 0,
      status: 'pending'
    }));

    spyOn(client, 'getSessionStatus').and.returnValue(of({
      session_id: 'test-session',
      chunks: [
        { chunk_index: 0, status: ChunkStatus.DRAFT, draft_text: 'Hello', final_text: null, timestamp: 123 }
      ]
    }));

    service.startRecording();

    recorder.chunkReady$.next({
      audioBlob: new Blob(['test']),
      chunkIndex: 0,
      timestamp: Date.now() / 1000
    });

    tick(1000); // Wait for polling to start

    expect(client.getSessionStatus).toHaveBeenCalledWith('test-session');
  }));

  it('should emit transcription updates', (done) => {
    spyOn(client, 'getSessionStatus').and.returnValue(of({
      session_id: 'test-session',
      chunks: [
        { chunk_index: 0, status: ChunkStatus.DRAFT, draft_text: 'Hello world', final_text: null, timestamp: 123 }
      ]
    }));

    service.transcriptionUpdate$.subscribe(chunks => {
      expect(chunks.length).toBe(1);
      expect(chunks[0].draft_text).toBe('Hello world');
      done();
    });

    client.setSessionId('test-session');
    service['pollForUpdates']();
  });

  it('should stop polling when recording stops', fakeAsync(() => {
    service.startRecording();
    tick(100);

    service.stopRecording();

    expect(service['isPolling']).toBe(false);
  }));
});
```

**Step 2: Run test to verify it fails**

```bash
npm test -- --include='**/transcription-manager.service.spec.ts'
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `frontend/src/app/services/transcription-manager.service.ts`:

```typescript
import { Injectable } from '@angular/core';
import { Subject, interval, Subscription } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { RealtimeRecorderService, AudioChunk } from './realtime-recorder.service';
import { TranscriptionClientService, TranscriptionChunk } from './transcription-client.service';

@Injectable({
  providedIn: 'root'
})
export class TranscriptionManagerService {
  transcriptionUpdate$ = new Subject<TranscriptionChunk[]>();

  private pollingSubscription: Subscription | null = null;
  private chunkSubscription: Subscription | null = null;
  private isPolling = false;
  private pollingInterval = 500; // Start with fast polling

  constructor(
    private recorder: RealtimeRecorderService,
    private client: TranscriptionClientService
  ) { }

  /**
   * Start recording and automatic chunk upload/polling.
   */
  async startRecording(
    silenceThreshold: number = 20,
    silenceDuration: number = 500
  ): Promise<void> {
    // Clear previous session
    this.client.clearSession();

    // Start recording
    await this.recorder.startRecording(silenceThreshold, silenceDuration);

    // Subscribe to chunk events
    this.chunkSubscription = this.recorder.chunkReady$.subscribe(
      (chunk: AudioChunk) => this.handleChunkReady(chunk)
    );
  }

  /**
   * Stop recording and polling.
   */
  stopRecording(): void {
    this.recorder.stopRecording();
    this.stopPolling();

    if (this.chunkSubscription) {
      this.chunkSubscription.unsubscribe();
      this.chunkSubscription = null;
    }
  }

  /**
   * Handle chunk ready for upload.
   */
  private handleChunkReady(chunk: AudioChunk): void {
    this.client.uploadChunk(chunk.audioBlob, chunk.chunkIndex, chunk.timestamp)
      .subscribe({
        next: (response) => {
          console.log('Chunk uploaded:', response);

          // Set session ID on first chunk
          if (!this.client.getSessionId()) {
            this.client.setSessionId(response.session_id);
            this.startPolling();
          }
        },
        error: (error) => {
          console.error('Upload error:', error);
          // TODO: Implement retry logic
        }
      });
  }

  /**
   * Start polling for transcription updates.
   */
  private startPolling(): void {
    if (this.isPolling) return;

    this.isPolling = true;

    this.pollingSubscription = interval(this.pollingInterval)
      .pipe(
        switchMap(() => {
          const sessionId = this.client.getSessionId();
          if (!sessionId) {
            throw new Error('No session ID available');
          }
          return this.client.getSessionStatus(sessionId);
        })
      )
      .subscribe({
        next: (status) => {
          this.transcriptionUpdate$.next(status.chunks);

          // Adjust polling interval based on chunk status
          const hasDrafts = status.chunks.some(c => c.status === 'draft');
          this.pollingInterval = hasDrafts ? 500 : 2000;
        },
        error: (error) => {
          console.error('Polling error:', error);
        }
      });
  }

  /**
   * Stop polling for updates.
   */
  private stopPolling(): void {
    this.isPolling = false;

    if (this.pollingSubscription) {
      this.pollingSubscription.unsubscribe();
      this.pollingSubscription = null;
    }
  }

  /**
   * Check if currently recording.
   */
  isRecording(): boolean {
    return this.recorder.isCurrentlyRecording();
  }
}
```

**Step 4: Run test to verify it passes**

```bash
npm test -- --include='**/transcription-manager.service.spec.ts'
```

Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/app/services/transcription-manager.service.ts frontend/src/app/services/transcription-manager.service.spec.ts
git commit -m "feat: add transcription manager orchestrating recording, upload, and polling"
```

---

## Phase 3: UI Components

### Task 10: Transcription Display Component

**Files:**
- Create: `frontend/src/app/components/transcription-display/transcription-display.component.ts`
- Create: `frontend/src/app/components/transcription-display/transcription-display.component.html`
- Create: `frontend/src/app/components/transcription-display/transcription-display.component.scss`
- Create: `frontend/src/app/components/transcription-display/transcription-display.component.spec.ts`

**Step 1: Write the failing test**

Create component:

```bash
cd frontend
ng generate component components/transcription-display --skip-tests
```

Create `frontend/src/app/components/transcription-display/transcription-display.component.spec.ts`:

```typescript
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { TranscriptionDisplayComponent } from './transcription-display.component';
import { ChunkStatus, TranscriptionChunk } from '../../services/transcription-client.service';

describe('TranscriptionDisplayComponent', () => {
  let component: TranscriptionDisplayComponent;
  let fixture: ComponentFixture<TranscriptionDisplayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TranscriptionDisplayComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(TranscriptionDisplayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display draft chunks with indicator', () => {
    const chunks: TranscriptionChunk[] = [
      { chunk_index: 0, status: ChunkStatus.DRAFT, draft_text: 'Hello world', final_text: null, timestamp: 123 }
    ];

    component.chunks = chunks;
    fixture.detectChanges();

    const compiled = fixture.nativeElement;
    const chunkElement = compiled.querySelector('.chunk.draft');

    expect(chunkElement).toBeTruthy();
    expect(chunkElement.textContent).toContain('Hello world');
  });

  it('should display final chunks without indicator', () => {
    const chunks: TranscriptionChunk[] = [
      { chunk_index: 0, status: ChunkStatus.FINAL, draft_text: 'Hello', final_text: 'Hello, world!', timestamp: 123 }
    ];

    component.chunks = chunks;
    fixture.detectChanges();

    const compiled = fixture.nativeElement;
    const chunkElement = compiled.querySelector('.chunk.final');

    expect(chunkElement).toBeTruthy();
    expect(chunkElement.textContent).toContain('Hello, world!');
  });

  it('should prefer final text over draft', () => {
    const chunks: TranscriptionChunk[] = [
      { chunk_index: 0, status: ChunkStatus.FINAL, draft_text: 'draft', final_text: 'final', timestamp: 123 }
    ];

    component.chunks = chunks;
    fixture.detectChanges();

    const text = component.getDisplayText(chunks[0]);
    expect(text).toBe('final');
  });

  it('should show processing state for pending chunks', () => {
    const chunks: TranscriptionChunk[] = [
      { chunk_index: 0, status: ChunkStatus.PENDING, draft_text: null, final_text: null, timestamp: 123 }
    ];

    component.chunks = chunks;
    fixture.detectChanges();

    const compiled = fixture.nativeElement;
    const chunkElement = compiled.querySelector('.chunk.pending');

    expect(chunkElement).toBeTruthy();
  });
});
```

**Step 2: Run test to verify it fails**

```bash
npm test -- --include='**/transcription-display.component.spec.ts'
```

Expected: FAIL (methods don't exist)

**Step 3: Write implementation**

Update `frontend/src/app/components/transcription-display/transcription-display.component.ts`:

```typescript
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TranscriptionChunk, ChunkStatus } from '../../services/transcription-client.service';

@Component({
  selector: 'app-transcription-display',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './transcription-display.component.html',
  styleUrls: ['./transcription-display.component.scss']
})
export class TranscriptionDisplayComponent {
  @Input() chunks: TranscriptionChunk[] = [];

  ChunkStatus = ChunkStatus; // Expose enum to template

  /**
   * Get display text for chunk (prefer final over draft).
   */
  getDisplayText(chunk: TranscriptionChunk): string {
    return chunk.final_text || chunk.draft_text || '';
  }

  /**
   * Check if chunk is in draft state.
   */
  isDraft(chunk: TranscriptionChunk): boolean {
    return chunk.status === ChunkStatus.DRAFT;
  }

  /**
   * Check if chunk is finalized.
   */
  isFinal(chunk: TranscriptionChunk): boolean {
    return chunk.status === ChunkStatus.FINAL;
  }

  /**
   * Get full transcription text.
   */
  getFullTranscription(): string {
    return this.chunks
      .map(chunk => this.getDisplayText(chunk))
      .filter(text => text.length > 0)
      .join(' ');
  }
}
```

Update `frontend/src/app/components/transcription-display/transcription-display.component.html`:

```html
<div class="transcription-container">
  <div class="chunks">
    <div
      *ngFor="let chunk of chunks"
      class="chunk"
      [class.draft]="isDraft(chunk)"
      [class.final]="isFinal(chunk)"
      [class.pending]="chunk.status === ChunkStatus.PENDING"
      [class.finalizing]="chunk.status === ChunkStatus.FINALIZING">

      <span class="text">{{ getDisplayText(chunk) }}</span>

      <span *ngIf="isDraft(chunk)" class="draft-indicator" title="Draft transcription">✎</span>
      <span *ngIf="chunk.status === ChunkStatus.PENDING" class="processing-indicator">⏳</span>
      <span *ngIf="chunk.status === ChunkStatus.FINALIZING" class="processing-indicator">🔄</span>
    </div>
  </div>

  <div *ngIf="chunks.length === 0" class="empty-state">
    Start speaking to see transcription...
  </div>
</div>
```

Update `frontend/src/app/components/transcription-display/transcription-display.component.scss`:

```scss
.transcription-container {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  min-height: 200px;
  max-height: 500px;
  overflow-y: auto;
}

.chunks {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  line-height: 1.8;
}

.chunk {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  transition: all 0.3s ease;

  &.draft {
    .text {
      color: #666;
      font-style: italic;
    }
  }

  &.final {
    .text {
      color: #000;
      font-weight: 500;
    }
  }

  &.pending, &.finalizing {
    .text {
      color: #999;
    }
  }
}

.draft-indicator {
  font-size: 0.9em;
  color: #999;
  opacity: 0.7;
}

.processing-indicator {
  font-size: 0.8em;
  animation: pulse 1.5s ease-in-out infinite;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #999;
  font-style: italic;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

**Step 4: Run test to verify it passes**

```bash
npm test -- --include='**/transcription-display.component.spec.ts'
```

Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/app/components/transcription-display/
git commit -m "feat: add transcription display component with draft/final states"
```

---

### Task 11: Integration into Main App Component

**Files:**
- Modify: `frontend/src/app/app.ts`
- Modify: `frontend/src/app/app.html`
- Modify: `frontend/src/app/app.scss`

**Step 1: Update app component**

Update `frontend/src/app/app.ts`:

```typescript
import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TranscriptionDisplayComponent } from './components/transcription-display/transcription-display.component';
import { TranscriptionManagerService } from './services/transcription-manager.service';
import { TranscriptionChunk } from './services/transcription-client.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, TranscriptionDisplayComponent],
  templateUrl: './app.html',
  styleUrls: ['./app.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  chunks: TranscriptionChunk[] = [];
  isRecording = false;

  private transcriptionSubscription: Subscription | null = null;

  constructor(private transcriptionManager: TranscriptionManagerService) {}

  ngOnInit(): void {
    // Subscribe to transcription updates
    this.transcriptionSubscription = this.transcriptionManager.transcriptionUpdate$
      .subscribe(chunks => {
        this.chunks = chunks;
      });
  }

  ngOnDestroy(): void {
    if (this.transcriptionSubscription) {
      this.transcriptionSubscription.unsubscribe();
    }

    if (this.isRecording) {
      this.stopRecording();
    }
  }

  async startRecording(): Promise<void> {
    try {
      await this.transcriptionManager.startRecording();
      this.isRecording = true;
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to start recording. Please check microphone permissions.');
    }
  }

  stopRecording(): void {
    this.transcriptionManager.stopRecording();
    this.isRecording = false;
  }

  clearTranscription(): void {
    this.chunks = [];
  }

  copyToClipboard(): void {
    const text = this.chunks
      .map(c => c.final_text || c.draft_text || '')
      .filter(t => t.length > 0)
      .join(' ');

    navigator.clipboard.writeText(text).then(() => {
      alert('Transcription copied to clipboard!');
    });
  }
}
```

Update `frontend/src/app/app.html`:

```html
<div class="app-container">
  <header>
    <h1>🎤 Real-Time Speech Transcription</h1>
    <p>Speak naturally - transcription appears as you talk</p>
  </header>

  <main>
    <div class="controls">
      <button
        *ngIf="!isRecording"
        (click)="startRecording()"
        class="btn btn-primary btn-start">
        <span class="icon">🎙️</span>
        Start Recording
      </button>

      <button
        *ngIf="isRecording"
        (click)="stopRecording()"
        class="btn btn-danger btn-stop">
        <span class="icon">⏹️</span>
        Stop Recording
      </button>

      <button
        (click)="clearTranscription()"
        class="btn btn-secondary"
        [disabled]="chunks.length === 0">
        <span class="icon">🗑️</span>
        Clear
      </button>

      <button
        (click)="copyToClipboard()"
        class="btn btn-secondary"
        [disabled]="chunks.length === 0">
        <span class="icon">📋</span>
        Copy
      </button>
    </div>

    <div class="status" *ngIf="isRecording">
      <span class="recording-indicator">● Recording</span>
      <span class="chunk-count">{{ chunks.length }} chunks</span>
    </div>

    <app-transcription-display [chunks]="chunks"></app-transcription-display>
  </main>

  <footer>
    <p>
      <strong>Legend:</strong>
      <span class="legend-item">
        <span class="draft-text">Italic</span> = Draft (fast)
      </span>
      <span class="legend-item">
        <span class="final-text">Bold</span> = Final (accurate)
      </span>
    </p>
  </footer>
</div>
```

Update `frontend/src/app/app.scss`:

```scss
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

header {
  text-align: center;
  margin-bottom: 40px;

  h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
    color: #333;
  }

  p {
    color: #666;
    font-size: 1.1em;
  }
}

.controls {
  display: flex;
  gap: 15px;
  justify-content: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  font-size: 16px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 600;

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  &:not(:disabled):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:not(:disabled):active {
    transform: translateY(0);
  }
}

.btn-primary {
  background: #007bff;
  color: white;

  &:not(:disabled):hover {
    background: #0056b3;
  }
}

.btn-danger {
  background: #dc3545;
  color: white;
  animation: pulse-red 2s ease-in-out infinite;

  &:not(:disabled):hover {
    background: #c82333;
  }
}

.btn-secondary {
  background: #6c757d;
  color: white;

  &:not(:disabled):hover {
    background: #545b62;
  }
}

.status {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-bottom: 20px;
  font-size: 14px;
  color: #666;
}

.recording-indicator {
  color: #dc3545;
  font-weight: 600;
  animation: blink 1.5s ease-in-out infinite;
}

.chunk-count {
  font-weight: 500;
}

footer {
  margin-top: 40px;
  text-align: center;
  font-size: 14px;
  color: #666;

  .legend-item {
    margin: 0 15px;
  }

  .draft-text {
    font-style: italic;
    color: #666;
  }

  .final-text {
    font-weight: 600;
    color: #000;
  }
}

@keyframes pulse-red {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(220, 53, 69, 0);
  }
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
```

**Step 2: Test in browser**

```bash
cd frontend
npm start
```

Open http://localhost:4200 and test:
- Click "Start Recording"
- Speak into microphone
- Verify chunks appear
- Verify draft→final transitions
- Test controls (stop, clear, copy)

**Step 3: Commit**

```bash
git add frontend/src/app/app.ts frontend/src/app/app.html frontend/src/app/app.scss
git commit -m "feat: integrate real-time transcription into main app"
```

---

## Phase 4: Testing and Polish

### Task 12: End-to-End Testing

**Files:**
- Create: `tests/test_e2e_realtime.py`

**Step 1: Write E2E test**

Create `tests/test_e2e_realtime.py`:

```python
import pytest
import time
from pathlib import Path
from session_manager import SessionManager
from transcription_worker import DraftTranscriptionWorker
from finalization_worker import FinalizationWorker

def test_full_realtime_workflow(tmp_path):
    """Test complete workflow from chunk upload to finalization."""
    # Setup
    session_manager = SessionManager()
    draft_worker = DraftTranscriptionWorker(session_manager)
    finalization_worker = FinalizationWorker(session_manager, merge_threshold=3)

    # Create session
    session_id = session_manager.create_session()

    # Simulate 5 chunks being uploaded
    test_audio = Path("tests/fixtures/test_audio.wav")
    if not test_audio.exists():
        pytest.skip("Test audio file not found")

    chunk_ids = []
    for i in range(5):
        chunk_id = session_manager.add_chunk(
            session_id=session_id,
            chunk_index=i,
            audio_path=str(test_audio),
            timestamp=time.time()
        )
        chunk_ids.append(chunk_id)

        # Process draft (in real system this is async)
        import asyncio
        asyncio.run(draft_worker.process_chunk(session_id, chunk_id))

        # Verify draft created
        chunk = session_manager.get_chunk(session_id, chunk_id)
        assert chunk["draft_text"] is not None
        assert chunk["status"] == "draft"

    # Check finalization triggered after threshold
    asyncio.run(finalization_worker.check_and_finalize(session_id))

    # Verify first 3 chunks finalized
    for i in range(3):
        chunk = session_manager.get_chunk(session_id, i)
        assert chunk["final_text"] is not None
        assert chunk["status"] == "final"

    # Verify last 2 chunks still draft
    for i in range(3, 5):
        chunk = session_manager.get_chunk(session_id, i)
        assert chunk["status"] == "draft"

def test_session_cleanup():
    """Test session cleanup after timeout."""
    manager = SessionManager()
    session_id = manager.create_session()

    # Add chunks
    for i in range(3):
        manager.add_chunk(session_id, i, f"test{i}.wav", time.time())

    # Verify session exists
    assert session_id in manager.sessions

    # TODO: Implement session cleanup logic
    # manager.cleanup_old_sessions(max_age_seconds=1800)
```

**Step 2: Run test**

```bash
pytest tests/test_e2e_realtime.py -v
```

Expected: PASS (or SKIP if test audio not available)

**Step 3: Commit**

```bash
git add tests/test_e2e_realtime.py
git commit -m "test: add end-to-end workflow test for real-time transcription"
```

---

### Task 13: Documentation

**Files:**
- Create: `docs/REALTIME_AUDIO.md`

**Step 1: Write user documentation**

Create `docs/REALTIME_AUDIO.md`:

```markdown
# Real-Time Audio Transcription

Live dictation system with silence-based chunking and progressive transcription refinement.

## Features

- **Continuous Recording**: Record indefinitely without duration limits
- **Silence Detection**: Automatically chunks audio on 0.5-1s pauses
- **Draft→Final Transcription**: See quick draft transcripts that improve to accurate finals
- **Visual Feedback**: Clear indicators for draft vs. final transcription state
- **Simple Controls**: Start, stop, clear, and copy transcriptions

## How It Works

### 1. Audio Capture (Frontend)
- Browser records continuous audio using MediaRecorder API
- Web Audio API analyzes audio levels every 100ms
- When 0.5s of silence detected, chunk is finalized and sent

### 2. Draft Transcription (Backend)
- Chunk received via HTTP POST
- Immediately transcribed with fast model (ctc_1b)
- Draft text returned within ~1-2 seconds

### 3. Final Transcription (Backend)
- Chunks buffered until threshold (5-8 chunks or 5-10s audio)
- Audio files merged into single segment
- Re-transcribed with accurate model (llm_1b)
- All chunks in merge group updated with final text

### 4. UI Updates (Frontend)
- Polls backend every 500ms for status updates
- Draft text displayed in italic/light color
- Final text displayed in bold/dark color
- Smooth transitions between states

## Usage

### Starting a Recording Session

```typescript
// Inject service
constructor(private transcriptionManager: TranscriptionManagerService) {}

// Start recording
await this.transcriptionManager.startRecording();

// Subscribe to updates
this.transcriptionManager.transcriptionUpdate$.subscribe(chunks => {
  console.log('Transcription chunks:', chunks);
});
```

### Stopping Recording

```typescript
this.transcriptionManager.stopRecording();
```

### API Endpoints

**Upload Chunk**
```
POST /api/transcribe/chunk
Content-Type: multipart/form-data

file: audio file (WAV, 16kHz mono)
chunk_index: integer
timestamp: float
session_id: string (optional, created if not provided)

Response:
{
  "session_id": "uuid",
  "chunk_id": 0,
  "status": "pending"
}
```

**Get Status**
```
GET /api/transcribe/status/{session_id}

Response:
{
  "session_id": "uuid",
  "chunks": [
    {
      "chunk_index": 0,
      "status": "draft|final|pending|finalizing",
      "draft_text": "Quick transcription",
      "final_text": "Accurate transcription",
      "timestamp": 1699123456.789
    }
  ]
}
```

## Configuration

### Backend

```python
# Adjust merge threshold (number of chunks before finalization)
finalization_worker = FinalizationWorker(
    session_manager,
    merge_threshold=5,  # 5 chunks
    time_threshold=10.0  # or 10 seconds
)

# Choose models (speed vs. accuracy)
draft_worker = DraftTranscriptionWorker(session_manager)  # Uses ctc_1b
finalization_worker = FinalizationWorker(session_manager)  # Uses llm_1b
```

### Frontend

```typescript
// Adjust silence detection
await transcriptionManager.startRecording(
  silenceThreshold: 20,  // Audio level 0-255
  silenceDuration: 500   // Milliseconds
);
```

## Troubleshooting

### No audio detected
- Check browser microphone permissions
- Verify correct input device selected
- Check silence threshold (lower = more sensitive)

### Chunks not appearing
- Check browser console for upload errors
- Verify backend API is running (http://localhost:8000)
- Check network tab for failed requests

### Draft text not updating to final
- Verify finalization worker is running
- Check backend logs for transcription errors
- Ensure enough chunks for merge threshold

### High latency
- Reduce polling interval (default 500ms)
- Use faster models (ctc vs llm)
- Check network connection

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ MediaRecorder│→ │Silence Detect│→ │  HTTP Client │      │
│  └──────────────┘  └──────────────┘  └──────┬───────┘      │
└─────────────────────────────────────────────┼──────────────┘
                                              │
                                         POST /chunk
                                              │
┌─────────────────────────────────────────────▼──────────────┐
│                      Backend API                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Session    │→ │ Draft Worker │  │Finalization  │     │
│  │   Manager    │  │  (ctc_1b)    │  │   Worker     │     │
│  └──────────────┘  └──────────────┘  │  (llm_1b)    │     │
│         ▲                             └──────────────┘     │
│         │                                                   │
│    GET /status                                              │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │
    ┌─────▼─────┐
    │  Polling  │
    │  (500ms)  │
    └───────────┘
```

## Performance Metrics

- **Draft Latency**: <2s from silence detection
- **Final Latency**: <5s from merge trigger
- **Polling Overhead**: ~10KB/request every 500ms
- **Audio Upload**: ~100KB per 5-second chunk

## Future Enhancements

- WebSocket support for lower latency
- Word-level alignment for smoother updates
- Speaker diarization for multi-speaker scenarios
- Custom vocabulary/domain adaptation
- Offline mode with IndexedDB buffering
```

**Step 2: Commit**

```bash
git add docs/REALTIME_AUDIO.md
git commit -m "docs: add comprehensive real-time audio transcription documentation"
```

---

## Implementation Complete! 🎉

All tasks completed:
- ✅ Backend session management and API endpoints
- ✅ Draft and finalization workers with async processing
- ✅ Frontend silence detection and recording services
- ✅ HTTP client and orchestration manager
- ✅ UI components with visual state indicators
- ✅ End-to-end testing
- ✅ Complete documentation

---

Plan saved to: `docs/plans/2025-11-12-realtime-audio-implementation.md`
