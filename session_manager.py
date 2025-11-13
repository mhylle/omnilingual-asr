from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import threading


class ChunkStatus(str, Enum):
    PENDING = "pending"
    DRAFT = "draft"
    FINALIZING = "finalizing"
    FINAL = "final"


class SessionManager:
    """Manages transcription sessions and chunks with thread safety."""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def create_session(self) -> str:
        """Create new session and return session ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self.sessions[session_id] = {
                "chunks": {},  # Changed from list to dict for O(1) access
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
        """Add new chunk to session.

        Args:
            session_id: Session identifier
            chunk_index: Non-negative chunk index
            audio_path: Non-empty path to audio file
            timestamp: Positive timestamp

        Raises:
            ValueError: If session not found, chunk already exists, or invalid inputs
        """
        # Input validation
        if chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
        if not audio_path or not audio_path.strip():
            raise ValueError("audio_path cannot be empty")
        if timestamp < 0:
            raise ValueError(f"timestamp must be positive, got {timestamp}")

        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            # Check for duplicate chunk_index
            chunks = self.sessions[session_id]["chunks"]
            if chunk_index in chunks:
                raise ValueError(f"Chunk {chunk_index} already exists in session {session_id}")

            chunk = {
                "chunk_index": chunk_index,
                "audio_path": audio_path,
                "timestamp": timestamp,
                "status": ChunkStatus.PENDING,
                "draft_text": None,
                "final_text": None
            }

            chunks[chunk_index] = chunk
        return chunk_index

    def get_chunk(self, session_id: str, chunk_index: int) -> Dict:
        """Get specific chunk from session.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index to retrieve

        Returns:
            Chunk dictionary

        Raises:
            ValueError: If session or chunk not found
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            chunks = self.sessions[session_id]["chunks"]
            if chunk_index not in chunks:
                raise ValueError(f"Chunk {chunk_index} not found in session {session_id}")

            # Return a copy to prevent external modification
            return chunks[chunk_index].copy()

    def update_chunk_draft(self, session_id: str, chunk_index: int, draft_text: str):
        """Update chunk with draft transcription.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index to update
            draft_text: Draft transcription text

        Raises:
            ValueError: If session or chunk not found
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            chunks = self.sessions[session_id]["chunks"]
            if chunk_index not in chunks:
                raise ValueError(f"Chunk {chunk_index} not found in session {session_id}")

            chunks[chunk_index]["draft_text"] = draft_text
            chunks[chunk_index]["status"] = ChunkStatus.DRAFT

    def update_chunk_final(self, session_id: str, chunk_index: int, final_text: str):
        """Update chunk with final transcription.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index to update
            final_text: Final transcription text

        Raises:
            ValueError: If session or chunk not found
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            chunks = self.sessions[session_id]["chunks"]
            if chunk_index not in chunks:
                raise ValueError(f"Chunk {chunk_index} not found in session {session_id}")

            chunks[chunk_index]["final_text"] = final_text
            chunks[chunk_index]["status"] = ChunkStatus.FINAL

    def get_all_chunks(self, session_id: str) -> List[Dict]:
        """Get all chunks for session, sorted by chunk_index.

        Args:
            session_id: Session identifier

        Returns:
            List of chunk dictionaries sorted by chunk_index

        Raises:
            ValueError: If session not found
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            chunks = self.sessions[session_id]["chunks"]
            # Return sorted list of chunks (sorted by chunk_index)
            return [chunks[idx].copy() for idx in sorted(chunks.keys())]

    def get_chunks_for_finalization(self, session_id: str) -> List[Dict]:
        """Get chunks that need finalization (status = DRAFT).

        Args:
            session_id: Session identifier

        Returns:
            List of chunks with DRAFT status, sorted by chunk_index

        Raises:
            ValueError: If session not found
        """
        all_chunks = self.get_all_chunks(session_id)
        return [c for c in all_chunks if c["status"] == ChunkStatus.DRAFT]

    def delete_session(self, session_id: str):
        """Delete a session and all its chunks.

        Args:
            session_id: Session identifier

        Raises:
            ValueError: If session not found
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            del self.sessions[session_id]
