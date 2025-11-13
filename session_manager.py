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
