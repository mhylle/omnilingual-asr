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
