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
