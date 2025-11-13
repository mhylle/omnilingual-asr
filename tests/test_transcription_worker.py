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
