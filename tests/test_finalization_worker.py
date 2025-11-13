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
