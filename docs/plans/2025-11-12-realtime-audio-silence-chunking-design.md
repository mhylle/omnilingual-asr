# Real-Time Audio Transcription with Silence-Based Chunking

**Date:** 2025-11-12
**Purpose:** Live dictation with aggressive silence-based chunking and draft/final transcription
**Priority:** Accuracy over latency

## Overview

A three-layer architecture for real-time audio transcription that provides immediate draft feedback while ensuring accuracy through background re-transcription of merged audio chunks.

### Design Goals

- **Live dictation** experience with continuous recording
- **Aggressive chunking** on 0.5-1s silence periods for frequent updates
- **Accuracy-first** approach using draft + finalize strategy
- **HTTP polling** for simplicity and debuggability
- **Seamless UX** with visual states showing draft vs. final transcriptions

## Architecture

### Layer 1: Browser Recording (Angular)

**Audio Capture:**
- MediaRecorder API configured for 16kHz mono (matches backend ASR requirements)
- Echo cancellation and noise suppression enabled
- Continuous recording to buffer

**Silence Detection:**
- Web Audio API analyser monitors audio energy levels every 100ms
- Silence threshold detection triggers chunk finalization after 0.5s
- Fallback: Force chunk at 10s max duration even without silence
- Adaptive threshold calibration based on ambient noise

**Chunking Strategy:**
- Finalize chunk on 0.5-1s silence detection
- Include metadata: session_id, chunk_index, timestamp
- Handle edge cases: very short chunks (<0.5s) merge with next

**HTTP Communication:**
- POST `/api/transcribe/chunk` - Upload audio chunk (WAV, 16kHz mono)
- GET `/api/transcribe/status/{session_id}` - Poll every 500ms for updates
- Smart polling: Fast (500ms) when drafts exist, slower (2s) when all finalized
- Retry logic: 3 attempts with exponential backoff for failed uploads

### Layer 2: Backend Processing (FastAPI)

**Session Management:**
```python
sessions = {
    "session_uuid": {
        "chunks": [
            {
                "chunk_id": 0,
                "audio_path": "uploads/session_uuid_chunk_0.wav",
                "draft_text": "Hello world",
                "final_text": None,
                "status": "draft",  # draft | finalizing | final
                "timestamp": 1699123456.789
            }
        ],
        "buffer": []  # Chunks awaiting merge for finalization
    }
}
```

**API Endpoints:**

`POST /api/transcribe/chunk`
- Accept: audio file (multipart/form-data), session_id, chunk_index
- Save audio to uploads directory
- Queue for draft transcription (immediate)
- Add to merge buffer for finalization
- Return: 200 OK with chunk_id

`GET /api/transcribe/status/{session_id}`
- Return: Array of all chunks with current status and text
- Frontend uses this to update UI progressively

**Processing Queues:**

*Draft Queue (Fast Path):*
- Process individual chunks immediately
- Use `ctc_1b` model for speed
- Store result as `draft_text`
- Mark status as `"draft"`
- Target: <2s processing time

*Finalization Queue (Accuracy Path):*
- Buffer chunks until threshold: 5-10 seconds of audio OR 5-8 chunks
- Merge audio files using pydub
- Re-transcribe merged audio with `llm_1b` model
- Update all affected chunks with `final_text`
- Mark status as `"final"`
- Target: <5s processing time

**Background Workers:**
- Use asyncio tasks for concurrent draft and finalization processing
- Non-blocking API responses

**Error Handling:**
- Accept out-of-order chunks (sort by timestamp when merging)
- Handle gaps gracefully (transcribe available chunks)
- Session timeout: 30 minutes inactivity
- Cleanup: Delete temporary audio files after finalization
- Resource limits: Max sessions per user, max audio storage per session
- Rate limiting on chunk uploads

### Layer 3: Transcription (Meta Omnilingual ASR)

**Two-Stage Transcription:**

*Stage 1 - Draft (Fast):*
```python
async def draft_transcribe(chunk_path, session_id, chunk_id):
    transcriber = Transcriber(model="ctc_1b")
    result = transcriber.transcribe(chunk_path)

    sessions[session_id]["chunks"][chunk_id]["draft_text"] = result[0]
    sessions[session_id]["chunks"][chunk_id]["status"] = "draft"
```

*Stage 2 - Final (Accurate):*
```python
async def finalize_transcription(session_id, chunk_ids):
    # Merge audio from chunk_ids
    chunk_paths = [sessions[session_id]["chunks"][i]["audio_path"]
                   for i in chunk_ids]
    merged_audio = merge_chunks(chunk_paths)

    # Re-transcribe with better model
    transcriber = Transcriber(model="llm_1b")
    final_result = transcriber.transcribe(merged_audio)

    # Update all chunks in this merge group
    for chunk_id in chunk_ids:
        sessions[session_id]["chunks"][chunk_id]["final_text"] = final_result[0]
        sessions[session_id]["chunks"][chunk_id]["status"] = "final"
```

**Audio Merging:**
- Use pydub AudioSegment for concatenation
- Sort chunks by timestamp before merging
- Export as WAV 16kHz mono
- Clean up merged temporary files after transcription

**Merge Triggers:**
- Time-based: Every 5-10 seconds of accumulated audio
- Chunk-based: Every 5-8 chunks (whichever comes first)
- Session end: Finalize all remaining chunks when recording stops

## Data Flow

```
User speaks
  ↓
Silence detected (0.5s)
  ↓
Chunk finalized and sent via POST
  ↓
Backend receives chunk
  ├─→ Draft Queue: ctc_1b transcription → "draft_text" (~1-2s)
  └─→ Merge Buffer: accumulate for finalization
        ↓
      Buffer reaches threshold (5-10s or 5-8 chunks)
        ↓
      Merge audio files
        ↓
      Finalization Queue: llm_1b transcription → "final_text" (~3-5s)
        ↓
      Update all affected chunks

Frontend polls GET /status/{session_id} every 500ms
  ↓
Receives updates (draft → final)
  ↓
UI updates: Replace draft with final text
```

## UI/UX Design

### Transcription Display

**Visual States:**
```html
<div class="transcription-chunk"
     [class.draft]="chunk.status === 'draft'"
     [class.final]="chunk.status === 'final'"
     [class.processing]="chunk.status === 'finalizing'">
  {{ chunk.final_text || chunk.draft_text }}
  <span *ngIf="chunk.status === 'draft'" class="draft-indicator">✎</span>
</div>
```

**Styling:**
- Draft: Light gray, italic, subtle pulse animation
- Final: Normal weight, dark color, smooth fade-in transition
- Processing: Spinner icon for chunks awaiting finalization

### Recording Interface

**Visual Feedback:**
- Pulsing microphone icon while recording
- Audio level visualization (moving bars or waveform)
- Silence detection indicator (color change when silence detected)
- Chunk counter (total chunks sent)

**User Controls:**
- Start/Stop recording (primary action)
- Clear/Reset transcription
- Download/Copy final text
- Optional: Pause recording

### Polling Strategy

**Smart Polling:**
```typescript
pollInterval = 500; // Initial fast polling

pollForUpdates() {
  const hasDrafts = this.chunks.some(c => c.status === 'draft');

  if (hasDrafts) {
    this.pollInterval = 500; // Fast when processing
  } else {
    this.pollInterval = 2000; // Slow when stable
  }

  // Stop when recording ended AND all chunks finalized
  if (!this.isRecording && !hasDrafts) {
    return;
  }

  setTimeout(() => this.pollForUpdates(), this.pollInterval);
}
```

## Implementation Notes

### Technology Stack

**Frontend:**
- Angular (existing)
- MediaRecorder API
- Web Audio API (AnalyserNode)
- RxJS for polling observables

**Backend:**
- FastAPI (existing)
- asyncio for background workers
- pydub for audio merging
- In-memory storage (or Redis for production)

**Transcription:**
- Meta Omnilingual ASR (existing)
- Models: ctc_1b (draft), llm_1b (final)

### Edge Cases Handled

1. **Continuous speech without pauses:** Force chunk at 10s max
2. **Background noise:** Adaptive silence threshold calibration
3. **Very short chunks:** Skip or merge with next chunk
4. **Network failures:** Retry logic with exponential backoff
5. **Out-of-order chunks:** Sort by timestamp when merging
6. **Session isolation:** UUID-based session management
7. **Resource exhaustion:** Limits on sessions, storage, rate limiting

### Future Enhancements

- **Word-level alignment:** Replace individual words instead of chunks (using forced alignment)
- **Speaker diarization:** Detect multiple speakers in merged chunks
- **Language auto-detection:** Remove manual language selection
- **WebSocket upgrade:** For lower latency if needed
- **Punctuation restoration:** Post-process finalized text
- **Custom vocabulary:** Domain-specific word boosting

## Success Metrics

- Draft transcription latency: <2s from silence detection
- Final transcription latency: <5s from merge trigger
- Transcription accuracy: >95% WER (Word Error Rate)
- UI responsiveness: <100ms for draft text display
- Session stability: Handle 30+ minute continuous recording
- Chunk upload success rate: >99% (with retries)

## Testing Strategy

1. **Unit tests:** Silence detection algorithm, audio merging, chunk management
2. **Integration tests:** End-to-end flow from recording to finalized transcription
3. **Performance tests:** Concurrent sessions, long recordings, network failures
4. **UX tests:** Latency perception, draft→final transitions, error recovery
5. **Edge case tests:** Continuous speech, background noise, network interruptions
