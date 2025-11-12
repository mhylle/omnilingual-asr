#!/usr/bin/env python3
"""
Omnilingual ASR REST API

FastAPI server for speech recognition supporting 1,600+ languages.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import tempfile
import shutil
import logging
from datetime import datetime
from pydub import AudioSegment

from transcriber import Transcriber
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global transcriber instance (lazy loaded)
_transcriber: Optional[Transcriber] = None
_current_model: Optional[str] = None


def get_transcriber(model: str = settings.default_model) -> Transcriber:
    """
    Get or create transcriber instance.

    Reuses existing transcriber if same model is requested.
    """
    global _transcriber, _current_model

    if _transcriber is None or _current_model != model:
        logger.info(f"Loading model: {model}")
        _transcriber = Transcriber(model=model)
        _current_model = model

    return _transcriber


def convert_to_wav(input_path: Path, output_path: Path = None) -> Path:
    """
    Convert audio file to WAV format (16kHz, mono) required by ASR pipeline.

    Args:
        input_path: Path to input audio file (any format)
        output_path: Optional output path. If None, creates a new temp file.

    Returns:
        Path to converted WAV file

    Raises:
        Exception: If conversion fails
    """
    try:
        logger.info(f"Converting audio: {input_path.name} ({input_path.suffix})")

        # Load audio in any format
        audio = AudioSegment.from_file(str(input_path))

        # Convert to mono
        if audio.channels > 1:
            logger.info(f"Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)

        # Convert to 16kHz sample rate
        if audio.frame_rate != 16000:
            logger.info(f"Resampling from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)

        # Create output path if not provided
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            output_path = Path(temp_file.name)
            temp_file.close()

        # Export as WAV
        audio.export(
            str(output_path),
            format="wav",
            parameters=["-acodec", "pcm_s16le"]  # 16-bit PCM
        )

        logger.info(f"Converted to WAV: {output_path} ({audio.duration_seconds:.1f}s)")
        return output_path

    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise Exception(f"Failed to convert audio to WAV format: {str(e)}")


async def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Save uploaded file to temporary location.

    Args:
        upload_file: FastAPI upload file

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file save fails
    """
    try:
        # Validate file extension
        file_ext = Path(upload_file.filename).suffix.lower()
        if file_ext not in settings.allowed_audio_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {settings.allowed_audio_formats}"
            )

        # Create temp file
        suffix = file_ext or ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)

        # Save uploaded content
        with temp_file as f:
            shutil.copyfileobj(upload_file.file, f)

        logger.info(f"Saved upload to: {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Create upload directory
    Path(settings.upload_dir).mkdir(exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": _transcriber is not None,
        "current_model": _current_model
    }


@app.get("/models")
async def list_models():
    """List available ASR models."""
    return {
        "models": list(Transcriber.AVAILABLE_MODELS.keys()),
        "details": {
            "ctc": {
                "ctc_300m": {"params": "300M", "type": "CTC", "speed": "fastest"},
                "ctc_1b": {"params": "1B", "type": "CTC", "speed": "fast", "recommended": True},
                "ctc_3b": {"params": "3B", "type": "CTC", "speed": "medium"},
                "ctc_7b": {"params": "7B", "type": "CTC", "speed": "slower"}
            },
            "llm": {
                "llm_300m": {"params": "300M", "type": "LLM", "language_aware": True},
                "llm_1b": {"params": "1B", "type": "LLM", "language_aware": True},
                "llm_3b": {"params": "3B", "type": "LLM", "language_aware": True},
                "llm_7b": {"params": "7B", "type": "LLM", "language_aware": True, "vram": "~17GB"},
                "llm_7b_zs": {"params": "7B", "type": "LLM", "zero_shot": True}
            }
        },
        "default": settings.default_model
    }


@app.get("/languages")
async def list_languages():
    """List common supported languages."""
    return {
        "total_supported": "1600+",
        "common_languages": Transcriber.COMMON_LANGUAGES,
        "format": "{language_code}_{script}",
        "examples": {
            "english": "eng_Latn",
            "spanish": "spa_Latn",
            "chinese_simplified": "cmn_Hans",
            "arabic": "arb_Arab"
        }
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Query(settings.default_model, description="Model to use"),
    language: Optional[str] = Query(None, description="Language code or name"),
):
    """
    Transcribe a single audio file.

    Args:
        file: Audio file (WAV, FLAC, MP3, WebM, etc.)
        model: Model name (e.g., 'ctc_1b', 'llm_1b')
        language: Language code (e.g., 'eng_Latn', 'english')

    Returns:
        Transcription result with detailed timing breakdown

    Limitations:
        - Audio must be ≤40 seconds
        - Max file size: 50MB

    Note:
        Audio is automatically converted to WAV 16kHz mono for processing.
    """
    temp_path = None
    wav_path = None

    # Start timing
    request_start = datetime.utcnow()
    timings = {}

    try:
        # Validate model
        if model not in Transcriber.AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Use /models to see available options."
            )

        # Save uploaded file
        upload_start = datetime.utcnow()
        temp_path = await save_upload_file(file)
        upload_end = datetime.utcnow()
        timings['upload_save'] = (upload_end - upload_start).total_seconds()

        # Convert to WAV format (16kHz mono) required by ASR pipeline
        logger.info(f"Converting {file.filename} to WAV format for transcription")
        conversion_start = datetime.utcnow()
        wav_path = convert_to_wav(temp_path)
        conversion_end = datetime.utcnow()
        timings['audio_conversion'] = (conversion_end - conversion_start).total_seconds()

        # Get transcriber (may include model loading)
        model_load_start = datetime.utcnow()
        model_was_loaded = _transcriber is not None and _current_model == model
        transcriber = get_transcriber(model)
        model_load_end = datetime.utcnow()
        timings['model_load'] = (model_load_end - model_load_start).total_seconds()
        timings['model_was_cached'] = model_was_loaded

        # Transcribe
        logger.info(f"Transcribing {file.filename} with model {model}")
        transcription_start = datetime.utcnow()

        transcriptions = transcriber.transcribe(
            [wav_path],
            language=language
        )

        transcription_end = datetime.utcnow()
        timings['transcription'] = (transcription_end - transcription_start).total_seconds()

        # Calculate total time
        request_end = datetime.utcnow()
        timings['total'] = (request_end - request_start).total_seconds()

        return JSONResponse({
            "success": True,
            "transcription": transcriptions[0],
            "metadata": {
                "filename": file.filename,
                "model": model,
                "language": language,
                "processing_time": f"{timings['total']:.2f}s",
                "timestamp": request_end.isoformat()
            },
            "timings": {
                "upload_and_save": f"{timings['upload_save']:.3f}s",
                "audio_conversion": f"{timings['audio_conversion']:.3f}s",
                "model_loading": f"{timings['model_load']:.3f}s",
                "transcription": f"{timings['transcription']:.2f}s",
                "total_backend": f"{timings['total']:.2f}s",
                "model_was_cached": timings['model_was_cached']
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        if settings.cleanup_uploads:
            for path in [temp_path, wav_path]:
                if path and path.exists():
                    try:
                        path.unlink()
                        logger.info(f"Cleaned up: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {path}: {e}")


@app.post("/transcribe/batch")
async def transcribe_batch(
    files: List[UploadFile] = File(..., description="Audio files to transcribe"),
    model: str = Query(settings.default_model, description="Model to use"),
    language: Optional[str] = Query(None, description="Language code or name"),
    batch_size: int = Query(settings.default_batch_size, description="Batch size")
):
    """
    Transcribe multiple audio files in batch.

    Args:
        files: List of audio files (WAV, FLAC, MP3, WebM, etc.)
        model: Model name
        language: Language code
        batch_size: Processing batch size (1-10)

    Returns:
        List of transcription results

    Note:
        Audio files are automatically converted to WAV 16kHz mono for processing.
    """
    temp_paths = []
    wav_paths = []

    try:
        # Validate
        if len(files) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Max: {settings.max_batch_size}"
            )

        if batch_size > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Max: {settings.max_batch_size}"
            )

        # Save and convert all files
        logger.info(f"Processing {len(files)} files for batch transcription")
        for upload_file in files:
            # Save uploaded file
            temp_path = await save_upload_file(upload_file)
            temp_paths.append(temp_path)

            # Convert to WAV
            wav_path = convert_to_wav(temp_path)
            wav_paths.append(wav_path)

        # Get transcriber
        transcriber = get_transcriber(model)

        # Transcribe batch
        logger.info(f"Batch transcribing {len(files)} files with model {model}")
        start_time = datetime.utcnow()

        transcriptions = transcriber.transcribe(
            wav_paths,
            language=language,
            batch_size=batch_size
        )

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Build results
        results = []
        for file, text in zip(files, transcriptions):
            results.append({
                "filename": file.filename,
                "transcription": text
            })

        return JSONResponse({
            "success": True,
            "count": len(results),
            "results": results,
            "metadata": {
                "model": model,
                "language": language,
                "batch_size": batch_size,
                "processing_time": f"{duration:.2f}s",
                "avg_time_per_file": f"{duration/len(files):.2f}s",
                "timestamp": end_time.isoformat()
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup all temp files
        if settings.cleanup_uploads:
            for path_list in [temp_paths, wav_paths]:
                for path in path_list:
                    if path and path.exists():
                        try:
                            path.unlink()
                            logger.info(f"Cleaned up: {path}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup {path}: {e}")


@app.get("/info")
async def api_info():
    """Get detailed API information."""
    return {
        "api": {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": settings.app_description
        },
        "capabilities": {
            "languages": "1600+",
            "max_audio_duration": f"{settings.max_audio_duration}s",
            "max_file_size": f"{settings.max_file_size / (1024*1024):.0f}MB",
            "max_batch_size": settings.max_batch_size,
            "supported_formats": settings.allowed_audio_formats
        },
        "endpoints": {
            "health": "GET /health",
            "models": "GET /models",
            "languages": "GET /languages",
            "transcribe": "POST /transcribe",
            "batch_transcribe": "POST /transcribe/batch",
            "docs": "GET /docs"
        },
        "default_model": settings.default_model
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers
    )
