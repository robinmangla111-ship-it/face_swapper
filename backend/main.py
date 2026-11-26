"""
Face Swapper API Server
========================
FastAPI application providing REST endpoints and WebSocket for real-time
progress updates. Handles video upload, face discovery, swap execution,
and result download.
"""

import asyncio
import json
import logging
import shutil
import sys
import uuid

# Fix for Windows Python 3.12 ConnectionResetError (WinError 10054) on WebSocket disconnect
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        def _silence_connection_lost(self, exc):
            try:
                _orig_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                pass
        _ProactorBasePipeTransport._call_connection_lost = _silence_connection_lost
    except Exception:
        pass
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiofiles
import uvicorn
from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException,
    UploadFile, WebSocket, WebSocketDisconnect, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

import config
from core.pipeline import FaceSwapPipeline, SwapJob
from logger import (
    write_log, start_new_session_log, list_log_files,
    read_log_file, delete_all_logs, delete_single_log
)

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("face_swapper")

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Face Swapper API",
    description="Production-ready face swapping tool with parallel video processing",
    version="1.0.0",
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    write_log(f"Global Error on {request.method} {request.url.path}: {exc}", level="ERROR", exc=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}", "error_type": type(exc).__name__}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
# In production, these would be stored in Redis/DB
pipeline = FaceSwapPipeline()
videos: Dict[str, dict] = {}           # video_id → {path, info, ...}
source_faces: Dict[str, dict] = {}     # face_id → {path, thumbnail, ...}
jobs: Dict[str, SwapJob] = {}          # job_id → SwapJob
ws_connections: Dict[str, List[WebSocket]] = {}   # job_id → [WebSocket, ...]
main_loop: Optional[asyncio.AbstractEventLoop] = None


def ensure_video(video_id: str) -> dict:
    """Get video info from memory or recover from uploads folder on disk."""
    if video_id in videos:
        return videos[video_id]
    for ext in config.SUPPORTED_VIDEO_FORMATS:
        candidate = config.UPLOADS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            try:
                info = pipeline.get_video_info(str(candidate), video_id)
                videos[video_id] = {
                    "id": video_id,
                    "path": str(candidate),
                    "filename": f"{video_id}{ext}",
                    "info": info.to_dict(),
                }
                return videos[video_id]
            except Exception as e:
                logger.warning(f"Error auto-restoring video {candidate}: {e}")
    write_log(f"Video {video_id} not found in memory store or disk", level="ERROR")
    raise HTTPException(404, f"Video not found: {video_id}")


def ensure_source_face(face_id: str) -> dict:
    """Get source face info from memory or recover from uploads folder on disk."""
    if face_id in source_faces:
        return source_faces[face_id]
    for ext in config.SUPPORTED_IMAGE_FORMATS:
        candidate = config.UPLOADS_DIR / f"face_{face_id}{ext}"
        if candidate.exists():
            try:
                face_info = pipeline.validate_source_face(str(candidate))
                source_faces[face_id] = {
                    "id": str(face_id),
                    "path": str(candidate),
                    "filename": candidate.name,
                    "thumbnail": str(face_info["thumbnail"]),
                    "confidence": float(face_info["confidence"]),
                }
                return source_faces[face_id]
            except Exception as e:
                logger.warning(f"Error auto-restoring source face {candidate}: {e}")
    write_log(f"Source face {face_id} not found in memory store or disk", level="ERROR")
    raise HTTPException(404, f"Source face not found: {face_id}")


# Serve outputs and character library images
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")
app.mount("/library-images", StaticFiles(directory=str(config.LIBRARY_IMAGES_DIR)), name="library-images")

from core.character_library import CharacterLibrary
character_library = CharacterLibrary()


@app.on_event("startup")
async def startup_event():
    """Pre-warm AI detection model so first request has zero cold-start delay."""
    global main_loop
    try:
        main_loop = asyncio.get_running_loop()
    except Exception:
        main_loop = asyncio.get_event_loop()
    logger.info("Pre-warming FaceAnalysis model in background...")
    await main_loop.run_in_executor(None, lambda: pipeline.detector._ensure_loaded())
    logger.info("FaceAnalysis model pre-warmed and ready.")


# ─── Request/Response Models ─────────────────────────────────────────────────

class SwapRequest(BaseModel):
    video_id: str
    source_face_id: str
    character_id: Union[int, str]
    enhance: bool = True
    num_workers: Optional[int] = None
    swap_mode: str = "auto"
    fal_key: Optional[str] = None

    @field_validator("character_id", mode="before")
    @classmethod
    def parse_character_id(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: str
    frames_processed: int
    total_frames: int
    elapsed_seconds: float
    eta_seconds: float
    error: Optional[str]
    output_path: Optional[str]


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "ffmpeg": config.FFMPEG_PATH is not None,
        "gpu": config.USE_GPU,
        "execution_providers": config.EXECUTION_PROVIDERS,
        "max_workers": config.EFFECTIVE_MAX_WORKERS,
    }


# ─── Video Upload ────────────────────────────────────────────────────────────

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for face swapping."""
    safe_stem = "".join(c for c in (Path(file.filename or "video").stem) if c.isalnum() or c in "-_")[:20] or "video"
    start_new_session_log(f"upload_{safe_stem}")
    write_log(f"Received video upload request: {file.filename}")

    # Validate file extension
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in config.SUPPORTED_VIDEO_FORMATS:
        write_log(f"Unsupported format: {ext}", level="WARNING")
        raise HTTPException(400, f"Unsupported format: {ext}. "
                                  f"Supported: {config.SUPPORTED_VIDEO_FORMATS}")

    # Generate unique ID
    video_id = str(uuid.uuid4())[:12]
    filename = f"{video_id}{ext}"
    save_path = config.UPLOADS_DIR / filename
    write_log(f"Allocated video_id: {video_id}, saving to: {save_path}")

    # Save uploaded file
    try:
        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
        write_log(f"File saved to disk ({save_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        write_log(f"Failed to save uploaded file: {e}", level="ERROR", exc=e)
        raise HTTPException(500, f"Failed to save file: {e}")

    # Validate and get video info
    try:
        write_log("Probing video metadata with VideoProcessor...")
        video_info = pipeline.get_video_info(str(save_path), video_id)
        write_log(f"Video parsed: {video_info.width}x{video_info.height} @ {video_info.fps}fps, "
                  f"{video_info.total_frames} frames, {video_info.duration:.1f}s")
    except Exception as e:
        write_log(f"Invalid video file during metadata probe: {e}", level="ERROR", exc=e)
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid video file: {e}")

    # Store in state
    videos[video_id] = {
        "id": video_id,
        "path": str(save_path),
        "filename": file.filename,
        "info": video_info.to_dict(),
    }

    write_log(f"Video {video_id} registered successfully.")
    return {
        "video_id": video_id,
        "filename": file.filename,
        "info": video_info.to_dict(),
    }


# Lock to prevent concurrent ONNX model inference crashes (c0000005)
discovery_lock = asyncio.Lock()

# ─── Face Discovery ──────────────────────────────────────────────────────────

@app.get("/api/faces/{video_id}")
async def detect_faces(video_id: str):
    """Discover unique characters/faces in an uploaded video."""
    write_log(f"Character discovery requested for video: {video_id}")
    video_data = ensure_video(video_id)

    try:
        async with discovery_lock:
            # Check cache first inside lock
            if video_id in pipeline._character_cache:
                write_log(f"Returning cached characters for {video_id}")
                characters = pipeline._character_cache[video_id]
            else:
                write_log("Starting frame sampling and face clustering in executor...")
                global main_loop
                loop = main_loop or asyncio.get_event_loop()
                characters = await loop.run_in_executor(
                    None,
                    lambda: pipeline.discover_characters(
                        video_path=video_data["path"],
                        video_id=video_id,
                    )
                )
                write_log(f"Character discovery complete: {len(characters)} unique characters found.")
    except Exception as e:
        write_log(f"Face discovery failed: {e}", level="ERROR", exc=e)
        raise HTTPException(500, f"Face detection failed: {e}")

    chars_data = []
    for c in characters:
        cd = c.to_dict()
        chars_data.append({
            "id": int(cd["id"]),
            "label": str(cd["label"]),
            "face_count": int(cd["face_count"]),
            "thumbnail": str(cd["thumbnail"]),
            "best_score": float(cd["best_score"]),
        })

    return {
        "video_id": str(video_id),
        "characters": chars_data,
        "total_characters": int(len(chars_data)),
    }


# ─── Source Face Upload ───────────────────────────────────────────────────────

@app.post("/api/upload/source-face")
async def upload_source_face(file: UploadFile = File(...)):
    """Upload a source face image (the face to swap in)."""
    ext = Path(file.filename or "face.jpg").suffix.lower()
    if ext not in config.SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(400, f"Unsupported format: {ext}. "
                                  f"Supported: {config.SUPPORTED_IMAGE_FORMATS}")

    face_id = str(uuid.uuid4())[:12]
    filename = f"face_{face_id}{ext}"
    save_path = config.UPLOADS_DIR / filename

    try:
        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    # Validate face
    try:
        face_info = pipeline.validate_source_face(str(save_path))
    except ValueError as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Face validation failed: {e}")

    source_faces[face_id] = {
        "id": str(face_id),
        "path": str(save_path),
        "filename": str(file.filename or ""),
        "thumbnail": str(face_info["thumbnail"]),
        "confidence": float(face_info["confidence"]),
    }

    logger.info(f"Source face uploaded: {face_id} ({file.filename}, "
                f"confidence={face_info['confidence']})")

    return {
        "face_id": str(face_id),
        "filename": str(file.filename or ""),
        "thumbnail": str(face_info["thumbnail"]),
        "confidence": float(face_info["confidence"]),
        "face_count": int(face_info["face_count"]),
        "swap_mode": str(face_info.get("swap_mode", "neural")),
    }


# ─── Character Library (Hollywood, Bollywood, Avengers, Mythology, Custom) ───

@app.get("/api/library")
async def list_library_characters(category: Optional[str] = None):
    """List all characters in the preset & custom library."""
    return {
        "characters": character_library.list_characters(category=category),
    }


@app.post("/api/library")
async def add_library_character(
    file: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form("Custom"),
    swap_mode: str = Form("auto"),
):
    """Add a new character portrait to the persistent library."""
    ext = Path(file.filename or "char.jpg").suffix.lower()
    if ext not in config.SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    char_id = f"lib_{str(uuid.uuid4())[:8]}"
    filename = f"{char_id}{ext}"
    save_path = config.LIBRARY_IMAGES_DIR / filename

    try:
        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(500, f"Failed to save character image: {e}")

    try:
        face_info = pipeline.validate_source_face(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid character image: {e}")

    effective_mode = swap_mode if swap_mode in ("neural", "mask_overlay") else face_info.get("swap_mode", "auto")
    entry = character_library.add_character(
        name=name,
        category=category,
        filename=filename,
        swap_mode=effective_mode,
        is_custom=True,
        char_id=char_id,
    )
    return {"character": entry}


@app.post("/api/library/{char_id}/select")
async def select_library_character(char_id: str):
    """Select a character from the library as the active source face."""
    char = character_library.get_character(char_id)
    if not char:
        raise HTTPException(404, f"Library character not found: {char_id}")

    img_path = Path(char["image_path"])
    face_info = pipeline.validate_source_face(str(img_path))

    # Copy into uploads so ensure_source_face can always auto-recover it
    face_id = f"lib_{char_id}"
    ext = img_path.suffix.lower() or ".jpg"
    dest_path = config.UPLOADS_DIR / f"face_{face_id}{ext}"
    if not dest_path.exists():
        import shutil
        shutil.copy2(img_path, dest_path)

    effective_mode = char.get("swap_mode", "auto")
    if effective_mode == "auto":
        effective_mode = face_info.get("swap_mode", "neural")

    source_faces[face_id] = {
        "id": face_id,
        "path": str(dest_path),
        "filename": char["name"],
        "thumbnail": str(face_info["thumbnail"]),
        "confidence": float(face_info["confidence"]),
        "swap_mode": effective_mode,
    }

    return {
        "face_id": face_id,
        "filename": char["name"],
        "thumbnail": str(face_info["thumbnail"]),
        "confidence": float(face_info["confidence"]),
        "face_count": int(face_info["face_count"]),
        "swap_mode": effective_mode,
    }


@app.delete("/api/library/{char_id}")
async def delete_library_character(char_id: str):
    """Delete a character from the library."""
    deleted = character_library.delete_character(char_id)
    if not deleted:
        raise HTTPException(404, f"Library character not found: {char_id}")
    return {"deleted": True, "id": char_id}


# ─── Start Face Swap ─────────────────────────────────────────────────────────

@app.post("/api/swap")
async def start_swap(request: SwapRequest, background_tasks: BackgroundTasks):
    """Start a face swap processing job."""
    # Validate and ensure inputs (auto-restores from disk if backend reloaded)
    video_data = ensure_video(request.video_id)
    face_data = ensure_source_face(request.source_face_id)

    # Create job
    job_id = str(uuid.uuid4())[:12]
    job = SwapJob(
        job_id=job_id,
        video_id=request.video_id,
        video_path=video_data["path"],
        source_face_path=face_data["path"],
        character_id=request.character_id,
        swap_mode=request.swap_mode or face_data.get("swap_mode", "auto"),
        fal_key=request.fal_key,
    )
    jobs[job_id] = job

    # Start processing in background
    background_tasks.add_task(
        _run_swap_job, job, request.enhance, request.num_workers
    )

    logger.info(f"Swap job started: {job_id}")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Processing started. Use WebSocket or polling for progress.",
    }


async def _run_swap_job(job: SwapJob, enhance: bool, num_workers: Optional[int]):
    """Background task to run face swap processing."""
    global main_loop
    loop = main_loop or asyncio.get_event_loop()

    def progress_callback(processed, total):
        # Notify WebSocket clients safely across thread boundary
        try:
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    _broadcast_progress(job.job_id, job.to_dict()),
                    loop
                )
        except Exception:
            pass

    try:
        # Run processing in a thread pool to not block the event loop
        await loop.run_in_executor(
            None,
            lambda: pipeline.process_video(
                job=job,
                num_workers=num_workers,
                enhance=enhance,
                progress_callback=progress_callback,
            )
        )

        # Final broadcast
        await _broadcast_progress(job.job_id, job.to_dict())

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.error(f"Swap job {job.job_id} failed: {e}")
        try:
            await _broadcast_progress(job.job_id, job.to_dict())
        except Exception:
            pass


# ─── Job Status ───────────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the current status of a swap job."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    return jobs[job_id].to_dict()


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": [j.to_dict() for j in jobs.values()]}


# ─── WebSocket Progress ──────────────────────────────────────────────────────

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = []
    ws_connections[job_id].append(websocket)

    logger.info(f"WebSocket connected for job {job_id}")

    try:
        # Send current status immediately
        if job_id in jobs:
            await websocket.send_json(jobs[job_id].to_dict())

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong or close)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
            except WebSocketDisconnect:
                break

    finally:
        if job_id in ws_connections:
            ws_connections[job_id] = [
                ws for ws in ws_connections[job_id] if ws != websocket
            ]
        logger.info(f"WebSocket disconnected for job {job_id}")


async def _broadcast_progress(job_id: str, data: dict):
    """Broadcast progress update to all WebSocket clients for a job."""
    if job_id not in ws_connections:
        return

    disconnected = []
    for ws in ws_connections[job_id]:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected
    for ws in disconnected:
        ws_connections[job_id].remove(ws)


# ─── Download Result ──────────────────────────────────────────────────────────

@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download the processed video."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job not found: {job_id}")

    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(400, f"Job not completed. Status: {job.status}")

    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        path=str(output_path),
        filename=f"swapped_{job_id}.mp4",
        media_type="video/mp4",
    )


# ─── Logs & Diagnostics ───────────────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs_list():
    """List all session log files."""
    return {"logs": list_log_files()}


@app.get("/api/logs/latest")
async def get_latest_log():
    """Get contents of the most recent log file."""
    return read_log_file()


@app.get("/api/logs/{filename}")
async def get_log_content(filename: str):
    """Get contents of a specific log file."""
    return read_log_file(filename)


@app.delete("/api/logs")
async def clear_logs():
    """Delete all older log files."""
    deleted_count = delete_all_logs()
    write_log(f"All older log files cleared ({deleted_count} files removed).")
    return {"deleted": deleted_count, "message": f"Successfully deleted {deleted_count} log files"}


@app.delete("/api/logs/{filename}")
async def delete_log(filename: str):
    """Delete a single log file."""
    success = delete_single_log(filename)
    if not success:
        raise HTTPException(404, f"Log file not found: {filename}")
    return {"deleted": filename, "success": True}


# ─── Cleanup ──────────────────────────────────────────────────────────────────

@app.delete("/api/cleanup")
async def cleanup():
    """Clean up all temporary files and cached data."""
    cleaned = {"uploads": 0, "outputs": 0, "temp": 0}

    for d_name, d_path in [("uploads", config.UPLOADS_DIR),
                            ("outputs", config.OUTPUTS_DIR),
                            ("temp", config.TEMP_DIR)]:
        for f in d_path.iterdir():
            if f.name == ".gitkeep":
                continue
            try:
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()
                cleaned[d_name] += 1
            except Exception:
                pass

    videos.clear()
    source_faces.clear()
    jobs.clear()
    pipeline.clear_cache()

    return {"cleaned": cleaned}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Face Swapper API Server")
    logger.info("=" * 60)
    logger.info(f"  FFmpeg:    {config.FFMPEG_PATH or 'NOT FOUND'}")
    logger.info(f"  GPU:       {config.USE_GPU}")
    logger.info(f"  Providers: {config.EXECUTION_PROVIDERS}")
    logger.info(f"  Workers:   {config.EFFECTIVE_MAX_WORKERS}")
    logger.info(f"  Models:    {config.MODELS_DIR}")
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        workers=1,  # Single worker (processing is CPU-bound internally)
    )
