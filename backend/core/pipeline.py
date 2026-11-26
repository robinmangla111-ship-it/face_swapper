"""
Face Swap Pipeline
===================
Orchestrates the complete face swapping workflow:
  1. Character discovery (detect & cluster faces in video)
  2. Source face loading
  3. Video processing (sequential or parallel)
  4. Audio extraction & muxing
  5. Final output assembly

This is the main entry point for all face swapping operations.
"""

import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from core.face_detector import FaceDetector
from core.face_matcher import Character, FaceMatcher
from core.face_swapper import FaceSwapper
from core.video_processor import VideoInfo, VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class SwapJob:
    """Tracks the state of a face swap job."""
    job_id: str
    video_id: str
    video_path: str
    source_face_path: str
    character_id: int
    swap_mode: str = "auto"       # auto, neural, mask_overlay, full_body
    fal_key: Optional[str] = None
    output_path: str = ""
    status: str = "pending"       # pending, processing, completed, failed
    progress: float = 0.0         # 0.0 to 1.0
    current_step: str = ""
    frames_processed: int = 0
    total_frames: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None
    eta_seconds: float = 0.0

    def to_dict(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            "job_id": str(self.job_id),
            "status": str(self.status),
            "progress": float(round(float(self.progress), 3)),
            "current_step": str(self.current_step),
            "frames_processed": int(self.frames_processed),
            "total_frames": int(self.total_frames),
            "elapsed_seconds": float(round(float(elapsed), 1)),
            "eta_seconds": float(round(float(self.eta_seconds), 1)),
            "error": str(self.error) if self.error is not None else None,
            "output_path": str(self.output_path) if self.status == "completed" else None,
        }


class FaceSwapPipeline:
    """
    Complete face swapping pipeline.
    Handles character discovery, face swapping, and video assembly.
    """

    def __init__(self):
        from config import (FACE_DET_SIZE, FACE_DET_THRESH,
                             FACE_SIMILARITY_THRESH, FACE_SAMPLE_RATE)

        self.detector = FaceDetector(det_size=FACE_DET_SIZE, det_thresh=FACE_DET_THRESH)
        self.swapper = FaceSwapper()
        self.video_proc = VideoProcessor()
        self.matcher = FaceMatcher(similarity_threshold=FACE_SIMILARITY_THRESH)

        self._sample_rate = FACE_SAMPLE_RATE
        self._similarity_threshold = FACE_SIMILARITY_THRESH

        # Cache: video_id → characters
        self._character_cache: Dict[str, List[Character]] = {}
        self._video_info_cache: Dict[str, VideoInfo] = {}

    # ─── Character Discovery ──────────────────────────────────────────────────

    def discover_characters(self, video_path: str,
                            video_id: Optional[str] = None,
                            sample_rate: Optional[int] = None,
                            progress_callback: Optional[Callable] = None
                            ) -> List[Character]:
        """
        Discover unique characters/faces in a video.

        Samples frames, detects faces, clusters by identity,
        and returns a list of unique characters with thumbnails.

        Args:
            video_path: Path to video file
            video_id: Optional ID for caching results
            sample_rate: Override default frame sampling rate
            progress_callback: Function(step_description, progress_pct)

        Returns:
            List of Character objects
        """
        logger.info(f"Discovering characters in: {video_path}")

        if video_id and video_id in self._character_cache:
            logger.info(f"Returning cached characters for {video_id}")
            return self._character_cache[video_id]

        if progress_callback:
            progress_callback("Analyzing video...", 0.0)

        # Get video info
        video_info = self.video_proc.get_video_info(video_path)
        if video_id:
            self._video_info_cache[video_id] = video_info

        logger.info(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps}fps, "
                     f"{video_info.total_frames} frames, {video_info.duration:.1f}s")

        # Sample frames: sample every 12 frames up to 35 frames to capture all scenes and poses
        rate = sample_rate or 12
        if progress_callback:
            progress_callback("Sampling frames...", 0.1)

        sampled_frames = self.video_proc.sample_frames(
            video_path, every_n=rate, max_samples=35
        )

        # Detect faces in each sampled frame
        all_face_data = []
        for i, (frame_idx, frame) in enumerate(sampled_frames):
            faces = self.detector.detect_faces(frame)
            for face in faces:
                crop = self.detector.get_face_crop(frame, face)
                all_face_data.append({
                    'face': face,
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'embedding': face.embedding.copy(),
                    'crop': crop,
                    'det_score': face.det_score,
                })

            if progress_callback and i % 5 == 0:
                pct = 0.1 + 0.7 * (i / len(sampled_frames))
                progress_callback(f"Detecting faces... ({i}/{len(sampled_frames)} frames)", pct)

        logger.info(f"Detected {len(all_face_data)} face instances across "
                     f"{len(sampled_frames)} sampled frames")

        # Cluster faces into characters
        if progress_callback:
            progress_callback("Clustering characters...", 0.85)

        characters = self.matcher.cluster_faces(all_face_data)

        # Cache results
        if video_id:
            self._character_cache[video_id] = characters

        if progress_callback:
            progress_callback("Done!", 1.0)

        return characters

    # ─── Face Swap Processing ─────────────────────────────────────────────────

    def process_video(self, job: SwapJob,
                      num_workers: Optional[int] = None,
                      enhance: bool = False,
                      progress_callback: Optional[Callable] = None) -> str:
        """
        Full face swap pipeline: process video with face swapping.

        Args:
            job: SwapJob with video_path, source_face_path, character_id
            num_workers: Number of parallel workers (None = auto)
            enhance: Whether to run face enhancement (requires GFPGAN)
            progress_callback: Function(processed_frames, total_frames)

        Returns:
            Path to output video file
        """
        from config import TEMP_DIR, OUTPUTS_DIR, AUTO_PARALLEL_THRESHOLD

        job.status = "processing"
        job.start_time = time.time()

        try:
            # ── Step 1: Validate inputs ───────────────────────────────────
            job.current_step = "Validating inputs..."
            logger.info(f"Starting swap job {job.job_id}")

            video_info = self.video_proc.get_video_info(job.video_path)
            job.total_frames = video_info.total_frames

            # Validate source face (supports both neural human faces and stylized/masked overlays)
            source_img = cv2.imread(job.source_face_path)
            if source_img is None:
                raise ValueError("Cannot read source face image")
            effective_mode = job.swap_mode
            if effective_mode in ("auto", "neural"):
                source_faces = self.detector.detect_faces_robust(source_img)
                if source_faces:
                    effective_mode = "neural"
                elif effective_mode == "auto":
                    effective_mode = "mask_overlay"
                    logger.info("No human ArcFace detected in source image; using Full Mask/Character Overlay mode.")
                else:
                    raise ValueError("No human face detected in source image")

            # Get character embedding
            characters = self._character_cache.get(job.video_id, [])
            if not characters:
                logger.info(f"Characters not cached for {job.video_id}, discovering characters now...")
                characters = self.discover_characters(job.video_path, job.video_id)

            target_char = None
            for char in characters:
                if char.id == job.character_id:
                    target_char = char
                    break

            if target_char is None or target_char.centroid is None:
                # If ID didn't match, fallback to the primary character (ID 0)
                if characters and characters[0].centroid is not None:
                    target_char = characters[0]
                else:
                    raise ValueError(f"Character {job.character_id} not found")

            # Collect all representative embeddings for the target character across poses
            target_embeddings = []
            if target_char.centroid is not None:
                target_embeddings.append(target_char.centroid)
            for emb in target_char.embeddings:
                if emb is not None and len(emb) > 0:
                    target_embeddings.append(emb)

            char_embedding_array = np.array(target_embeddings, dtype=np.float32)

            # Collect competitor character embeddings to avoid swapping wrong actors
            other_embeddings = []
            for c in characters:
                if str(c.id) != str(target_char.id):
                    if c.centroid is not None:
                        other_embeddings.append(c.centroid)
                    for emb in c.embeddings:
                        if emb is not None and len(emb) > 0:
                            other_embeddings.append(emb)

            other_embedding_array = np.array(other_embeddings, dtype=np.float32) if other_embeddings else None
            is_single_character = (len(characters) <= 1)

            logger.info(f"Target character: {target_char.label} "
                         f"({target_char.face_count} detections, {len(target_embeddings)} pose embeddings, "
                         f"competitor_embeddings={len(other_embeddings)}, "
                         f"single_character_mode={is_single_character})")

            # ── Step 2: Extract audio ─────────────────────────────────────
            job.current_step = "Extracting audio..."
            logger.info("Extracting audio track...")
            audio_path = self.video_proc.extract_audio(job.video_path)

            # ── Step 3: Process frames ────────────────────────────────────
            job.current_step = "Swapping faces..."
            job_temp_dir = str(TEMP_DIR / f"job_{job.job_id}")
            Path(job_temp_dir).mkdir(parents=True, exist_ok=True)

            temp_video_path = str(Path(job_temp_dir) / "processed_video.mp4")

            from core.chunk_manager import ChunkManager
            chunk_mgr = ChunkManager(num_workers=num_workers)

            def _progress_wrapper(processed, total):
                job.frames_processed = processed
                job.total_frames = total
                if total > 0:
                    job.progress = processed / total
                    elapsed = time.time() - job.start_time
                    if processed > 0:
                        job.eta_seconds = (elapsed / processed) * (total - processed)
                if progress_callback:
                    try:
                        progress_callback(processed, total)
                    except Exception:
                        pass

            # Decide: full_body vs parallel vs sequential
            if effective_mode == "full_body":
                from core.character_swapper import CharacterSwapper
                char_swapper = CharacterSwapper(fal_key=job.fal_key)
                if char_swapper.has_cloud_gpu:
                    job.current_step = "Running Cloud AI Full-Body Replacement..."
                    logger.info("Using Cloud AI (fal.ai) for full-body character replacement")
                    char_swapper.process_full_body_cloud(
                        video_path=job.video_path,
                        character_image_path=job.source_face_path,
                        output_path=temp_video_path,
                        progress_callback=_progress_wrapper,
                    )
                else:
                    job.current_step = "Running Local Full-Body Character Alignment..."
                    logger.info("Using Local CPU full-body character alignment (no FAL_KEY set)")
                    char_swapper.process_full_body_local(
                        video_path=job.video_path,
                        character_image_path=job.source_face_path,
                        output_path=temp_video_path,
                        progress_callback=_progress_wrapper,
                    )
            else:
                # Decide: parallel or sequential
                use_parallel = video_info.total_frames > AUTO_PARALLEL_THRESHOLD
                num_w = num_workers or chunk_mgr.num_workers

                if use_parallel and num_w > 1:
                    # Parallel processing
                    chunks = chunk_mgr.create_chunks(video_info.total_frames)
                    logger.info(f"Using parallel processing: {len(chunks)} chunks, "
                                f"{num_w} workers")

                    results = chunk_mgr.process_parallel(
                        video_path=job.video_path,
                        source_face_path=job.source_face_path,
                        char_embedding=char_embedding_array,
                        chunks=chunks,
                        temp_dir=job_temp_dir,
                        similarity_threshold=self._similarity_threshold,
                        enhance_enabled=enhance,
                        always_swap_if_single=is_single_character,
                        other_embeddings=other_embedding_array,
                        swap_mode=effective_mode,
                        progress_callback=_progress_wrapper,
                    )

                    # Check for failures
                    failed = [r for r in results if not r.success]
                    if len(failed) == len(results):
                        raise RuntimeError("All chunks failed to process")

                    # Merge successful chunks
                    job.current_step = "Merging chunks..."
                    chunk_paths = [r.output_path for r in results if r.success]
                    if len(chunk_paths) > 1:
                        self.video_proc.concat_videos(chunk_paths, temp_video_path)
                    else:
                        shutil.copy2(chunk_paths[0], temp_video_path)

                else:
                    # Sequential processing
                    logger.info("Using sequential processing")
                    chunk_mgr.process_sequential(
                        video_path=job.video_path,
                        source_face_path=job.source_face_path,
                        char_embedding=char_embedding_array,
                        output_path=temp_video_path,
                        similarity_threshold=self._similarity_threshold,
                        enhance_enabled=enhance,
                        always_swap_if_single=is_single_character,
                        other_embeddings=other_embedding_array,
                        swap_mode=effective_mode,
                        progress_callback=_progress_wrapper,
                    )

            # ── Step 4: Mux audio + video ─────────────────────────────────
            job.current_step = "Assembling final video..."
            output_filename = f"swapped_{job.job_id}.mp4"
            output_path = str(OUTPUTS_DIR / output_filename)

            self.video_proc.merge_video_audio(
                temp_video_path, audio_path, output_path
            )

            # ── Step 5: Cleanup & finalize ────────────────────────────────
            job.current_step = "Cleaning up..."
            try:
                shutil.rmtree(job_temp_dir, ignore_errors=True)
                if audio_path and Path(audio_path).exists():
                    Path(audio_path).unlink(missing_ok=True)
            except Exception:
                pass

            job.output_path = output_path
            job.status = "completed"
            job.progress = 1.0
            job.end_time = time.time()
            job.current_step = "Done!"

            elapsed = job.end_time - job.start_time
            logger.info(f"Job {job.job_id} completed in {elapsed:.1f}s → {output_path}")

            return output_path

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = time.time()
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            raise

    # ─── Utility Methods ──────────────────────────────────────────────────────

    def get_video_info(self, video_path: str, video_id: Optional[str] = None) -> VideoInfo:
        """Get video metadata, with caching."""
        if video_id and video_id in self._video_info_cache:
            return self._video_info_cache[video_id]
        info = self.video_proc.get_video_info(video_path)
        if video_id:
            self._video_info_cache[video_id] = info
        return info

    def validate_source_face(self, image_path: str) -> dict:
        """Validate a source face image (human or stylized/masked character) and return face info."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image file")

        faces = self.detector.detect_faces_robust(img)
        if faces:
            best_face = faces[0]
            crop = self.detector.get_face_crop(img, best_face)
            thumbnail = FaceDetector.encode_face_thumbnail(crop, size=200)
            return {
                "face_count": int(len(faces)),
                "thumbnail": str(thumbnail),
                "confidence": float(round(float(best_face.det_score), 3)),
                "swap_mode": "neural",
            }

        # Stylized / Masked character fallback (e.g., Iron Man helmet, Hanuman)
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = img[y0:y0 + side, x0:x0 + side]
        thumbnail = FaceDetector.encode_face_thumbnail(crop, size=200)
        return {
            "face_count": 1,
            "thumbnail": str(thumbnail),
            "confidence": 1.0,
            "swap_mode": "mask_overlay",
        }

    def clear_cache(self, video_id: Optional[str] = None):
        """Clear cached data."""
        if video_id:
            self._character_cache.pop(video_id, None)
            self._video_info_cache.pop(video_id, None)
        else:
            self._character_cache.clear()
            self._video_info_cache.clear()
