"""
Chunk Manager Module
=====================
Splits video processing into chunks for parallel execution.
Uses multiprocessing to process multiple frame ranges simultaneously.
Handles progress tracking across workers.
"""

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""
    chunk_id: int
    output_path: str
    start_frame: int
    end_frame: int
    frames_processed: int
    processing_time: float
    success: bool
    error: Optional[str] = None


def _process_chunk_worker(args: tuple) -> dict:
    """
    Worker function for processing a video chunk.
    Runs in a separate process with its own model instances.

    Args (unpacked from tuple):
        video_path, source_face_path, char_embedding_list,
        start_frame, end_frame, output_chunk_path,
        similarity_threshold, enhance_enabled, chunk_id, progress_file

    Returns:
        Dict with chunk processing results
    """
    (video_path, source_face_path, char_embedding_list,
     start_frame, end_frame, output_chunk_path,
     similarity_threshold, enhance_enabled, chunk_id, progress_file, *extra) = args
    always_swap_if_single = extra[0] if extra else False
    other_embedding_list = extra[1] if len(extra) > 1 else None
    swap_mode = extra[2] if len(extra) > 2 else "auto"
    other_embeddings = np.array(other_embedding_list, dtype=np.float32) if other_embedding_list is not None else None

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    start_time = time.time()

    try:
        from core.face_detector import FaceDetector
        from core.face_swapper import FaceSwapper

        # Initialize models in this worker process
        detector = FaceDetector()
        swapper = FaceSwapper()

        enhancer = None
        if enhance_enabled:
            try:
                from core.face_enhancer import FaceEnhancer
                enhancer = FaceEnhancer()
            except Exception:
                pass

        # Load source face (supports both neural human faces and stylized/masked overlays)
        source_img = cv2.imread(source_face_path)
        if source_img is None:
            raise ValueError(f"Cannot read source face: {source_face_path}")

        source_face = None
        overlay_bgra, overlay_kps = None, None
        effective_mode = swap_mode

        if effective_mode in ("auto", "neural"):
            source_faces = detector.detect_faces_robust(source_img)
            if source_faces:
                source_face = source_faces[0]
                effective_mode = "neural"
            elif effective_mode == "auto":
                effective_mode = "mask_overlay"
            else:
                raise ValueError("No human face detected in source image for neural swap")

        if effective_mode == "mask_overlay":
            overlay_bgra, overlay_kps = detector.prepare_mask_overlay_source(source_face_path)

        # Reconstruct character embedding
        char_embedding = np.array(char_embedding_list, dtype=np.float32)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create high-quality video writer for this chunk (H.264 direct pipe)
        from core.video_processor import create_video_writer
        writer = create_video_writer(output_chunk_path, width, height, fps, crf=15)

        if not writer.isOpened():
            raise RuntimeError(f"Cannot create VideoWriter: {output_chunk_path}")

        frames_processed = 0
        total_chunk_frames = end_frame - start_frame
        swapper.reset_tracker()

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in this frame
            faces = detector.detect_faces(frame)

            # Swap matching faces
            if faces:
                frame, swapped_boxes = swapper.swap_faces_in_frame(
                    frame, faces, source_face,
                    char_embedding, similarity_threshold,
                    always_swap_if_single=always_swap_if_single,
                    other_embeddings=other_embeddings,
                    swap_mode=effective_mode,
                    overlay_bgra=overlay_bgra,
                    overlay_kps=overlay_kps,
                )

                # Enhance swapped face regions for studio sharpness
                if enhancer is not None and effective_mode == "neural" and swapped_boxes:
                    try:
                        frame = enhancer.enhance_frame(frame, swapped_boxes)
                    except Exception as e:
                        logger.warning(f"Enhancement error: {e}")

            writer.write(frame)
            frames_processed += 1

            # Write progress to file (every 10 frames)
            if progress_file and frames_processed % 10 == 0:
                try:
                    with open(progress_file, 'w') as f:
                        f.write(f"{frames_processed}/{total_chunk_frames}")
                except Exception:
                    pass

        cap.release()
        writer.release()

        # Final progress update
        if progress_file:
            try:
                with open(progress_file, 'w') as f:
                    f.write(f"{frames_processed}/{total_chunk_frames}")
            except Exception:
                pass

        elapsed = time.time() - start_time
        return {
            "chunk_id": chunk_id,
            "output_path": output_chunk_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_processed": frames_processed,
            "processing_time": elapsed,
            "success": True,
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Chunk {chunk_id} failed: {e}")
        return {
            "chunk_id": chunk_id,
            "output_path": output_chunk_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_processed": 0,
            "processing_time": elapsed,
            "success": False,
            "error": str(e),
        }


class ChunkManager:
    """
    Manages parallel chunk-based video processing.
    Splits video into frame ranges, processes in parallel, and merges results.
    """

    def __init__(self, num_workers: Optional[int] = None):
        from config import EFFECTIVE_MAX_WORKERS
        self.num_workers = num_workers or EFFECTIVE_MAX_WORKERS
        logger.info(f"ChunkManager initialized with {self.num_workers} workers")

    def create_chunks(self, total_frames: int,
                       min_chunk_size: int = 200) -> List[Tuple[int, int]]:
        """
        Split frame range into chunks for parallel processing.

        Args:
            total_frames: Total number of frames in the video
            min_chunk_size: Minimum frames per chunk

        Returns:
            List of (start_frame, end_frame) tuples
        """
        from config import AUTO_PARALLEL_THRESHOLD

        # For short videos, use single chunk
        if total_frames <= AUTO_PARALLEL_THRESHOLD:
            return [(0, total_frames)]

        # Calculate optimal number of chunks
        num_chunks = min(self.num_workers, total_frames // min_chunk_size)
        num_chunks = max(1, num_chunks)

        frames_per_chunk = total_frames // num_chunks
        chunks = []

        for i in range(num_chunks):
            start = i * frames_per_chunk
            end = (i + 1) * frames_per_chunk if i < num_chunks - 1 else total_frames
            chunks.append((start, end))

        logger.info(f"Created {len(chunks)} chunks from {total_frames} frames:")
        for i, (s, e) in enumerate(chunks):
            logger.info(f"  Chunk {i}: frames {s}-{e} ({e - s} frames)")

        return chunks

    def process_parallel(self, video_path: str,
                         source_face_path: str,
                         char_embedding: np.ndarray,
                         chunks: List[Tuple[int, int]],
                         temp_dir: str,
                         similarity_threshold: float = 0.22,
                         enhance_enabled: bool = False,
                         always_swap_if_single: bool = False,
                         other_embeddings: Optional[np.ndarray] = None,
                         swap_mode: str = "auto",
                         progress_callback: Optional[Callable] = None) -> List[ChunkResult]:
        """
        Process video chunks in parallel using multiprocessing.

        Args:
            video_path: Input video file path
            source_face_path: Path to source face image
            char_embedding: Character's centroid embedding (or multi-embedding array)
            chunks: List of (start_frame, end_frame) tuples
            temp_dir: Directory for temporary chunk files
            similarity_threshold: Face matching threshold (default 0.22)
            enhance_enabled: Whether to run face enhancement
            always_swap_if_single: If True, swap all faces in single-character videos
            other_embeddings: Embeddings of other characters to avoid false swaps
            progress_callback: Function called with (processed_frames, total_frames)

        Returns:
            List of ChunkResult objects
        """
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        # Prepare worker arguments
        worker_args = []
        for chunk_id, (start, end) in enumerate(chunks):
            chunk_output = str(temp_path / f"chunk_{chunk_id:04d}.mp4")
            progress_file = str(temp_path / f"progress_{chunk_id:04d}.txt")

            worker_args.append((
                str(video_path),
                str(source_face_path),
                char_embedding.tolist() if hasattr(char_embedding, 'tolist') else char_embedding,
                start, end,
                chunk_output,
                similarity_threshold,
                enhance_enabled,
                chunk_id,
                progress_file,
                always_swap_if_single,
                other_embeddings.tolist() if (other_embeddings is not None and hasattr(other_embeddings, 'tolist')) else other_embeddings,
                swap_mode,
            ))

        total_frames = sum(end - start for start, end in chunks)

        # Use multiprocessing if more than 1 chunk
        if len(chunks) > 1:
            logger.info(f"Processing {len(chunks)} chunks with "
                        f"{min(len(chunks), self.num_workers)} workers...")

            # For parallel processing, use spawn context for Windows compatibility
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=min(len(chunks), self.num_workers)) as pool:
                # Launch async workers
                async_results = [
                    pool.apply_async(_process_chunk_worker, (args,))
                    for args in worker_args
                ]

                # Monitor progress
                while not all(r.ready() for r in async_results):
                    if progress_callback:
                        processed = self._read_progress_files(temp_path, len(chunks))
                        progress_callback(processed, total_frames)
                    time.sleep(1)

                # Collect results
                results = []
                for ar in async_results:
                    try:
                        result_dict = ar.get(timeout=3600)
                        results.append(ChunkResult(**result_dict))
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                        results.append(ChunkResult(
                            chunk_id=-1, output_path="", start_frame=0,
                            end_frame=0, frames_processed=0,
                            processing_time=0, success=False, error=str(e)
                        ))

        else:
            # Single chunk — process in current process
            logger.info("Processing single chunk (sequential mode)...")
            result_dict = _process_chunk_worker(worker_args[0])
            results = [ChunkResult(**result_dict)]

            if progress_callback:
                progress_callback(total_frames, total_frames)

        # Sort by chunk_id to maintain order
        results.sort(key=lambda r: r.chunk_id)

        # Log summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_time = sum(r.processing_time for r in results)

        logger.info(f"Processing complete: {len(successful)} succeeded, "
                     f"{len(failed)} failed, total time: {total_time:.1f}s")

        if failed:
            for r in failed:
                logger.error(f"  Chunk {r.chunk_id} failed: {r.error}")

        return results

    def process_sequential(self, video_path: str, source_face_path: str,
                            char_embedding: np.ndarray,
                            output_path: str,
                            similarity_threshold: float = 0.22,
                            enhance_enabled: bool = False,
                            always_swap_if_single: bool = False,
                            other_embeddings: Optional[np.ndarray] = None,
                            swap_mode: str = "auto",
                            progress_callback: Optional[Callable] = None) -> str:
        """
        Process video sequentially (single process, memory efficient).
        Used for short videos or low-memory machines.
        """
        from config import TEMP_DIR
        from core.face_detector import FaceDetector
        from core.face_swapper import FaceSwapper

        detector = FaceDetector()
        swapper = FaceSwapper()

        enhancer = None
        if enhance_enabled:
            try:
                from core.face_enhancer import FaceEnhancer
                enhancer = FaceEnhancer()
            except Exception:
                logger.warning("Face enhancer not available")

        # Load source face (supports both neural human faces and stylized/masked overlays)
        source_img = cv2.imread(source_face_path)
        if source_img is None:
            raise ValueError(f"Cannot read source face: {source_face_path}")

        source_face = None
        overlay_bgra, overlay_kps = None, None
        effective_mode = swap_mode

        if effective_mode in ("auto", "neural"):
            source_faces = detector.detect_faces_robust(source_img)
            if source_faces:
                source_face = source_faces[0]
                effective_mode = "neural"
            elif effective_mode == "auto":
                effective_mode = "mask_overlay"
            else:
                raise ValueError("No human face detected in source image for neural swap")

        if effective_mode == "mask_overlay":
            overlay_bgra, overlay_kps = detector.prepare_mask_overlay_source(source_face_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # High-quality H.264 writer
        from core.video_processor import create_video_writer
        writer = create_video_writer(output_path, width, height, fps, crf=15)

        frame_idx = 0
        swapper.reset_tracker()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect_faces(frame)
            if faces:
                frame, swapped_boxes = swapper.swap_faces_in_frame(
                    frame, faces, source_face,
                    char_embedding, similarity_threshold,
                    always_swap_if_single=always_swap_if_single,
                    other_embeddings=other_embeddings,
                    swap_mode=effective_mode,
                    overlay_bgra=overlay_bgra,
                    overlay_kps=overlay_kps,
                )
                if enhancer is not None and effective_mode == "neural" and swapped_boxes:
                    try:
                        frame = enhancer.enhance_frame(frame, swapped_boxes)
                    except Exception as e:
                        logger.warning(f"Enhancement error: {e}")

            writer.write(frame)
            frame_idx += 1

            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        writer.release()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        return output_path

    @staticmethod
    def _read_progress_files(temp_dir: Path, num_chunks: int) -> int:
        """Read progress files from all workers and sum up processed frames."""
        total = 0
        for i in range(num_chunks):
            progress_file = temp_dir / f"progress_{i:04d}.txt"
            if progress_file.exists():
                try:
                    content = progress_file.read_text().strip()
                    if '/' in content:
                        processed = int(content.split('/')[0])
                        total += processed
                except Exception:
                    pass
        return total
