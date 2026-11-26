"""
Full-Body Character Replacement Engine
======================================
Supports two backends for replacing an entire character (body + outfit + head):
  1. Cloud Diffusion (fal.ai Wan-Animate / Kling) when FAL_KEY is configured:
     True photorealistic pose-driven whole-body character replacement on GPU.
  2. Local Offline Body-Scale Overlay (CPU):
     Tracks detected person bounding box + face keypoints to estimate full-body
     bounding box, then scales the character image to cover head-to-waist/hips
     using standard 7x face-height human proportion scaling.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _extract_full_body_bgra(char_img_path: str) -> Optional[np.ndarray]:
    """
    Load a character image and return it as BGRA with the foreground extracted.
    Supports:
      - RGBA PNG  → use alpha channel directly
      - Non-transparent image → GrabCut to separate foreground (character) from background
    Returns None if extraction fails.
    """
    path = Path(char_img_path)
    if not path.exists():
        return None

    # Load with alpha if PNG
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        # Has alpha channel — use directly
        bgra = img.copy()
        # Zero out near-transparent pixels for cleaner mask
        alpha = bgra[:, :, 3]
        bgra[:, :, 3] = np.where(alpha > 10, alpha, 0).astype(np.uint8)
        return bgra

    # RGB/BGR image — use GrabCut with central foreground rect
    bgr = img[:, :, :3] if img.shape[2] >= 3 else img
    h, w = bgr.shape[:2]

    # Assume character occupies ~90% of image center
    margin_x = max(2, w // 15)
    margin_y = max(2, h // 15)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Smooth mask edges
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = fg_mask
        return bgra
    except Exception as e:
        logger.warning(f"GrabCut failed: {e}, using full image")
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255  # Fully opaque fallback
        return bgra


def _estimate_body_bbox(face_bbox: np.ndarray, frame_shape: tuple,
                         body_scale_x: float = 2.5, body_scale_y: float = 7.0) -> tuple:
    """
    Given a detected face bounding box, estimate where the full body is.
    Uses standard 7x face-height for body height, 2.5x face-width for body width.

    Returns: (x1, y1, x2, y2) body bounding box, clamped to frame
    """
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in face_bbox]
    face_w = x2 - x1
    face_h = y2 - y1

    # Body width is wider than face
    body_w = face_w * body_scale_x
    # Body top is at the top of the head (~0.3x face_h above detected face top)
    body_top = y1 - face_h * 0.3
    # Body bottom is below the detected face by 7x face height
    body_bot = y1 + face_h * body_scale_y

    # Center horizontally on the face center
    face_cx = (x1 + x2) / 2.0
    bx1 = face_cx - body_w / 2.0
    bx2 = face_cx + body_w / 2.0

    # Clamp to frame
    bx1 = max(0, int(bx1))
    bx2 = min(fw, int(bx2))
    by1 = max(0, int(body_top))
    by2 = min(fh, int(body_bot))

    return bx1, by1, bx2, by2


def _overlay_character_at_body(frame: np.ndarray, char_bgra: np.ndarray,
                                body_bbox: tuple, smooth_alpha: float = 0.0) -> np.ndarray:
    """
    Scale char_bgra to fit the body bounding box and alpha-composite onto frame.
    smooth_alpha: blend ratio for temporal smoothing of alpha channel (0 = disabled).
    """
    bx1, by1, bx2, by2 = body_bbox
    bw = bx2 - bx1
    bh = by2 - by1
    if bw <= 0 or bh <= 0:
        return frame

    # Resize character image to fit body bounding box
    resized = cv2.resize(char_bgra, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0

    # Soft-feather edges
    alpha_2d = cv2.GaussianBlur(alpha[:, :, 0], (5, 5), sigmaX=1.5)
    alpha = alpha_2d[:, :, np.newaxis]

    if np.max(alpha) < 0.02:
        return frame

    fg = resized[:, :, :3].astype(np.float32)
    bg = frame[by1:by2, bx1:bx2].astype(np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)

    result = frame.copy()
    result[by1:by2, bx1:bx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


class CharacterSwapper:
    """Handles full-body character replacement via Cloud AI or Local CPU alignment."""

    def __init__(self, fal_key: Optional[str] = None):
        from config import FAL_KEY
        self.fal_key = (fal_key or FAL_KEY or "").strip()

    @property
    def has_cloud_gpu(self) -> bool:
        return bool(self.fal_key)

    def process_full_body_cloud(
        self,
        video_path: str,
        character_image_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Run true photorealistic whole-body character replacement using fal.ai Wan-Animate.
        """
        import base64
        import httpx

        if not self.fal_key:
            raise ValueError("FAL_KEY is required for Cloud AI full-body character replacement.")

        headers = {
            "Authorization": f"Key {self.fal_key}",
            "Content-Type": "application/json",
        }

        def _to_data_uri(path: str, mime: str) -> str:
            raw = Path(path).read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return f"data:{mime};base64,{b64}"

        if progress_callback:
            progress_callback(10, 100)

        video_uri = _to_data_uri(video_path, "video/mp4")
        ext = Path(character_image_path).suffix.lower().lstrip(".") or "png"
        mime = "image/png" if ext == "png" else "image/jpeg"
        image_uri = _to_data_uri(character_image_path, mime)

        endpoint = "https://queue.fal.run/fal-ai/wan-2.2-animate/replace"
        payload = {
            "video_url": video_uri,
            "image_url": image_uri,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"Cloud AI submission failed ({resp.status_code}): {resp.text[:300]}")
            data = resp.json()
            status_url = data.get("status_url")
            response_url = data.get("response_url")
            if not status_url or not response_url:
                raise RuntimeError(f"Unexpected fal.ai queue response: {data}")

            pct = 20
            while True:
                time.sleep(3.0)
                st = client.get(status_url, headers=headers).json()
                state = st.get("status", "")
                if state == "COMPLETED":
                    break
                if state in ("FAILED", "CANCELLED"):
                    raise RuntimeError(f"Cloud AI job failed: {st}")
                pct = min(90, pct + 3)
                if progress_callback:
                    progress_callback(pct, 100)

            res = client.get(response_url, headers=headers).json()
            out_video_url = (
                res.get("video", {}).get("url")
                or res.get("output", {}).get("url")
            )
            if not out_video_url:
                raise RuntimeError(f"Cloud AI returned no video URL: {res}")

            dl = client.get(out_video_url, timeout=180.0)
            dl.raise_for_status()
            Path(output_path).write_bytes(dl.content)

        if progress_callback:
            progress_callback(100, 100)
        return output_path

    def process_full_body_local(
        self,
        video_path: str,
        character_image_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        body_scale_x: float = 2.5,
        body_scale_y: float = 7.0,
    ) -> str:
        """
        Local CPU full-body character overlay replacement.

        Strategy:
        1. Detect person's face in each frame using InsightFace
        2. From face bbox, estimate where the full body is using standard human proportions
           (body height ≈ 7× face height, body width ≈ 2.5× face width)
        3. Scale the character image to fit the estimated body bounding box
        4. Alpha-composite the character over the person with temporal smoothing

        This replaces the ENTIRE person silhouette with the character image,
        including body, costume/suit, and head.
        """
        from core.face_detector import FaceDetector
        from core.video_processor import create_video_writer

        logger.info(f"Full-body local replacement: {character_image_path} → {video_path}")

        # Extract character foreground
        char_bgra = _extract_full_body_bgra(character_image_path)
        if char_bgra is None:
            raise ValueError(f"Cannot load character image: {character_image_path}")
        logger.info(f"Character image loaded: {char_bgra.shape}, "
                    f"alpha_max={char_bgra[:,:,3].max()}")

        detector = FaceDetector()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = create_video_writer(output_path, width, height, fps, crf=15)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create VideoWriter: {output_path}")

        # Temporal smoothing state for body bbox
        smooth_bbox: Optional[np.ndarray] = None   # [x1, y1, x2, y2] smoothed
        miss_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = detector.detect_faces(frame)
            best_face = None
            if faces:
                # Pick highest confidence face
                best_face = max(faces, key=lambda f: f.det_score)
                if best_face.det_score < 0.25:
                    best_face = None

            if best_face is not None:
                # Estimate full body bounding box from face
                raw_bbox = np.array(_estimate_body_bbox(
                    best_face.bbox, frame.shape,
                    body_scale_x=body_scale_x,
                    body_scale_y=body_scale_y,
                ), dtype=np.float32)

                # Temporal smoothing (EMA)
                if smooth_bbox is None:
                    smooth_bbox = raw_bbox
                else:
                    smooth_bbox = 0.6 * raw_bbox + 0.4 * smooth_bbox
                miss_count = 0
            else:
                miss_count += 1
                if miss_count > 15:
                    smooth_bbox = None   # Lost track, reset

            if smooth_bbox is not None:
                bbox_int = tuple(int(v) for v in smooth_bbox)
                frame = _overlay_character_at_body(frame, char_bgra, bbox_int)

            writer.write(frame)
            frame_idx += 1

            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        writer.release()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        logger.info(f"Full-body local replacement complete: {frame_idx} frames written → {output_path}")
        return output_path
