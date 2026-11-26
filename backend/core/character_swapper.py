"""
Full-Body Character Replacement Engine
======================================
Supports two backends for replacing an entire character (body + outfit + head):
  1. Cloud Diffusion (fal.ai Wan-Animate / Kling) when FAL_KEY is configured:
     True photorealistic pose-driven whole-body character replacement on GPU.
  2. Local Offline Proportional Character Alignment (CPU):
     Aligns full costume/character silhouette to the tracked target actor's face
     and head coordinates while strictly preserving aspect ratio and natural human
     proportions. Applies exponential moving average (EMA) temporal smoothing
     to eliminate jitter across frames.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _extract_full_body_bgra(char_img_path: str) -> Optional[np.ndarray]:
    """
    Load a character image and return it as BGRA with the foreground extracted.
    Supports:
      - RGBA PNG with genuine alpha channel -> use alpha directly
      - High black/dark background ratio (>35%) -> direct luminance thresholding with smooth edge feathering
      - Natural/complex background -> GrabCut foreground segmentation
    """
    path = Path(char_img_path)
    if not path.exists():
        return None

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # 1. Check genuine RGBA transparency (at least 2% of pixels are transparent/semi-transparent)
    if img.shape[2] == 4 and (img[:, :, 3] < 240).sum() > (h * w * 0.02):
        bgra = img.copy()
        bgra[:, :, 3] = np.where(bgra[:, :, 3] > 15, bgra[:, :, 3], 0).astype(np.uint8)
        return bgra

    bgr = img[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2. Black or dark studio background detection
    black_ratio = float((gray < 25).mean())
    if black_ratio > 0.35:
        mask = np.where(gray > 20, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.2)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        return bgra

    # 3. GrabCut fallback for non-transparent photo backgrounds
    margin_x = max(2, w // 15)
    margin_y = max(2, h // 15)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = fg_mask
        return bgra
    except Exception as e:
        logger.warning(f"GrabCut failed: {e}, using full image")
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra


def get_costume_anchor(bgra: np.ndarray) -> Tuple[int, int, int]:
    """
    Finds the head center (cx, cy) and head height (head_h) in a costume silhouette.
    Works for both human faces and stylized masks/helmets by analyzing the upper silhouette.
    """
    h, w = bgra.shape[:2]
    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 25)
    if len(ys) == 0:
        return w // 2, h // 8, max(24, h // 7)

    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    fig_h = max_y - min_y

    # Head is located in the top 16% of the silhouette
    head_region_y = min_y + max(10, int(fig_h * 0.16))
    head_ys, head_xs = np.where(alpha[min_y:head_region_y, :] > 25)
    if len(head_xs) > 0:
        head_cx = int(head_xs.mean())
        head_cy = min_y + int(head_ys.mean())
    else:
        head_cx = (min_x + max_x) // 2
        head_cy = min_y + int(fig_h * 0.08)

    head_h = max(24, int(fig_h / 7.2))
    return head_cx, head_cy, head_h


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
        char_embedding: Optional[Union[np.ndarray, list]] = None,
        other_embeddings: Optional[Union[np.ndarray, list]] = None,
        always_swap_if_single: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        head_multiplier: float = 1.25,
    ) -> str:
        """
        Local CPU full-body character replacement.

        1. Accurately identifies the target person using face embeddings (does not swap competitor actors).
        2. Aligns costume head anchor to the target face center with strict aspect-ratio preservation.
        3. Applies exponential moving average (EMA) temporal smoothing on position and scale to eliminate jitter.
        4. Composites the costume onto the target body with feathered transparency blending.
        """
        from core.face_detector import FaceDetector
        from core.video_processor import create_video_writer

        logger.info(f"Starting local full-body replacement: {character_image_path} -> {video_path}")

        # 1. Load costume and find head anchor
        char_bgra = _extract_full_body_bgra(character_image_path)
        if char_bgra is None:
            raise ValueError(f"Cannot load character image: {character_image_path}")

        costume_cx, costume_cy, costume_h = get_costume_anchor(char_bgra)
        char_h, char_w = char_bgra.shape[:2]
        logger.info(f"Costume loaded: {char_w}x{char_h}, head_anchor=({costume_cx}, {costume_cy}), head_h={costume_h}")

        # 2. Parse reference embeddings
        ref_embeddings = []
        if char_embedding is not None:
            arr = np.asarray(char_embedding, dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                ref_embeddings = [arr]
            elif arr.ndim == 2 and arr.size > 0:
                ref_embeddings = [row for row in arr]

        oth_embeddings = []
        if other_embeddings is not None:
            arr_oth = np.asarray(other_embeddings, dtype=np.float32)
            if arr_oth.ndim == 1 and arr_oth.size > 0:
                oth_embeddings = [arr_oth]
            elif arr_oth.ndim == 2 and arr_oth.size > 0:
                oth_embeddings = [row for row in arr_oth]

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

        # Smoothed state: [x0, y0, nw, nh]
        smoothed_rect: Optional[np.ndarray] = None
        last_face_center: Optional[Tuple[float, float]] = None
        miss_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect_faces(frame)
            target_face = None

            if faces:
                if always_swap_if_single or not ref_embeddings:
                    # Pick face closest to previous tracking center, or highest confidence
                    if last_face_center is not None:
                        target_face = min(
                            faces,
                            key=lambda f: np.hypot(
                                (f.bbox[0] + f.bbox[2]) / 2.0 - last_face_center[0],
                                (f.bbox[1] + f.bbox[3]) / 2.0 - last_face_center[1]
                            )
                        )
                    else:
                        target_face = max(faces, key=lambda f: f.det_score)
                else:
                    # Multi-face discrimination: match against target embeddings
                    best_score = -1.0
                    for face in faces:
                        sim_tgt = max(
                            (FaceDetector.compute_similarity(face.embedding, emb) for emb in ref_embeddings),
                            default=0.0
                        )
                        sim_oth = max(
                            (FaceDetector.compute_similarity(face.embedding, emb) for emb in oth_embeddings),
                            default=0.0
                        ) if oth_embeddings else 0.0

                        # Reject if closer to a competitor character
                        if oth_embeddings and sim_oth > sim_tgt:
                            continue

                        # Spatial bonus
                        spatial_bonus = 0.0
                        if last_face_center is not None and miss_count <= 15:
                            cx = (face.bbox[0] + face.bbox[2]) / 2.0
                            cy = (face.bbox[1] + face.bbox[3]) / 2.0
                            dist = np.hypot(cx - last_face_center[0], cy - last_face_center[1])
                            diag = np.hypot(width, height)
                            norm_dist = dist / max(diag, 1.0)
                            if norm_dist < 0.25:
                                spatial_bonus = 0.25 * (1.0 - (norm_dist / 0.25))

                        score = sim_tgt + spatial_bonus
                        if score > best_score:
                            best_score = score
                            target_face = face

                    if best_score < 0.18 and last_face_center is not None and miss_count <= 15:
                        # Fallback to spatial tracking if embedding was temporarily missed
                        closest_face = min(
                            faces,
                            key=lambda f: np.hypot(
                                (f.bbox[0] + f.bbox[2]) / 2.0 - last_face_center[0],
                                (f.bbox[1] + f.bbox[3]) / 2.0 - last_face_center[1]
                            )
                        )
                        cx = (closest_face.bbox[0] + closest_face.bbox[2]) / 2.0
                        cy = (closest_face.bbox[1] + closest_face.bbox[3]) / 2.0
                        if np.hypot(cx - last_face_center[0], cy - last_face_center[1]) < width * 0.15:
                            target_face = closest_face

            if target_face is not None:
                fx1, fy1, fx2, fy2 = target_face.bbox
                fcx = (fx1 + fx2) / 2.0
                fcy = (fy1 + fy2) / 2.0
                fh = fy2 - fy1

                # Calculate aspect-ratio preserved scaling
                scale = (fh / max(costume_h, 1.0)) * head_multiplier
                curr_nw = max(10, int(char_w * scale))
                curr_nh = max(10, int(char_h * scale))
                curr_x0 = int(fcx - (costume_cx * scale))
                curr_y0 = int(fcy - (costume_cy * scale))

                curr_rect = np.array([curr_x0, curr_y0, curr_nw, curr_nh], dtype=np.float32)

                # Temporal smoothing (EMA)
                if smoothed_rect is None:
                    smoothed_rect = curr_rect
                else:
                    smoothed_rect = 0.65 * curr_rect + 0.35 * smoothed_rect

                last_face_center = (fcx, fcy)
                miss_count = 0
            else:
                miss_count += 1
                if miss_count > 15:
                    smoothed_rect = None
                    last_face_center = None

            # Render character overlay onto frame
            if smoothed_rect is not None:
                x0, y0, nw, nh = [int(v) for v in smoothed_rect]
                if nw > 10 and nh > 10:
                    resized_char = cv2.resize(char_bgra, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

                    rx1 = max(0, x0)
                    ry1 = max(0, y0)
                    rx2 = min(width, x0 + nw)
                    ry2 = min(height, y0 + nh)

                    cx1 = rx1 - x0
                    cy1 = ry1 - y0
                    cx2 = cx1 + (rx2 - rx1)
                    cy2 = cy1 + (ry2 - ry1)

                    if rx2 > rx1 and ry2 > ry1 and cx2 > cx1 and cy2 > cy1:
                        costume_roi = resized_char[cy1:cy2, cx1:cx2]
                        alpha = costume_roi[:, :, 3:4].astype(np.float32) / 255.0

                        # Smooth outer alpha border
                        alpha_2d = cv2.GaussianBlur(alpha[:, :, 0], (5, 5), sigmaX=1.2)
                        alpha = alpha_2d[:, :, np.newaxis]

                        if np.max(alpha) > 0.02:
                            fg = costume_roi[:, :, :3].astype(np.float32)
                            bg = frame[ry1:ry2, rx1:rx2].astype(np.float32)
                            blended = fg * alpha + bg * (1.0 - alpha)
                            frame[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)

            writer.write(frame)
            frame_idx += 1

            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        writer.release()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        logger.info(f"Full-body replacement complete: {frame_idx} frames written -> {output_path}")
        return output_path
