"""
Face Detection & Analysis Module
=================================
Wraps InsightFace's FaceAnalysis for face detection, landmark extraction,
and embedding computation. Used for both character discovery and per-frame
face detection during swapping.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load insightface to avoid import-time errors
_face_analysis = None


class FaceData:
    """
    Serializable face data container.
    Extracted from InsightFace Face objects for cross-process communication.
    """

    def __init__(self, bbox: np.ndarray, kps: np.ndarray,
                 embedding: np.ndarray, det_score: float,
                 gender: Optional[int] = None, age: Optional[int] = None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kps = np.array(kps, dtype=np.float32)
        self.embedding = np.array(embedding, dtype=np.float32)
        self.det_score = float(det_score)
        self.gender = gender
        self.age = age

    @classmethod
    def from_insightface(cls, face) -> "FaceData":
        """Create FaceData from an InsightFace Face object."""
        return cls(
            bbox=face.bbox,
            kps=face.kps,
            embedding=face.embedding,
            det_score=face.det_score,
            gender=getattr(face, "gender", None),
            age=getattr(face, "age", None),
        )

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "kps": self.kps.tolist(),
            "embedding": self.embedding.tolist(),
            "det_score": self.det_score,
            "gender": self.gender,
            "age": self.age,
        }


class FaceDetector:
    """
    Detects faces in images using InsightFace's FaceAnalysis (buffalo_l).
    Provides face crops, embeddings, and landmarks.
    """

    def __init__(self, model_name: str = "buffalo_l",
                 det_size: Tuple[int, int] = (640, 640),
                 det_thresh: Optional[float] = None,
                 providers: Optional[List[str]] = None):
        from config import EXECUTION_PROVIDERS, FACE_DET_THRESH
        self._providers = providers or EXECUTION_PROVIDERS
        self._model_name = model_name
        self._det_size = det_size
        self._det_thresh = det_thresh if det_thresh is not None else FACE_DET_THRESH
        self._app = None

    def _ensure_loaded(self):
        """Lazy-load the face analysis model."""
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            logger.info(f"Loading FaceAnalysis model: {self._model_name} (det_thresh={self._det_thresh})")
            self._app = FaceAnalysis(
                name=self._model_name,
                providers=self._providers,
            )
            self._app.prepare(ctx_id=0, det_size=self._det_size, det_thresh=self._det_thresh)
            logger.info("FaceAnalysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FaceAnalysis: {e}")
            raise RuntimeError(
                f"Could not load face analysis model '{self._model_name}'. "
                f"Make sure insightface is installed: pip install insightface"
            ) from e

    def detect_faces(self, image: np.ndarray, max_faces: int = 20) -> list:
        """
        Detect all faces in an image.

        Args:
            image: BGR image as numpy array (HxWxC)
            max_faces: Maximum number of faces to return

        Returns:
            List of InsightFace Face objects, sorted by detection score
        """
        self._ensure_loaded()
        if image is None or image.size == 0:
            return []

        try:
            faces = self._app.get(image)
            
            # Close-up / extreme crop fallback:
            # If no face is detected (e.g. face touches edges of frame in tight zoom shots),
            # pad the frame with reflective border so RetinaFace can capture full facial contours.
            if not faces and image.shape[0] >= 64 and image.shape[1] >= 64:
                h, w = image.shape[:2]
                pad_y = int(h * 0.25)
                pad_x = int(w * 0.25)
                padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
                p_faces = self._app.get(padded)
                if p_faces:
                    faces = []
                    for pf in p_faces:
                        x1 = max(0.0, float(pf.bbox[0] - pad_x))
                        y1 = max(0.0, float(pf.bbox[1] - pad_y))
                        x2 = min(float(w), float(pf.bbox[2] - pad_x))
                        y2 = min(float(h), float(pf.bbox[3] - pad_y))
                        if x2 > x1 and y2 > y1:
                            pf.bbox[0] = x1
                            pf.bbox[1] = y1
                            pf.bbox[2] = x2
                            pf.bbox[3] = y2
                            pf.kps[:, 0] -= pad_x
                            pf.kps[:, 1] -= pad_y
                            faces.append(pf)

            # Sort by detection score (highest first)
            faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
            return faces[:max_faces]
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

    def detect_faces_with_data(self, image: np.ndarray,
                                max_faces: int = 20) -> List[FaceData]:
        """Detect faces and return serializable FaceData objects."""
        faces = self.detect_faces(image, max_faces)
        return [FaceData.from_insightface(f) for f in faces]

    def get_face_crop(self, image: np.ndarray, face,
                      margin: float = 0.3) -> np.ndarray:
        """
        Crop a face from the image with margin.

        Args:
            image: Full frame (BGR)
            face: Face object or FaceData with .bbox attribute
            margin: Extra margin around the face (0.3 = 30%)

        Returns:
            Cropped face image (BGR)
        """
        bbox = face.bbox.astype(int)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        mx = int(face_w * margin)
        my = int(face_h * margin)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return crop

    def get_best_face(self, faces: list):
        """Get the face with highest detection score (most frontal/clear)."""
        if not faces:
            return None
        return max(faces, key=lambda f: f.det_score)

    def detect_faces_robust(self, image: np.ndarray, max_faces: int = 20) -> list:
        """
        Multi-pass face detection for source images and challenging frames.
        Tries standard detection first, then falls back to lower threshold (0.12)
        and upscaled/padded image if no face is found.
        """
        faces = self.detect_faces(image, max_faces=max_faces)
        if faces or image is None or image.size == 0:
            return faces

        self._ensure_loaded()
        try:
            # Lower threshold temporarily for stylized / obscured / low-contrast faces
            self._app.det_model.det_thresh = 0.12
            h, w = image.shape[:2]
            pad_y, pad_x = int(h * 0.2), int(w * 0.2)
            padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
            p_faces = self._app.get(padded)
            if p_faces:
                result = []
                for pf in p_faces:
                    x1 = max(0.0, float(pf.bbox[0] - pad_x))
                    y1 = max(0.0, float(pf.bbox[1] - pad_y))
                    x2 = min(float(w), float(pf.bbox[2] - pad_x))
                    y2 = min(float(h), float(pf.bbox[3] - pad_y))
                    if x2 > x1 and y2 > y1:
                        pf.bbox[0], pf.bbox[1], pf.bbox[2], pf.bbox[3] = x1, y1, x2, y2
                        pf.kps[:, 0] -= pad_x
                        pf.kps[:, 1] -= pad_y
                        result.append(pf)
                result = sorted(result, key=lambda f: f.det_score, reverse=True)
                return result[:max_faces]
        except Exception as e:
            logger.warning(f"Robust face detection fallback failed: {e}")
        finally:
            self._app.det_model.det_thresh = self._det_thresh

        return []

    def prepare_mask_overlay_source(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare a stylized/masked character image (e.g., Iron Man helmet, Hanuman)
        for full-head/mask overlay swapping.

        Returns:
            Tuple of (bgra_image, 5x2_keypoints_float32)
        """
        raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read character image: {image_path}")

        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        elif raw.shape[2] == 3:
            # No alpha channel: extract clean foreground via GrabCut + elliptical feathering
            h, w = raw.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (max(1, int(w * 0.08)), max(1, int(h * 0.05)),
                    max(2, int(w * 0.84)), max(2, int(h * 0.90)))
            try:
                cv2.grabCut(raw, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
                fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                # Fallback to soft oval if GrabCut wiped out too much or kept everything
                fg_ratio = float(np.count_nonzero(fg_mask)) / max(1, h * w)
                if fg_ratio < 0.12 or fg_ratio > 0.96:
                    fg_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.ellipse(fg_mask, (w // 2, h // 2), (int(w * 0.40), int(h * 0.45)), 0, 0, 360, 255, -1)
                else:
                    # Smooth alpha edges
                    k = max(5, (min(h, w) // 40) | 1)
                    fg_mask = cv2.GaussianBlur(fg_mask, (k, k), 0)
            except Exception:
                fg_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(fg_mask, (w // 2, h // 2), (int(w * 0.40), int(h * 0.45)), 0, 0, 360, 255, -1)

            raw = np.dstack([raw, fg_mask])

        bgr = raw[:, :, :3]
        h, w = bgr.shape[:2]

        # Try detecting facial keypoints if character has human-like features
        faces = self.detect_faces_robust(bgr, max_faces=1)
        if faces and hasattr(faces[0], "kps") and faces[0].kps is not None:
            kps = np.array(faces[0].kps, dtype=np.float32)
        else:
            # Canonical centered portrait keypoints (left_eye, right_eye, nose, left_mouth, right_mouth)
            kps = np.array([
                [w * 0.36, h * 0.42],
                [w * 0.64, h * 0.42],
                [w * 0.50, h * 0.56],
                [w * 0.39, h * 0.71],
                [w * 0.61, h * 0.71],
            ], dtype=np.float32)

        return raw, kps

    @staticmethod
    def compute_similarity(embedding1: np.ndarray,
                           embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two face embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    @staticmethod
    def encode_face_thumbnail(image: np.ndarray, size: int = 200) -> str:
        """Encode a face crop as base64 JPEG string for API responses with high-quality scaling."""
        import base64
        if image is None or image.size == 0:
            return ""
        # Resize to thumbnail with high-quality interpolation
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        thumb = cv2.resize(image, (new_w, new_h), interpolation=interp)
        # Encode with high JPEG quality (95%)
        _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode("utf-8")
