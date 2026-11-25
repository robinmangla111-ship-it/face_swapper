"""
Color Correction Module
========================
Adjusts color and lighting of swapped faces to match the target frame.
Prevents the "pasted sticker" look by harmonizing skin tones and lighting.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ColorCorrector:
    """
    Applies color correction to blended face regions.
    Uses histogram matching and seamless cloning for natural results.
    """

    @staticmethod
    def histogram_match_color(source: np.ndarray,
                               target: np.ndarray) -> np.ndarray:
        """
        Match the color distribution of source to target using
        histogram matching in LAB color space.

        Args:
            source: Image whose colors to adjust (BGR)
            target: Reference image with target colors (BGR)

        Returns:
            Color-corrected source image (BGR)
        """
        if source.size == 0 or target.size == 0:
            return source

        try:
            # Convert to LAB color space
            source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Compute mean and std for each channel
            for i in range(3):
                src_mean, src_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
                tgt_mean, tgt_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()

                if src_std < 1e-6:
                    continue

                # Normalize, scale, and shift
                source_lab[:, :, i] = (
                    (source_lab[:, :, i] - src_mean) * (tgt_std / src_std) + tgt_mean
                )

            # Clip and convert back
            source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
            return result

        except Exception as e:
            logger.warning(f"Histogram matching failed: {e}")
            return source

    @staticmethod
    def seamless_blend(frame: np.ndarray,
                       swapped_face: np.ndarray,
                       face_bbox: np.ndarray,
                       blend_mode: str = "normal") -> np.ndarray:
        """
        Blend a swapped face into the frame using Poisson seamless cloning.

        Args:
            frame: Original full frame (BGR)
            swapped_face: Frame with swapped face (BGR, same size as frame)
            face_bbox: Face bounding box [x1, y1, x2, y2]
            blend_mode: "normal" for cv2.NORMAL_CLONE, "mixed" for MIXED_CLONE

        Returns:
            Blended frame
        """
        try:
            bbox = face_bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return swapped_face

            # Create mask for the face region
            face_w, face_h = x2 - x1, y2 - y1
            mask = np.zeros((h, w), dtype=np.uint8)

            # Elliptical mask for natural blending
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            axes = (face_w // 2, face_h // 2)
            cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

            # Gaussian blur the mask edges for feathering
            mask = cv2.GaussianBlur(mask, (21, 21), 11)

            clone_flag = cv2.NORMAL_CLONE if blend_mode == "normal" else cv2.MIXED_CLONE

            result = cv2.seamlessClone(
                swapped_face, frame, mask,
                (center_x, center_y), clone_flag
            )
            return result

        except Exception as e:
            logger.warning(f"Seamless blending failed: {e}, using simple blend")
            return ColorCorrector.simple_blend(frame, swapped_face, face_bbox)

    @staticmethod
    def simple_blend(frame: np.ndarray,
                     swapped_frame: np.ndarray,
                     face_bbox: np.ndarray,
                     feather_amount: int = 15) -> np.ndarray:
        """
        Simple alpha blending with feathered edges.
        Fallback when seamless cloning fails.

        Args:
            frame: Original frame (BGR)
            swapped_frame: Frame with swapped face (BGR, same size)
            face_bbox: Face bounding box [x1, y1, x2, y2]
            feather_amount: Gaussian blur kernel size for feathering

        Returns:
            Blended frame
        """
        h, w = frame.shape[:2]
        bbox = face_bbox.astype(int)
        x1, y1, x2, y2 = bbox

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Create soft mask
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0

        # Feather edges
        ksize = feather_amount * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_amount)

        # Blend
        mask_3ch = mask[:, :, np.newaxis]
        result = (swapped_frame * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)
        return result

    @staticmethod
    def correct_face_in_frame(frame: np.ndarray,
                               swapped_frame: np.ndarray,
                               face_bbox: np.ndarray) -> np.ndarray:
        """
        Full color correction pipeline for a swapped face:
        1. Histogram matching for color consistency
        2. Feathered blending for smooth edges

        Args:
            frame: Original frame
            swapped_frame: Frame after face swap
            face_bbox: Face bounding box

        Returns:
            Color-corrected and blended result
        """
        bbox = face_bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return swapped_frame

        # Extract face regions
        original_face = frame[y1:y2, x1:x2]
        swapped_face = swapped_frame[y1:y2, x1:x2]

        # Color correct the swapped face to match original surroundings
        corrected_face = ColorCorrector.histogram_match_color(
            swapped_face, original_face
        )

        # Place corrected face back
        result = swapped_frame.copy()
        result[y1:y2, x1:x2] = corrected_face

        # Feathered blend
        result = ColorCorrector.simple_blend(frame, result, face_bbox)

        return result
