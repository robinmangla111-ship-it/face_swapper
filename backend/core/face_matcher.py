"""
Face Matching & Character Clustering Module
=============================================
Groups detected faces across video frames into unique "characters"
using embedding similarity. Filters noise and merges similar clusters
to accurately identify only truly unique people.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.face_detector import FaceDetector

logger = logging.getLogger(__name__)


@dataclass
class Character:
    """Represents a unique person/character detected in the video."""
    id: int
    label: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    thumbnail: Optional[np.ndarray] = None       # Best face crop (BGR)
    thumbnail_b64: str = ""                        # Base64-encoded thumbnail
    face_count: int = 0
    best_score: float = 0.0                        # Best detection confidence
    best_quality: float = 0.0                      # Resolution * score quality score
    frame_indices: List[int] = field(default_factory=list)

    def update_centroid(self):
        """Recompute the centroid embedding from all embeddings."""
        if self.embeddings:
            self.centroid = np.mean(self.embeddings, axis=0)
            norm = np.linalg.norm(self.centroid)
            if norm > 0:
                self.centroid = self.centroid / norm

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "label": str(self.label),
            "face_count": int(self.face_count),
            "thumbnail": str(self.thumbnail_b64),
            "best_score": float(round(float(self.best_score), 3)),
        }


class FaceMatcher:
    """
    Clusters detected faces into unique characters using cosine similarity.
    Filters noise, picks high-resolution thumbnails, and merges similar clusters.
    """

    def __init__(self, similarity_threshold: float = 0.28):
        """
        Args:
            similarity_threshold: Minimum cosine similarity to merge faces
                                  into the same character cluster (default 0.28).
        """
        self.similarity_threshold = similarity_threshold
        self.characters: List[Character] = []

    def cluster_faces(self, face_data_list: List[dict]) -> List[Character]:
        """
        Cluster a list of face detections into unique characters.

        Args:
            face_data_list: List of dicts with keys:
                - 'face': InsightFace Face object or FaceData
                - 'frame_idx': Frame index
                - 'frame': The full frame (BGR image)
                - 'embedding': Face embedding (np.ndarray)
                - 'crop': Cropped face image (BGR)
                - 'det_score': float detection score

        Returns:
            List of unique Character objects.
        """
        self.characters = []

        if not face_data_list:
            return self.characters

        # Step 1: Filter out tiny face artifacts (< 32px), low confidence (< 0.60), and abnormal aspect ratios
        valid_faces = []
        for f in face_data_list:
            bbox = f.get('face').bbox if hasattr(f.get('face'), 'bbox') else None
            score = float(f.get('det_score', 0.5))
            if bbox is not None:
                w = float(bbox[2] - bbox[0])
                h = float(bbox[3] - bbox[1])
                aspect = w / max(h, 1e-5)
                # Normal human face: taller than wide or roughly square (0.45 <= aspect <= 1.25)
                # Face must be large enough to swap cleanly (max dimension >= 32px) and confident (>= 0.60)
                if max(w, h) >= 32 and score >= 0.60 and 0.45 <= aspect <= 1.25:
                    valid_faces.append(f)
            else:
                if score >= 0.60:
                    valid_faces.append(f)

        # Fallback if too strict
        if not valid_faces:
            valid_faces = [f for f in face_data_list if float(f.get('det_score', 0)) >= 0.5]
        if not valid_faces:
            valid_faces = face_data_list

        logger.info(f"Clustering {len(valid_faces)} face detections (from {len(face_data_list)} total)...")

        # Step 2: Incremental clustering using multi-embedding portfolio matching
        for face_info in valid_faces:
            embedding = face_info['embedding']
            det_score = float(face_info.get('det_score', 0.5))
            frame_idx = face_info.get('frame_idx', 0)
            crop = face_info.get('crop')

            if embedding is None or len(embedding) == 0:
                continue

            crop_quality = 0.0
            if crop is not None and crop.size > 0:
                ch, cw = crop.shape[:2]
                caspect = cw / max(ch, 1e-5)
                # Face aspect penalty: ideal portrait aspect is ~0.72
                aspect_penalty = max(0.2, min(1.0, 1.0 - abs(caspect - 0.72)))
                crop_quality = float(cw * ch) * (det_score ** 2) * aspect_penalty

            # Find best matching existing character by checking both centroid and recent pose portfolio
            best_match_idx = -1
            best_similarity = -1.0

            for i, char in enumerate(self.characters):
                if char.centroid is None:
                    continue
                # Centroid similarity
                sim_centroid = FaceDetector.compute_similarity(embedding, char.centroid)
                # Max similarity against portfolio of poses
                sim_portfolio = max(
                    (FaceDetector.compute_similarity(embedding, e) for e in char.embeddings[-8:]),
                    default=-1.0
                )
                sim = max(sim_centroid, sim_portfolio)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_idx = i

            # Relaxed threshold (0.22) for multi-angle matching to prevent splitting the same person
            if best_similarity >= max(0.20, self.similarity_threshold - 0.06) and best_match_idx >= 0:
                # Add to existing character cluster
                char = self.characters[best_match_idx]
                char.embeddings.append(embedding)
                char.face_count += 1
                char.frame_indices.append(frame_idx)
                char.update_centroid()

                # Update thumbnail if this crop has better resolution/quality
                if crop_quality > char.best_quality and crop is not None:
                    char.best_quality = crop_quality
                    char.thumbnail = crop.copy()
                    char.best_score = max(char.best_score, det_score)
            else:
                # Create new character cluster
                new_char = Character(
                    id=len(self.characters),
                    label=f"Person {len(self.characters) + 1}",
                    embeddings=[embedding],
                    face_count=1,
                    best_score=det_score,
                    best_quality=crop_quality,
                    thumbnail=crop.copy() if crop is not None else None,
                    frame_indices=[frame_idx],
                )
                new_char.update_centroid()
                self.characters.append(new_char)

        # Step 3: Aggressive co-occurrence and cross-portfolio merging
        # If two clusters never appear simultaneously in the same frame, they are likely the same person across camera angles
        merged = True
        while merged and len(self.characters) > 1:
            merged = False
            for i in range(len(self.characters)):
                if merged:
                    break
                for j in range(i + 1, len(self.characters)):
                    c1 = self.characters[i]
                    c2 = self.characters[j]
                    if c1.centroid is None or c2.centroid is None:
                        continue
                    
                    sim_centroid = FaceDetector.compute_similarity(c1.centroid, c2.centroid)
                    # Portfolio cross similarity
                    sim_cross = max(
                        (FaceDetector.compute_similarity(e1, e2) for e1 in c1.embeddings[-5:] for e2 in c2.embeddings[-5:]),
                        default=-1.0
                    )
                    effective_sim = max(sim_centroid, sim_cross)
                    
                    frames1 = set(c1.frame_indices)
                    frames2 = set(c2.frame_indices)
                    co_occurred = not frames1.isdisjoint(frames2)
                    
                    # If they never appear together in the same frame, merge if similarity >= 0.17
                    # If they co-occur, only merge if similarity is very high (>= 0.35, e.g. mirror reflection)
                    should_merge = (effective_sim >= 0.17 and not co_occurred) or (effective_sim >= 0.35)
                    
                    if should_merge:
                        c1.embeddings.extend(c2.embeddings)
                        c1.face_count += c2.face_count
                        c1.frame_indices.extend(c2.frame_indices)
                        c1.update_centroid()
                        if c2.best_quality > c1.best_quality and c2.thumbnail is not None:
                            c1.thumbnail = c2.thumbnail
                            c1.best_quality = c2.best_quality
                        c1.best_score = max(c1.best_score, c2.best_score)
                        self.characters.pop(j)
                        merged = True
                        break

        # Step 4: Prune transient background detections (passersby / false detections)
        total_detections = sum(c.face_count for c in self.characters)
        kept_chars = []
        for c in self.characters:
            # Keep if dominant or has significant representation (>= 3 detections or >= 15% of total faces or high confidence)
            if c.face_count >= 3 or c.best_score >= 0.85 or (total_detections > 0 and c.face_count / total_detections >= 0.15):
                kept_chars.append(c)

        if not kept_chars and self.characters:
            kept_chars = [max(self.characters, key=lambda c: (c.face_count, c.best_quality))]

        self.characters = kept_chars[:5]  # Limit to top 5 prominent unique characters max

        # Step 5: Post-process: encode high-res thumbnails and sort by prominence
        for char in self.characters:
            if char.thumbnail is not None:
                char.thumbnail_b64 = FaceDetector.encode_face_thumbnail(
                    char.thumbnail, size=200
                )

        # Sort by face frequency (most prominent character first)
        self.characters.sort(key=lambda c: (c.face_count, c.best_quality), reverse=True)

        # Re-assign clean sequential IDs
        for i, char in enumerate(self.characters):
            char.id = i
            char.label = f"Person {i + 1}"

        logger.info(f"Discovered {len(self.characters)} truly unique character(s):")
        for char in self.characters:
            logger.info(f"  {char.label}: {char.face_count} detections, "
                        f"best score={char.best_score:.3f}")

        return self.characters

    def _merge_similar_characters(self, merge_threshold: float = 0.40):
        """Merge character clusters that have high centroid similarity."""
        merged = True
        while merged and len(self.characters) > 1:
            merged = False
            for i in range(len(self.characters)):
                if merged:
                    break
                for j in range(i + 1, len(self.characters)):
                    c1 = self.characters[i]
                    c2 = self.characters[j]
                    if c1.centroid is not None and c2.centroid is not None:
                        sim = FaceDetector.compute_similarity(c1.centroid, c2.centroid)
                        if sim >= merge_threshold:
                            # Merge c2 into c1
                            c1.embeddings.extend(c2.embeddings)
                            c1.face_count += c2.face_count
                            c1.frame_indices.extend(c2.frame_indices)
                            c1.update_centroid()
                            if c2.best_quality > c1.best_quality and c2.thumbnail is not None:
                                c1.thumbnail = c2.thumbnail
                                c1.best_quality = c2.best_quality
                            c1.best_score = max(c1.best_score, c2.best_score)
                            self.characters.pop(j)
                            merged = True
                            break

    def get_character_embedding(self, character_id: int) -> Optional[np.ndarray]:
        """Get the centroid embedding for a character."""
        for char in self.characters:
            if char.id == character_id:
                return char.centroid
        return None

    def match_face_to_character(self, face_embedding: np.ndarray,
                                 character_id: int) -> bool:
        """Check if a face embedding matches a specific character."""
        char_embedding = self.get_character_embedding(character_id)
        if char_embedding is None:
            return False
        similarity = FaceDetector.compute_similarity(face_embedding, char_embedding)
        return similarity >= self.similarity_threshold
