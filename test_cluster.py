import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from core.face_detector import FaceDetector
from core.video_processor import VideoProcessor
from core.face_matcher import FaceMatcher

vp = VideoProcessor()
detector = FaceDetector()
video_path = 'backend/uploads/26b21988-43a.mp4'

samples = vp.sample_frames(video_path, every_n=10, max_samples=60)
print('Sampled frames:', len(samples))

# Filter faces
raw_faces = []
for f_idx, frame in samples:
    faces = detector.detect_faces(frame)
    for face in faces:
        bbox = face.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        raw_faces.append({'face': face, 'frame_idx': f_idx, 'frame': frame, 'crop': detector.get_face_crop(frame, face), 'score': float(face.det_score), 'w': w, 'h': h, 'embedding': face.embedding.copy()})

print('Total raw faces:', len(raw_faces))

# Filter: min dimension >= 35, score >= 0.60
filtered = [f for f in raw_faces if max(f['w'], f['h']) >= 35 and f['score'] >= 0.60]
print('Filtered faces (>=35px, score>=0.60):', len(filtered))

matcher = FaceMatcher(similarity_threshold=0.38)
chars = matcher.cluster_faces(filtered if len(filtered) > 0 else raw_faces)
print(f'Discovered {len(chars)} unique character(s):')
for c in chars:
    print(f"  {c.label}: {c.face_count} faces, best_score={c.best_score:.3f}, thumb size={c.thumbnail.shape if c.thumbnail is not None else None}")
