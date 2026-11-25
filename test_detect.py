import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import cv2
import numpy as np
from core.face_detector import FaceDetector
from core.video_processor import VideoProcessor
from core.face_matcher import FaceMatcher

vp = VideoProcessor()
detector = FaceDetector()
video_path = 'backend/uploads/26b21988-43a.mp4'

samples = vp.sample_frames(video_path, every_n=15, max_samples=60)
print('Sampled frames:', len(samples))

all_faces = []
for f_idx, frame in samples:
    faces = detector.detect_faces(frame)
    for face in faces:
        bbox = face.bbox
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        all_faces.append({'face': face, 'frame_idx': f_idx, 'score': float(face.det_score), 'w': w, 'h': h, 'embedding': face.embedding.copy()})

print('Total face detections:', len(all_faces))
for idx, item in enumerate(all_faces):
    print(f"Detection {idx}: frame={item['frame_idx']}, score={item['score']:.3f}, size={item['w']}x{item['h']}")

if len(all_faces) >= 2:
    for i in range(min(len(all_faces), 8)):
        for j in range(i+1, min(len(all_faces), 8)):
            s = FaceDetector.compute_similarity(all_faces[i]['embedding'], all_faces[j]['embedding'])
            print(f"Similarity {i} vs {j}: {s:.3f}")
