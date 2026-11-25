"""
Face Swapper — Gradio App for Hugging Face Spaces
===================================================
Standalone Gradio interface that can be deployed to Hugging Face Spaces
with free GPU (ZeroGPU). Also works locally.

Deploy: Push this directory to a Hugging Face Space (SDK: Gradio)
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Try to import spaces for ZeroGPU (only on HF Spaces)
try:
    import spaces
    ON_HF_SPACES = True
except ImportError:
    ON_HF_SPACES = False
    # Create a no-op decorator
    class spaces:
        @staticmethod
        def GPU(func=None, duration=120):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator


def discover_characters_ui(video_path):
    """Step 1: Detect characters in uploaded video."""
    if video_path is None:
        return None, "Please upload a video first."

    from core.face_detector import FaceDetector
    from core.face_matcher import FaceMatcher
    from core.video_processor import VideoProcessor

    try:
        video_proc = VideoProcessor()
        detector = FaceDetector()
        matcher = FaceMatcher(similarity_threshold=0.45)

        # Sample frames
        samples = video_proc.sample_frames(video_path, every_n=30, max_samples=60)

        # Detect faces
        all_face_data = []
        for frame_idx, frame in samples:
            faces = detector.detect_faces(frame)
            for face in faces:
                crop = detector.get_face_crop(frame, face)
                all_face_data.append({
                    'face': face,
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'embedding': face.embedding.copy(),
                    'crop': crop,
                    'det_score': face.det_score,
                })

        characters = matcher.cluster_faces(all_face_data)

        if not characters:
            return None, "No faces detected in the video."

        # Create gallery of character thumbnails
        thumbnails = []
        labels = []
        for char in characters:
            if char.thumbnail is not None:
                # Convert BGR to RGB for Gradio
                thumb_rgb = cv2.cvtColor(char.thumbnail, cv2.COLOR_BGR2RGB)
                # Resize for display
                thumb_rgb = cv2.resize(thumb_rgb, (200, 200))
                thumbnails.append(thumb_rgb)
                labels.append(f"{char.label} ({char.face_count} detections)")

        gallery_items = [(img, label) for img, label in zip(thumbnails, labels)]

        info = f"Found {len(characters)} character(s). Select the one to swap."
        return gallery_items, info

    except Exception as e:
        return None, f"Error: {str(e)}"


@spaces.GPU(duration=300)
def swap_faces_ui(video_path, source_face_image, character_index, enhance):
    """Step 2: Perform face swap."""
    if video_path is None:
        return None, "Please upload a video."
    if source_face_image is None:
        return None, "Please upload a source face image."

    from core.face_detector import FaceDetector
    from core.face_matcher import FaceMatcher
    from core.face_swapper import FaceSwapper
    from core.video_processor import VideoProcessor

    try:
        detector = FaceDetector()
        swapper = FaceSwapper()
        video_proc = VideoProcessor()
        matcher = FaceMatcher(similarity_threshold=0.45)

        # Re-discover characters to get embeddings
        samples = video_proc.sample_frames(video_path, every_n=30, max_samples=60)
        all_face_data = []
        for frame_idx, frame in samples:
            faces = detector.detect_faces(frame)
            for face in faces:
                crop = detector.get_face_crop(frame, face)
                all_face_data.append({
                    'face': face,
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'embedding': face.embedding.copy(),
                    'crop': crop,
                    'det_score': face.det_score,
                })

        characters = matcher.cluster_faces(all_face_data)
        if not characters:
            return None, "No faces found."

        char_idx = int(character_index) if character_index is not None else 0
        if char_idx >= len(characters):
            char_idx = 0
        char_embedding = characters[char_idx].centroid

        # Load source face
        source_bgr = cv2.cvtColor(np.array(source_face_image), cv2.COLOR_RGB2BGR)
        source_faces = detector.detect_faces(source_bgr)
        if not source_faces:
            return None, "No face detected in source image."
        source_face = source_faces[0]

        # Extract audio
        audio_path = video_proc.extract_audio(video_path)

        # Process video frame by frame
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_dir = tempfile.mkdtemp()
        temp_video = os.path.join(output_dir, "processed.mp4")
        output_path = os.path.join(output_dir, "output.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect_faces(frame)
            if faces:
                frame = swapper.swap_faces_in_frame(
                    frame, faces, source_face,
                    char_embedding, 0.45
                )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # Mux audio
        video_proc.merge_video_audio(temp_video, audio_path, output_path)

        return output_path, f"Done! Processed {frame_idx} frames."

    except Exception as e:
        return None, f"Error: {str(e)}"


# ─── Gradio Interface ────────────────────────────────────────────────────────

with gr.Blocks(
    title="Face Swapper",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown("# 🎭 Face Swapper\nUpload a video, select a character, and swap their face!")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            detect_btn = gr.Button("🔍 Detect Characters", variant="primary")

        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Detected Characters", columns=3, height=300)
            char_info = gr.Textbox(label="Info", interactive=False)
            char_index = gr.Number(label="Character Index to Swap (0-based)", value=0, precision=0)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            source_face = gr.Image(label="Source Face (the new face)", type="pil")
            enhance_toggle = gr.Checkbox(label="Enhance faces (GFPGAN)", value=False)
            swap_btn = gr.Button("🔄 Swap Faces!", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Result")
            status_text = gr.Textbox(label="Status", interactive=False)

    # Wire up events
    detect_btn.click(
        fn=discover_characters_ui,
        inputs=[video_input],
        outputs=[gallery, char_info],
    )

    swap_btn.click(
        fn=swap_faces_ui,
        inputs=[video_input, source_face, char_index, enhance_toggle],
        outputs=[output_video, status_text],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
