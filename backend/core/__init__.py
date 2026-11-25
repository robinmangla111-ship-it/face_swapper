"""
Face Swapper Core Engine
========================
Modular face swapping pipeline with parallel processing support.

Modules:
    face_detector   - Face detection and embedding extraction (InsightFace)
    face_swapper    - Face swapping using inswapper_128 model
    face_enhancer   - Post-swap face enhancement (GFPGAN)
    face_matcher    - Character clustering and matching
    video_processor - Video I/O and FFmpeg operations
    chunk_manager   - Parallel chunk-based video processing
    color_corrector - Color/lighting correction for blending
    pipeline        - Full orchestration pipeline
"""

__version__ = "1.0.0"
