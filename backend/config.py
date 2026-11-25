"""
Configuration for the Face Swapper application.
Paths, model URLs, processing parameters.
"""
import os
import sys
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
LIBRARY_DIR = BASE_DIR / "library"
LIBRARY_IMAGES_DIR = LIBRARY_DIR / "images"

# Create directories on import
for _d in [MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, TEMP_DIR, LIBRARY_DIR, LIBRARY_IMAGES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Model Paths & URLs ──────────────────────────────────────────────────────
INSWAPPER_MODEL_PATH = MODELS_DIR / "inswapper_128.onnx"
INSWAPPER_DOWNLOAD_URLS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/thebiglaskowski/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
]

# InsightFace buffalo_l downloads automatically to ~/.insightface/models/buffalo_l/
FACE_ANALYSIS_MODEL = "buffalo_l"

# GFPGAN ONNX model for high-resolution face restoration
GFPGAN_MODEL_PATH = MODELS_DIR / "GFPGANv1.4.onnx"

# ─── Face Detection Settings ─────────────────────────────────────────────────
FACE_DET_SIZE = (640, 640)          # Detection input resolution
FACE_DET_THRESH = 0.25              # Minimum detection confidence (0.25 for profile/motion coverage)
FACE_SIMILARITY_THRESH = 0.18       # Cosine similarity for character matching
FACE_SAMPLE_RATE = 12               # Sample every Nth frame for discovery

# ─── Video Processing ────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 500             # Maximum upload size in MB
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_CODEC = "libx264"
VIDEO_CRF = "15"                     # Quality: 15=visually lossless studio grade (0=lossless, 23=default)
VIDEO_PRESET = "medium"              # Encoding speed: ultrafast, fast, medium, slow
AUDIO_CODEC = "copy"                 # Copy audio without re-encoding
OUTPUT_FORMAT = "mp4"
PIXEL_FORMAT = "yuv420p"

# ─── Parallel Processing ─────────────────────────────────────────────────────
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(1, CPU_COUNT - 1)
CHUNK_MIN_FRAMES = 200              # Don't create chunks smaller than this
AUTO_PARALLEL_THRESHOLD = 500       # Use parallel processing if > this many frames

# Memory-based worker scaling
try:
    import psutil
    AVAILABLE_MEMORY_GB = psutil.virtual_memory().total / (1024 ** 3)
except ImportError:
    AVAILABLE_MEMORY_GB = 8  # Assume 8GB if psutil not available

# Each worker needs ~1GB for models; limit workers based on available RAM
MEMORY_PER_WORKER_GB = 1.0
MAX_WORKERS_BY_MEMORY = max(1, int((AVAILABLE_MEMORY_GB - 2) / MEMORY_PER_WORKER_GB))
EFFECTIVE_MAX_WORKERS = min(MAX_WORKERS, MAX_WORKERS_BY_MEMORY)

# ─── Face Enhancement ────────────────────────────────────────────────────────
ENHANCE_ENABLED = True               # Enable GFPGAN face enhancement
ENHANCE_UPSCALE = 1                  # Upscale factor (1 = no upscale, just enhance)
ENHANCE_WEIGHT = 0.7                 # Blending weight (0=original, 1=fully enhanced)

# ─── ONNX Runtime ────────────────────────────────────────────────────────────
def get_execution_providers():
    """Detect available ONNX Runtime execution providers."""
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
        # Prefer CUDA for dedicated NVIDIA GPUs; use CPUExecutionProvider for rock-solid stability
        if 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    except Exception:
        return ['CPUExecutionProvider']

EXECUTION_PROVIDERS = get_execution_providers()
USE_GPU = 'CUDAExecutionProvider' in EXECUTION_PROVIDERS

# ─── FFmpeg Path ──────────────────────────────────────────────────────────────
def get_ffmpeg_path():
    """Get path to FFmpeg binary. Try system PATH first, then imageio-ffmpeg."""
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Fallback to direct search in site-packages
        binaries_dir = BASE_DIR.parent / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
        if binaries_dir.exists():
            for f in binaries_dir.glob("ffmpeg*.exe"):
                return str(f)
        return None

FFMPEG_PATH = get_ffmpeg_path()

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
