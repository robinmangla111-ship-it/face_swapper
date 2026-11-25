"""
Model Downloader
=================
Downloads required AI models for the face swapper:
  - inswapper_128.onnx (face swapping model)
  - buffalo_l (InsightFace face analysis — auto-downloaded by insightface)
  - GFPGANv1.4.pth (face enhancement — auto-downloaded by gfpgan)

Usage:
    python models/download_models.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_file(url: str, dest_path: str, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    from tqdm import tqdm
    import httpx

    dest = Path(dest_path)
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = str(dest) + ".tmp"

    print(f"  ↓ Downloading: {desc}")
    print(f"    URL: {url}")

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(temp_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=f"    {dest.name}") as pbar:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Rename temp to final
        Path(temp_path).rename(dest)
        print(f"  ✓ Saved: {dest_path} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        Path(temp_path).unlink(missing_ok=True)
        return False


def download_inswapper():
    """Download the inswapper_128 face swapping model."""
    from config import INSWAPPER_MODEL_PATH, INSWAPPER_DOWNLOAD_URLS

    print("\n[1/3] inswapper_128.onnx (Face Swapping Model)")

    for url in INSWAPPER_DOWNLOAD_URLS:
        if download_file(url, str(INSWAPPER_MODEL_PATH), "inswapper_128.onnx"):
            return True

    print("\n  ⚠ Automatic download failed. Please download manually:")
    print("    1. Go to: https://huggingface.co/deepinsight/inswapper/tree/main")
    print(f"    2. Download 'inswapper_128.onnx' to: {INSWAPPER_MODEL_PATH}")
    return False


def download_buffalo_l():
    """Trigger InsightFace buffalo_l model download."""
    print("\n[2/3] buffalo_l (Face Detection & Analysis)")

    try:
        from insightface.app import FaceAnalysis
        import numpy as np

        print("  ↓ Loading InsightFace (will auto-download buffalo_l)...")
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))

        # Quick test
        test_img = np.zeros((320, 320, 3), dtype=np.uint8)
        app.get(test_img)

        print("  ✓ buffalo_l model ready")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Try: pip install insightface onnxruntime")
        return False


def download_gfpgan():
    """Check if GFPGAN is available (optional)."""
    print("\n[3/3] GFPGAN v1.4 (Face Enhancement — Optional)")

    try:
        import torch
        from gfpgan import GFPGANer
        print("  ✓ GFPGAN is installed (model will auto-download on first use)")
        return True
    except ImportError:
        print("  ⚠ GFPGAN not installed (optional — for higher quality)")
        print("    Install with: pip install torch torchvision gfpgan")
        return False


def verify_models():
    """Verify all models are ready."""
    from config import INSWAPPER_MODEL_PATH

    print("\n" + "=" * 50)
    print("MODEL STATUS")
    print("=" * 50)

    all_ok = True

    # Check inswapper
    if INSWAPPER_MODEL_PATH.exists():
        size = INSWAPPER_MODEL_PATH.stat().st_size / 1e6
        print(f"  ✓ inswapper_128.onnx ({size:.1f} MB)")
    else:
        print(f"  ✗ inswapper_128.onnx — MISSING")
        all_ok = False

    # Check buffalo_l
    buffalo_path = Path.home() / ".insightface" / "models" / "buffalo_l"
    if buffalo_path.exists() and any(buffalo_path.iterdir()):
        print(f"  ✓ buffalo_l (in {buffalo_path})")
    else:
        print(f"  ✗ buffalo_l — MISSING")
        all_ok = False

    # Check GFPGAN
    try:
        import gfpgan
        print(f"  ✓ GFPGAN (installed)")
    except ImportError:
        print(f"  ⚠ GFPGAN — not installed (optional)")

    # Check FFmpeg
    from config import FFMPEG_PATH
    if FFMPEG_PATH:
        print(f"  ✓ FFmpeg ({FFMPEG_PATH})")
    else:
        print(f"  ✗ FFmpeg — NOT FOUND")
        all_ok = False

    print("=" * 50)
    if all_ok:
        print("All required models are ready! ✓")
    else:
        print("Some models are missing. See above for instructions.")
    print()

    return all_ok


if __name__ == "__main__":
    print("=" * 50)
    print("  Face Swapper — Model Downloader")
    print("=" * 50)

    s1 = download_inswapper()
    s2 = download_buffalo_l()
    s3 = download_gfpgan()

    verify_models()
