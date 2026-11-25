# Face Swapper

A production-ready face swapping tool with a modern web interface.

## Features

- 🎥 **Video Upload** — Upload any video format (MP4, AVI, MOV, MKV, WebM)
- 👤 **Character Detection** — Auto-detect all unique faces/characters in the video
- 🔄 **Face Selection** — Choose which character's face to swap
- ✨ **High Quality** — GFPGAN face enhancement for natural-looking results
- 🔊 **Audio Preserved** — Original audio track is kept perfectly intact
- ⚡ **Parallel Processing** — Multi-threaded chunk processing for long videos
- 🌐 **Web Interface** — Modern Next.js frontend with real-time progress
- ☁️ **Cloud Deploy** — Gradio version for free GPU on Hugging Face Spaces

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### Installation

```bash
# Clone and enter directory
cd face_swapper

# Option 1: Use the install script (Windows)
scripts\install.bat

# Option 2: Manual installation
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r backend/requirements.txt

# Optional: Install GFPGAN for enhanced quality
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gfpgan

# Download AI models
python backend/models/download_models.py

# Install frontend
cd frontend && npm install && cd ..
```

### Running

```bash
# Option 1: Use the start script (Windows)
scripts\start.bat

# Option 2: Manual start
# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Frontend
cd frontend && npm run dev
```

Then open **http://localhost:3000** in your browser.

## Architecture

```
face_swapper/
├── backend/              # Python FastAPI backend
│   ├── core/             # Face swap engine
│   │   ├── face_detector.py    # InsightFace face detection
│   │   ├── face_swapper.py     # inswapper_128 face swap
│   │   ├── face_enhancer.py    # GFPGAN enhancement
│   │   ├── face_matcher.py     # Character clustering
│   │   ├── video_processor.py  # FFmpeg video operations
│   │   ├── chunk_manager.py    # Parallel processing
│   │   ├── color_corrector.py  # Color/lighting correction
│   │   └── pipeline.py         # Orchestration
│   ├── main.py           # FastAPI server
│   └── config.py         # Configuration
├── frontend/             # Next.js frontend
├── gradio_app/           # Gradio version for HF Spaces
└── scripts/              # Installation & startup scripts
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload/video` | Upload a video file |
| GET | `/api/faces/{video_id}` | Detect characters |
| POST | `/api/upload/source-face` | Upload source face |
| POST | `/api/swap` | Start swap job |
| GET | `/api/jobs/{job_id}` | Get job status |
| WS | `/ws/progress/{job_id}` | Real-time progress |
| GET | `/api/download/{job_id}` | Download result |

## AI Models Used

| Model | Purpose | Size |
|-------|---------|------|
| buffalo_l | Face detection & analysis | ~330MB |
| inswapper_128 | Face swapping | ~550MB |
| GFPGAN v1.4 | Face enhancement (optional) | ~350MB |

## Cloud Deployment (Free GPU)

Deploy to Hugging Face Spaces for free GPU processing:

```bash
cd gradio_app
# Push to your HF Space repo
```

## License

For personal and research use only. The inswapper model weights are
licensed for non-commercial use. See individual model licenses for details.
