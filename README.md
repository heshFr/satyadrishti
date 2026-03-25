# Satya Drishti — AI Deepfake & Coercion Detection

A privacy-first, multimodal deepfake and coercion detection system that protects users against voice cloning scams, deepfake video calls, and digital manipulation. All processing runs locally — no data leaves your machine.

## What It Does

- **Media Forensics Scanner** — Upload any image, audio, or video for instant AI analysis across 5 forensic dimensions (face geometry, frequency patterns, AI signatures, metadata, neural detection)
- **Real-Time Call Protection** — Monitor live calls via WebSocket for voice cloning, coercion tactics, and identity fraud with instant threat alerts
- **Voice Print Enrollment** — Register family members' voice prints so the system can verify caller identity
- **Evidence & Reporting** — Save forensic reports, track incidents, and generate court-ready PDF evidence packages

## Architecture

```
satyadrishti/
├── engine/                 # Python ML Engine (4 modules)
│   ├── audio/              # AST voice deepfake detection (Audio Spectrogram Transformer)
│   ├── video/              # ViT-B/16 (spatial) + R3D-18 (temporal) video analysis
│   ├── text/               # DeBERTaV3 + LoRA coercion detection (4-class: safe/urgency/financial/combined)
│   ├── fusion/             # Cross-Attention Transformer (multi-modal fusion)
│   ├── image_forensics/    # 5-check image forensics pipeline (ELA, frequency, metadata, ViT, compression)
│   └── data/               # Dataset loaders (JSON manifest format)
├── server/                 # FastAPI backend (REST + WebSocket)
│   ├── app.py              # Main server with /api/analyze/* and /ws/live endpoints
│   ├── inference_engine.py # Model orchestration with graceful degradation
│   ├── routes/             # Auth, scans, cases, contact endpoints
│   └── ...                 # Config, auth, rate limiting, logging, reports
├── frontend/               # React 19 + Vite + TypeScript + Tailwind CSS
│   └── src/
│       ├── pages/          # 15 pages (Landing, Scanner, Dashboard, CallProtection, etc.)
│       ├── components/     # 20 components (3D background, glass cards, waveform viz)
│       └── hooks/          # Real-time call protection hook (WebSocket + WebAudio)
├── configs/                # Training YAML configs
├── scripts/                # Training, data download, preprocessing, ONNX export
└── tests/                  # API + ML engine tests (pytest)
```

## Prerequisites

- **Python 3.10+** (backend + ML)
- **Node.js 18+** (frontend)
- **CUDA GPU** (optional, for ML inference — CPU fallback available)

## Quick Start

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/YOUR_USERNAME/satyadrishti.git
cd satyadrishti

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -e ".[dev]"
```

### 2. Start the Backend Server

```bash
python -m server.app
```

The API server starts on `http://localhost:8000`. API docs are available at `/docs` (Swagger UI) and `/redoc`.

> **Note:** The server works without ML models installed — it returns stub responses via the `HAS_ML` flag for graceful degradation. To get real analysis results, you need trained model weights in the `models/` directory.

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend starts on `http://localhost:3000`. Open it in your browser.

### 4. Use the App

1. **Scan Media** — Go to `/scanner`, upload an image/audio/video, and click "Scan Now"
2. **Call Protection** — Go to `/call-protection`, grant microphone access, and start monitoring
3. **Voice Prints** — Go to `/voice-prints` to enroll family members' voices
4. **Dashboard** — Go to `/dashboard` for system status and quick actions

## AI Models

| Engine | Model | Input | Output |
|--------|-------|-------|--------|
| Audio | Audio Spectrogram Transformer (AST) | Raw waveform 16kHz | Bonafide/Spoof + 768d embedding |
| Video | ViT-B/16 (spatial) + R3D-18 (temporal) | 16 frames @ 224x224 | Real/Fake classification |
| Text | DeBERTaV3 + LoRA (r=16, alpha=32) | Text up to 512 tokens | 4-class coercion detection |
| Fusion | Cross-Attention Transformer | Projected 256d embeddings | Combined threat score |
| Image | ViT-B/16 + ELA + Frequency + Metadata | Single image | 5-check forensic analysis |

Model weights (`.pt`, `.onnx`) are not included in the repository. See `scripts/` for training scripts and `configs/` for training configurations.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze/media` | Upload file for forensic analysis |
| POST | `/api/analyze/text` | Analyze text for coercion patterns |
| POST | `/api/analyze/audio` | Analyze audio for voice deepfakes |
| POST | `/api/analyze/video` | Analyze video for visual deepfakes |
| POST | `/api/analyze/multimodal` | Multi-modal fusion analysis |
| WS | `/ws/live` | Real-time streaming call protection |
| GET | `/api/scans` | List scan history |
| POST | `/api/cases` | Create investigation case |
| POST | `/api/auth/login` | JWT authentication |

Full API documentation: `http://localhost:8000/docs`

## Development

```bash
# Run tests
pytest                          # all tests
pytest tests/test_api.py        # API tests only
pytest tests/test_engines.py    # ML engine tests only

# Lint & format
ruff check .                    # lint
ruff check --fix .              # auto-fix
black .                         # format

# Build frontend for production
cd frontend && npm run build

# Export models to ONNX with INT8 quantization
python scripts/export_onnx.py
```

## Tech Stack

**Backend:** Python 3.10, FastAPI, PyTorch, ONNX Runtime, SQLAlchemy, JWT Auth

**Frontend:** React 19, TypeScript, Vite, Tailwind CSS, Framer Motion, Three.js, React Three Fiber

**ML Models:** AST, ViT-B/16, R3D-18, DeBERTaV3, CLIP, Cross-Attention Transformer

## Privacy

All analysis runs locally on your device. No data is uploaded to external servers. Audio from call protection is processed in-memory and discarded after analysis. Files are never stored permanently unless you choose to save scan results.

## License

Proprietary — All Rights Reserved.
