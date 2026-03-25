# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satya Drishti is a multimodal deepfake and coercion detection system with a Python ML engine, FastAPI backend, and React frontend. It fuses audio, video, and text analysis for real-time threat detection, designed for edge deployment via ONNX/INT8 quantization.

## Development Commands

### Python Backend

```bash
# Setup (Python 3.10+ required)
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"

# Run API server (port 8000)
python -m server.app

# Tests
pytest                       # all tests
pytest tests/test_api.py     # API endpoint tests only
pytest tests/test_engines.py # ML engine tests only
pytest tests/test_engines.py::test_audio_engine_raw_net3  # single test

# Linting & formatting
ruff check .                 # lint (rules: E, F, I, W)
ruff check --fix .           # auto-fix
black .                      # format
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # Vite dev server on port 3000
npm run build    # tsc + vite build
npm run preview  # preview production build
```

## Architecture

### ML Engine (`engine/`)

Four independent modules with separate training pipelines, fused via cross-attention:

| Module | Path             | Model                                  | Input                       | Output                                    |
| ------ | ---------------- | -------------------------------------- | --------------------------- | ----------------------------------------- |
| Audio  | `engine/audio/`  | AST (pretrained VoxCelebSpoof)         | Raw waveform 16kHz          | 2-class (bonafide/spoof), 768d embedding  |
| Video  | `engine/video/`  | ViT-B/16 (spatial) + R3D-18 (temporal) | 16 frames @ 224x224         | 2-class (real/fake)                       |
| Text   | `engine/text/`   | DeBERTaV3 + LoRA (r=16, alpha=32)      | Text up to 512 tokens       | 4-class (safe/urgency/financial/combined) |
| Fusion | `engine/fusion/` | Cross-Attention Transformer            | Projected embeddings (256d) | 4-class threat + fusion embeddings        |

`engine/image_forensics/` runs a 5-check pipeline (face geometry, GAN frequency analysis, EXIF metadata, AI pattern classifier) for static image analysis.

Training scripts are in `scripts/` with YAML configs in `configs/`.

### Backend (`server/`)

FastAPI server with REST + WebSocket endpoints:

- `POST /api/analyze/{text,audio,video,multimodal,media}` - per-modality and fused analysis
- `WS /ws/live` - real-time streaming analysis for call protection
- `server/inference_engine.py` orchestrates model loading with `HAS_ML` flag for graceful degradation when models aren't available

### Frontend (`frontend/`)

React 19 + Vite + TypeScript + Tailwind CSS SPA. Path alias `@` maps to `./src`.

Key routes (React Router v7): `/` (dashboard), `/scanner` (media forensics), `/history`, `/settings`, `/advanced`.

Scanner flow: DropZone upload -> AnalysisProgress -> ScanVerdict + ForensicDetails.

CORS allows `localhost:3000` and `localhost:5173`.

## Code Style

- **Python:** line-length unlimited, target py310. Ruff for linting, Black for formatting.
- **TypeScript:** strict mode, ES2020 target. Tailwind with dark mode (class-based). Custom color palette defined in `frontend/tailwind.config.ts`.

## Key Conventions

- Model weights (`.pt`, `.pth`, `.onnx`, `.bin`, `.safetensors`) and `datasets/`/`models/` directories are gitignored — never commit them.
- ML modules use lazy loading and return stub responses when models aren't loaded (`HAS_ML` flag).
- The data module (`engine/data/dataloader.py`) uses JSON manifest files to define dataset splits with labels: 0=safe, 1=deepfake, 2=coercion, 3=both.
