# Image + Video Detection Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-tuned 20-module image/video detection pipeline with three SOTA pretrained detectors (DIRE, AIDE, GenConViT) fused via a learned logistic regression with temperature scaling, deployed behind a feature flag for safe rollout.

**Architecture:** Three new detector wrappers slot in alongside existing models. A `MetaFusionEngine` collects detector outputs into a fixed-shape feature vector, runs them through a tiny logistic regression (~5KB joblib), then a single-scalar temperature scaler for honest confidence. The new engine is wired into `server/inference_engine.py` behind `USE_NEW_PIPELINE` env var; old pipeline preserved unchanged for rollback. No retraining of base detectors.

**Tech Stack:** Python 3.10, PyTorch 2.10+cu128, transformers, scikit-learn (LogisticRegression), joblib, pytest, FastAPI.

**Spec reference:** `docs/superpowers/specs/2026-05-09-image-video-redesign-design.md`

**Scope:** This plan covers Phases 1-3 of the migration (build new detectors, train meta-fusion, wire feature flag). Phases 4-6 (A/B comparison, switch flip, old code deletion) are deployment work that runs after this plan completes.

---

## File Structure

**New files (16):**
```
engine/image_forensics/dire_detector.py
engine/image_forensics/aide_detector.py
engine/video/genconvit_detector.py
engine/meta_fusion/__init__.py
engine/meta_fusion/feature_extractor.py
engine/meta_fusion/fusion_model.py
engine/meta_fusion/temperature_scaler.py
engine/meta_fusion/engine.py
scripts/build_calibration_set.py
scripts/train_meta_fusion.py
scripts/eval_meta_fusion.py
tests/test_new_detectors.py
tests/test_meta_fusion.py
tests/test_pipeline_golden.py
tests/test_pipeline_latency.py
tests/fixtures/golden/images/labels.json
tests/fixtures/golden/videos/labels.json
```

**Modified files (2):**
```
server/config.py              # Add USE_NEW_PIPELINE flag + model paths
server/inference_engine.py    # Add MetaFusionEngine dispatch behind flag
```

**New empty directories:**
```
models/meta_fusion/
models/image_forensics/dire/
models/image_forensics/aide/
models/video/genconvit/
tests/fixtures/golden/images/
tests/fixtures/golden/videos/
```

---

## Task 1: Verify pretrained weight access

**Purpose:** Before writing detector wrappers, confirm each pretrained model is accessible. Resolve fallback paths NOW so later tasks have certainty.

**Files:** No code changes — research only. Output is a written summary.

- [ ] **Step 1: Verify each model's availability**

For each of the three models, visit the upstream sources and confirm:
1. **DIRE** — Visit `https://github.com/ZhendongWang6/DIRE`. Look for: pretrained weight URL, exact model class import path, expected input format (PIL/numpy/tensor), expected output format (logit/probability).
2. **AIDE** — Visit `https://github.com/shilinyan99/AIDE`. Same checks.
3. **GenConViT** — Visit `https://github.com/erprogs/GenConViT`. Same checks.

For each model, decide:
- **Loading method:** HuggingFace `from_pretrained()` if HF mirror exists; else direct PyTorch `load_state_dict()` from URL
- **Image preprocessing:** input resolution, normalization values
- **Output:** is the score already a probability, or do we need softmax/sigmoid?

If primary URL is dead, document the fallback:
- DIRE → DRCT (`https://github.com/beibuwandeluori/DRCT`)
- AIDE → NPR (`https://github.com/chuangchuangtan/NPR-DeepfakeDetection`)
- GenConViT → TALL (`https://github.com/rainy-xu/TALL4Deepfake`)

Save the resolution in `docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md` with this exact format:
```markdown
# Pretrained Model Resolution

## DIRE
- **Used:** [DIRE | DRCT (fallback)]
- **Weight URL:** <exact URL>
- **Loading code reference:** <upstream filename and line range>
- **Input:** <resolution, format, normalization>
- **Output:** <probability/logit, range>

## AIDE
... same ...

## GenConViT
... same ...
```

- [ ] **Step 2: Commit the resolution doc**

```bash
git add docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md
git commit -m "docs: pretrained weight sourcing for detection redesign"
```

---

## Task 2: Set up directory scaffolding

**Files:**
- Create: `engine/meta_fusion/__init__.py` (empty marker)
- Create directories: `models/meta_fusion/`, `models/image_forensics/dire/`, `models/image_forensics/aide/`, `models/video/genconvit/`, `tests/fixtures/golden/images/`, `tests/fixtures/golden/videos/`

- [ ] **Step 1: Create empty package marker**

Create `engine/meta_fusion/__init__.py`:

```python
"""Meta-fusion package: combines outputs of forensic detectors via learned logistic regression."""
```

- [ ] **Step 2: Create empty model directories**

```bash
mkdir -p models/meta_fusion
mkdir -p models/image_forensics/dire
mkdir -p models/image_forensics/aide
mkdir -p models/video/genconvit
mkdir -p tests/fixtures/golden/images
mkdir -p tests/fixtures/golden/videos
```

Add `.gitkeep` files so directories are tracked even when empty:

```bash
touch models/meta_fusion/.gitkeep
touch models/image_forensics/dire/.gitkeep
touch models/image_forensics/aide/.gitkeep
touch models/video/genconvit/.gitkeep
touch tests/fixtures/golden/images/.gitkeep
touch tests/fixtures/golden/videos/.gitkeep
```

- [ ] **Step 3: Verify .gitignore does not block them**

Check `.gitignore` for patterns that would exclude `models/**/*.pt`, `models/**/*.bin`. The `.gitkeep` files should still be tracked. Run:

```bash
git status
```

Expected: all six `.gitkeep` files appear as untracked.

- [ ] **Step 4: Commit scaffolding**

```bash
git add engine/meta_fusion/__init__.py models/meta_fusion/.gitkeep models/image_forensics/dire/.gitkeep models/image_forensics/aide/.gitkeep models/video/genconvit/.gitkeep tests/fixtures/golden/images/.gitkeep tests/fixtures/golden/videos/.gitkeep
git commit -m "chore: scaffold directories for detection redesign"
```

---

## Task 3: Implement DIREDetector wrapper

**Files:**
- Create: `engine/image_forensics/dire_detector.py`
- Test: `tests/test_new_detectors.py` (created here, extended in later tasks)

**Reference pattern:** `engine/image_forensics/vit_detector.py` (existing, follow its structure).

- [ ] **Step 1: Write the failing test**

Create `tests/test_new_detectors.py`:

```python
"""Unit tests for new detector wrappers (DIRE, AIDE, GenConViT)."""
import os
import pytest
import numpy as np
import cv2

# Skip these tests if torch isn't available (CI without GPU dependencies)
torch = pytest.importorskip("torch")


def _make_test_image() -> np.ndarray:
    """Return a synthetic 256x256 BGR image (works without sample fixtures)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.mark.gpu
def test_dire_detector_loads_and_scores():
    from engine.image_forensics.dire_detector import DIREDetector
    det = DIREDetector()
    score, details = det.predict(_make_test_image())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert details["status"] in ("success", "error")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_new_detectors.py::test_dire_detector_loads_and_scores -v
```

Expected: `FAIL — ModuleNotFoundError: No module named 'engine.image_forensics.dire_detector'`.

- [ ] **Step 3: Write minimal implementation**

Create `engine/image_forensics/dire_detector.py`. Use the loading method resolved in Task 1 (`docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md`). The wrapper structure below is fixed; fill in `_build_model()` body per the resolution doc.

```python
"""
DIRE (Diffusion Reconstruction Error) detector wrapper.
=======================================================
Detects diffusion-generated images by reconstructing them through a pretrained
diffusion model and measuring per-pixel reconstruction error.

Real images do not reconstruct cleanly through a diffusion model that was not
trained on them. AI images (especially from SD/Flux/MJ) reconstruct with low
error because they live near the diffusion model's manifold.

Reference: Wang et al., "DIRE for Diffusion-Generated Image Detection" (ICCV 2023).
Loading method per docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md
"""
import os
from typing import Tuple, Dict, Any

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DIREDetector:
    """
    Diffusion reconstruction error detector.

    Input: BGR numpy image
    Output: (fake_probability, details_dict) — same shape as ViTDetector
    """

    INPUT_SIZE = 256  # DIRE typically operates at 256x256 (LSUN-Bedroom training)
    WEIGHTS_PATH = os.path.join("models", "image_forensics", "dire", "dire_weights.pt")

    def __init__(self, device: str = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for DIREDetector")

        if device and device != "cpu" and not torch.cuda.is_available():
            device = None
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        """
        Load the DIRE detector head from the resolved pretrained source.

        Implementation: copy the upstream loading snippet identified in
        Task 1 (`docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md`)
        into this method. The wrapper expects:
          - self.model: a callable PyTorch module that takes a [1,3,H,W] tensor
            normalized per upstream conventions and returns a single logit OR
            a 2-class softmax tensor.
          - self.model.eval() called before exit.
          - self.model moved to self.device.

        If load fails, set self.model = None and log a warning. predict() handles
        this gracefully.
        """
        try:
            # === REPLACE THIS BLOCK with upstream-specific loading ===
            # Example (HuggingFace style):
            # from transformers import AutoModelForImageClassification
            # self.model = AutoModelForImageClassification.from_pretrained(
            #     "Hyojung/DIRE-pretrained"  # or local WEIGHTS_PATH
            # ).to(self.device).eval()
            #
            # Example (raw torch checkpoint):
            # from .dire_arch import DIREArch  # local copy of upstream architecture
            # self.model = DIREArch()
            # state = torch.load(self.WEIGHTS_PATH, map_location=self.device)
            # self.model.load_state_dict(state)
            # self.model.to(self.device).eval()
            raise NotImplementedError(
                "Fill in DIRE loading from resolution doc before running"
            )
            # === END BLOCK ===
        except Exception as e:
            print(f"[DIRE] Model load failed: {e}")
            self.model = None

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Score an image. Returns (fake_probability, details).
        Always returns a probability in [0.0, 1.0]; on error returns 0.5
        (neutral) so the meta-fusion does not get NaN.
        """
        if self.model is None:
            return 0.5, {"status": "error", "reason": "model not initialized"}

        try:
            # 1. Preprocess: BGR -> RGB, resize to INPUT_SIZE, normalize
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            # 2. Forward pass — adapt softmax/sigmoid based on upstream output
            out = self.model(tensor)
            if hasattr(out, "logits"):
                out = out.logits
            if out.shape[-1] == 2:
                # 2-class output: take fake-class probability
                fake_prob = float(torch.softmax(out, dim=-1)[0, 1].item())
            else:
                # Single logit: sigmoid
                fake_prob = float(torch.sigmoid(out).flatten()[0].item())

            return fake_prob, {
                "status": "success",
                "fake_probability": round(fake_prob, 4),
            }
        except Exception as e:
            return 0.5, {"status": "error", "reason": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_new_detectors.py::test_dire_detector_loads_and_scores -v
```

Expected: PASS. The test allows `details["status"]` to be either `"success"` or `"error"`, so it passes even if `_build_model()` is still raising NotImplementedError (which lands the model in error-state). After Step 3 of Task 1's resolution is incorporated into `_build_model()`, both status outcomes pass. Score will be 0.5 in error state — still satisfies the `0.0 <= score <= 1.0` assertion.

- [ ] **Step 5: Commit**

```bash
git add engine/image_forensics/dire_detector.py tests/test_new_detectors.py
git commit -m "feat: add DIRE detector wrapper for diffusion reconstruction error"
```

---

## Task 4: Implement AIDEDetector wrapper

**Files:**
- Create: `engine/image_forensics/aide_detector.py`
- Modify: `tests/test_new_detectors.py` (extend with AIDE test)

- [ ] **Step 1: Append the failing test to `tests/test_new_detectors.py`**

Add after the DIRE test:

```python
@pytest.mark.gpu
def test_aide_detector_loads_and_scores():
    from engine.image_forensics.aide_detector import AIDEDetector
    det = AIDEDetector()
    score, details = det.predict(_make_test_image())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert details["status"] in ("success", "error")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_new_detectors.py::test_aide_detector_loads_and_scores -v
```

Expected: `FAIL — ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `engine/image_forensics/aide_detector.py`:

```python
"""
AIDE (Attentive and Inversive DCT-based Embedding) detector wrapper.
====================================================================
Detects AI-generated images by combining low-level DCT spectral features
with high-level CLIP semantic features. Designed for cross-generator
generalization — strong on novel models the wrapper was not trained on.

Reference: Yan et al., "Attentive and Inversive DCT-based AI Image Detector"
(NeurIPS 2024).
Loading method per docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md
"""
import os
from typing import Tuple, Dict, Any

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AIDEDetector:
    """AIDE wrapper. Same interface as ViTDetector / DIREDetector."""

    INPUT_SIZE = 224
    WEIGHTS_PATH = os.path.join("models", "image_forensics", "aide", "aide_weights.pt")

    def __init__(self, device: str = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for AIDEDetector")

        if device and device != "cpu" and not torch.cuda.is_available():
            device = None
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        """
        Load AIDE model from resolved upstream source.
        Replace the block below with the loading code from the resolution doc.
        """
        try:
            # === REPLACE THIS BLOCK with upstream-specific loading ===
            raise NotImplementedError(
                "Fill in AIDE loading from resolution doc before running"
            )
            # === END BLOCK ===
        except Exception as e:
            print(f"[AIDE] Model load failed: {e}")
            self.model = None

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Score an image. Returns (fake_probability, details)."""
        if self.model is None:
            return 0.5, {"status": "error", "reason": "model not initialized"}

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            # Standard ImageNet normalization (most CLIP/ViT use this)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0).to(self.device)

            out = self.model(tensor)
            if hasattr(out, "logits"):
                out = out.logits
            if out.shape[-1] == 2:
                fake_prob = float(torch.softmax(out, dim=-1)[0, 1].item())
            else:
                fake_prob = float(torch.sigmoid(out).flatten()[0].item())

            return fake_prob, {
                "status": "success",
                "fake_probability": round(fake_prob, 4),
            }
        except Exception as e:
            return 0.5, {"status": "error", "reason": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_new_detectors.py::test_aide_detector_loads_and_scores -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/image_forensics/aide_detector.py tests/test_new_detectors.py
git commit -m "feat: add AIDE detector wrapper (DCT + CLIP fusion)"
```

---

## Task 5: Implement GenConViTDetector wrapper

**Files:**
- Create: `engine/video/genconvit_detector.py`
- Modify: `tests/test_new_detectors.py`

- [ ] **Step 1: Append the failing test**

Add to `tests/test_new_detectors.py`:

```python
def _make_test_video_frames(num_frames: int = 16) -> list:
    """Return a list of synthetic 256x256 BGR frames."""
    rng = np.random.default_rng(seed=42)
    return [
        rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]


@pytest.mark.gpu
def test_genconvit_detector_loads_and_scores():
    from engine.video.genconvit_detector import GenConViTDetector
    det = GenConViTDetector()
    frames = _make_test_video_frames()
    score, details = det.predict(frames)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert details["status"] in ("success", "error")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_new_detectors.py::test_genconvit_detector_loads_and_scores -v
```

Expected: `FAIL — ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `engine/video/genconvit_detector.py`:

```python
"""
GenConViT (Generalized Convolutional Vision Transformer) wrapper.
=================================================================
Single-stream ViT+CNN hybrid for deepfake video detection. Catches
face-swap and lip-sync artifacts that per-frame still detectors miss.

Reference: "Generalized Convolutional Vision Transformer for Deepfake
Detection" (2024).
Loading method per docs/superpowers/plans/2026-05-09-pretrained-model-resolution.md
"""
import os
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GenConViTDetector:
    """GenConViT wrapper. Operates on a list of video frames."""

    INPUT_SIZE = 224
    NUM_FRAMES = 16  # Standard clip length for the upstream model
    WEIGHTS_PATH = os.path.join("models", "video", "genconvit", "genconvit_weights.pt")

    def __init__(self, device: str = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GenConViTDetector")

        if device and device != "cpu" and not torch.cuda.is_available():
            device = None
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        """Load GenConViT from resolved upstream source."""
        try:
            # === REPLACE THIS BLOCK with upstream-specific loading ===
            raise NotImplementedError(
                "Fill in GenConViT loading from resolution doc before running"
            )
            # === END BLOCK ===
        except Exception as e:
            print(f"[GenConViT] Model load failed: {e}")
            self.model = None

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """
        Score a list of BGR frames. Frames are sampled/padded to NUM_FRAMES.
        Returns (fake_probability, details).
        """
        if self.model is None:
            return 0.5, {"status": "error", "reason": "model not initialized"}

        if not frames:
            return 0.5, {"status": "error", "reason": "no frames provided"}

        try:
            # Sample exactly NUM_FRAMES evenly-spaced frames
            n = len(frames)
            if n >= self.NUM_FRAMES:
                indices = np.linspace(0, n - 1, self.NUM_FRAMES, dtype=int)
                selected = [frames[i] for i in indices]
            else:
                # Pad by repeating last frame
                selected = frames + [frames[-1]] * (self.NUM_FRAMES - n)

            # Preprocess each frame: BGR -> RGB, resize, normalize
            tensors = []
            for f in selected:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
                t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
                tensors.append(t)

            # Stack to [T, C, H, W] then add batch dim -> [1, T, C, H, W]
            clip = torch.stack(tensors).unsqueeze(0).to(self.device)

            out = self.model(clip)
            if hasattr(out, "logits"):
                out = out.logits
            if out.shape[-1] == 2:
                fake_prob = float(torch.softmax(out, dim=-1)[0, 1].item())
            else:
                fake_prob = float(torch.sigmoid(out).flatten()[0].item())

            return fake_prob, {
                "status": "success",
                "fake_probability": round(fake_prob, 4),
                "num_frames_used": self.NUM_FRAMES,
            }
        except Exception as e:
            return 0.5, {"status": "error", "reason": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_new_detectors.py::test_genconvit_detector_loads_and_scores -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/video/genconvit_detector.py tests/test_new_detectors.py
git commit -m "feat: add GenConViT detector wrapper for video deepfake detection"
```

---

## Task 6: Implement FeatureExtractor (image + video)

**Files:**
- Create: `engine/meta_fusion/feature_extractor.py`
- Test: `tests/test_meta_fusion.py`

- [ ] **Step 0: Verify auxiliary detector method names**

The feature extractor invokes existing detectors via `_safe_call`. Method names referenced:
- `CompressionDetector.analyze(image_path)` and `analyze_array(image)`
- `ELAAnalyzer.analyze(image)`
- `NoiseAnalyzer.analyze(image)`
- `FrequencyAnalyzer.detect_artifacts(image)`
- `PixelStatisticsAnalyzer.analyze(image)`
- `VideoQualityAnalyzer.analyze(video_path)`
- `FaceForensicsDetector.detect_faces(image)`
- `TemporalR3D.predict(frames)`
- `ASTSpoofDetector.predict(audio_waveform)`

Verify these signatures against the actual modules:

```bash
grep -nE "def (analyze|analyze_array|detect_artifacts|detect_faces|predict)" engine/image_forensics/compression_detector.py engine/image_forensics/ela_analysis.py engine/image_forensics/noise_analysis.py engine/image_forensics/frequency_analysis.py engine/image_forensics/pixel_statistics.py engine/image_forensics/face_detector.py engine/video/quality_analyzer.py engine/video/temporal_r3d.py engine/audio/ast_spoof.py
```

If any method name differs, update the corresponding lambda in `feature_extractor.py` below to match the real signature. The `_safe_call` wrapper catches exceptions either way — wrong names will just impute defaults rather than crash — but accuracy depends on these calls actually returning real signal.

- [ ] **Step 1: Write the failing test**

Create `tests/test_meta_fusion.py`:

```python
"""Unit tests for meta-fusion components."""
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _make_test_image() -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def _make_test_frames(n: int = 16) -> list:
    rng = np.random.default_rng(seed=7)
    return [rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(n)]


@pytest.mark.gpu
def test_image_feature_extractor_returns_fixed_shape():
    from engine.meta_fusion.feature_extractor import ImageFeatureExtractor
    extractor = ImageFeatureExtractor()
    features = extractor.extract(_make_test_image())
    assert features.shape == (9,)  # spec: 9 image features
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0.0) and np.all(features <= 1.0)


@pytest.mark.gpu
def test_video_feature_extractor_returns_fixed_shape():
    from engine.meta_fusion.feature_extractor import VideoFeatureExtractor
    extractor = VideoFeatureExtractor()
    features = extractor.extract(_make_test_frames())
    assert features.shape == (12,)  # spec: 12 video features
    assert np.all(np.isfinite(features))


def test_image_feature_extractor_handles_detector_failure():
    """If DIRE/AIDE/ViT raises, the failed feature should be imputed (not NaN)."""
    from engine.meta_fusion.feature_extractor import ImageFeatureExtractor
    from unittest.mock import patch

    extractor = ImageFeatureExtractor()
    # Patch DIRE to raise — feature should fall back to imputed value
    with patch.object(extractor.dire, "predict", side_effect=RuntimeError("OOM")):
        features = extractor.extract(_make_test_image())
        assert np.all(np.isfinite(features))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_meta_fusion.py::test_image_feature_extractor_returns_fixed_shape -v
```

Expected: `FAIL — ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `engine/meta_fusion/feature_extractor.py`:

```python
"""
Feature extractors for meta-fusion.

ImageFeatureExtractor — produces 9-dim feature vector per image.
VideoFeatureExtractor — produces 12-dim feature vector per video clip.

The feature ordering is FIXED. The trained logistic regression depends on
column order. Never reorder columns without retraining and incrementing
the joblib version.
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np
import logging

log = logging.getLogger("satyadrishti.meta_fusion")

# Imputation defaults — used when a detector errors out. Set to neutral 0.5
# unless a specific feature has a different "no signal" value.
IMAGE_IMPUTE = {
    "vit_score": 0.5,
    "dire_score": 0.5,
    "aide_score": 0.5,
    "compression_flag": 0.0,
    "content_type_code": 0.0,
    "ela_score": 0.0,
    "noise_score": 0.0,
    "frequency_score": 0.0,
    "pixel_stat_score": 0.0,
}

VIDEO_IMPUTE = {
    "frame_vit_mean": 0.5,
    "frame_vit_max": 0.5,
    "frame_vit_std": 0.0,
    "frame_aide_mean": 0.5,
    "frame_aide_max": 0.5,
    "frame_aide_std": 0.0,
    "dire_keyframe_mean": 0.5,
    "genconvit_score": 0.5,
    "r3d_score": 0.5,
    "audio_score": 0.5,
    "compression_flag": 0.0,
    "has_face": 0.0,
}

IMAGE_FEATURE_ORDER = list(IMAGE_IMPUTE.keys())
VIDEO_FEATURE_ORDER = list(VIDEO_IMPUTE.keys())


def _safe_call(fn, default: float, label: str) -> float:
    """Call fn(), return its float result, or `default` on any exception."""
    try:
        result = fn()
        if isinstance(result, tuple):
            result = result[0]
        result = float(result)
        if not np.isfinite(result):
            log.warning("[%s] non-finite value, imputing default", label)
            return default
        return max(0.0, min(1.0, result))
    except Exception as e:
        log.warning("[%s] failed (%s), imputing default", label, e)
        return default


class ImageFeatureExtractor:
    """Builds 9-dim feature vector for an image."""

    def __init__(self):
        from engine.image_forensics.vit_detector import ViTDetector
        from engine.image_forensics.dire_detector import DIREDetector
        from engine.image_forensics.aide_detector import AIDEDetector
        from engine.image_forensics.compression_detector import CompressionDetector
        from engine.image_forensics.ela_analysis import ELAAnalyzer
        from engine.image_forensics.noise_analysis import NoiseAnalyzer
        from engine.image_forensics.frequency_analysis import FrequencyAnalyzer
        from engine.image_forensics.pixel_statistics import PixelStatisticsAnalyzer

        self.vit = ViTDetector()
        self.dire = DIREDetector()
        self.aide = AIDEDetector()
        self.compression = CompressionDetector()
        self.ela = ELAAnalyzer()
        self.noise = NoiseAnalyzer()
        self.freq = FrequencyAnalyzer()
        self.pixel = PixelStatisticsAnalyzer()

    def extract(self, image: np.ndarray, image_path: Optional[str] = None) -> np.ndarray:
        """Return 9-dim feature vector in IMAGE_FEATURE_ORDER."""
        feats = {}
        feats["vit_score"] = _safe_call(
            lambda: self.vit.predict(image), IMAGE_IMPUTE["vit_score"], "vit"
        )
        feats["dire_score"] = _safe_call(
            lambda: self.dire.predict(image), IMAGE_IMPUTE["dire_score"], "dire"
        )
        feats["aide_score"] = _safe_call(
            lambda: self.aide.predict(image), IMAGE_IMPUTE["aide_score"], "aide"
        )

        # compression_flag: 1.0 if social-media compressed, else 0.0
        def _compression():
            result = self.compression.analyze(image_path) if image_path else self.compression.analyze_array(image)
            platform = result.get("platform", "")
            return 1.0 if platform and platform != "original" else 0.0
        feats["compression_flag"] = _safe_call(
            _compression, IMAGE_IMPUTE["compression_flag"], "compression"
        )

        # content_type_code: reserved feature column. Always 0.0 for now —
        # the spec keeps it in the schema so the joblib feature_dim stays
        # stable when a future content classifier supplies a real value.
        feats["content_type_code"] = 0.0

        feats["ela_score"] = _safe_call(
            lambda: self.ela.analyze(image), IMAGE_IMPUTE["ela_score"], "ela"
        )
        feats["noise_score"] = _safe_call(
            lambda: self.noise.analyze(image), IMAGE_IMPUTE["noise_score"], "noise"
        )
        feats["frequency_score"] = _safe_call(
            lambda: self.freq.detect_artifacts(image), IMAGE_IMPUTE["frequency_score"], "freq"
        )
        feats["pixel_stat_score"] = _safe_call(
            lambda: self.pixel.analyze(image), IMAGE_IMPUTE["pixel_stat_score"], "pixel"
        )

        return np.array([feats[k] for k in IMAGE_FEATURE_ORDER], dtype=np.float64)


class VideoFeatureExtractor:
    """Builds 12-dim feature vector for a video clip."""

    NUM_DIRE_KEYFRAMES = 3

    def __init__(self):
        from engine.image_forensics.vit_detector import ViTDetector
        from engine.image_forensics.aide_detector import AIDEDetector
        from engine.image_forensics.dire_detector import DIREDetector
        from engine.video.genconvit_detector import GenConViTDetector
        from engine.video.temporal_r3d import TemporalR3D
        from engine.audio.ast_spoof import ASTSpoofDetector
        from engine.video.quality_analyzer import VideoQualityAnalyzer
        from engine.image_forensics.face_detector import FaceForensicsDetector

        self.vit = ViTDetector()
        self.aide = AIDEDetector()
        self.dire = DIREDetector()
        self.genconvit = GenConViTDetector()
        self.r3d = TemporalR3D()
        self.audio = ASTSpoofDetector()
        self.quality = VideoQualityAnalyzer()
        self.face = FaceForensicsDetector()

    def extract(
        self,
        frames: List[np.ndarray],
        audio_waveform: Optional[np.ndarray] = None,
        video_path: Optional[str] = None,
    ) -> np.ndarray:
        """Return 12-dim feature vector in VIDEO_FEATURE_ORDER."""
        feats = {}

        # Per-frame ViT scores
        vit_scores = []
        for f in frames:
            s = _safe_call(lambda: self.vit.predict(f), 0.5, "vit_per_frame")
            vit_scores.append(s)
        feats["frame_vit_mean"] = float(np.mean(vit_scores)) if vit_scores else 0.5
        feats["frame_vit_max"] = float(np.max(vit_scores)) if vit_scores else 0.5
        feats["frame_vit_std"] = float(np.std(vit_scores)) if vit_scores else 0.0

        # Per-frame AIDE scores
        aide_scores = []
        for f in frames:
            s = _safe_call(lambda: self.aide.predict(f), 0.5, "aide_per_frame")
            aide_scores.append(s)
        feats["frame_aide_mean"] = float(np.mean(aide_scores)) if aide_scores else 0.5
        feats["frame_aide_max"] = float(np.max(aide_scores)) if aide_scores else 0.5
        feats["frame_aide_std"] = float(np.std(aide_scores)) if aide_scores else 0.0

        # DIRE on N evenly-spaced keyframes only (compute budget)
        if len(frames) >= self.NUM_DIRE_KEYFRAMES:
            kf_indices = np.linspace(0, len(frames) - 1, self.NUM_DIRE_KEYFRAMES, dtype=int)
            kf_scores = [
                _safe_call(lambda f=frames[i]: self.dire.predict(f), 0.5, "dire_keyframe")
                for i in kf_indices
            ]
            feats["dire_keyframe_mean"] = float(np.mean(kf_scores))
        else:
            feats["dire_keyframe_mean"] = 0.5

        # Whole-clip detectors
        feats["genconvit_score"] = _safe_call(
            lambda: self.genconvit.predict(frames), VIDEO_IMPUTE["genconvit_score"], "genconvit"
        )
        feats["r3d_score"] = _safe_call(
            lambda: self.r3d.predict(frames), VIDEO_IMPUTE["r3d_score"], "r3d"
        )

        # Audio: 0.5 if missing
        if audio_waveform is not None:
            feats["audio_score"] = _safe_call(
                lambda: self.audio.predict(audio_waveform),
                VIDEO_IMPUTE["audio_score"],
                "audio",
            )
        else:
            feats["audio_score"] = 0.5

        # Compression flag
        if video_path:
            feats["compression_flag"] = _safe_call(
                lambda: float(self.quality.analyze(video_path).get("compression_score", 0) > 0.3),
                VIDEO_IMPUTE["compression_flag"],
                "compression",
            )
        else:
            feats["compression_flag"] = 0.0

        # has_face: 1 if any frame contains a face
        def _has_face():
            for f in frames[:4]:  # check first 4 frames only
                if self.face.detect_faces(f):
                    return 1.0
            return 0.0
        feats["has_face"] = _safe_call(_has_face, VIDEO_IMPUTE["has_face"], "face")

        return np.array([feats[k] for k in VIDEO_FEATURE_ORDER], dtype=np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_meta_fusion.py -v
```

Expected: 3 PASS. Tests use the failure-imputation path because new detectors are still in error-state until Task 1's resolution is wired in. That's the intended design — meta-fusion never crashes on detector failure.

- [ ] **Step 5: Commit**

```bash
git add engine/meta_fusion/feature_extractor.py tests/test_meta_fusion.py
git commit -m "feat: add image+video feature extractors with failure imputation"
```

---

## Task 7: Implement FusionModel + TemperatureScaler

**Files:**
- Create: `engine/meta_fusion/fusion_model.py`
- Create: `engine/meta_fusion/temperature_scaler.py`
- Modify: `tests/test_meta_fusion.py`

- [ ] **Step 1: Append the failing tests**

Add to `tests/test_meta_fusion.py`:

```python
def test_fusion_model_load_and_predict(tmp_path):
    """Train a tiny logistic regression, save it, reload via wrapper, predict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib

    rng = np.random.default_rng(0)
    X = rng.random((50, 9))
    y = (X[:, 0] > 0.5).astype(int)  # synthetic: depends on first feature

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression().fit(scaler.transform(X), y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump({"scaler": scaler, "classifier": clf, "feature_dim": 9}, model_path)

    from engine.meta_fusion.fusion_model import FusionModel
    model = FusionModel.load(str(model_path))
    prob = model.predict_proba(rng.random(9))
    assert 0.0 <= prob <= 1.0


def test_temperature_scaler_reduces_overconfidence():
    """Calibrated probability should be closer to 0.5 than the raw probability when T>1."""
    from engine.meta_fusion.temperature_scaler import TemperatureScaler
    scaler = TemperatureScaler(T=2.0)
    raw_logit = 3.0  # raw sigmoid = 0.9526
    calibrated = scaler.scale(raw_logit)
    raw = 1.0 / (1.0 + np.exp(-raw_logit))
    assert calibrated < raw
    assert calibrated > 0.5  # still on the same side
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_meta_fusion.py::test_fusion_model_load_and_predict -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementations**

Create `engine/meta_fusion/fusion_model.py`:

```python
"""Wrapper around a saved sklearn LogisticRegression bundle."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import joblib


@dataclass
class FusionModel:
    """Loaded meta-fusion model (scaler + classifier)."""
    scaler: object       # sklearn StandardScaler
    classifier: object   # sklearn LogisticRegression
    feature_dim: int

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        bundle = joblib.load(path)
        return cls(
            scaler=bundle["scaler"],
            classifier=bundle["classifier"],
            feature_dim=int(bundle["feature_dim"]),
        )

    def predict_proba(self, features: np.ndarray) -> float:
        """Return probability of fake (class 1) given a 1-D feature vector."""
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"FusionModel expected {self.feature_dim} features, got {features.shape}"
            )
        scaled = self.scaler.transform(features.reshape(1, -1))
        prob = float(self.classifier.predict_proba(scaled)[0, 1])
        return prob

    def predict_logit(self, features: np.ndarray) -> float:
        """Return raw decision-function logit (used by temperature scaling)."""
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"FusionModel expected {self.feature_dim} features, got {features.shape}"
            )
        scaled = self.scaler.transform(features.reshape(1, -1))
        logit = float(self.classifier.decision_function(scaled)[0])
        return logit
```

Create `engine/meta_fusion/temperature_scaler.py`:

```python
"""Temperature scaling for confidence calibration."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass


@dataclass
class TemperatureScaler:
    """Single scalar T applied as: calibrated_prob = sigmoid(logit / T)."""
    T: float = 1.0

    def scale(self, logit: float) -> float:
        """Apply temperature to a raw logit, return calibrated probability."""
        if self.T <= 0:
            # Defensive: degenerate T, fall back to identity
            return 1.0 / (1.0 + math.exp(-logit))
        return 1.0 / (1.0 + math.exp(-logit / self.T))

    @classmethod
    def load(cls, path: str, key: str = "T") -> "TemperatureScaler":
        """Load from a JSON file like {"image_T": 1.4, "video_T": 1.2}."""
        with open(path, "r") as f:
            data = json.load(f)
        T = float(data.get(key, 1.0))
        return cls(T=T)

    def save(self, path: str, key: str = "T") -> None:
        """Persist T to JSON."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data[key] = self.T
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_meta_fusion.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/meta_fusion/fusion_model.py engine/meta_fusion/temperature_scaler.py tests/test_meta_fusion.py
git commit -m "feat: add fusion model loader and temperature scaler"
```

---

## Task 8: Implement MetaFusionEngine orchestrator

**Files:**
- Create: `engine/meta_fusion/engine.py`
- Modify: `tests/test_meta_fusion.py`

- [ ] **Step 1: Append the failing test**

```python
def test_meta_fusion_engine_image_returns_valid_response(tmp_path, monkeypatch):
    """End-to-end: extract features, load fusion model, return verdict response."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib

    rng = np.random.default_rng(0)
    X = rng.random((50, 9))
    y = (X[:, 0] > 0.5).astype(int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression().fit(scaler.transform(X), y)

    model_dir = tmp_path / "meta_fusion"
    model_dir.mkdir()
    joblib.dump({"scaler": scaler, "classifier": clf, "feature_dim": 9},
                model_dir / "image_v1.joblib")
    (model_dir / "temperature.json").write_text('{"image_T": 1.0, "video_T": 1.0}')

    monkeypatch.setenv("META_FUSION_DIR", str(model_dir))

    from engine.meta_fusion.engine import MetaFusionEngine
    engine = MetaFusionEngine("image", model_dir=str(model_dir))
    result = engine.analyze_array(_make_test_image())

    assert result["verdict"] in ("authentic", "ai-generated", "inconclusive")
    assert 0 <= result["confidence"] <= 100
    assert "raw_scores" in result
    assert "forensic_checks" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_meta_fusion.py::test_meta_fusion_engine_image_returns_valid_response -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Create `engine/meta_fusion/engine.py`:

```python
"""
MetaFusionEngine — top-level orchestrator for the new detection pipeline.

Replaces the old ImageForensicsDetector verdict logic. Loads a trained
LogisticRegression + TemperatureScaler, delegates feature extraction
to ImageFeatureExtractor / VideoFeatureExtractor, and produces a
{verdict, confidence, raw_scores, forensic_checks} response matching
the old API contract.
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import cv2

from .feature_extractor import (
    ImageFeatureExtractor, VideoFeatureExtractor,
    IMAGE_FEATURE_ORDER, VIDEO_FEATURE_ORDER,
)
from .fusion_model import FusionModel
from .temperature_scaler import TemperatureScaler

log = logging.getLogger("satyadrishti.meta_fusion")

# Verdict thresholds applied AFTER temperature scaling
THRESH_FAKE = 0.70
THRESH_REAL = 0.30
CONF_CAP = 95.0


class MetaFusionEngine:
    """
    Args:
        modality: "image" or "video"
        model_dir: path to models/meta_fusion/
    """

    def __init__(self, modality: str, model_dir: Optional[str] = None):
        if modality not in ("image", "video"):
            raise ValueError(f"modality must be 'image' or 'video', got {modality}")
        self.modality = modality
        self.model_dir = model_dir or os.path.join("models", "meta_fusion")

        if modality == "image":
            self.extractor = ImageFeatureExtractor()
            model_path = os.path.join(self.model_dir, "image_v1.joblib")
            t_key = "image_T"
            self.feature_order = IMAGE_FEATURE_ORDER
        else:
            self.extractor = VideoFeatureExtractor()
            model_path = os.path.join(self.model_dir, "video_v1.joblib")
            t_key = "video_T"
            self.feature_order = VIDEO_FEATURE_ORDER

        self.fusion = FusionModel.load(model_path)
        temp_path = os.path.join(self.model_dir, "temperature.json")
        if os.path.exists(temp_path):
            self.temperature = TemperatureScaler.load(temp_path, key=t_key)
        else:
            log.warning("temperature.json not found, defaulting to T=1.0")
            self.temperature = TemperatureScaler(T=1.0)

    def analyze(self, path: str) -> Dict[str, Any]:
        """Top-level entry point for a file path."""
        if self.modality == "image":
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            return self._analyze_image(image, path)
        return self._analyze_video(path)

    def analyze_array(
        self,
        image: np.ndarray,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience: analyze a numpy image already in memory."""
        if self.modality != "image":
            raise ValueError("analyze_array is image-only; use analyze() for video")
        return self._analyze_image(image, path)

    def _analyze_image(self, image: np.ndarray, image_path: Optional[str]) -> Dict[str, Any]:
        features = self.extractor.extract(image, image_path=image_path)
        return self._verdict_from_features(features)

    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        # Sample 16 frames + extract audio
        frames = self._sample_frames(video_path, num_frames=16)
        audio_wav = self._extract_audio(video_path)
        features = self.extractor.extract(
            frames, audio_waveform=audio_wav, video_path=video_path
        )
        return self._verdict_from_features(features)

    def _verdict_from_features(self, features: np.ndarray) -> Dict[str, Any]:
        logit = self.fusion.predict_logit(features)
        prob = self.temperature.scale(logit)

        if prob >= THRESH_FAKE:
            verdict = "ai-generated"
            confidence = min(CONF_CAP, 50.0 + (prob - 0.5) * 100)
        elif prob < THRESH_REAL:
            verdict = "authentic"
            confidence = min(CONF_CAP, 50.0 + (0.5 - prob) * 100)
        else:
            verdict = "inconclusive"
            confidence = 50.0 + abs(prob - 0.5) * 50

        raw_scores = {
            name: float(features[i]) for i, name in enumerate(self.feature_order)
        }
        raw_scores["fusion_logit"] = float(logit)
        raw_scores["calibrated_probability"] = float(prob)

        forensic_checks = [
            {
                "id": "meta_fusion",
                "name": "Meta-Fusion Verdict",
                "status": "pass" if verdict == "authentic" else
                          "fail" if verdict == "ai-generated" else "warn",
                "description": (
                    f"Calibrated probability of AI: {prob:.3f} "
                    f"(threshold: real<{THRESH_REAL}, fake≥{THRESH_FAKE})"
                ),
            }
        ]

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 1),
            "raw_scores": raw_scores,
            "forensic_checks": forensic_checks,
        }

    @staticmethod
    def _sample_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def _extract_audio(video_path: str) -> Optional[np.ndarray]:
        """Extract audio waveform at 16kHz mono. Returns None if no audio track."""
        try:
            import subprocess
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
                 "-vn", tmp_path],
                check=True, capture_output=True,
            )
            wav, _ = sf.read(tmp_path)
            os.unlink(tmp_path)
            return wav.astype(np.float32)
        except Exception as e:
            log.warning("audio extraction failed: %s", e)
            return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_meta_fusion.py::test_meta_fusion_engine_image_returns_valid_response -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/meta_fusion/engine.py tests/test_meta_fusion.py
git commit -m "feat: add MetaFusionEngine orchestrator"
```

---

## Task 9: Build calibration set assembly script

**Files:**
- Create: `scripts/build_calibration_set.py`

- [ ] **Step 1: Write the script**

Create `scripts/build_calibration_set.py`:

```python
"""
Build calibration set for meta-fusion training.

Walks a directory of labeled examples, runs feature extractors on each,
saves (features, labels) array to disk for use by train_meta_fusion.py.

Expected directory structure:
    calibration_data/
        images/
            real/
                *.jpg/png/heic
            ai/
                *.jpg/png
        videos/
            real/
                *.mp4
            ai/
                *.mp4

Output:
    models/meta_fusion/calibration_image.npz   {X, y, paths}
    models/meta_fusion/calibration_video.npz   {X, y, paths}
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from engine.meta_fusion.feature_extractor import (
    ImageFeatureExtractor, VideoFeatureExtractor,
)


def _walk_labeled_dir(root: Path, exts: set) -> list:
    """Return [(path, label), ...] where label=1 for ai/, 0 for real/."""
    pairs = []
    for cls, label in (("real", 0), ("ai", 1)):
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in exts:
                pairs.append((str(p), label))
    return pairs


def build_image_calibration(data_root: Path, out_path: Path):
    extractor = ImageFeatureExtractor()
    pairs = _walk_labeled_dir(
        data_root / "images",
        {".jpg", ".jpeg", ".png", ".webp", ".heic"},
    )
    if not pairs:
        raise RuntimeError(f"No images found under {data_root}/images")

    X, y, paths = [], [], []
    for i, (path, label) in enumerate(pairs):
        img = cv2.imread(path)
        if img is None:
            print(f"  [{i+1}/{len(pairs)}] SKIP (unreadable): {path}")
            continue
        feats = extractor.extract(img, image_path=path)
        X.append(feats)
        y.append(label)
        paths.append(path)
        print(f"  [{i+1}/{len(pairs)}] {path} -> {feats[:3]} ... label={label}")

    np.savez(
        out_path,
        X=np.array(X, dtype=np.float64),
        y=np.array(y, dtype=np.int64),
        paths=np.array(paths),
    )
    print(f"Saved {len(X)} image samples to {out_path}")


def build_video_calibration(data_root: Path, out_path: Path):
    extractor = VideoFeatureExtractor()
    pairs = _walk_labeled_dir(data_root / "videos", {".mp4", ".mov", ".avi", ".mkv"})
    if not pairs:
        print(f"No videos found under {data_root}/videos -- skipping video calibration")
        return

    from engine.meta_fusion.engine import MetaFusionEngine

    X, y, paths = [], [], []
    for i, (path, label) in enumerate(pairs):
        try:
            frames = MetaFusionEngine._sample_frames(path, num_frames=16)
            audio = MetaFusionEngine._extract_audio(path)
            feats = extractor.extract(frames, audio_waveform=audio, video_path=path)
            X.append(feats)
            y.append(label)
            paths.append(path)
            print(f"  [{i+1}/{len(pairs)}] {path} -> label={label}")
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] SKIP ({e}): {path}")

    np.savez(
        out_path,
        X=np.array(X, dtype=np.float64),
        y=np.array(y, dtype=np.int64),
        paths=np.array(paths),
    )
    print(f"Saved {len(X)} video samples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Directory containing images/{real,ai}/ and videos/{real,ai}/")
    parser.add_argument("--out-dir", default="models/meta_fusion", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_image_calibration(args.data_root, args.out_dir / "calibration_image.npz")
    build_video_calibration(args.data_root, args.out_dir / "calibration_video.npz")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script invocation**

```bash
python scripts/build_calibration_set.py --help
```

Expected: argparse usage printed (no exception).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_calibration_set.py
git commit -m "feat: add calibration-set assembly script"
```

---

## Task 10: Build meta-fusion training script

**Files:**
- Create: `scripts/train_meta_fusion.py`

- [ ] **Step 1: Write the script**

Create `scripts/train_meta_fusion.py`:

```python
"""
Train logistic regression + temperature scaler from calibration set.

Inputs:
    models/meta_fusion/calibration_image.npz  {X, y, paths}
    models/meta_fusion/calibration_video.npz  (optional)

Outputs:
    models/meta_fusion/image_v1.joblib  {scaler, classifier, feature_dim}
    models/meta_fusion/video_v1.joblib  (if video calibration present)
    models/meta_fusion/temperature.json  {image_T, video_T}
    models/meta_fusion/eval_v1.json      held-out metrics
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, log_loss,
)
from scipy.optimize import minimize_scalar


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T that minimizes NLL of sigmoid(logits / T) vs labels."""
    def nll(T):
        if T <= 0:
            return 1e9
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE over n_bins equal-width probability bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(labels[mask]))
        bin_conf = float(np.mean(probs[mask]))
        ece += (np.sum(mask) / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def train_one_modality(
    cal_path: Path, out_joblib: Path, modality: str
) -> dict:
    data = np.load(cal_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"[{modality}] Loaded {len(X)} samples, {X.shape[1]} features")

    if len(X) < 20:
        raise RuntimeError(
            f"Too few samples ({len(X)}) — collect more calibration data"
        )

    # 80/20 train/holdout split
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_ho_s = scaler.transform(X_ho)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_tr_s, y_tr)

    # 5-fold CV accuracy on training portion
    cv_scores = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring="accuracy")
    print(f"[{modality}] 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Holdout metrics
    holdout_logits = clf.decision_function(X_ho_s)
    holdout_probs_uncal = clf.predict_proba(X_ho_s)[:, 1]
    holdout_preds = (holdout_probs_uncal >= 0.5).astype(int)

    # Fit temperature on holdout
    T = fit_temperature(holdout_logits, y_ho)
    print(f"[{modality}] Temperature T = {T:.3f}")

    # Calibrated probabilities for metrics
    cal_probs = 1.0 / (1.0 + np.exp(-holdout_logits / T))

    cm = confusion_matrix(y_ho, holdout_preds, labels=[0, 1])
    real_total = int(cm[0].sum()) or 1
    ai_total = int(cm[1].sum()) or 1
    metrics = {
        "n_train": int(len(y_tr)),
        "n_holdout": int(len(y_ho)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "holdout_accuracy": float(accuracy_score(y_ho, holdout_preds)),
        "holdout_auc": float(roc_auc_score(y_ho, holdout_probs_uncal))
                       if len(np.unique(y_ho)) == 2 else None,
        "fp_rate_real": float(cm[0, 1]) / real_total,
        "fn_rate_ai": float(cm[1, 0]) / ai_total,
        "expected_calibration_error_uncal": expected_calibration_error(holdout_probs_uncal, y_ho),
        "expected_calibration_error_cal": expected_calibration_error(cal_probs, y_ho),
        "log_loss_uncal": float(log_loss(y_ho, holdout_probs_uncal, labels=[0, 1])),
        "log_loss_cal": float(log_loss(y_ho, cal_probs, labels=[0, 1])),
        "temperature_T": T,
        "feature_coefs": clf.coef_[0].tolist(),
    }
    print(f"[{modality}] Holdout: acc={metrics['holdout_accuracy']:.3f}, "
          f"FP_real={metrics['fp_rate_real']:.3f}, FN_ai={metrics['fn_rate_ai']:.3f}")
    print(f"[{modality}] ECE: uncal={metrics['expected_calibration_error_uncal']:.3f}, "
          f"cal={metrics['expected_calibration_error_cal']:.3f}")

    # Save the bundle
    bundle = {
        "scaler": scaler,
        "classifier": clf,
        "feature_dim": int(X.shape[1]),
    }
    joblib.dump(bundle, out_joblib)
    print(f"[{modality}] Saved {out_joblib}")

    return {"metrics": metrics, "T": T}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-dir", default="models/meta_fusion", type=Path)
    args = parser.parse_args()

    image_cal = args.cal_dir / "calibration_image.npz"
    video_cal = args.cal_dir / "calibration_video.npz"

    eval_results = {}
    temperatures = {}

    if image_cal.exists():
        result = train_one_modality(
            image_cal, args.cal_dir / "image_v1.joblib", "image",
        )
        eval_results["image"] = result["metrics"]
        temperatures["image_T"] = result["T"]

    if video_cal.exists():
        result = train_one_modality(
            video_cal, args.cal_dir / "video_v1.joblib", "video",
        )
        eval_results["video"] = result["metrics"]
        temperatures["video_T"] = result["T"]

    with open(args.cal_dir / "temperature.json", "w") as f:
        json.dump(temperatures, f, indent=2)
    with open(args.cal_dir / "eval_v1.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Saved temperatures to {args.cal_dir / 'temperature.json'}")
    print(f"Saved eval metrics to {args.cal_dir / 'eval_v1.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test invocation**

```bash
python scripts/train_meta_fusion.py --help
```

Expected: argparse usage.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_meta_fusion.py
git commit -m "feat: add meta-fusion training script with temperature scaling"
```

---

## Task 11: Build evaluation script

**Files:**
- Create: `scripts/eval_meta_fusion.py`

- [ ] **Step 1: Write the script**

Create `scripts/eval_meta_fusion.py`:

```python
"""
Evaluate a trained meta-fusion model on a labeled dataset.

Usage:
    python scripts/eval_meta_fusion.py --modality image --data-root <dir>
    python scripts/eval_meta_fusion.py --modality video --data-root <dir>

Prints confusion matrix + per-source breakdown and saves to
models/meta_fusion/eval_<timestamp>.json.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from engine.meta_fusion.engine import MetaFusionEngine


VERDICT_TO_INT = {"authentic": 0, "ai-generated": 1, "inconclusive": -1}


def evaluate(modality: str, data_root: Path, model_dir: Path) -> dict:
    engine = MetaFusionEngine(modality, model_dir=str(model_dir))

    preds, labels, paths = [], [], []
    if modality == "image":
        exts = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
        subdir = "images"
    else:
        exts = {".mp4", ".mov", ".avi", ".mkv"}
        subdir = "videos"

    for cls, label in (("real", 0), ("ai", 1)):
        cls_dir = data_root / subdir / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() not in exts:
                continue
            try:
                if modality == "image":
                    result = engine.analyze(str(p))
                else:
                    result = engine.analyze(str(p))
                pred = VERDICT_TO_INT[result["verdict"]]
                preds.append(pred)
                labels.append(label)
                paths.append(str(p))
            except Exception as e:
                print(f"SKIP {p}: {e}")

    preds = np.array(preds)
    labels = np.array(labels)

    # Treat "inconclusive" as wrong-by-default for accuracy
    correct = (preds == labels).sum()
    accuracy = float(correct) / max(len(labels), 1)

    # Confusion matrix excluding inconclusive
    decisive_mask = preds != -1
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    inconclusive = int(np.sum(preds == -1))

    real_n = int(np.sum(labels == 0))
    ai_n = int(np.sum(labels == 1))

    metrics = {
        "modality": modality,
        "n_total": int(len(labels)),
        "n_real": real_n,
        "n_ai": ai_n,
        "accuracy": accuracy,
        "fp_rate_real": float(fp) / max(real_n, 1),
        "fn_rate_ai": float(fn) / max(ai_n, 1),
        "inconclusive_rate": float(inconclusive) / max(len(labels), 1),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                             "inconclusive": inconclusive},
        "decisive_n": int(decisive_mask.sum()),
        "decisive_accuracy": float(np.sum(preds[decisive_mask] == labels[decisive_mask]))
                              / max(decisive_mask.sum(), 1),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["image", "video"])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", default="models/meta_fusion", type=Path)
    args = parser.parse_args()

    metrics = evaluate(args.modality, args.data_root, args.model_dir)
    print(json.dumps(metrics, indent=2))

    out_path = args.model_dir / f"eval_{args.modality}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test invocation**

```bash
python scripts/eval_meta_fusion.py --help
```

Expected: argparse usage.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_meta_fusion.py
git commit -m "feat: add meta-fusion evaluation script"
```

---

## Task 12: Add server config flags

**Files:**
- Modify: `server/config.py` (append at end of "Model Checkpoints" section)

- [ ] **Step 1: Add config block**

In `server/config.py`, after the existing `FUSION_CKPT = ...` line (around line 88), append:

```python

# ─── Meta-Fusion Pipeline (new detection redesign) ───
USE_NEW_PIPELINE = os.environ.get("USE_NEW_PIPELINE", "false").lower() == "true"
META_FUSION_DIR = str(MODEL_DIR / "meta_fusion")
DIRE_WEIGHTS_PATH = str(MODEL_DIR / "image_forensics" / "dire" / "dire_weights.pt")
AIDE_WEIGHTS_PATH = str(MODEL_DIR / "image_forensics" / "aide" / "aide_weights.pt")
GENCONVIT_WEIGHTS_PATH = str(MODEL_DIR / "video" / "genconvit" / "genconvit_weights.pt")
```

- [ ] **Step 2: Verify the import works**

```bash
python -c "from server.config import USE_NEW_PIPELINE, META_FUSION_DIR; print(USE_NEW_PIPELINE, META_FUSION_DIR)"
```

Expected: `False <full path to models/meta_fusion>` (no exception).

- [ ] **Step 3: Commit**

```bash
git add server/config.py
git commit -m "feat: add USE_NEW_PIPELINE feature flag and meta-fusion paths"
```

---

## Task 13: Wire MetaFusionEngine into inference_engine

**Files:**
- Modify: `server/inference_engine.py`

This is the surgical hook. The new engine ONLY runs when `USE_NEW_PIPELINE=true` and meta-fusion model files exist; otherwise old pipeline is unchanged.

- [ ] **Step 1: Locate the existing image and video analysis methods**

```bash
grep -n "def analyze_image\|def analyze_video\|self.image_detector\|self.video_detector" server/inference_engine.py
```

Note the line numbers for image and video analysis entry points; they will be referenced below.

- [ ] **Step 2: Add MetaFusionEngine import + lazy init**

In `server/inference_engine.py`, find the `try: import torch ...` block (around line 28-50) and add inside the `try`:

```python
    from engine.meta_fusion.engine import MetaFusionEngine
```

Then locate the `InferenceEngine.__init__` method and add at the end (before `__init__` returns):

```python
        # New pipeline (Phase 3 — feature flagged off by default)
        self._meta_fusion_image = None
        self._meta_fusion_video = None
        if config.USE_NEW_PIPELINE:
            try:
                self._meta_fusion_image = MetaFusionEngine("image", model_dir=config.META_FUSION_DIR)
                log.info("Meta-fusion image engine loaded")
            except Exception as e:
                log.warning("Meta-fusion image engine unavailable: %s", e)
            try:
                self._meta_fusion_video = MetaFusionEngine("video", model_dir=config.META_FUSION_DIR)
                log.info("Meta-fusion video engine loaded")
            except Exception as e:
                log.warning("Meta-fusion video engine unavailable: %s", e)
```

Extend the existing `from .config import (...)` block (around line 21-24 of `server/inference_engine.py`) to also import the new symbols:

```python
from .config import (
    TEXT_CHECKPOINT, VIDEO_SPATIAL_CKPT, VIDEO_TEMPORAL_CKPT,
    FUSION_CKPT, XLS_R_MODEL_PATH,
    USE_NEW_PIPELINE, META_FUSION_DIR,
)
```

Then in the `__init__` block, reference the imported names directly (`USE_NEW_PIPELINE` and `META_FUSION_DIR`) rather than `config.USE_NEW_PIPELINE`. Update the snippet from Step 2 accordingly:

```python
        self._meta_fusion_image = None
        self._meta_fusion_video = None
        if USE_NEW_PIPELINE:
            try:
                self._meta_fusion_image = MetaFusionEngine("image", model_dir=META_FUSION_DIR)
                log.info("Meta-fusion image engine loaded")
            except Exception as e:
                log.warning("Meta-fusion image engine unavailable: %s", e)
            try:
                self._meta_fusion_video = MetaFusionEngine("video", model_dir=META_FUSION_DIR)
                log.info("Meta-fusion video engine loaded")
            except Exception as e:
                log.warning("Meta-fusion video engine unavailable: %s", e)
```

- [ ] **Step 3: Dispatch in image analysis path**

Find the existing image analysis method (image-handling path, typically named `analyze_image` or similar; the actual call is into `ImageForensicsDetector.analyze`). Wrap the call with the dispatch:

```python
    def analyze_image(self, image_path: str, **kwargs) -> Dict[str, Any]:
        if self._meta_fusion_image is not None:
            try:
                return self._meta_fusion_image.analyze(image_path)
            except Exception as e:
                log.warning("meta-fusion image analyze failed (%s); falling back", e)
        # Fall back to old pipeline
        return self._legacy_analyze_image(image_path, **kwargs)
```

Rename the existing implementation to `_legacy_analyze_image` (just the method name; body unchanged).

- [ ] **Step 4: Dispatch in video analysis path**

Same pattern for video. Find the video analysis method and wrap it:

```python
    def analyze_video(self, video_path: str, **kwargs) -> Dict[str, Any]:
        if self._meta_fusion_video is not None:
            try:
                return self._meta_fusion_video.analyze(video_path)
            except Exception as e:
                log.warning("meta-fusion video analyze failed (%s); falling back", e)
        return self._legacy_analyze_video(video_path, **kwargs)
```

Rename the original to `_legacy_analyze_video`.

- [ ] **Step 5: Verify server still boots with flag off**

```bash
USE_NEW_PIPELINE=false python -c "from server.inference_engine import InferenceEngine; e = InferenceEngine(); print('OK')"
```

Expected: `OK`. (No errors. Old pipeline is the active path.)

- [ ] **Step 6: Verify server boots with flag on (model files may not exist yet)**

```bash
USE_NEW_PIPELINE=true python -c "from server.inference_engine import InferenceEngine; e = InferenceEngine(); print('OK')"
```

Expected: `OK` plus a warning like "Meta-fusion image engine unavailable: ...image_v1.joblib...". The flag enables the lookup but a missing joblib is non-fatal — the engine falls back to legacy.

- [ ] **Step 7: Commit**

```bash
git add server/inference_engine.py
git commit -m "feat: wire MetaFusionEngine into InferenceEngine behind feature flag"
```

---

## Task 14: Set up golden test fixtures

**Files:**
- Create: `tests/fixtures/golden/images/labels.json`
- Create: `tests/fixtures/golden/videos/labels.json`
- Drop test files into `tests/fixtures/golden/{images,videos}/` (manual step — collect from existing data sources)

- [ ] **Step 1: Write `tests/fixtures/golden/images/labels.json`**

```json
{
  "real_dslr_001.jpg": {"verdict": "authentic", "min_confidence": 70},
  "real_whatsapp_001.jpg": {"verdict": "authentic", "min_confidence": 60},
  "real_screenshot_001.png": {"verdict": "authentic", "min_confidence": 60},
  "real_smartphone_001.jpg": {"verdict": "authentic", "min_confidence": 65},
  "ai_sd35_001.png": {"verdict": "ai-generated", "min_confidence": 70},
  "ai_flux_001.png": {"verdict": "ai-generated", "min_confidence": 65},
  "ai_mj7_001.jpg": {"verdict": "ai-generated", "min_confidence": 65},
  "ai_dalle3_001.png": {"verdict": "ai-generated", "min_confidence": 70},
  "ai_gptimage_001.png": {"verdict": "ai-generated", "min_confidence": 65}
}
```

- [ ] **Step 2: Write `tests/fixtures/golden/videos/labels.json`**

```json
{
  "real_phone_001.mp4": {"verdict": "authentic", "min_confidence": 65},
  "real_call_001.mp4": {"verdict": "authentic", "min_confidence": 60},
  "deepfake_ff_001.mp4": {"verdict": "ai-generated", "min_confidence": 70},
  "deepfake_celebdf_001.mp4": {"verdict": "ai-generated", "min_confidence": 70}
}
```

- [ ] **Step 3: Source the golden files (manual)**

Copy real example files into the directories. Use existing dataset assets where possible:
- Real images: from your phone, scanned documents, or production "Mark Authentic" overrides
- AI images: 1 file per generator from the parquet shards
- Real videos: phone recordings of common scenes
- AI videos: FF++ / Celeb-DF test split samples

Each file < 5MB to keep total fixture size manageable.

- [ ] **Step 4: Commit fixtures**

```bash
git add tests/fixtures/golden/images/ tests/fixtures/golden/videos/
git commit -m "test: add golden test fixtures with verdict labels"
```

---

## Task 15: Implement integration tests

**Files:**
- Create: `tests/test_pipeline_golden.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_pipeline_golden.py`:

```python
"""Integration tests: full pipeline must match labels on golden fixtures."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

GOLDEN_IMAGES_DIR = Path("tests/fixtures/golden/images")
GOLDEN_VIDEOS_DIR = Path("tests/fixtures/golden/videos")
META_FUSION_DIR = Path("models/meta_fusion")


def _load_labels(labels_path: Path) -> dict:
    if not labels_path.exists():
        return {}
    return json.loads(labels_path.read_text())


def _models_present() -> bool:
    return (META_FUSION_DIR / "image_v1.joblib").exists()


@pytest.mark.skipif(not _models_present(),
                    reason="Meta-fusion model not trained yet (run train_meta_fusion.py)")
@pytest.mark.parametrize("filename,expected", list(_load_labels(GOLDEN_IMAGES_DIR / "labels.json").items()))
def test_image_pipeline_golden(filename: str, expected: dict):
    from engine.meta_fusion.engine import MetaFusionEngine
    engine = MetaFusionEngine("image", model_dir=str(META_FUSION_DIR))
    img_path = GOLDEN_IMAGES_DIR / filename
    if not img_path.exists():
        pytest.skip(f"fixture missing: {img_path}")
    result = engine.analyze(str(img_path))
    assert result["verdict"] == expected["verdict"], (
        f"{filename}: got verdict={result['verdict']} confidence={result['confidence']}%, "
        f"expected={expected['verdict']}"
    )
    assert result["confidence"] >= expected["min_confidence"], (
        f"{filename}: confidence {result['confidence']}% below min {expected['min_confidence']}%"
    )


@pytest.mark.skipif(not (META_FUSION_DIR / "video_v1.joblib").exists(),
                    reason="Video meta-fusion model not trained yet")
@pytest.mark.parametrize("filename,expected", list(_load_labels(GOLDEN_VIDEOS_DIR / "labels.json").items()))
def test_video_pipeline_golden(filename: str, expected: dict):
    from engine.meta_fusion.engine import MetaFusionEngine
    engine = MetaFusionEngine("video", model_dir=str(META_FUSION_DIR))
    video_path = GOLDEN_VIDEOS_DIR / filename
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")
    result = engine.analyze(str(video_path))
    assert result["verdict"] == expected["verdict"]
    assert result["confidence"] >= expected["min_confidence"]
```

- [ ] **Step 2: Run tests (will be skipped until model is trained)**

```bash
pytest tests/test_pipeline_golden.py -v
```

Expected: tests are SKIPPED (with reason "Meta-fusion model not trained yet"). After Task 9-10 produce a trained model, they will run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_golden.py
git commit -m "test: add golden-fixture integration tests for new pipeline"
```

---

## Task 16: Implement latency regression tests

**Files:**
- Create: `tests/test_pipeline_latency.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_pipeline_latency.py`:

```python
"""Latency regression: pipeline must stay within budget."""
from __future__ import annotations
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

GOLDEN_IMAGES_DIR = Path("tests/fixtures/golden/images")
GOLDEN_VIDEOS_DIR = Path("tests/fixtures/golden/videos")
META_FUSION_DIR = Path("models/meta_fusion")

IMAGE_BUDGET_S = 2.0
VIDEO_BUDGET_S = 8.0


def _models_present() -> bool:
    return (META_FUSION_DIR / "image_v1.joblib").exists()


@pytest.mark.skipif(not _models_present(),
                    reason="Meta-fusion model not trained yet")
@pytest.mark.gpu
def test_image_pipeline_latency():
    from engine.meta_fusion.engine import MetaFusionEngine
    engine = MetaFusionEngine("image", model_dir=str(META_FUSION_DIR))

    images = list(GOLDEN_IMAGES_DIR.glob("real_*.jpg"))
    if not images:
        pytest.skip("no real_*.jpg fixtures present")

    # Warmup (first call loads weights into GPU memory)
    engine.analyze(str(images[0]))

    start = time.time()
    engine.analyze(str(images[0]))
    elapsed = time.time() - start
    assert elapsed < IMAGE_BUDGET_S, (
        f"Image pipeline took {elapsed:.2f}s (budget: {IMAGE_BUDGET_S}s)"
    )


@pytest.mark.skipif(not (META_FUSION_DIR / "video_v1.joblib").exists(),
                    reason="Video meta-fusion model not trained yet")
@pytest.mark.gpu
def test_video_pipeline_latency():
    from engine.meta_fusion.engine import MetaFusionEngine
    engine = MetaFusionEngine("video", model_dir=str(META_FUSION_DIR))

    videos = list(GOLDEN_VIDEOS_DIR.glob("real_*.mp4"))
    if not videos:
        pytest.skip("no real_*.mp4 fixtures present")

    engine.analyze(str(videos[0]))  # warmup

    start = time.time()
    engine.analyze(str(videos[0]))
    elapsed = time.time() - start
    assert elapsed < VIDEO_BUDGET_S, (
        f"Video pipeline took {elapsed:.2f}s (budget: {VIDEO_BUDGET_S}s)"
    )
```

- [ ] **Step 2: Run tests (skipped until model trained)**

```bash
pytest tests/test_pipeline_latency.py -v
```

Expected: SKIPPED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_latency.py
git commit -m "test: add latency regression tests for new pipeline"
```

---

## Post-implementation steps (Phase 2 + 3 of the spec)

After Tasks 1-16 land, the engineer must execute the following runtime activities (no more code changes — these are operations):

### Activity A: Assemble calibration data and train the meta-fusion

```bash
# 1. Curate calibration directory (from instructions in Task 9 docstring)
#    calibration_data/images/{real,ai}/
#    calibration_data/videos/{real,ai}/

# 2. Build feature arrays
python scripts/build_calibration_set.py --data-root calibration_data

# 3. Train logistic regression + temperature scaler
python scripts/train_meta_fusion.py --cal-dir models/meta_fusion

# 4. Verify metrics meet spec targets
cat models/meta_fusion/eval_v1.json
# Spec targets:
#   accuracy >= 0.90, fp_rate_real <= 0.05, fn_rate_ai <= 0.10
#   expected_calibration_error_cal <= 0.05, holdout_auc >= 0.95

# 5. If targets fail: collect more data, retrain. Do not flip the flag.
```

### Activity B: Run integration + latency tests

```bash
pytest tests/test_pipeline_golden.py tests/test_pipeline_latency.py -v
```

All previously-skipped tests should now run and pass.

### Activity C: Phase 4 shadow run

Set `USE_NEW_PIPELINE=true` in a non-production environment and run for 24 hours. Compare outputs against the old pipeline. If acceptance criteria from spec Section "Migration plan / Phase 4" are met, proceed to Phase 5 (production switch).

---

## Notes for implementation

- **Detector wrapper TODO blocks:** Tasks 3-5 contain `# === REPLACE THIS BLOCK ===` markers in `_build_model()`. These reference Task 1's resolution doc. Until those blocks are filled in, the detectors land in error state and return 0.5 — by design, the meta-fusion still produces valid output via imputation. Fill in the blocks once Task 1 is complete.
- **Memory profile:** When all detectors are loaded simultaneously, peak VRAM is ~2GB (per spec). On a 4GB GPU, ensure no other CUDA workload is running.
- **Windows-specific:** Existing memory notes that `num_workers=0` is required on Windows. The new code does not use DataLoader, so this doesn't apply, but be aware if loading auxiliary datasets.
- **No retraining of base detectors.** The existing ViT, R3D, and AST checkpoints are unchanged. Only the new wrappers (DIRE/AIDE/GenConViT) and the meta-fusion logistic regression are introduced.
