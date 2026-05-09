# Image + Video Detection — Redevelopment from Scratch

**Started:** 2026-05-09
**Goal:** Rebuild image/video deepfake detection without retraining base models. Replace hand-tuned 20-module pipeline with SOTA pretrained ensemble + learned meta-fusion. Approach 1 picked from brainstorm.

**Status:** Brainstorming in progress. Sections approved by user as we go.

---

## Section 1 — Architecture overview ✅ APPROVED

**Image pipeline (replaces current 20-module stack):**

```
Input image
   ↓
[Compression Detector]  ← keep (just flags WhatsApp/IG/etc., feeds meta-fusion)
   ↓
┌──────────────────────────────────────────────┐
│  Run 3 detectors in parallel:                │
│   • Existing ViT-B/16 (your trained model)   │
│   • DIRE (diffusion reconstruction error)    │
│   • AIDE (DCT + semantic, NeurIPS 2024)      │
└──────────────────────────────────────────────┘
   ↓
[Meta-Fusion: Logistic Regression]
  Inputs: 3 detector scores + compression flag + content type
   ↓
[Temperature scaling for calibrated confidence]
   ↓
Single threshold → verdict (authentic / ai-generated / inconclusive)
```

**Video pipeline (replaces current 16-module stack):**

```
Input video
   ↓
Sample 16 frames + extract audio
   ↓
┌──────────────────────────────────────────────┐
│  Run 2 detectors:                            │
│   • Existing Temporal R3D-18 (AUC 0.98)      │
│   • GenConViT (2024, single-stream)          │
└──────────────────────────────────────────────┘
   ↓
Per-frame image pipeline above (avg of 16 frames)
   ↓
[Meta-Fusion: Logistic Regression]
  Inputs: 2 video scores + avg-image score + audio score (if voice present)
   ↓
Calibrated probability → verdict
```

**Key principle:** All "smarts" live in **one place** — the logistic regression. No more compression dampening, threshold zones, gray-zone fallbacks scattered across `detector.py`. Detectors output raw scores. The meta-model decides.

---

## Section 2 — Image ensemble specifics ✅ APPROVED

**Three detectors, each with a clear job:**

| Detector | Role | Inference | Memory | Module file |
|---|---|---|---|---|
| **Existing ViT-B/16** (your trained model) | Specialist — fine-tuned on YOUR training distribution (110-120 parquet shards) | ~80ms | 344MB | `vit_detector.py` (keep) |
| **DIRE** (Diffusion Reconstruction Error, ICCV 2023) | Diffusion specialist — runs image through pretrained diffusion model, measures noise→denoise reconstruction error. AI images reconstruct cleanly; real images don't | ~500-1000ms | ~500MB | `dire_detector.py` (new) |
| **AIDE** (NeurIPS 2024) | Generalist — combines DCT low-level + CLIP high-level features. Strong cross-generator generalization on novel models | ~200ms | ~200MB | `aide_detector.py` (new) |

**Why these three:**
- ViT covers your training distribution (SD 3.5, Flux, SDXL, MJ 6/7, GPT-image, DALL-E, Imagen)
- DIRE catches diffusion-specific physics that pixel-level models miss
- AIDE provides cross-generator generalization for novel models you haven't seen

**Total budget:** ~1s parallel inference, ~1GB GPU memory. Fits your 4GB VRAM with headroom.

**Pretrained weights sourcing:**
- DIRE: official GitHub `ZhendongWang6/DIRE` weights, ~500MB. Mirror on HuggingFace if needed.
- AIDE: official GitHub `shilinyan99/AIDE` weights + paper checkpoints.
- Verify both are still accessible during the implementation phase. If either is gone, fallback to closest equivalent (e.g., **NPR** for AIDE, **DRCT** for DIRE).

**TTA stays on ViT** (already implemented, 5-view average). Don't add TTA to DIRE (too slow) or AIDE (its DCT branch is already augmentation-equivalent).

**What happens to the 17 other image_forensics modules:**
- **Kept as features** for meta-fusion: `compression_detector`, `ela_analysis`, `noise_analysis`, `frequency_analysis`, `pixel_statistics` — outputs become inputs to logistic regression. Logistic regression learns their weights (or zero-weights them).
- **Deleted (or archived):** `gan_fingerprint`, `inpainting_detector`, `spectral_analyzer`, `color_forensics`, `texture_forensics`, `reconstruction_detector`, `upsampling_detector`, `efficientnet_detector`, `clip_detector`, `content_classifier`. Most are duplicative of what AIDE/DIRE do better, or were never validated.
- **Verdict logic in `detector.py` deleted entirely:** `_compute_verdict()`, `_compute_verdict_v2()`, all threshold zones, gray-zone bias, compression dampening. Replaced by meta-fusion call.

**Net code reduction:** ~3000 lines deleted, ~600 lines added (3 new detector modules + 1 fusion module).

---

## Section 3 — Video ensemble specifics ✅ APPROVED

**Core compute tradeoff:** running the full image ensemble (3 detectors) on 16 frames = 48 inferences. Too slow. Solution: tier detectors by speed.

**Five signals fed into video meta-fusion:**

| # | Signal | What it does | Compute |
|---|---|---|---|
| 1 | **Per-frame ViT + AIDE (16 frames)** | Fast image detectors batched across all frames. Returns mean + max + std of frame scores | ~800ms (GPU batched) |
| 2 | **DIRE (3 keyframes)** | Slow diffusion detector — only on 3 evenly-spaced keyframes, not all 16. Catches diffusion-generated video frames | ~3s |
| 3 | **GenConViT** (whole clip) | Single-stream ViT+CNN purpose-built for deepfake video (2024). Catches face-swap/lip-sync artifacts that per-frame detectors miss | ~1s |
| 4 | **Existing Temporal R3D-18** (whole clip) | Your trained 3D CNN, AUC 0.98. Catches temporal motion inconsistencies | ~500ms |
| 5 | **Audio AST** (if voice present) | Existing audio spoof detector. Voice-faked videos always need synthetic audio | ~300ms |

**Total per-video budget:** ~6s sequential / ~4s with smart parallelism. Acceptable for forensic analysis (system was never real-time).

**Memory:** Image models (~1GB) + GenConViT (~500MB) + R3D (127MB) + AST (~400MB) = ~2GB. Fits 4GB VRAM.

**What the meta-fusion sees per video — 12 features:**
```
features = [
    frame_vit_mean, frame_vit_max, frame_vit_std,
    frame_aide_mean, frame_aide_max, frame_aide_std,
    dire_keyframe_mean,
    genconvit_score,
    r3d_score,
    audio_score (or 0.5 if no voice),
    compression_flag (from quality_analyzer),
    has_face (from face_detector),
]
```

The `_std` features matter: real videos have consistent frames; deepfakes flicker. Max + std capture this without needing a separate "consistency" module.

**Pretrained weights:**
- GenConViT: official GitHub `erprogs/GenConViT`, paper "Generalized Convolutional Vision Transformer for Deepfake Detection" (2024)
- Fallback if unavailable: **TALL** (CVPR 2024) or **AltFreezing** (CVPR 2023)

**`engine/video/` module changes:**

Keep: `temporal_r3d.py`, `quality_analyzer.py`, `two_stream.py` (refactored as orchestrator)
New: `genconvit_detector.py`
Deleted: `spatial_vit.py`, `temporal_x3d.py`, `rppg.py`, `rppg_analyzer.py`, `av_sync_analyzer.py`, `clip_temporal_drift.py`, `face_mesh_analyzer.py`, `forgery_localization.py`, `lighting_consistency.py`, `micro_expression_analyzer.py`, `pupil_analyzer.py`, `temporal_frequency.py`, `ai_video_detector.py`

**Net code reduction:** ~5000 lines deleted, ~400 lines added.

**Compute decision (locked in):** DIRE runs on 3 keyframes only. "Deep scan" mode (DIRE on all 16) deferred to future feature.

---

## Section 4 — Meta-fusion layer ✅ APPROVED

The single component that replaces ~3000 lines of hand-tuned thresholds. Gets ALL the smarts.

### Model choice: Logistic Regression (with reasoning)

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Logistic Regression** ✅ | Trains in <1s on 200 examples. Coefficients interpretable ("ViT got 2.3x weight, ELA got 0.1x"). Easy retrain. Cannot overfit small data | Linear — can't model "if compressed AND high ELA" interactions | **Pick this** |
| Gradient Boosting (XGBoost) | Captures interactions. ~2-3% more accurate | Harder to debug. Risk of overfitting on 200 examples | Reject — premature complexity |
| Neural meta-learner | Most expressive | Needs 10x more labeled data than we have | Reject |

### Calibration set construction

**Image set: ~200 examples (100 real + 100 AI). Video set: ~80 examples (40 real + 40 AI).**

| Bucket | Real sources | AI sources |
|---|---|---|
| Images | DSLR, smartphone HEIC, WhatsApp downloads, Instagram screenshots, scanned docs, video stills | SD 3.5, Flux, MJ 7, GPT-image-1, DALL-E 3, Imagen 4, Ideogram (parquet val split) |
| Videos | Phone recordings, news clips, vlogs, video calls | FF++ test, Celeb-DF test, recent Sora/Veo/Kling |

Sources in practice: existing parquet val split + FF++/Celeb-DF test splits + production failures from "Mark Authentic" overrides + ~30 hard cases (WhatsApp selfies, phone calls).

### Training procedure

```python
1. Run all detectors on calibration set → save (features, label) pairs
2. sklearn.LogisticRegression(C=1.0, class_weight='balanced').fit(X, y)
3. 5-fold cross-validation → realistic accuracy estimate
4. Save weights → models/meta_fusion/image_v1.joblib (~5KB file)
```

### Temperature scaling for confidence

```python
T = optimize_NLL(temperature_scale(logits / T), labels)
calibrated_prob = sigmoid(logit / T)
```

Fits in 10 seconds. Result: "85% confident" actually means right 85% of the time.

### Verdict thresholds (after calibration)

- `prob ≥ 0.70` → **ai-generated**
- `0.30 ≤ prob < 0.70` → **inconclusive** (shrunk zone)
- `prob < 0.30` → **authentic**

### Retraining cadence

When new failure found: add to calibration set → refit logistic regression → deploy new `.joblib` (`<1 minute, no GPU`). System improves continuously without retraining heavy detectors.

### Code locations

```
engine/meta_fusion/
├── __init__.py
├── feature_extractor.py    # Run all detectors, build feature vector
├── fusion_model.py         # Load .joblib, predict probability
├── temperature_scaler.py   # Apply T scaling
└── train_fusion.py         # One-shot training script

models/meta_fusion/
├── image_v1.joblib         # ~5KB
├── video_v1.joblib         # ~5KB
└── temperature.json        # {image_T: 1.4, video_T: 1.2}
```

### Failure modes

1. **Detector errors out** → NaN feature → impute with training-set mean. Log warning.
2. **No face in image** → `has_face = 0` already handled.
3. **Audio missing in video** → `audio_score = 0.5` (neutral).
4. **Temperature scaling collapses** → fall back T=1, log warning.

---

## Section 5 — Migration plan & server integration ✅ APPROVED

### Six-phase rollout

| Phase | Goal | Duration | Risk |
|---|---|---|---|
| **1** | Add new detectors **alongside** old (no behavior change yet) | 2 days | Low |
| **2** | Build calibration set, fit logistic regression + temperature | 2 days | Med (data quality) |
| **3** | Wire new pipeline behind feature flag `USE_NEW_PIPELINE` | 1 day | Low |
| **4** | A/B compare old vs new on calibration set + production traffic | 1 day | Low |
| **5** | Flip default to new pipeline, mark old as deprecated | 0.5 day | **High** (rollback path matters) |
| **6** | Delete old code after 1 week of stable production | 0.5 day | Low |

**Total:** ~7 days work, ~14 days calendar including A/B observation window.

### Phase 1 — Add alongside (new files only)

```
NEW FILES:
  engine/image_forensics/dire_detector.py
  engine/image_forensics/aide_detector.py
  engine/video/genconvit_detector.py
  engine/meta_fusion/__init__.py
  engine/meta_fusion/feature_extractor.py
  engine/meta_fusion/fusion_model.py
  engine/meta_fusion/temperature_scaler.py
  engine/meta_fusion/train_fusion.py
  scripts/build_calibration_set.py
  scripts/train_meta_fusion.py
  models/meta_fusion/    (empty, populated Phase 2)
```

Each new detector must run end-to-end on one test image before proceeding. If a pretrained weight is unavailable, swap to fallback (NPR, DRCT, TALL).

### Phase 3 — Feature flag

```python
# server/config.py
USE_NEW_PIPELINE = bool(os.getenv("USE_NEW_PIPELINE", "false").lower() == "true")

# server/inference_engine.py
class InferenceEngine:
    def __init__(self):
        self.image_detector = ImageForensicsDetector()  # old
        if config.USE_NEW_PIPELINE:
            self.meta_fusion_image = MetaFusionEngine("image")
            self.meta_fusion_video = MetaFusionEngine("video")

    def analyze_image(self, path):
        if config.USE_NEW_PIPELINE:
            return self.meta_fusion_image.analyze(path)
        return self.image_detector.analyze(path)
```

**Critical:** API response shape stays identical. Frontend doesn't change.

### Phase 4 — A/B acceptance criteria (flip-the-switch gate)

- New pipeline accuracy ≥ old + 5 percentage points
- New pipeline FP rate on real WhatsApp/smartphone subset ≤ 5%
- New pipeline FN rate on AI subset ≤ 10%
- No production crashes during 24h shadow run

If criteria fail: stop, diagnose, don't ship.

### Phase 5 — Rollback path

```bash
USE_NEW_PIPELINE=false  # in .env, no code change needed
# Restart server, old pipeline back online
```

Old code stays in tree for 1 week post-switch.

### Server endpoint changes

**No new endpoints. No changed schemas.**

| Endpoint | Currently | After redesign |
|---|---|---|
| `POST /api/analyze/image` | `ImageForensicsDetector.analyze()` | `MetaFusionEngine("image").analyze()` |
| `POST /api/analyze/video` | Old video pipeline | `MetaFusionEngine("video").analyze()` |
| `POST /api/analyze/media` | Auto-detect | Same, internal routing only |
| `WS /ws/live` | Call protection | Unchanged |
| `POST /api/analyze/multimodal` | Cross-attention fusion | Unchanged |

"Mark Authentic" override stays — independent of which detection pipeline ran.

### Configuration additions

```python
# server/config.py
USE_NEW_PIPELINE = False
META_FUSION_DIR = "models/meta_fusion"
DIRE_WEIGHTS_PATH = "models/image_forensics/dire/weights.pt"
AIDE_WEIGHTS_PATH = "models/image_forensics/aide/weights.pt"
GENCONVIT_WEIGHTS_PATH = "models/video/genconvit/weights.pt"
```

### Phase 6 — Final cleanup

- 12 image_forensics modules deleted
- 13 video modules deleted
- `detector.py`: 1729 → ~400 lines (orchestrator + back-compat shim)
- **~8000 lines net deletion**

### What stays unchanged

- Audio pipeline (`engine/audio/`)
- Text pipeline (`engine/text/`)
- Multimodal fusion network (`engine/fusion/cross_attention.py`)
- Frontend (same API contract)
- Database schema, auth, rate limiting, case management

---

## Section 6 — Testing & validation plan ✅ APPROVED

### Three validation layers

| Layer | Purpose | Frequency | Owner |
|---|---|---|---|
| **Unit tests** | Each detector loads weights and produces a score in [0, 1] | Every commit (CI) | Automated |
| **Integration tests** | Full pipeline produces valid verdict on golden test cases | Every commit (CI) | Automated |
| **Calibration eval** | Measure accuracy + calibration error on held-out set | Before each meta-fusion retrain | Manual via script |

### Golden test set (committed to repo)

```
tests/fixtures/golden/
├── images/
│   ├── real_dslr_001.jpg          (real, DSLR)
│   ├── real_whatsapp_001.jpg      (real, WhatsApp compressed)
│   ├── real_screenshot_001.png    (real, phone screenshot)
│   ├── real_smartphone_001.heic   (real, iPhone)
│   ├── ai_sd35_001.png            (Stable Diffusion 3.5)
│   ├── ai_flux_001.png            (Flux)
│   ├── ai_mj7_001.jpg             (Midjourney 7)
│   ├── ai_dalle3_001.png          (DALL-E 3)
│   ├── ai_gptimage_001.png        (GPT-image-1)
│   └── labels.json
└── videos/
    ├── real_phone_001.mp4
    ├── real_call_001.mp4
    ├── deepfake_ff_001.mp4        (FaceForensics++)
    ├── deepfake_celebdf_001.mp4   (Celeb-DF)
    └── labels.json
```

Size: ~10 images + ~4 videos. Small enough to commit (<50MB). Large enough to catch regressions.

### Unit tests (`tests/test_new_detectors.py`)

- Each new detector loads weights, predicts in [0, 1]
- Meta-fusion returns valid verdict + confidence
- Detector-failure imputation: NaN replaced with training-set mean, pipeline doesn't crash

### Integration tests (`tests/test_pipeline_golden.py`)

For each golden file, assert verdict matches label. **Acceptance: ALL golden cases pass.** If one breaks, block merge.

### Calibration evaluation (`scripts/eval_meta_fusion.py`)

After each meta-fusion retrain, run on held-out 20% subset (never used for training):

```python
metrics = {
    "accuracy": ...,                    # target ≥ 90%
    "fp_rate_real": ...,                # target ≤ 5%
    "fn_rate_ai": ...,                  # target ≤ 10%
    "expected_calibration_error": ...,  # target ≤ 5%
    "auc": ...,                         # target ≥ 0.95
}
```

Save to `models/meta_fusion/eval_v1.json`. Diff against previous version to spot regressions.

### Production validation (Phase 4 shadow run)

24h shadow run: both pipelines run on every request, old returns user-facing verdict, new logged silently. After 24h, compare:

```sql
SELECT old_verdict, new_verdict, COUNT(*)
FROM scan_log
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY old_verdict, new_verdict;
```

Spot-check cases where new says "authentic" but old said "ai-generated" — these are the false positives we set out to fix.

### Latency regression check

```python
def test_image_pipeline_latency():
    elapsed = time_pipeline(image)
    assert elapsed < 2.0  # budget: 2s

def test_video_pipeline_latency():
    elapsed = time_pipeline(video)
    assert elapsed < 8.0  # budget: 8s
```

### Failure-case feedback loop (durability mechanism)

```
User clicks "Mark Authentic" override
    ↓
Backend logs (image_path, model_verdict, model_confidence, user_override)
    ↓
Weekly review of these cases
    ↓
Add to calibration set → refit logistic regression (1 min) → eval → deploy
    ↓
System learns from real failures, no model retraining needed
```

This is the closed loop that makes the redesign durable.

### CI additions

```yaml
- name: Test new detectors
  run: pytest tests/test_new_detectors.py -v
- name: Test pipeline on golden cases
  run: pytest tests/test_pipeline_golden.py -v
- name: Latency regression check
  run: pytest tests/test_pipeline_latency.py -v
```

---

## Status

All 6 sections approved by user (2026-05-09).

Next steps:
1. Write formal spec to `docs/superpowers/specs/2026-05-09-image-video-redesign-design.md` (this is the canonical design doc going forward)
2. User reviews formal spec
3. Hand off to writing-plans skill for implementation plan
