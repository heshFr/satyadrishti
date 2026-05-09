# Image + Video Detection Redesign — Design Spec

**Date:** 2026-05-09
**Status:** Approved through brainstorm. Awaiting user spec review before implementation planning.
**Author:** Claude (with user direction)

## Problem statement

The Satya Drishti image + video deepfake detection pipeline has accumulated ~20 image forensic modules and ~16 video forensic modules over the past month, fused via hand-tuned weights. Despite each module working in isolation, the system produces **inaccurate verdicts in production** — both false positives (real WhatsApp selfies flagged as AI) and false negatives. Recent fixes have been point-patches (threshold tweaks, override flags) on top of a shaky calibration foundation.

Diagnosis from code + memory:
- The base ViT-B/16 model is solidly trained (~110-120 of 206 parquet shards, hundreds of thousands of AI/real images covering SD 3.5, Flux, SDXL, MJ 6/7, GPT-image-1, DALL-E 3, Imagen 4). The model itself is **not** the bottleneck.
- The fusion layer is **hand-tuned weights** (2.5/1.2/0.6/0.4/0.3/0.3) never optimized against ground truth.
- Verdict logic in `engine/image_forensics/detector.py` (1729 lines) is a sprawl of threshold zones, gray-zone fallbacks, compression dampening rules, and conditional overrides — accreted incrementally over the project lifetime.

## Goal

Replace the layered hand-tuned pipeline with:
1. A **smaller, stronger ensemble** of off-the-shelf SOTA pretrained detectors (no training of base models from scratch).
2. A **learned meta-fusion** layer (logistic regression) that replaces all hand-tuned weights.
3. **Temperature scaling** for honestly calibrated confidence.
4. A **durable feedback loop** so the system improves from production failures without retraining heavy models.

Target metrics (measured on held-out calibration set):
- Accuracy ≥ 90% overall
- FP rate on real WhatsApp/smartphone subset ≤ 5%
- FN rate on AI subset ≤ 10%
- Expected Calibration Error ≤ 5%
- AUC ≥ 0.95

## Non-goals

- Retraining the existing ViT-B/16, Spatial ViT v2, or Temporal R3D-18 from scratch
- Changing the audio pipeline (`engine/audio/`)
- Changing the text pipeline (`engine/text/`)
- Changing the multimodal cross-attention fusion (`engine/fusion/cross_attention.py`)
- Changing API contracts, frontend, database schema, auth, rate limiting, or case management

## Architecture

### Image pipeline

```
Input image
   ↓
Compression Detector (flags WhatsApp/IG/etc., feeds meta-fusion)
   ↓
3 detectors in parallel:
   • Existing ViT-B/16 (your trained model)
   • DIRE (diffusion reconstruction error)
   • AIDE (DCT + semantic, NeurIPS 2024)
   ↓
Meta-Fusion: Logistic Regression
   Inputs: 3 detector scores + compression flag + content type + statistical features
   ↓
Temperature scaling
   ↓
Single threshold → verdict (authentic / ai-generated / inconclusive)
```

### Video pipeline

```
Input video
   ↓
Sample 16 frames + extract audio
   ↓
5 signals:
   1. Per-frame ViT + AIDE on 16 frames (mean, max, std)
   2. DIRE on 3 keyframes (mean)
   3. GenConViT on whole clip
   4. Temporal R3D-18 on whole clip
   5. Audio AST score (or 0.5 if no voice)
   ↓
Meta-Fusion: Logistic Regression on 12 features
   ↓
Temperature scaling
   ↓
Single threshold → verdict
```

### Component table

| Component | Status | Inference | Memory | Module file |
|---|---|---|---|---|
| ViT-B/16 (existing trained) | Keep | ~80ms | 344MB | `engine/image_forensics/vit_detector.py` |
| DIRE (new) | Add | ~500-1000ms | ~500MB | `engine/image_forensics/dire_detector.py` |
| AIDE (new) | Add | ~200ms | ~200MB | `engine/image_forensics/aide_detector.py` |
| Temporal R3D-18 (existing) | Keep | ~500ms | 127MB | `engine/video/temporal_r3d.py` |
| GenConViT (new) | Add | ~1s | ~500MB | `engine/video/genconvit_detector.py` |
| Audio AST (existing) | Keep | ~300ms | ~400MB | `engine/audio/ast_spoof.py` |
| Meta-fusion (new) | Add | <1ms | ~5KB | `engine/meta_fusion/fusion_model.py` |

**Memory ceiling:**
- Image-only request: ~1GB GPU (ViT + DIRE + AIDE)
- Video request: ~2GB GPU (image models + GenConViT + R3D + AST)
- Both fit 4GB VRAM with headroom

**Latency budget:**
- Image: ~1s parallel
- Video: ~6s sequential / ~4s parallel

## Pretrained weight sourcing

| Model | Primary source | Fallback |
|---|---|---|
| DIRE | GitHub `ZhendongWang6/DIRE` | DRCT |
| AIDE | GitHub `shilinyan99/AIDE` | NPR |
| GenConViT | GitHub `erprogs/GenConViT` | TALL or AltFreezing |

Each weight must be verified accessible during Phase 1; if primary unavailable, swap to fallback before continuing.

## Meta-fusion layer

### Model: Logistic Regression

- `sklearn.linear_model.LogisticRegression(C=1.0, class_weight='balanced')`
- Trains in <1 second on 200 examples
- Coefficients are interpretable (we can see which detector matters)
- Cannot overfit small data
- Easy to retrain on new failures

Rejected alternatives:
- **Gradient Boosting (XGBoost):** ~2-3% more accurate but harder to debug, risk of overfitting on 200 examples
- **Neural meta-learner:** Needs 10x more labeled data than we have

### Image features (9 features)

Three primary detector scores plus statistical features from kept modules:

```
[
    vit_score,           # primary detector
    dire_score,          # primary detector
    aide_score,          # primary detector
    compression_flag,    # 0 = original, 1 = social-media-compressed
    content_type_code,   # 0 = photo, 1 = screenshot, 2 = document, 3 = other
    ela_score,           # statistical
    noise_score,         # statistical
    frequency_score,     # statistical
    pixel_stat_score,    # statistical
]
```

The logistic regression learns which features matter; statistical features are likely zero-weighted for compressed inputs but useful for high-quality originals.

### Video features (12 features)

```
[
    frame_vit_mean, frame_vit_max, frame_vit_std,
    frame_aide_mean, frame_aide_max, frame_aide_std,
    dire_keyframe_mean,
    genconvit_score,
    r3d_score,
    audio_score (or 0.5),
    compression_flag,
    has_face,
]
```

The `_std` features capture frame consistency (deepfakes flicker, real videos don't) without a separate consistency module.

### Calibration set

- **Image set:** ~200 examples (100 real + 100 AI)
- **Video set:** ~80 examples (40 real + 40 AI)

Sources:
- Validation slice from the ~110-120 parquet shards already downloaded (NOT all 206 — user only ever had ~half the dataset)
- FF++ / Celeb-DF test splits (labeled videos)
- Production failures from "Mark Authentic" overrides
- Manually curated ~30 hard cases (WhatsApp selfies, video calls)

If the parquet val slice is too small to fill the 100-AI quota, supplement with fresh AI samples from public sources (Civitai showcases, MJ public gallery, etc.) — implementation phase will confirm exact counts available.

### Temperature scaling

```python
T = optimize_NLL(temperature_scale(logits / T), labels)
calibrated_prob = sigmoid(logit / T)
```

T is one scalar fitted on validation set. Result: when system says "85% confident", it's right 85% of the time.

### Verdict thresholds (after calibration)

- `prob ≥ 0.70` → ai-generated
- `0.30 ≤ prob < 0.70` → inconclusive (small zone)
- `prob < 0.30` → authentic

### Failure modes

| Mode | Handling |
|---|---|
| Detector errors out (e.g., DIRE OOM) | NaN feature → impute with training-set mean. Log warning |
| No face in image | `has_face = 0` already handled |
| Audio missing in video | `audio_score = 0.5` (neutral) |
| Temperature scaling collapses (T → 0 or ∞) | Fall back to T = 1, log warning |

## Module changes

### `engine/image_forensics/`

**Keep as features (output → meta-fusion input):**
- `compression_detector`, `ela_analysis`, `noise_analysis`, `frequency_analysis`, `pixel_statistics`, `face_detector`, `preprocessing`

**Keep with current role:**
- `vit_detector` (the existing trained model)
- `metadata_checker` (provenance flag, separate from forensic verdict)

**New:**
- `dire_detector.py`
- `aide_detector.py`

**Deleted:**
- `gan_fingerprint.py`, `inpainting_detector.py`, `spectral_analyzer.py`, `color_forensics.py`, `texture_forensics.py`, `reconstruction_detector.py`, `upsampling_detector.py`, `efficientnet_detector.py`, `clip_detector.py`, `content_classifier.py`

**Reduced:**
- `detector.py`: 1729 → ~400 lines (orchestrator + back-compat shim, not verdict logic)

### `engine/video/`

**Keep:**
- `temporal_r3d.py`, `quality_analyzer.py`, `two_stream.py` (refactored as orchestrator)

**New:**
- `genconvit_detector.py`

**Deleted:**
- `spatial_vit.py`, `temporal_x3d.py`, `rppg.py`, `rppg_analyzer.py`, `av_sync_analyzer.py`, `clip_temporal_drift.py`, `face_mesh_analyzer.py`, `forgery_localization.py`, `lighting_consistency.py`, `micro_expression_analyzer.py`, `pupil_analyzer.py`, `temporal_frequency.py`, `ai_video_detector.py`

### `engine/meta_fusion/` (new module)

```
engine/meta_fusion/
├── __init__.py
├── feature_extractor.py    # Run all detectors, build feature vector
├── fusion_model.py         # Load .joblib, predict probability
├── temperature_scaler.py   # Apply T scaling
└── train_fusion.py         # One-shot training script
```

### `models/meta_fusion/` (new directory)

```
models/meta_fusion/
├── image_v1.joblib         # ~5KB
├── video_v1.joblib         # ~5KB
├── temperature.json        # {image_T: 1.4, video_T: 1.2}
└── eval_v1.json            # Held-out metrics
```

### `scripts/`

**New:**
- `build_calibration_set.py` — collect labeled data, run detectors, save (features, label) pairs
- `train_meta_fusion.py` — fit logistic regression + temperature, save weights
- `eval_meta_fusion.py` — evaluate on held-out set, save metrics

### Server changes

- `server/config.py` — add `USE_NEW_PIPELINE` flag, model paths
- `server/inference_engine.py` — add `MetaFusionEngine` class with feature flag dispatch
- All other server files unchanged

### Total code change

- ~8000 lines deleted
- ~1000 lines added
- Net reduction: ~7000 lines

## Migration plan

### Six phases

| Phase | Goal | Duration | Risk |
|---|---|---|---|
| 1 | Add new detectors alongside old (no behavior change) | 2 days | Low |
| 2 | Build calibration set, fit logistic regression + temperature | 2 days | Med (data quality) |
| 3 | Wire new pipeline behind feature flag `USE_NEW_PIPELINE` | 1 day | Low |
| 4 | A/B compare old vs new on calibration set + 24h production shadow | 1 day | Low |
| 5 | Flip default to new pipeline, mark old as deprecated | 0.5 day | High (rollback path matters) |
| 6 | Delete old code after 1 week of stable production | 0.5 day | Low |

**Total:** ~7 days work, ~14 days calendar.

### Phase 4 acceptance criteria (flip-the-switch gate)

- New pipeline accuracy ≥ old + 5 percentage points
- New pipeline FP rate on real WhatsApp/smartphone subset ≤ 5%
- New pipeline FN rate on AI subset ≤ 10%
- No production crashes during 24h shadow run

If criteria fail: stop, diagnose, do not ship.

### Rollback path

```bash
USE_NEW_PIPELINE=false  # in .env
# Restart server, old pipeline back online
```

Old code stays in tree for 1 week post-switch. Phase 6 deletes it after stable production confirms.

## Testing & validation

### Three layers

| Layer | Purpose | Frequency |
|---|---|---|
| Unit tests | Detector loads + scores in [0, 1] | Every commit (CI) |
| Integration tests | Pipeline produces valid verdict on golden cases | Every commit (CI) |
| Calibration eval | Accuracy + ECE on held-out 20% | Before each meta-fusion retrain |

### Golden test set (`tests/fixtures/golden/`)

~10 images + ~4 videos, hand-picked, label-locked, committed to git (<50MB).

Image bucket: real DSLR, real WhatsApp, real screenshot, real iPhone HEIC, AI SD 3.5, AI Flux, AI MJ 7, AI DALL-E 3, AI GPT-image-1.

Video bucket: real phone, real call, FF++ deepfake, Celeb-DF deepfake.

### Calibration eval metrics (target values)

```python
{
    "accuracy": ≥ 0.90,
    "fp_rate_real": ≤ 0.05,
    "fn_rate_ai": ≤ 0.10,
    "expected_calibration_error": ≤ 0.05,
    "auc": ≥ 0.95,
}
```

### Latency regression checks

- Image pipeline: <2s per request
- Video pipeline: <8s per request

### Production validation (Phase 4 shadow run)

24 hours of dual-pipeline execution. Old returns user-facing verdict; new logged silently. Compare verdicts after 24h. Spot-check disagreements.

### Failure-case feedback loop (durability)

```
User clicks "Mark Authentic" override
  → Backend logs (image_path, model_verdict, confidence, user_override)
  → Weekly review
  → Add to calibration set, refit logistic regression (<1 min)
  → Re-eval, deploy new .joblib if metrics improve
```

System improves continuously without retraining heavy models.

## Open questions for implementation phase

1. DIRE / AIDE / GenConViT pretrained weight URLs need verification before Phase 1.
2. Calibration set assembly: do we have enough "Mark Authentic" production failures logged, or do we need to manually curate the hard cases?
3. Server `inference_engine.py` is currently consistency-weighted ensemble; the swap point needs a careful audit.
4. The `WS /ws/live` real-time call protection path uses image stills from video frames — confirm whether to use new or old pipeline there (recommendation: new, since same `MetaFusionEngine("image")` works).

## Approval

User approved sections 1-6 of the brainstorm on 2026-05-09 (see `redevelopment_from_scratch.md` for the section-by-section history).
