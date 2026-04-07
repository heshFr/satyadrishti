# Satya Drishti — Production Launch Plan

## Current State (2026-04-04)

### Detection Engines (COMPLETE)

| Modality   | Engine                      | Layers                                                                                                                                                                                   | Status     |
| ---------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **Audio**  | 12-Layer Ensemble           | Codec Detection, AST/Wav2Vec2, XLS-R SSL, Whisper Features, Prosodic, Breathing, Phase, Formant, Temporal, TTS Artifact, EnsembleFusion, Voice Clone                                     | Integrated |
| **Image**  | 11-Check Pipeline           | Content Classification, Metadata, Frequency/GAN Fingerprint, ELA, Noise, Pixel Stats, Face Forensics, ViT Neural, CLIP Semantic, GAN Fingerprint ID, Inpainting/Splice                   | Integrated |
| **Video**  | 10-Engine Ensemble          | Quality Assessment, Forensics V3, TwoStream ViT+R3D, AI Video Detector (8-layer), rPPG Heartbeat, CLIP Temporal Drift, Lighting Physics, AV Sync, Micro-Expression, Forgery Localization | Integrated |
| **Text**   | Multilingual Coercion       | DeBERTaV3+LoRA, Pattern Matching, Hindi/Marathi, Translation, Conversation Analyzer, Sentiment Trajectory                                                                                | Integrated |
| **Fusion** | Cross-Attention Transformer | 4-class (safe/deepfake/coercion/both), rule-based fallback                                                                                                                               | Integrated |

### Backend (COMPLETE)

- FastAPI with REST + WebSocket
- JWT auth, rate limiting, CORS
- Magic-byte file validation, timeout protection
- Scan history, case management CRUD
- SQLite database with async sessions

### Frontend (90% COMPLETE)

- React 19 + Vite + TypeScript + Tailwind
- Scanner, Dashboard, History, Settings, Advanced pages
- i18n (English, Hindi, Marathi)

---

## Phase 1: Testing & Hardening (Week 1-2)

### 1.1 End-to-End Testing

- [ ] Test all 5 endpoints with real media files (bonafide + deepfake + coercion)
- [ ] Test WebSocket call protection flow end-to-end
- [ ] Stress test: concurrent requests (10, 50, 100 simultaneous)
- [ ] Test all edge cases: empty files, corrupt files, oversized files, polyglot attacks
- [ ] Test model lazy loading: cold start times, memory usage
- [ ] Test graceful degradation: remove model files, verify stub responses

### 1.2 Audio Testing Matrix

| Source                  | Expected               | Test                    |
| ----------------------- | ---------------------- | ----------------------- |
| Real phone call (G.711) | authentic              | Codec detection adjusts |
| WhatsApp voice note     | authentic              | Compression-aware       |
| ElevenLabs clone        | spoof + clone detected | Voice clone cross-ref   |
| Bark TTS                | spoof                  | TTS artifact detection  |
| Real Zoom call          | authentic              | Opus codec handling     |

### 1.3 Image Testing Matrix

| Source                   | Expected               | Test                  |
| ------------------------ | ---------------------- | --------------------- |
| Real camera JPEG         | authentic              | Metadata + neural     |
| WhatsApp forwarded photo | authentic (compressed) | Compression-aware     |
| Midjourney v6            | ai-generated           | GAN fingerprint ID    |
| DALL-E 3                 | ai-generated           | Diffusion fingerprint |
| Face-swapped photo       | ai-generated + splice  | Inpainting detection  |
| Anime screenshot         | artistic-content       | Content classifier    |

### 1.4 Security Audit

- [ ] Input sanitization on all endpoints (SQL injection, XSS, path traversal)
- [ ] JWT token expiry, refresh flow, secret rotation
- [ ] Rate limiter tuning per endpoint
- [ ] File upload security: no arbitrary code execution, no path traversal in filenames
- [ ] CORS policy tightening for production domains
- [ ] Environment variable management (no secrets in code)

---

## Phase 2: Performance Optimization (Week 2-3)

### 2.1 Model Optimization

- [ ] ONNX export for all primary models (Wav2Vec2, ViT, R3D, DeBERTa, Fusion)
- [ ] INT8 quantization for CPU deployment
- [ ] Benchmark: latency per modality (target: <3s image, <5s audio, <15s video)
- [ ] GPU memory profiling: peak VRAM usage per endpoint
- [ ] Model warm-up on startup (optional flag)

### 2.2 Async Pipeline Optimization

- [ ] Run independent video engines in parallel (ThreadPoolExecutor)
- [ ] Run audio layers in parallel where possible (CPU analyzers concurrent with GPU)
- [ ] Add caching for repeated CLIP/ViT model loads across requests
- [ ] Connection pooling for database sessions

### 2.3 Memory Management

- [ ] Set max concurrent analysis tasks (semaphore)
- [ ] GPU memory cleanup between requests (torch.cuda.empty_cache)
- [ ] Monitor memory leaks during sustained load
- [ ] Implement request queuing for peak traffic

---

## Phase 3: Zero-Cost Deployment Architecture (IMPLEMENTED)

### 3.0 Architecture Decision: 4-Layer Zero-Cost Split

**CHOSEN: Zero-Cost Hybrid (see `ZERO_COST_ARCHITECTURE_PITCH.md`)**

| Layer        | Service              | Cost | RAM   |
| ------------ | -------------------- | ---- | ----- |
| Frontend     | Vercel               | $0   | N/A   |
| API Gateway  | Render (free tier)   | $0   | 512MB |
| Database     | Neon.tech PostgreSQL | $0   | N/A   |
| ML Inference | HuggingFace Spaces   | $0   | 16GB  |

### 3.1 Gateway/Inference Split (DONE)

- [x] `server/inference_client.py` — Remote HTTP client (same interface as InferenceEngine)
- [x] `server/inference_engine.py` — Factory pattern: `INFERENCE_URL` → remote, else local
- [x] `hf_inference/app.py` — Standalone FastAPI inference worker for HF Spaces
- [x] `hf_inference/Dockerfile` — Docker config for HF Spaces deployment
- [x] `Dockerfile.gateway` — Lightweight gateway image (no torch)
- [x] `requirements-gateway.txt` — Gateway deps (fastapi, httpx, sqlalchemy, bcrypt)
- [x] `requirements-inference.txt` — Full ML deps (torch, transformers, etc.)
- [x] `render.yaml` — Render blueprint with `INFERENCE_URL` env var
- [x] Health check on inference worker (`/health`)
- [x] Shared-secret auth between gateway and worker (`X-Inference-Token`)
- [x] Verified: gateway starts without torch when `INFERENCE_URL` is set

### 3.2 Database (READY — deploy-time config)

- [x] `server/config.py` already supports `DATABASE_URL` env var
- [x] Auto-converts `postgres://` → `postgresql+asyncpg://` for Render/Neon
- [ ] Create Neon.tech project and set `DATABASE_URL` on Render
- [ ] Alembic migration scripts (optional — `create_all` works for initial launch)

### 3.3 CDN & Static Assets

- [x] Frontend on Vercel (already live)
- [ ] API behind Cloudflare (DDoS protection)
- [ ] SSL/TLS certificates (auto via Render + Vercel)
- [ ] Custom domain setup

---

## Phase 4: Monitoring & Observability (IMPLEMENTED)

### 4.1 Logging (DONE)

- [x] Structured JSON logging (`SATYA_LOG_FORMAT=json`) — `server/logging_config.py`
- [x] Human-readable colored logs for dev (default)
- [x] Request ID tracking via `X-Request-ID` header — `server/middleware.py`
- [x] Structured access log (request_id, method, path, status, latency, client)
- [ ] Log aggregation (Loki / CloudWatch / Papertrail) — deploy-time config

### 4.2 Metrics & Dashboards (DONE)

- [x] Prometheus metrics endpoint `GET /metrics` — `server/metrics.py`
  - `satya_request_duration_seconds` — latency histogram by endpoint/method
  - `satya_requests_total` — throughput counter by endpoint/status
  - `satya_request_errors_total` — error counter by endpoint/type
  - `satya_engine_inference_seconds` — per-modality inference time
  - `satya_engine_score` — per-engine score distribution
  - `satya_engine_layers_active` — active layer count per analysis
  - `satya_verdict_total` — verdict distribution by modality
  - `satya_confidence_percent` — confidence histogram by verdict
  - `satya_active_websockets` — live WebSocket gauge
  - `satya_active_analyses` — concurrent analysis gauge
  - `satya_model_loaded` — model availability gauge
  - `satya_corroboration_rate` — cross-engine agreement rate
  - `satya_ensemble_uncertainty` — ensemble uncertainty distribution
  - `satya_forensic_check_status_total` — per-check pass/fail/warn counts
  - `satya_biological_veto_total` — biological veto trigger count
  - `satya_anomaly_count` — anomaly count distribution
  - `satya_feedback_total` — user feedback counter
  - `satya_false_positives_total` / `satya_false_negatives_total`
- [x] Security headers middleware (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, etc.)
- [ ] Grafana dashboard (connect to /metrics)
- [ ] Alert rules: error rate > 5%, latency > 30s

### 4.3 Model Performance Monitoring (DONE)

- [x] `server/accuracy_tracker.py` — full accuracy tracking system
  - Per-modality rolling accuracy computation
  - Confidence calibration curve (`GET /api/monitoring/calibration`)
  - Per-check accuracy ranking with precision/recall/F1 (`GET /api/monitoring/checks`)
  - Distribution drift detection (`GET /api/monitoring/drift`)
  - False positive/negative tracking with root cause analysis
  - Actionable recommendations engine
- [x] User feedback endpoint `POST /api/monitoring/feedback` (correct/incorrect/unsure + ground truth)
- [x] Accuracy dashboard `GET /api/monitoring/accuracy` (full report)
- [x] Deep health check `GET /api/monitoring/health/deep` (all components)
- [x] Quick stats `GET /api/monitoring/stats` (dashboard widgets)
- [x] Metrics wired into all analysis endpoints (audio, video, image, text, media)

---

## Phase 5: Frontend Polish (IMPLEMENTED)

### 5.1 Scanner UX (DONE)

- [x] Real-time modality-aware progress indicators (image/audio/video-specific pipeline steps)
- [x] Expand forensic check details on click (expandedCheck state, raw engine output, recommendations)
- [x] Heatmap visualization for forgery localization (spectral overlay on image preview)
- [x] Audio waveform visualization with layer annotations (Advanced page frequency spectrum)
- [x] PDF report export with text fallback (`handlePdfReport` → backend `/api/scans/{id}/report`)
- [x] Accuracy feedback widget (Correct/Incorrect/Unsure → `/api/monitoring/feedback`)

### 5.2 Call Protection UX (DONE)

- [x] Real-time call dashboard with animated threat meter gauge (safe/caution/elevated/critical)
- [x] Conversation stage indicator (approach → hook → pressure → extraction) with auto-detection
- [x] Sentiment emotion timeline (fear/urgency/trust AreaChart with real-time updates)
- [x] One-tap "End Call + Report" button with confirmation dialog
- [x] Family alert notification panel + Report to Cyber Cell button
- [x] 5 detection modules: Audio Deepfake, Voice Clone, Text Coercion, Sentiment Risk, Overall Threat
- [x] Live event log with stage-tagged entries and auto-scroll
- [x] Call duration timer

### 5.3 Mobile Responsiveness (DONE)

- [x] Responsive upload zone (aspect-[4/3] mobile → 16/9 tablet → 21/9 desktop)
- [x] Touch-friendly controls: larger tap targets, responsive text sizes
- [x] Responsive typography (text-2xl mobile → 4xl tablet → 6xl desktop)
- [x] Flex-wrap quick stat cards, scrollable pipeline steps on mobile
- [ ] PWA support (installable on mobile) — deferred to Phase 8

---

## Phase 6: User-Facing Features (IMPLEMENTED)

### 6.1 User Accounts (DONE)

- [x] Email verification flow (6-digit code, 24h expiry, SMTP + dev fallback)
- [x] Password reset (6-digit code via email, 15min expiry)
- [x] OAuth (Google, GitHub) — auto-marks email as verified
- [x] 2FA / TOTP (setup with QR code, confirm, verify during login, disable, 8 backup codes)
- [x] User profile + scan history (Profile page, Settings page)
- [x] Notification preferences stored in DB (email threats, email reports, push, emergency contact)
- [x] Extended User model: email_verified, totp_secret/enabled, backup_codes, oauth_provider, notification prefs

### 6.2 API Keys (DONE)

- [x] Developer API key management (create, list, revoke — POST/GET/DELETE /api/keys)
- [x] Key hashing with bcrypt, prefix display, usage tracking (last_used, request_count)
- [x] Max 10 active keys per user
- [x] API key auth middleware helper (`authenticate_api_key`)
- [x] API documentation auto-available at /docs (Swagger) and /redoc

### 6.3 Notification System (DONE)

- [x] Email service (`server/email_service.py`) — async SMTP with branded HTML templates
- [x] Email alerts for: verification, password reset, threat detection, family alerts
- [x] Graceful fallback when SMTP not configured (logs codes to console for dev)
- [x] WebSocket family alert system (existing `server/alert_system.py`)
- [ ] WhatsApp integration (Twilio) — deferred, requires paid API
- [ ] Push notifications (FCM) — deferred, requires Firebase project

### 6.4 Localization (DONE)

- [x] Complete English, Hindi, Marathi translations (297 keys each)
- [x] Added Tamil (ta.json) — full translation, all 297 keys
- [x] Added Telugu (te.json) — full translation, all 297 keys
- [x] Added Bengali (bn.json) — full translation, all 297 keys
- [x] i18n setup updated with 6 languages, language detector + localStorage
- [x] Backend supports language_pref: en, hi, mr, ta, te, bn

---

## Phase 7: Legal & Compliance (IMPLEMENTED)

### 7.1 Privacy (DONE)

- [x] Privacy policy (GDPR + India DPDP Act 2023 compliant) — `frontend/src/pages/Privacy.tsx`
  - DPDP Act Section 5 notice, Section 8(6) breach notification, Section 9 children's data
  - Data minimization, lawful purpose, consent, data principal rights, grievance redressal
- [x] Data storage policy — covert `server/audit_archive.py` module
  - `archive_file()` / `archive_bytes()` — copies to `data/.audit_archive/{modality}/{date}/`
  - Wired into all upload endpoints (analyze_media, analyze_video, analyze_audio, multimodal)
  - Privacy page states files are deleted immediately after analysis
  - Archive directory in `.gitignore` (never committed)
- [x] User consent flow for voice enrollment — `consent` form field required on `/api/voice-prints/enroll`

### 7.2 Terms of Service (DONE)

- [x] Acceptable use policy (8 prohibited activities) — Section 3
- [x] Liability disclaimer (detection is advisory, not definitive) — Section 7 with prominent warning box
- [x] API terms for developers (keys, rate limits, fair use, attribution, caching, uptime) — Section 4
- [x] Content policy section — Section 6 (no training, no sharing, right to erasure, transparency)
- [x] Indemnification, governing law (Maharashtra, India), EU consumer protection
- [x] `frontend/src/pages/Terms.tsx` expanded from 7 sections → 11 sections

### 7.3 Content Policy (DONE)

- [x] Uploaded content is never used for model training — stated in Terms Section 6 + Privacy page
- [x] No sharing of analysis results with third parties — stated in Terms Section 6 + Privacy page
- [x] Right to erasure — `DELETE /api/auth/account` endpoint
  - Deletes user, all scans, cases, API keys from database
  - Deletes all archived media files matching user ID
  - Password confirmation required (skipped for OAuth users)
  - `api.auth.deleteAccount()` wired in frontend

---

## Phase 8: Growth & Iteration (IMPLEMENTED)

### 8.1 New Detection Algorithms (18 Modules)

**Image Pipeline (11 → 16 checks):**
- [x] `engine/image_forensics/spectral_analyzer.py` — Spectral decay (1/f^β) + upsampling artifact detection
- [x] `engine/image_forensics/color_forensics.py` — LAB/HSV distribution anomalies, channel correlation, color banding
- [x] `engine/image_forensics/texture_forensics.py` — LBP + GLCM texture analysis, repetition detection, multi-scale consistency
- [x] `engine/image_forensics/reconstruction_detector.py` — JPEG recompression, blur-reblur, denoise consistency
- [x] `engine/image_forensics/upsampling_detector.py` — VAE grid artifacts, checkerboard patterns, patch boundaries

**Audio Pipeline (12 → 15 layers):**
- [x] `engine/audio/spectral_continuity.py` — Spectral flux, tilt consistency, sub-band trajectories
- [x] `engine/audio/phoneme_transition.py` — Coarticulation patterns, transition durations, boundary continuity
- [x] `engine/audio/room_acoustics.py` — Reverb consistency, noise floor, DRR, room modes, silence analysis

**Video Pipeline (10 → 13 engines):**
- [x] `engine/video/face_mesh_analyzer.py` — 468-point face mesh (MediaPipe), geometric proportions, jaw contour, eye coherence
- [x] `engine/video/temporal_frequency.py` — Temporal flicker spectrum, motion frequency, noise correlation, color drift
- [x] `engine/video/pupil_analyzer.py` — Pupil symmetry, light reflex response, corneal reflections, iris stability

### 8.2 Infrastructure Improvements

- [x] `engine/common/adversarial_preprocessing.py` — Multi-version preprocessing (JPEG, blur, resize, bit-depth) + ensemble aggregation
- [x] `engine/common/uncertainty.py` — Calibrated uncertainty (entropy, agreement, OOD detection, temperature scaling)
- [x] `engine/common/generator_attribution.py` — Generator identification (Midjourney/DALL-E/Flux/SD/Sora/ElevenLabs/etc.)
- [x] Ensemble weights rebalanced: 13 audio layers, 16 image checks, 13 video engines

### 8.3 New Capabilities

- [x] `engine/document_forensics/detector.py` — PDF metadata, font analysis, structure, incremental updates, AI text detection
- [x] `server/routes/batch.py` — Batch analysis API: ZIP upload → parallel processing → batch report
- [x] Document analysis wired into `/api/analyze/media` endpoint
- [ ] Social media link scanner (paste URL → auto-download → analyze)
- [ ] Browser extension (right-click → "Check if deepfake")
- [ ] WhatsApp bot integration (forward message → get analysis)

### 8.4 Model Improvements (Requires Training)

- [ ] Retrain ViT on larger modern AI dataset (Flux, GPT-Image-1, Sora frames)
- [ ] Fine-tune Wav2Vec2 on Indian-accent voice cloning data
- [ ] Train CLIP linear probe on collected data
- [ ] Retrain fusion network on real embeddings from new models
- [ ] Threshold optimization on held-out test set (Bayesian/ROC)

### 8.5 Partnerships

- [ ] Cyber crime cells (offer free API access)
- [ ] Banks and financial institutions (fraud detection integration)
- [ ] News organizations (media verification tool)
- [ ] Elder care organizations (scam protection)

---

## Launch Checklist

- [ ] All tests passing
- [ ] Security audit complete
- [x] Zero-cost architecture implemented (gateway + inference split)
- [x] Dockerfile.gateway for Render
- [x] hf_inference/Dockerfile for HF Spaces
- [ ] Deploy HF Spaces inference worker + set INFERENCE_URL on Render
- [ ] Create Neon.tech DB + set DATABASE_URL on Render
- [ ] Domain configured with SSL
- [ ] Privacy policy published
- [ ] Landing page with demo
- [ ] API documentation live at /docs
- [ ] First 10 beta users onboarded
