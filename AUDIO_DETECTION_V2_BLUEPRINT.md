# Satya Drishti Audio Detection V2 — Elite-Scientist Blueprint

## Executive Summary

The current system relies on a **single AST model** (MattyB95/AST-VoxCelebSpoof) trained on 2021-era data. Modern AI voice generators (ElevenLabs, VALL-E, CosyVoice, F5-TTS, GPT-SoVITS, RVC, Maya1, Dia2, MeloTTS) produce voices that defeat single-paradigm detectors. This document specifies a **7-layer detection architecture** with **calibrated ensemble fusion** that leaves no detectable artifact unexploited.

---

## Self-Asked Follow-Up Questions (100 Questions → Answers)

### Q1: Why does AST alone fail against modern TTS?
**A:** AST learns spectrogram patterns from VoxCelebSpoof (2021 data). Modern codec-based TTS (VALL-E) and flow-matching TTS (F5-TTS) produce fundamentally different artifact signatures that AST never saw during training.

### Q2: What's the single most impactful upgrade?
**A:** Adding an SSL (Self-Supervised Learning) frontend. XLS-R 300M achieves 17.4% mean EER across diverse conditions per Spoof-SUPERB 2026 benchmark — the #1 ranked feature representation.

### Q3: Why XLS-R over WavLM or HuBERT?
**A:** XLS-R's multilingual pretraining (128 languages) provides the strongest robustness under acoustic degradation (9.4% degraded EER vs WavLM's 12.1%). Critical for Indian telephony with diverse languages.

### Q4: Why not just retrain AST on more data?
**A:** No single detection paradigm works against all synthesis types. XLS-R excels on autoregressive TTS (Dia2: 7.07% EER) but struggles with flow-matching (MeloTTS: 27.10%). Whisper features catch what SSL misses. You need both.

### Q5: How do prosodic features help when spectral methods already exist?
**A:** Prosodic features (F0, jitter, shimmer, HNR) achieve 93% accuracy standalone AND survive adversarial attacks that drop spectral methods to 0.7% accuracy. They're the only adversarially-robust feature set proven in literature.

### Q6: What biological signals do AI voices still fail to replicate?
**A:** (1) Natural breathing patterns — periodicity, depth variation; (2) Micro-pauses at phrase boundaries matching cognitive load; (3) Phonation onset/offset transients matching vocal fold physics; (4) Jitter/shimmer from vocal fold biomechanics.

### Q7: What about phase-domain artifacts?
**A:** "Many spoofing cues for LA attacks are embedded in the phase domain." Vocoders produce phase patterns incompatible with natural glottal excitation. Group delay and instantaneous frequency features catch what magnitude-only analysis misses.

### Q8: Why formant analysis at consonant-vowel boundaries?
**A:** Phoneme-level features achieve 98.39% AUC on ASVspoof 2021 DF vs AASIST's 85.44%. Deepfake models fail to accurately replicate formant transitions during CV boundaries because they don't model vocal tract physics.

### Q9: How do we handle the 4GB VRAM constraint?
**A:** (1) Use XLS-R 300M (not Large); (2) ONNX + INT8 quantization; (3) Sequential model loading with unloading; (4) Lightweight backends (logistic regression, SVM) on SSL features; (5) CPU-only prosodic/phase/breathing analyzers.

### Q10: What's the latency budget for real-time detection?
**A:** Current system uses 3-second chunks. Pindrop achieves 99% accuracy with 2-second windows. We keep 3-second chunks but add multi-resolution analysis (also analyze at 1s and 5s windows for different artifact scales).

### Q11: Why does codec augmentation matter so much?
**A:** SAFE Challenge 2025 showed TPR drops from 0.87 (clean) to 0.39 (laundered audio). Training without codec simulation means real phone calls will defeat our detector. Top ASVspoof 5 teams ALL used codec augmentation.

### Q12: Which codecs must we simulate in training?
**A:** (1) OPUS — WebRTC/VoIP standard; (2) AMR-WB — cellular voice; (3) G.711 — PSTN telephony; (4) G.722 — HD voice; (5) SILK — Skype/WeChat; (6) EVS — modern cellular.

### Q13: How do we handle transcoding (multi-hop codec)?
**A:** Chain multiple codec simulations during augmentation: e.g., OPUS→AMR-WB→G.711 to simulate conference bridge transcoding. Each hop destroys different artifact frequencies.

### Q14: What about noise augmentation?
**A:** Add real-world noise at SNR 5-30dB: (1) car/traffic; (2) crowd/restaurant; (3) office/HVAC; (4) wind/outdoor; (5) TV/radio in background. Use ESC-50 or MUSAN noise datasets.

### Q15: What's RawBoost and why use it?
**A:** RawBoost is a raw waveform augmentation technique widely adopted by top ASVspoof systems. It adds controlled signal degradation (linear/non-linear convolutive noise, impulsive signal-dependent noise) to improve generalization.

### Q16: How do we structure the ensemble fusion?
**A:** Calibrated logistic regression on all detector outputs. Each detector outputs a probability; logistic regression learns optimal weights. This gives inherently calibrated confidence scores and handles detector disagreement gracefully.

### Q17: Should we use hard voting or soft scoring?
**A:** Soft scoring (probability fusion) always outperforms hard voting. Challenge-winning systems universally use score-level fusion with learned weights.

### Q18: How do we handle when detectors disagree?
**A:** High disagreement = high uncertainty. Report calibrated uncertainty to the user. If SSL says "fake" but prosodic says "real", the system should flag "uncertain" rather than giving false confidence.

### Q19: What about temporal consistency across a call?
**A:** Track speaker embedding stability over time. Real speakers have consistent but naturally varying embeddings. AI voices show either too-stable embeddings (unnaturally consistent) or sudden shifts (model switching). Also track F0 drift and emotional consistency.

### Q20: How does the existing RawNet3 fit in?
**A:** RawNet3 code already exists in `engine/audio/rawnet3.py` but isn't used in production. Integrate it as Layer 3 alongside AST — it learns SincConv filterbanks that capture different frequency discriminators than AST's spectrogram patches.

### Q21: What training data should we use?
**A:** (1) ASVspoof 2021 LA+DF (existing); (2) MLAAD v9 — 140 TTS models, 51 languages, 678 hours; (3) In-the-Wild dataset; (4) ASVspoof 5 (2024); (5) Generate our own samples using open-source TTS (GPT-SoVITS, RVC, XTTS, F5-TTS).

### Q22: How do we handle Hindi/Marathi/regional language voices?
**A:** XLS-R is pretrained on 128 languages including Indian languages. MLAAD v9 covers 51 languages. For specific Indian language coverage, generate synthetic samples using Indic TTS systems and add to training data.

### Q23: What about continual learning when new TTS systems emerge?
**A:** Implement retrieval-based zero-day detection (1.25% EER on unseen TTS systems). Also use UAP (Universal Adversarial Perturbation) for continual learning without catastrophic forgetting — new TTS samples can be incorporated without retraining from scratch.

### Q24: How do we evaluate improvements?
**A:** Benchmark on: (1) ASVspoof 2021 LA (standard); (2) In-the-Wild (realistic); (3) Our own test set with ElevenLabs/RVC/GPT-SoVITS samples; (4) Telephony-codec test set. Report EER and calibrated accuracy.

### Q25: What's the architecture for the prosodic analyzer?
**A:** Extract: (1) F0 contour via CREPE or pYIN; (2) Jitter (cycle-to-cycle F0 variation); (3) Shimmer (cycle-to-cycle amplitude variation); (4) HNR (Harmonics-to-Noise Ratio); (5) Speech rate; (6) Pause statistics. Feed into lightweight MLP classifier.

### Q26: How do we detect breathing patterns?
**A:** (1) Segment audio into voiced/unvoiced/silence using VAD + voicing detector; (2) Analyze breathing intervals — distribution, periodicity, duration; (3) Check for absence of breath noise between phrases (AI voices often skip breathing); (4) Measure breath-to-speech ratio.

### Q27: What about micro-pause analysis?
**A:** Measure: (1) Duration distribution of inter-word silences; (2) Correlation between pause duration and syntactic boundaries; (3) Cognitive load indicators — longer pauses before complex phrases; (4) Compare against human baseline statistics.

### Q28: How do we implement phase analysis?
**A:** (1) Compute STFT with phase; (2) Extract group delay = -d(phase)/d(frequency); (3) Extract instantaneous frequency = d(phase)/d(time); (4) Look for phase discontinuities at frame boundaries; (5) Measure phase coherence across harmonic frequencies.

### Q29: What formant features do we extract?
**A:** (1) F1, F2, F3 formant frequencies and bandwidths via LPC; (2) Formant transition rates at CV boundaries; (3) Vowel space area; (4) Formant-to-F0 ratios; (5) Cross-formant consistency with vocal tract length estimation.

### Q30: How do we handle the speaker embedding temporal consistency check?
**A:** (1) Extract ECAPA-TDNN embedding every 3 seconds (already done); (2) Compute sliding-window cosine similarity between consecutive embeddings; (3) Measure variance of similarity scores; (4) Flag if variance is too low (robotic consistency) or too high (model instability).

### Q31: What about emotional consistency tracking?
**A:** (1) Use a speech emotion recognition model (e.g., emotion2vec or SpeechBrain's emotion classifier); (2) Track emotional state over time; (3) Flag sudden emotional shifts that don't match content; (4) Check timing alignment between emotional expression and semantic content.

### Q32: How do we integrate all 7 layers without increasing latency beyond 3 seconds?
**A:** Run all analyzers in parallel using asyncio. CPU-bound analyzers (prosodic, breathing, phase, formant) run on CPU. GPU-bound models (AST, RawNet3, XLS-R) run sequentially on GPU or batched. Ensemble fusion is instant (<1ms).

### Q33: What's the memory footprint of all models loaded simultaneously?
**A:** AST (~350MB), XLS-R 300M (~1.2GB), RawNet3 (~100MB), Whisper features (reuse existing transcriber), ECAPA-TDNN (~80MB). Total GPU: ~1.7GB. Remaining ~2.3GB for inference. Prosodic/breathing/phase/formant analyzers are CPU-only (<100MB RAM each).

### Q34: Should we use ONNX for all models?
**A:** Yes for production inference. ONNX + INT8 quantization reduces memory by ~4x and increases throughput. Export each model to ONNX as part of the deployment pipeline.

### Q35: How do we handle the Whisper encoder as a feature extractor?
**A:** Reuse the existing faster-whisper model (already loaded for transcription). Extract encoder hidden states as features. This adds zero memory overhead since the model is already loaded.

### Q36: What classifier goes on top of SSL features?
**A:** Logistic regression for calibrated probabilities, or a 2-layer MLP (256→64→2) for slightly better accuracy. Both are lightweight (<1MB). Logistic regression has the advantage of inherently calibrated probability outputs.

### Q37: How do we train the ensemble fusion layer?
**A:** Collect outputs from all 7 analyzers on a validation set. Train logistic regression on the concatenated scores. Use cross-validation to prevent overfitting. Update periodically as new TTS systems emerge.

### Q38: What's the uncertainty estimation strategy?
**A:** (1) Calibrated probability from logistic regression; (2) Disagreement score = variance across detector outputs; (3) Entropy of ensemble prediction; (4) If entropy > threshold, report "uncertain" instead of binary verdict.

### Q39: How do we handle false positives on elderly/accented speakers?
**A:** (1) Prosodic features are inherently less biased than spectral features; (2) XLS-R's multilingual training reduces accent bias; (3) Add demographic-aware calibration thresholds; (4) If speaker is enrolled and verified, apply softer thresholds.

### Q40: What about false positives from hearing aids or voice prostheses?
**A:** These produce mechanical artifacts that can trigger false alarms. Mitigation: (1) If speaker is enrolled with their assistive device, baseline includes those artifacts; (2) Add "medical device" profile option that adjusts thresholds.

### Q41: How do we handle call recording consent in India?
**A:** The system already only records when spoof is detected. Indian Telegraph Act allows recording for security purposes with one-party consent. The user (our user) is one party to the call.

### Q42: What if the attacker uses a real voice for the first 30 seconds then switches to AI?
**A:** The temporal consistency tracker catches this — sudden change in speaker embedding, prosodic characteristics, or breathing pattern. Also, per-segment analysis detects the switch point.

### Q43: What about voice conversion attacks where the attacker speaks in their own voice but it's converted to the target?
**A:** Voice conversion preserves some source speaker characteristics in formant ratios and prosodic patterns. The formant analyzer catches vocal tract inconsistencies. Speaker verification (ECAPA-TDNN) may still detect mismatch if the conversion is imperfect.

### Q44: How do we handle the case where a legitimate caller uses a voice changer (e.g., for privacy)?
**A:** This will trigger detection. The system should report "voice modification detected" separately from "AI-generated voice detected." The user can then make an informed decision.

### Q45: What about attacks using real recorded speech (replay attacks)?
**A:** Replay attacks introduce channel artifacts (recording device characteristics, ambient noise differences). The phase analyzer and channel consistency checker can detect these. Also, speaker verification with liveness detection helps.

### Q46: How do we handle degraded audio quality (bad network, low bitrate)?
**A:** (1) Codec augmentation in training; (2) Signal quality estimation before analysis; (3) If quality is too low, report "insufficient quality for reliable detection" rather than a false verdict; (4) Request the caller to improve connection if possible.

### Q47: What's the minimum audio duration for reliable detection?
**A:** 1 second for spectral analysis, 3 seconds for prosodic analysis, 5 seconds for temporal consistency, 10+ seconds for breathing pattern analysis. The system provides increasingly confident verdicts as more audio accumulates.

### Q48: How do we aggregate per-chunk verdicts into a call-level decision?
**A:** (1) Bayesian update: each chunk updates the posterior probability; (2) Exponential moving average of per-chunk scores; (3) Track the maximum threat score (peak detection); (4) Count the number of "spoof" chunks vs "bonafide" chunks.

### Q49: What if only one 3-second chunk triggers detection but all others are clean?
**A:** This could be: (1) A false positive — suppress if only 1/20+ chunks triggers; (2) A mid-call voice switch — check temporal consistency; (3) Noise corruption — check signal quality for that segment. Use Bayesian smoothing to avoid single-chunk false alarms.

### Q50: How do we handle simultaneous speakers (caller + background voices)?
**A:** (1) Speaker diarization to isolate the primary caller's voice; (2) Source separation if needed; (3) Analyze only the primary speaker's segments; (4) Flag background voices separately.

### Q51: What about music or hold tones triggering false positives?
**A:** (1) Music detection model to identify non-speech segments; (2) Skip analysis for music/hold segments; (3) The existing VAD partially handles this.

### Q52: How do we handle very short utterances (single word answers)?
**A:** Short utterances have limited features for prosodic/breathing analysis. Weight spectral features higher for short segments. Accumulate evidence across multiple short utterances.

### Q53: What's the training pipeline for the prosodic classifier?
**A:** (1) Extract F0/jitter/shimmer/HNR from ASVspoof + MLAAD samples; (2) Train gradient boosted trees (XGBoost) — lightweight, fast; (3) Cross-validate across datasets; (4) Output calibrated probabilities.

### Q54: How do we extract F0 reliably?
**A:** Use CREPE (CNN-based F0 estimation) for high accuracy, or pYIN for speed. CREPE gives frame-level F0 with confidence. Fall back to pYIN for real-time if CREPE is too slow.

### Q55: What's the jitter calculation?
**A:** Jitter = mean absolute difference between consecutive F0 periods / mean F0 period. Variants: local jitter (adjacent periods), RAP (3-period average), PPQ (5-period average).

### Q56: What's the shimmer calculation?
**A:** Shimmer = mean absolute difference between consecutive amplitude peaks / mean amplitude peak. Similar variants as jitter.

### Q57: What's HNR (Harmonics-to-Noise Ratio)?
**A:** Ratio of periodic (harmonic) energy to aperiodic (noise) energy in voiced speech. Measured in dB. Lower HNR indicates breathier or rougher voice quality. AI voices often have unnaturally high HNR.

### Q58: How do we detect breathing sounds?
**A:** (1) Use energy and spectral centroid to identify low-energy, high-frequency-spread segments; (2) Breathing has characteristic spectral shape (broad, noise-like); (3) Check for periodicity of breathing events; (4) Measure inhalation vs exhalation ratio.

### Q59: What's the expected breathing rate for speech?
**A:** Normal conversational speech: 12-20 breaths per minute. Breathing occurs at phrase boundaries, prosodic pauses, and hesitation points. AI voices either skip breathing entirely or insert it at wrong locations.

### Q60: How do we compute group delay?
**A:** Group delay = -d(unwrapped_phase) / d(omega). Computed from STFT. Natural speech has smooth group delay; synthetic speech may show discontinuities, especially at vocoder frame boundaries.

### Q61: What about instantaneous frequency?
**A:** IF = d(unwrapped_phase) / d(time). Complements group delay (frequency vs time derivatives of phase). Vocoders often produce IF discontinuities at synthesis frame boundaries.

### Q62: How do we extract formants?
**A:** (1) Pre-emphasis (6dB/octave); (2) Windowed LPC analysis (order = 2 + sr/1000); (3) Find LPC roots; (4) Convert to frequencies; (5) Select roots corresponding to F1 (200-900 Hz), F2 (600-2800 Hz), F3 (1800-3500 Hz).

### Q63: What's a CV boundary?
**A:** Consonant-to-Vowel transition boundary. E.g., in "ba", the transition from /b/ to /a/. These boundaries reveal vocal tract dynamics that AI models struggle to replicate because they involve rapid, coordinated articulatory movements.

### Q64: How do we detect CV boundaries?
**A:** (1) Use phoneme alignment from Whisper or a forced aligner; (2) Identify consonant-vowel transitions; (3) Extract formant trajectories through the transition; (4) Measure transition rate (Hz/ms).

### Q65: What about the vocal tract length estimation?
**A:** VTL ≈ c / (4 * F1) where c = speed of sound. For a real speaker, VTL should be consistent across all vowels in an utterance. AI voices may show VTL inconsistencies because different vowels are generated independently.

### Q66: How do we implement multi-resolution analysis?
**A:** Analyze each audio chunk at 3 window sizes: (1) 1-second windows for short-term spectral artifacts; (2) 3-second windows for prosodic features; (3) 5+ second accumulated windows for temporal consistency and breathing. Each resolution captures different artifact scales.

### Q67: What about the AASIST architecture — should we implement it?
**A:** AASIST (graph attention network) is architecturally complex and achieves 1.13% EER on ASVspoof 2019. However, XLS-R + lightweight classifier achieves comparable performance with simpler implementation. Use XLS-R + MLP as our SSL detector.

### Q68: Should we use AASIST3 with Kolmogorov-Arnold Networks?
**A:** KAN-enhanced AASIST3 halved minDCF on ASVspoof 2024. However, KAN implementation complexity is high. For our resource-constrained setup, XLS-R + logistic regression provides better ROI. Consider AASIST3 for a future upgrade.

### Q69: What about Mamba/SSM architectures?
**A:** RawBMamba (1.19% EER) and Fake-Mamba are designed for real-time detection. They're promising but require custom implementations. XLS-R + MLP is more pragmatic for now. Monitor Mamba developments for V3.

### Q70: How do we handle the training data class imbalance?
**A:** Current training uses 4:1 weighted cross-entropy (bonafide:spoof). For multi-detector training, also use: (1) Focal loss for hard examples; (2) Balanced sampling per batch; (3) Synthetic data generation to balance classes.

### Q71: What augmentations should we apply during SSL fine-tuning?
**A:** (1) RawBoost waveform augmentation; (2) Speed perturbation (0.9x-1.1x); (3) SpecAug (frequency + time masking); (4) Codec simulation chain; (5) Additive noise (SNR 5-30dB); (6) Reverberation (room impulse responses).

### Q72: How do we prevent catastrophic forgetting when adding new TTS data?
**A:** (1) UAP (Universal Adversarial Perturbation) preserves historical spoofing distributions; (2) CADE with knowledge distillation loss; (3) Elastic Weight Consolidation (EWC); (4) Replay buffer of old training samples.

### Q73: What's the deployment architecture?
**A:** All models run on the local machine (edge deployment). No cloud required. Models are loaded lazily (current architecture preserved). ONNX + INT8 for production. Full PyTorch for development/training.

### Q74: How do we update models over-the-air?
**A:** Version model weights with semantic versioning. New weights downloaded to `models/audio_v2/`. Hot-swap via configuration change. Old weights kept as fallback.

### Q75: What monitoring/telemetry should we add?
**A:** (1) Detection rate per modality; (2) False positive rate (user feedback); (3) Processing latency per analyzer; (4) Model confidence distribution; (5) Unknown TTS signature detection (anomaly detection).

### Q76: How do we handle adversarial attacks on our detector?
**A:** (1) Prosodic features are inherently robust (proven in literature); (2) Ensemble disagreement detects targeted attacks on single models; (3) Input signal quality validation; (4) Rate limiting from suspicious sources.

### Q77: What about adding watermark/C2PA detection?
**A:** OpenAI and Google embed C2PA watermarks. Add a C2PA/JUMBF scanner to the audio pipeline. If found, it's a strong signal of synthetic origin. But absence doesn't mean authentic — open-source TTS doesn't watermark.

### Q78: Should we use a retrieval-based approach for zero-day detection?
**A:** Yes. The "Frustratingly Easy" paper achieves 1.25% EER on unseen TTS via k-NN retrieval. Build a knowledge base of known TTS signatures (embeddings from training data). At inference, compare against the knowledge base.

### Q79: How big should the retrieval knowledge base be?
**A:** Store 10K-50K embeddings (one per training sample). Use FAISS or Annoy for efficient k-NN search (<10ms for 50K embeddings). Updates are append-only — new TTS samples just add to the index.

### Q80: What about the WebSocket pipeline — how do we integrate new analyzers?
**A:** Each analyzer is wrapped as an async function. The WebSocket handler calls all analyzers in parallel via asyncio.gather(). Results are merged into the existing threat escalation mechanism.

### Q81: How do we update the threat escalation weights for new analyzers?
**A:** Current: audio spoof +0.3, coercion +0.25, voice mismatch +0.2. Add: prosodic anomaly +0.15, breathing anomaly +0.15, phase anomaly +0.1, formant anomaly +0.1, temporal inconsistency +0.2. Normalize so max single-chunk escalation ≈ 0.4.

### Q82: What should the frontend display for the new analyzers?
**A:** Add analyzer-level detail cards in the call protection UI: (1) Voice Authenticity Score (ensemble); (2) Prosodic Analysis (jitter/shimmer/HNR visualized); (3) Breathing Pattern (timeline visualization); (4) Speaker Consistency (stability graph over time).

### Q83: How do we test the new system?
**A:** (1) Unit tests per analyzer; (2) Integration test with known fake/real samples; (3) A/B test against current system on historical data; (4) EER measurement on ASVspoof 2021 + In-the-Wild; (5) Manual testing with ElevenLabs/RVC generated samples.

### Q84: What's the minimum viable version for V2?
**A:** Phase 1 (implement now): Prosodic analyzer + Breathing detector + Phase analyzer + Formant analyzer + RawNet3 integration + Ensemble fusion. These require NO model downloads and NO GPU memory.

### Q85: What requires model downloads?
**A:** Phase 2: XLS-R 300M (~1.2GB download), fine-tuning data (MLAAD). Phase 3: Retrieval knowledge base construction.

### Q86: How long will training take on RTX 3050?
**A:** XLS-R fine-tuning with AMP: ~2-4 hours for 10 epochs on ASVspoof 2021 LA. Prosodic classifier: <30 minutes (XGBoost on extracted features).

### Q87: What Python packages do we need to add?
**A:** (1) `praat-parselmouth` — F0/jitter/shimmer/HNR extraction via Praat; (2) `librosa` (already installed) — for spectrogram/phase features; (3) `scipy` (already installed) — for LPC/formant extraction; (4) `scikit-learn` — for logistic regression ensemble; (5) `crepe` (optional) — for neural F0 estimation.

### Q88: Should we use Praat or librosa for prosodic features?
**A:** Praat via `parselmouth` is the gold standard for prosodic analysis. Jitter/shimmer/HNR implementations in Praat are scientifically validated. Use parselmouth for prosodic features, librosa for spectral/phase features.

### Q89: How do we handle the fact that Praat is designed for clean audio?
**A:** Pre-process: (1) High-pass filter at 60Hz to remove mains hum; (2) Normalize amplitude; (3) Use Praat's robust F0 estimation with appropriate floor/ceiling (75-600Hz). Quality check: reject segments where Praat can't detect F0 reliably.

### Q90: What about real-time performance of parselmouth?
**A:** Parselmouth processes a 3-second audio chunk in <50ms. This is well within our latency budget. All prosodic features can be extracted in a single pass.

### Q91: How do we structure the code?
**A:** Create `engine/audio/` modules:
- `prosodic_analyzer.py` — F0, jitter, shimmer, HNR, speech rate
- `breathing_detector.py` — Breath detection, periodicity, distribution
- `phase_analyzer.py` — Group delay, instantaneous frequency, phase coherence
- `formant_analyzer.py` — Formant extraction, CV transition analysis, VTL consistency
- `temporal_tracker.py` — Speaker embedding stability, prosodic drift, emotional consistency
- `ensemble_fusion.py` — Calibrated logistic regression on all detector outputs
- `codec_augmentor.py` — Codec simulation for training augmentation

### Q92: What's the output format of each analyzer?
**A:** Each analyzer returns a dict:
```python
{
    "score": float,         # 0.0 (bonafide) to 1.0 (spoof)
    "confidence": float,    # How confident the analyzer is
    "features": dict,       # Raw feature values for forensic display
    "anomalies": list,      # Specific anomalies detected
}
```

### Q93: How does the ensemble fusion combine these?
**A:** Concatenate all scores into a feature vector. Logistic regression outputs:
```python
{
    "verdict": "bonafide" | "spoof" | "uncertain",
    "probability": float,      # Calibrated P(spoof)
    "uncertainty": float,      # Entropy of ensemble
    "per_analyzer": dict,      # Individual analyzer scores
}
```

### Q94: What about explainability — why did the system flag this call?
**A:** Report which analyzers triggered and why: "Voice flagged because: (1) Abnormal jitter pattern (jitter=0.002, expected 0.01-0.05); (2) No breathing detected in 30-second segment; (3) Phase discontinuities at 0.02s intervals consistent with vocoder frame boundaries."

### Q95: How do we handle model versioning?
**A:** Each model has a version in its config. The ensemble fusion layer is versioned separately. When any model is updated, the ensemble must be recalibrated on the validation set.

### Q96: What about A/B testing new detectors?
**A:** Run old and new systems in parallel. Log both outputs. Compare EER on accumulated real-world data. Only switch production to new system when it demonstrably outperforms.

### Q97: How do we handle the Indian telephony-specific challenges?
**A:** (1) Indian cellular uses AMR-WB/EVS primarily — ensure codec augmentation covers these; (2) JIO/Airtel VoLTE uses EVS codec; (3) WhatsApp calls use OPUS; (4) Landline uses G.711; (5) Regional language TTS models are less sophisticated — potentially easier to detect.

### Q98: What's the expected improvement over current system?
**A:** Conservative estimate: (1) EER on clean audio: <2% (from ~5-10% with AST alone); (2) EER on telephony audio: <8% (from ~20-30% with AST alone); (3) False positive rate on real speech: <1% (from ~3-5%); (4) Detection of modern TTS (ElevenLabs, RVC): >95% (from ~60-70%).

### Q99: What are the remaining gaps after V2?
**A:** (1) Video deepfake lip-sync detection (need audio-visual correlation); (2) Adversarial attacks specifically targeting our ensemble (need adversarial training); (3) Real-time on-device mobile deployment (need TFLite/CoreML export); (4) Zero-day TTS from unknown architectures (need continual learning pipeline).

### Q100: What's the production deployment checklist?
**A:** (1) All 7 analyzers implemented and tested; (2) Ensemble fusion trained and calibrated; (3) ONNX exports for GPU models; (4) Latency benchmarked <3s per chunk; (5) Memory profiled <4GB total; (6) Unit tests passing; (7) Integration tests with real calls; (8) A/B test against V1; (9) Gradual rollout with monitoring; (10) Documentation updated.

---

## Implementation Architecture

### Layer 1: SSL Backbone (XLS-R 300M)
```
Audio → XLS-R 300M (frozen or fine-tuned) → Weighted-sum layer aggregation → MLP(256→64→2) → P(spoof)
```
- **Why**: Best SSL frontend per Spoof-SUPERB benchmark (17.4% mean EER)
- **Catches**: Autoregressive TTS artifacts (Dia2: 7.07% EER)
- **Requirements**: Model download (~1.2GB), GPU inference

### Layer 2: Semantic Analysis (Whisper Encoder)
```
Audio → Whisper Encoder (reuse existing) → Extract hidden states → MLP(256→64→2) → P(spoof)
```
- **Why**: Complementary to SSL — catches non-autoregressive/flow-matching systems
- **Catches**: MeloTTS (17.05% EER), StyleTTS2
- **Requirements**: Zero additional memory (reuse transcriber)

### Layer 3: Spectrogram + Raw Waveform (AST + RawNet3)
```
Audio → AST (existing) → P(spoof)
Audio → RawNet3 (existing, unused) → P(spoof)
```
- **Why**: Complementary architectures capturing different artifact signatures
- **Catches**: Known vocoder patterns, GAN artifacts
- **Requirements**: Already implemented, just need integration

### Layer 4: Prosodic Forensics
```
Audio → Praat (parselmouth) → [F0, jitter, shimmer, HNR, speech_rate, pause_stats]
     → XGBoost/LogisticRegression → P(spoof)
```
- **Why**: 93% accuracy standalone, adversarially robust
- **Catches**: Missing vocal fold biomechanics, unnatural prosody
- **Requirements**: CPU only, <50ms per chunk

### Layer 5: Biological Signal Analysis
```
Audio → VAD + Voicing Detector → [breathing_intervals, breath_periodicity,
         breath_duration, inhalation_ratio, phonation_onset_offset]
     → Rule-based + statistical classifier → P(spoof)
```
- **Why**: Physiological traces AI can't replicate perfectly
- **Catches**: Missing breathing, wrong breath timing, no phonation transients
- **Requirements**: CPU only, <30ms per chunk

### Layer 6: Phase & Formant Forensics
```
Audio → STFT(with phase) → [group_delay, instantaneous_freq, phase_coherence]
Audio → LPC → [F1, F2, F3, bandwidths, VTL_consistency, CV_transition_rates]
     → MLP/Rules → P(spoof)
```
- **Why**: Phase domain has untapped spoofing cues; formant analysis achieves 98.39% AUC
- **Catches**: Vocoder phase artifacts, incorrect formant dynamics
- **Requirements**: CPU only, <40ms per chunk

### Layer 7: Temporal Consistency
```
Audio stream → Per-chunk:
  → ECAPA-TDNN embedding (192d) → Sliding window similarity → Stability score
  → F0 contour → Long-range drift analysis
  → Emotion classifier → Consistency score
     → Combined temporal anomaly score
```
- **Why**: Catches long-range inconsistencies invisible in short chunks
- **Catches**: Mid-call voice switches, embedding instability, emotional discontinuity
- **Requirements**: Accumulated over call duration

### Ensemble Fusion
```
[Layer1_score, Layer2_score, Layer3a_score, Layer3b_score,
 Layer4_score, Layer5_score, Layer6_score, Layer7_score]
     → Calibrated Logistic Regression
     → P(spoof), uncertainty, per_analyzer_detail
```

---

## Training Augmentation Pipeline

```python
def augment_audio(waveform, sr=16000):
    # 1. RawBoost (signal degradation)
    waveform = rawboost(waveform, algo=random.choice([1,2,3,4,5]))

    # 2. Speed perturbation
    speed = random.uniform(0.9, 1.1)
    waveform = librosa.effects.time_stretch(waveform, rate=speed)

    # 3. Codec simulation (random chain)
    codecs = random.sample(['opus', 'amr_wb', 'g711', 'g722'], k=random.randint(1,2))
    for codec in codecs:
        waveform = simulate_codec(waveform, codec)

    # 4. Noise injection
    noise = load_random_noise()  # ESC-50 or MUSAN
    snr = random.uniform(5, 30)
    waveform = add_noise(waveform, noise, snr)

    # 5. Reverberation
    if random.random() < 0.3:
        rir = load_random_rir()
        waveform = convolve(waveform, rir)

    return waveform
```

---

## File Structure

```
engine/audio/
├── ast_spoof.py              # [EXISTING] AST detector
├── rawnet3.py                # [EXISTING] RawNet3 detector
├── features.py               # [EXISTING] LFCC features
├── speaker_verify.py         # [EXISTING] ECAPA-TDNN
├── transcriber.py            # [EXISTING] Whisper transcription
├── prosodic_analyzer.py      # [NEW] F0/jitter/shimmer/HNR
├── breathing_detector.py     # [NEW] Breath pattern analysis
├── phase_analyzer.py         # [NEW] Group delay/IF/phase coherence
├── formant_analyzer.py       # [NEW] Formant extraction/CV boundaries
├── temporal_tracker.py       # [NEW] Embedding stability/prosodic drift
├── ensemble_fusion.py        # [NEW] Calibrated multi-detector fusion
├── codec_augmentor.py        # [NEW] Training codec simulation
└── ssl_detector.py           # [NEW] XLS-R/Whisper feature detectors
```

---

## Implementation Priority

### Phase 1 — CPU-Only Analyzers (No downloads, immediate impact)
1. `prosodic_analyzer.py` — parselmouth-based
2. `breathing_detector.py` — VAD + spectral analysis
3. `phase_analyzer.py` — STFT phase features
4. `formant_analyzer.py` — LPC formant extraction
5. `temporal_tracker.py` — Embedding consistency tracker
6. `ensemble_fusion.py` — Score fusion with uncertainty
7. Integrate RawNet3 into production pipeline

### Phase 2 — Model Downloads & Training
8. `ssl_detector.py` — XLS-R 300M fine-tuning
9. `codec_augmentor.py` — Training augmentation pipeline
10. Retrain all models with codec augmentation
11. Train ensemble fusion on validation data

### Phase 3 — Advanced Features
12. Retrieval-based zero-day detection
13. Continual learning pipeline
14. ONNX export for all new models
15. Benchmark suite (ASVspoof 2021 + In-the-Wild + custom)
