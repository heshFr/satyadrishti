"""
Satya Drishti — Inference Engine
================================
Orchestrates all ML models for multimodal deepfake and coercion detection.
Models are lazy-loaded on first request to conserve memory.

Engines:
  - Audio: 9-Layer Ensemble (AST + XLS-R SSL + Whisper Features + Prosodic +
           Breathing + Phase + Formant + Temporal + EnsembleFusion)
  - Text: DeBERTaV3 + LoRA for coercion/manipulation detection
  - Video: Two-stream ViT-B/16 + R3D-18 for video deepfake detection
  - Forensics: Image forensics pipeline (ViT + frequency + metadata)
  - Fusion: Cross-attention transformer for multimodal threat assessment
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional

from .config import (
    TEXT_CHECKPOINT, VIDEO_SPATIAL_CKPT, VIDEO_TEMPORAL_CKPT,
    FORENSICS_CKPT, FORENSICS_PRETRAINED, FUSION_CKPT,
    XLS_R_MODEL_PATH,
)

log = logging.getLogger("satyadrishti.engine")

# Try imports but fail gracefully if ml dependencies aren't ready
try:
    import torch
    import numpy as np
    from engine.audio.ast_spoof import ASTSpoofDetector
    from engine.text.coercion_detector import CoercionDetector, COERCION_LABELS
    from engine.image_forensics.detector import ImageForensicsDetector
    from engine.video.two_stream import TwoStreamDetector
    from engine.video.quality_analyzer import VideoQualityAnalyzer
    from engine.fusion.cross_attention import MultimodalFusionNetwork
    from engine.audio.transcriber import RealTimeTranscriber
    from engine.audio.speaker_verify import SpeakerVerifier
    HAS_ML = True
except ImportError as e:
    HAS_ML = False
    log.warning("ML dependencies missing (%s). Running in stub mode.", e)

# 9-Layer Audio Forensics — CPU-based analyzers (graceful fallback)
try:
    from engine.audio.prosodic_analyzer import ProsodicAnalyzer
    HAS_PROSODIC = True
except ImportError:
    HAS_PROSODIC = False

try:
    from engine.audio.breathing_detector import BreathingDetector
    HAS_BREATHING = True
except ImportError:
    HAS_BREATHING = False

try:
    from engine.audio.phase_analyzer import PhaseAnalyzer
    HAS_PHASE = True
except ImportError:
    HAS_PHASE = False

try:
    from engine.audio.formant_analyzer import FormantAnalyzer
    HAS_FORMANT = True
except ImportError:
    HAS_FORMANT = False

try:
    from engine.audio.temporal_tracker import TemporalTracker
    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False

try:
    from engine.audio.ensemble_fusion import EnsembleFusion
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

try:
    from engine.audio.ssl_detector import SSLDetector
    HAS_SSL = True
except ImportError:
    HAS_SSL = False

try:
    from engine.audio.whisper_features import WhisperFeatureExtractor
    HAS_WHISPER_FEATURES = True
except ImportError:
    HAS_WHISPER_FEATURES = False

# Threat class labels from the fusion network
THREAT_LABELS = {
    0: "safe",
    1: "deepfake",
    2: "coercion",
    3: "deepfake_and_coercion",
}


class InferenceEngine:
    @staticmethod
    def _select_device():
        """Pick CUDA only if enough free VRAM (>= 800 MB), else CPU."""
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                free_mb = free / (1024 ** 2)
                if free_mb >= 1500:
                    return torch.device("cuda")
                log.info("CUDA available but only %.0f MB free — using CPU for inference", free_mb)
            except Exception:
                pass  # older drivers may not support mem_get_info
        return torch.device("cpu")

    def __init__(self):
        if HAS_ML:
            self.device = self._select_device()
        else:
            self.device = "cpu"

        self.models: Dict[str, Any] = {
            "audio": None,
            "text": None,
            "video": None,
            "forensics": None,
            "fusion": None,
        }

        self.transcriber = None
        self.speaker_verifier = None

        # 9-Layer Audio Forensics analyzers (CPU-based, lightweight)
        self._prosodic: Optional[Any] = None
        self._breathing: Optional[Any] = None
        self._phase: Optional[Any] = None
        self._formant: Optional[Any] = None
        self._temporal_tracker: Optional[Any] = None
        self._ensemble_fusion: Optional[Any] = None
        self._ssl_detector: Optional[Any] = None
        self._whisper_features: Optional[Any] = None

        self._loading_locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for thread-safe model loading."""
        if key not in self._loading_locks:
            self._loading_locks[key] = asyncio.Lock()
        return self._loading_locks[key]

    # ─── Lazy Model Loaders ───

    async def get_audio_model(self) -> Optional[Any]:
        """Lazy load AST spoof detector."""
        async with self._get_lock("audio"):
            if self.models["audio"] is None and HAS_ML:
                log.info("Loading Audio (AST-VoxCelebSpoof) on %s...", self.device)
                model = ASTSpoofDetector()
                model.to(self.device)
                model.eval()
                self.models["audio"] = model
                log.info("Audio model ready.")
        return self.models["audio"]

    async def get_text_model(self) -> Optional[Any]:
        """Lazy load DeBERTaV3 + LoRA coercion detector with trained weights."""
        async with self._get_lock("text"):
            if self.models["text"] is None and HAS_ML:
                if os.path.isdir(TEXT_CHECKPOINT):
                    log.info("Loading Text (DeBERTaV3 + LoRA) on %s...", self.device)
                    model = CoercionDetector(checkpoint_dir=TEXT_CHECKPOINT)
                    model.model.eval()
                    model.model.to(self.device)
                    self.models["text"] = model
                    log.info("Text model ready.")
                else:
                    log.warning("Text checkpoint not found at %s", TEXT_CHECKPOINT)
        return self.models["text"]

    async def get_video_model(self) -> Optional[Any]:
        """Lazy load two-stream video deepfake detector."""
        async with self._get_lock("video"):
            if self.models["video"] is None and HAS_ML:
                spatial_ok = os.path.exists(VIDEO_SPATIAL_CKPT)
                temporal_ok = os.path.exists(VIDEO_TEMPORAL_CKPT)

                if spatial_ok and temporal_ok:
                    log.info("Loading Video (ViT-B/16 + R3D-18) on %s...", self.device)
                    self.models["video"] = TwoStreamDetector(
                        spatial_ckpt=VIDEO_SPATIAL_CKPT,
                        temporal_ckpt=VIDEO_TEMPORAL_CKPT,
                        device=self.device,
                    )
                    log.info("Video model ready.")
                else:
                    missing = []
                    if not spatial_ok:
                        missing.append(f"spatial ({VIDEO_SPATIAL_CKPT})")
                    if not temporal_ok:
                        missing.append(f"temporal ({VIDEO_TEMPORAL_CKPT})")
                    log.warning("Video checkpoints missing: %s", ', '.join(missing))
        return self.models["video"]

    async def get_forensics_model(self) -> Optional[Any]:
        """Lazy load image forensics pipeline (ViT-B/16 + frequency + metadata)."""
        async with self._get_lock("forensics"):
            if self.models["forensics"] is None and HAS_ML:
                self.models["forensics"] = ImageForensicsDetector(
                    model_path=FORENSICS_CKPT,
                    pretrained_dir=FORENSICS_PRETRAINED,
                    device=str(self.device),
                )
        return self.models["forensics"]

    async def get_fusion_model(self) -> Optional[Any]:
        """Lazy load cross-attention fusion network."""
        async with self._get_lock("fusion"):
            if self.models["fusion"] is None and HAS_ML:
                if os.path.exists(FUSION_CKPT):
                    log.info("Loading Fusion network on %s...", self.device)
                    fusion = MultimodalFusionNetwork(
                        audio_embed_dim=768,
                        video_embed_dim=768,
                        text_embed_dim=768,
                        latent_dim=256,
                        num_heads=8,
                        num_layers=4,
                        num_classes=4,
                    )
                    ckpt = torch.load(FUSION_CKPT, map_location=self.device, weights_only=False)
                    fusion.load_state_dict(ckpt["model_state_dict"])
                    fusion.to(self.device).eval()
                    self.models["fusion"] = fusion
                    log.info("Fusion network ready.")
                else:
                    log.warning("Fusion checkpoint not found at %s", FUSION_CKPT)
        return self.models["fusion"]

    async def get_transcriber(self):
        """Lazy load real-time transcription engine."""
        async with self._get_lock("transcriber"):
            if self.transcriber is None and HAS_ML:
                try:
                    self.transcriber = RealTimeTranscriber(
                        model_size="small",
                        device="auto",
                    )
                    if self.transcriber.is_available:
                        log.info("Real-time transcriber ready.")
                    else:
                        self.transcriber = None
                except Exception as e:
                    log.warning(f"Transcriber not available: {e}")
        return self.transcriber

    async def get_speaker_verifier(self):
        """Lazy load speaker verification engine."""
        async with self._get_lock("speaker_verify"):
            if self.speaker_verifier is None and HAS_ML:
                try:
                    self.speaker_verifier = SpeakerVerifier(device="auto")
                    if self.speaker_verifier.is_available:
                        log.info("Speaker verification ready.")
                    else:
                        self.speaker_verifier = None
                except Exception as e:
                    log.warning(f"Speaker verification not available: {e}")
        return self.speaker_verifier

    # ─── 9-Layer Audio Analyzer Loaders ───

    def _get_prosodic(self):
        if self._prosodic is None and HAS_PROSODIC:
            self._prosodic = ProsodicAnalyzer()
            log.info("ProsodicAnalyzer initialized (Layer 4)")
        return self._prosodic

    def _get_breathing(self):
        if self._breathing is None and HAS_BREATHING:
            self._breathing = BreathingDetector()
            log.info("BreathingDetector initialized (Layer 5)")
        return self._breathing

    def _get_phase(self):
        if self._phase is None and HAS_PHASE:
            self._phase = PhaseAnalyzer()
            log.info("PhaseAnalyzer initialized (Layer 6)")
        return self._phase

    def _get_formant(self):
        if self._formant is None and HAS_FORMANT:
            self._formant = FormantAnalyzer()
            log.info("FormantAnalyzer initialized (Layer 6)")
        return self._formant

    def _get_temporal_tracker(self):
        if self._temporal_tracker is None and HAS_TEMPORAL:
            self._temporal_tracker = TemporalTracker()
            log.info("TemporalTracker initialized (Layer 7)")
        return self._temporal_tracker

    def _get_ssl_detector(self):
        if self._ssl_detector is None and HAS_SSL:
            if os.path.isdir(XLS_R_MODEL_PATH):
                self._ssl_detector = SSLDetector(
                    model_path=XLS_R_MODEL_PATH,
                    device="cpu",  # 1.2GB model — run on CPU to save GPU VRAM
                )
                if self._ssl_detector.is_available:
                    log.info("SSLDetector (XLS-R 300M) initialized (Layer 2)")
                else:
                    self._ssl_detector = None
            else:
                log.warning("XLS-R 300M model not found at %s", XLS_R_MODEL_PATH)
        return self._ssl_detector

    def _get_whisper_features(self):
        if self._whisper_features is None and HAS_WHISPER_FEATURES:
            # Share model with transcriber if already loaded
            shared = None
            if self.transcriber is not None and hasattr(self.transcriber, 'model') and self.transcriber.model is not None:
                # faster-whisper's WhisperModel is on transcriber.model
                # But RealTimeTranscriber wraps it — access the underlying model
                shared = self.transcriber.model if hasattr(self.transcriber, 'model') else None
            self._whisper_features = WhisperFeatureExtractor(
                model_size="small",
                device="auto",
                shared_model=shared,
            )
            if self._whisper_features.is_available:
                log.info("WhisperFeatureExtractor initialized (Layer 3)")
            else:
                self._whisper_features = None
        return self._whisper_features

    def _get_ensemble_fusion(self):
        if self._ensemble_fusion is None and HAS_ENSEMBLE:
            self._ensemble_fusion = EnsembleFusion()
            log.info("EnsembleFusion initialized")
        return self._ensemble_fusion

    # ─── Audio Analysis (9-Layer Ensemble) ───

    async def analyze_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Full 9-layer audio forensics analysis.

        Layers:
          1. AST (Audio Spectrogram Transformer) — spectrogram-based spoof detection
          2. XLS-R 300M SSL backbone — self-supervised representation analysis
          3. Whisper encoder features — log-mel + encoder hidden state analysis
          4. Prosodic Forensics — jitter, shimmer, HNR, F0, speech rate
          5. Biological Signals — breathing patterns, periodicity
          6. Phase & Formant — group delay, instantaneous frequency, formant analysis
          7. Temporal Consistency — speaker embedding stability (single-shot mode)

        All layer scores are fused via calibrated EnsembleFusion.
        """
        import io
        import soundfile as sf

        # ── Step 0: Read waveform ──
        try:
            waveform, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
        except Exception as e:
            return {"error": f"Could not read audio: {e}"}

        duration = len(waveform) / sr
        analyzer_outputs: Dict[str, dict] = {}
        forensic_checks = []
        raw_scores: Dict[str, float] = {}
        layers_run = []

        # ── Layer 1: AST Spectrogram Transformer ──
        ast_model = await self.get_audio_model()
        if ast_model:
            try:
                ast_result = ast_model.predict(waveform, sr)
                # AST is_spoof → score 0-1 (higher = more likely spoof)
                spoof_prob = 0.0
                probs = ast_result.get("probabilities", {})
                for k, v in probs.items():
                    if "spoof" in k.lower() or "fake" in k.lower() or "synthetic" in k.lower():
                        spoof_prob = max(spoof_prob, v)

                analyzer_outputs["ast"] = {
                    "score": spoof_prob,
                    "confidence": ast_result["confidence"],
                }

                status = "fail" if spoof_prob > 0.6 else "warn" if spoof_prob > 0.4 else "pass"
                forensic_checks.append({
                    "id": "ast_spectrogram",
                    "name": "AST Spectrogram Analysis",
                    "status": status,
                    "description": (
                        f"Audio Spectrogram Transformer: {spoof_prob*100:.1f}% synthetic probability. "
                        f"Trained on VoxCelebSpoof for known vocoder patterns."
                    ),
                })
                raw_scores["ast_spoof_prob"] = round(spoof_prob, 4)
                layers_run.append("ast")
            except Exception as e:
                log.warning("AST analysis failed: %s", e)
                forensic_checks.append({
                    "id": "ast_spectrogram",
                    "name": "AST Spectrogram Analysis",
                    "status": "info",
                    "description": f"AST analysis unavailable: {e}",
                })

        # ── Layer 2: XLS-R 300M SSL Backbone ──
        ssl_detector = self._get_ssl_detector()
        if ssl_detector and duration >= 0.5:
            try:
                ssl_result = await asyncio.to_thread(ssl_detector.analyze, waveform, sr)
                ssl_score = ssl_result.get("score", 0.5)
                ssl_conf = ssl_result.get("confidence", 0.5)
                ssl_anomalies = ssl_result.get("anomalies", [])

                analyzer_outputs["ssl"] = {
                    "score": ssl_score,
                    "confidence": ssl_conf,
                    "anomalies": ssl_anomalies,
                }

                status = "fail" if ssl_score > 0.6 else "warn" if ssl_score > 0.4 else "pass"

                detail_parts = []
                ssl_features = ssl_result.get("features", {})
                if "temporal_similarity_mean" in ssl_features:
                    detail_parts.append(f"Temporal coherence: {ssl_features['temporal_similarity_mean']:.3f}")
                if "frame_diversity" in ssl_features:
                    detail_parts.append(f"Frame diversity: {ssl_features['frame_diversity']:.3f}")
                if "activation_kurtosis" in ssl_features:
                    detail_parts.append(f"Kurtosis: {ssl_features['activation_kurtosis']:.2f}")

                anomaly_text = ""
                if ssl_anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in ssl_anomalies[:3])}."

                forensic_checks.append({
                    "id": "ssl_backbone",
                    "name": "XLS-R 300M Self-Supervised Analysis",
                    "status": status,
                    "description": (
                        f"SSL score: {ssl_score*100:.1f}% synthetic. "
                        f"{'; '.join(detail_parts)}. "
                        f"Analyzes 24-layer transformer representations trained on 128 languages — "
                        f"real speech produces distinct distributional patterns in SSL feature space."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["ssl_score"] = round(ssl_score, 4)
                layers_run.append("ssl")
            except Exception as e:
                log.warning("SSL analysis failed: %s", e)
                forensic_checks.append({
                    "id": "ssl_backbone",
                    "name": "XLS-R 300M Self-Supervised Analysis",
                    "status": "info",
                    "description": f"SSL analysis unavailable: {e}",
                })

        # ── Layer 3: Whisper Encoder Features ──
        whisper_feat = self._get_whisper_features()
        if whisper_feat and duration >= 0.5:
            try:
                wf_result = await asyncio.to_thread(whisper_feat.analyze, waveform, sr)
                wf_score = wf_result.get("score", 0.5)
                wf_conf = wf_result.get("confidence", 0.5)
                wf_anomalies = wf_result.get("anomalies", [])

                analyzer_outputs["whisper_features"] = {
                    "score": wf_score,
                    "confidence": wf_conf,
                    "anomalies": wf_anomalies,
                }

                status = "fail" if wf_score > 0.6 else "warn" if wf_score > 0.4 else "pass"

                detail_parts = []
                wf_features = wf_result.get("features", {})
                if "mel_spectral_flatness" in wf_features:
                    detail_parts.append(f"Spectral flatness: {wf_features['mel_spectral_flatness']:.3f}")
                if "encoder_temporal_sim_mean" in wf_features:
                    detail_parts.append(f"Encoder coherence: {wf_features['encoder_temporal_sim_mean']:.3f}")
                if "modulation_4hz_power" in wf_features:
                    detail_parts.append(f"Syllabic modulation: {wf_features['modulation_4hz_power']:.4f}")
                if "delta_energy_std" in wf_features:
                    detail_parts.append(f"Energy dynamics: {wf_features['delta_energy_std']:.3f}")

                anomaly_text = ""
                if wf_anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in wf_anomalies[:3])}."

                forensic_checks.append({
                    "id": "whisper_features",
                    "name": "Whisper Encoder Feature Analysis",
                    "status": status,
                    "description": (
                        f"Whisper score: {wf_score*100:.1f}% synthetic. "
                        f"{'; '.join(detail_parts)}. "
                        f"Extracts log-mel spectrogram + encoder hidden states from Whisper (680K hours training) — "
                        f"captures spectral, modulation, and temporal patterns."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["whisper_features_score"] = round(wf_score, 4)
                layers_run.append("whisper_features")
            except Exception as e:
                log.warning("Whisper feature analysis failed: %s", e)
                forensic_checks.append({
                    "id": "whisper_features",
                    "name": "Whisper Encoder Feature Analysis",
                    "status": "info",
                    "description": f"Whisper feature analysis unavailable: {e}",
                })

        # ── Layer 4: Prosodic Forensics ──
        prosodic = self._get_prosodic()
        if prosodic and duration >= 0.5:
            try:
                p_result = await asyncio.to_thread(prosodic.analyze, waveform, sr)
                analyzer_outputs["prosodic"] = {
                    "score": p_result.get("score", 0.5),
                    "confidence": p_result.get("confidence", 0.5),
                    "anomalies": p_result.get("anomalies", []),
                }
                p_score = p_result.get("score", 0.5)
                anomalies = p_result.get("anomalies", [])
                status = "fail" if p_score > 0.6 else "warn" if p_score > 0.4 else "pass"

                detail_parts = []
                features = p_result.get("features", {})
                if "jitter_local" in features:
                    detail_parts.append(f"Jitter: {features['jitter_local']:.4f}")
                if "shimmer_local" in features:
                    detail_parts.append(f"Shimmer: {features['shimmer_local']:.4f}")
                if "hnr_mean" in features:
                    detail_parts.append(f"HNR: {features['hnr_mean']:.1f}dB")
                if "f0_cv" in features:
                    detail_parts.append(f"F0 CV: {features['f0_cv']:.3f}")

                anomaly_text = ""
                if anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in anomalies[:3])}."

                forensic_checks.append({
                    "id": "prosodic_forensics",
                    "name": "Prosodic Forensics (F0/Jitter/Shimmer/HNR)",
                    "status": status,
                    "description": (
                        f"Prosodic score: {p_score*100:.1f}% synthetic. "
                        f"{'; '.join(detail_parts)}."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["prosodic_score"] = round(p_score, 4)
                layers_run.append("prosodic")
            except Exception as e:
                log.warning("Prosodic analysis failed: %s", e)
                forensic_checks.append({
                    "id": "prosodic_forensics",
                    "name": "Prosodic Forensics (F0/Jitter/Shimmer/HNR)",
                    "status": "info",
                    "description": f"Prosodic analysis unavailable: {e}",
                })

        # ── Layer 5: Breathing Detection ──
        breathing = self._get_breathing()
        if breathing and duration >= 1.0:
            try:
                b_result = await asyncio.to_thread(breathing.analyze, waveform, sr)
                analyzer_outputs["breathing"] = {
                    "score": b_result.get("score", 0.5),
                    "confidence": b_result.get("confidence", 0.5),
                    "anomalies": b_result.get("anomalies", []),
                }
                b_score = b_result.get("score", 0.5)
                b_anomalies = b_result.get("anomalies", [])
                status = "fail" if b_score > 0.6 else "warn" if b_score > 0.4 else "pass"

                features = b_result.get("features", {})
                breath_count = features.get("breath_count", 0)
                breath_rate = features.get("breaths_per_minute", 0)

                anomaly_text = ""
                if b_anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in b_anomalies[:3])}."

                forensic_checks.append({
                    "id": "breathing_detection",
                    "name": "Biological Signals (Breathing Patterns)",
                    "status": status,
                    "description": (
                        f"Breathing score: {b_score*100:.1f}% synthetic. "
                        f"Detected {breath_count} breaths ({breath_rate:.1f}/min). "
                        f"Natural range: 12-20/min."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["breathing_score"] = round(b_score, 4)
                layers_run.append("breathing")
            except Exception as e:
                log.warning("Breathing detection failed: %s", e)
                forensic_checks.append({
                    "id": "breathing_detection",
                    "name": "Biological Signals (Breathing Patterns)",
                    "status": "info",
                    "description": f"Breathing analysis unavailable: {e}",
                })

        # ── Layer 6a: Phase Domain Forensics ──
        phase = self._get_phase()
        if phase and duration >= 0.1:
            try:
                ph_result = await asyncio.to_thread(phase.analyze, waveform, sr)
                analyzer_outputs["phase"] = {
                    "score": ph_result.get("score", 0.5),
                    "confidence": ph_result.get("confidence", 0.5),
                    "anomalies": ph_result.get("anomalies", []),
                }
                ph_score = ph_result.get("score", 0.5)
                ph_anomalies = ph_result.get("anomalies", [])
                status = "fail" if ph_score > 0.6 else "warn" if ph_score > 0.4 else "pass"

                anomaly_text = ""
                if ph_anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in ph_anomalies[:3])}."

                forensic_checks.append({
                    "id": "phase_forensics",
                    "name": "Phase Domain Forensics (Group Delay/IF)",
                    "status": status,
                    "description": (
                        f"Phase score: {ph_score*100:.1f}% synthetic. "
                        f"Analyzes group delay, instantaneous frequency, and phase coherence — "
                        f"vocoders produce patterns incompatible with natural glottal excitation."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["phase_score"] = round(ph_score, 4)
                layers_run.append("phase")
            except Exception as e:
                log.warning("Phase analysis failed: %s", e)
                forensic_checks.append({
                    "id": "phase_forensics",
                    "name": "Phase Domain Forensics (Group Delay/IF)",
                    "status": "info",
                    "description": f"Phase analysis unavailable: {e}",
                })

        # ── Layer 6b: Formant Analysis ──
        formant = self._get_formant()
        if formant and duration >= 0.1:
            try:
                f_result = await asyncio.to_thread(formant.analyze, waveform, sr)
                analyzer_outputs["formant"] = {
                    "score": f_result.get("score", 0.5),
                    "confidence": f_result.get("confidence", 0.5),
                    "anomalies": f_result.get("anomalies", []),
                }
                f_score = f_result.get("score", 0.5)
                f_anomalies = f_result.get("anomalies", [])
                status = "fail" if f_score > 0.6 else "warn" if f_score > 0.4 else "pass"

                features = f_result.get("features", {})
                vtl_consistency = features.get("vtl_consistency", 0)

                anomaly_text = ""
                if f_anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in f_anomalies[:3])}."

                forensic_checks.append({
                    "id": "formant_analysis",
                    "name": "Formant Analysis (LPC/VTL/CV Boundaries)",
                    "status": status,
                    "description": (
                        f"Formant score: {f_score*100:.1f}% synthetic. "
                        f"VTL consistency: {vtl_consistency:.2f}. "
                        f"Deepfake models fail to replicate formant transitions at consonant-vowel boundaries."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["formant_score"] = round(f_score, 4)
                layers_run.append("formant")
            except Exception as e:
                log.warning("Formant analysis failed: %s", e)
                forensic_checks.append({
                    "id": "formant_analysis",
                    "name": "Formant Analysis (LPC/VTL/CV Boundaries)",
                    "status": "info",
                    "description": f"Formant analysis unavailable: {e}",
                })

        # ── Layer 7: Temporal Consistency (single-shot mode) ──
        # For non-streaming audio, we split into chunks and track consistency
        temporal = self._get_temporal_tracker()
        if temporal and duration >= 3.0 and ast_model:
            try:
                tracker = TemporalTracker()  # fresh instance per analysis
                chunk_duration = 3.0
                chunk_samples = int(chunk_duration * sr)
                n_chunks = max(1, int(len(waveform) / chunk_samples))
                last_t_result = None

                for i in range(min(n_chunks, 10)):  # cap at 10 chunks
                    start = i * chunk_samples
                    end = min(start + chunk_samples, len(waveform))
                    chunk = waveform[start:end]

                    if len(chunk) < sr:  # skip chunks < 1s
                        continue

                    # Extract AST embedding for this chunk
                    try:
                        inputs = ast_model.preprocess(chunk, sr)
                        input_values = inputs["input_values"].to(self.device)
                        with torch.no_grad():
                            emb = ast_model.extract_embedding(input_values)
                        emb_np = emb.cpu().numpy().flatten()

                        # Get prosodic F0 stats for this chunk if available
                        f0_stats = None
                        if prosodic:
                            try:
                                chunk_prosodic = prosodic.analyze(chunk, sr)
                                f0_feat = chunk_prosodic.get("features", {})
                                if "f0_mean" in f0_feat:
                                    f0_stats = {
                                        "f0_mean": f0_feat["f0_mean"],
                                        "f0_std": f0_feat.get("f0_std", 0),
                                    }
                            except Exception:
                                pass

                        last_t_result = tracker.update(emb_np, f0_stats=f0_stats)
                    except Exception:
                        continue

                if last_t_result is None:
                    raise ValueError("No chunks were successfully analyzed")

                t_score = last_t_result.get("score", 0.5)
                t_confidence = last_t_result.get("confidence", 0.3)

                analyzer_outputs["temporal"] = {
                    "score": t_score,
                    "confidence": t_confidence,
                    "anomalies": last_t_result.get("anomalies", []),
                }

                status = "fail" if t_score > 0.6 else "warn" if t_score > 0.4 else "pass"
                t_anomalies = last_t_result.get("anomalies", [])
                anomaly_text = ""
                if t_anomalies:
                    anomaly_text = f" Anomalies: {', '.join(str(a) for a in t_anomalies[:3])}."

                forensic_checks.append({
                    "id": "temporal_consistency",
                    "name": "Temporal Consistency (Speaker Stability)",
                    "status": status,
                    "description": (
                        f"Temporal score: {t_score*100:.1f}% synthetic. "
                        f"Analyzed {n_chunks} chunks ({duration:.1f}s). "
                        f"Tracks embedding drift, F0 consistency, and sudden voice changes."
                        f"{anomaly_text}"
                    ),
                })
                raw_scores["temporal_score"] = round(t_score, 4)
                layers_run.append("temporal")
            except Exception as e:
                log.warning("Temporal analysis failed: %s", e)
                forensic_checks.append({
                    "id": "temporal_consistency",
                    "name": "Temporal Consistency (Speaker Stability)",
                    "status": "info",
                    "description": f"Temporal analysis unavailable: {e}",
                })

        # ── Ensemble Fusion ──
        ensemble = self._get_ensemble_fusion()
        if ensemble and analyzer_outputs:
            try:
                fusion_result = ensemble.fuse(analyzer_outputs)
                ensemble_verdict = fusion_result.get("verdict", "uncertain")
                ensemble_prob = fusion_result.get("probability", 0.5)
                ensemble_conf = fusion_result.get("confidence", 0.5)
                ensemble_uncertainty = fusion_result.get("uncertainty", 0.5)
                per_analyzer = fusion_result.get("per_analyzer", {})
                explanation = fusion_result.get("explanation", [])

                # Map ensemble verdict to our standard verdicts
                if ensemble_verdict == "spoof":
                    verdict = "spoof"
                elif ensemble_verdict == "bonafide":
                    verdict = "authentic"
                else:
                    verdict = "uncertain"

                confidence = round(ensemble_conf * 100, 1)

                # Add ensemble summary as a forensic check
                forensic_checks.insert(0, {
                    "id": "ensemble_verdict",
                    "name": f"9-Layer Ensemble Verdict ({len(layers_run)} layers active)",
                    "status": "fail" if verdict == "spoof" else "pass" if verdict == "authentic" else "warn",
                    "description": (
                        f"Fused probability: {ensemble_prob*100:.1f}% synthetic | "
                        f"Confidence: {ensemble_conf*100:.1f}% | "
                        f"Uncertainty: {ensemble_uncertainty*100:.1f}% | "
                        f"Active layers: {', '.join(layers_run)}"
                    ),
                })

                # Store per-analyzer breakdown in raw_scores
                raw_scores["ensemble_probability"] = round(ensemble_prob, 4)
                raw_scores["ensemble_confidence"] = round(ensemble_conf, 4)
                raw_scores["ensemble_uncertainty"] = round(ensemble_uncertainty, 4)
                for name, info in per_analyzer.items():
                    raw_scores[f"{name}_weight"] = round(info.get("weight", 0), 4)
                    raw_scores[f"{name}_contribution"] = round(info.get("contribution", 0), 4)

                log.info(
                    "Audio 9-Layer Ensemble: verdict=%s prob=%.3f conf=%.3f uncertainty=%.3f layers=%s",
                    verdict, ensemble_prob, ensemble_conf, ensemble_uncertainty,
                    ",".join(layers_run),
                )

                return {
                    "status": "success",
                    "verdict": verdict,
                    "confidence": confidence,
                    "forensic_checks": forensic_checks,
                    "raw_scores": raw_scores,
                    "details": {
                        "ensemble_probability": round(ensemble_prob, 4),
                        "per_analyzer": per_analyzer,
                        "explanation": explanation,
                        "layers_active": layers_run,
                        "layers_total": 9,
                        "duration_seconds": round(duration, 2),
                    },
                }
            except Exception as e:
                log.error("Ensemble fusion failed: %s", e)

        # ── Fallback: AST-only result if ensemble failed ──
        if "ast" in analyzer_outputs:
            ast_score = analyzer_outputs["ast"]["score"]
            verdict = "spoof" if ast_score > 0.6 else "authentic" if ast_score < 0.4 else "uncertain"
            return {
                "status": "success",
                "verdict": verdict,
                "confidence": round(analyzer_outputs["ast"]["confidence"] * 100, 1),
                "forensic_checks": forensic_checks,
                "raw_scores": raw_scores,
                "details": {
                    "layers_active": layers_run,
                    "layers_total": 9,
                    "duration_seconds": round(duration, 2),
                    "note": "Ensemble fusion unavailable — AST-only result",
                },
            }

        return {"error": "No audio analysis engines available"}

    async def extract_audio_embedding(self, audio_data: bytes) -> Optional[Any]:
        """Extract 768d audio embedding for fusion."""
        model = await self.get_audio_model()
        if not model:
            return None

        try:
            import io
            import soundfile as sf

            waveform, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
            inputs = model.preprocess(waveform, sr)
            input_values = inputs["input_values"].to(self.device)
            return model.extract_embedding(input_values)  # (1, 768)
        except Exception as e:
            log.error("Audio embedding extraction failed: %s", e)
            return None

    # ─── Text Analysis ───

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for coercion/manipulation patterns (multilingual)."""
        from engine.text.multilingual import detect_language, check_coercion_patterns, translate_to_english

        # Step 1: Detect language
        lang = detect_language(text)

        # Step 2: Fast pattern-based check for Indian languages
        pattern_result = check_coercion_patterns(text, lang)

        # Step 3: Get ML prediction
        analysis_text = text
        translation_note = None

        if lang not in ("en", "hinglish"):
            # Translate to English for the ML model
            translated = translate_to_english(text, lang)
            if translated:
                analysis_text = translated
                translation_note = f"Translated from {lang} to English for analysis"
            else:
                # Translation failed — rely on pattern matching only
                if pattern_result["coercion_score"] > 0.5:
                    return {
                        "status": "success",
                        "verdict": "coercion_detected",
                        "threat_level": "high" if pattern_result["coercion_score"] > 0.7 else "moderate",
                        "confidence": round(pattern_result["coercion_score"] * 100, 1),
                        "detected_patterns": pattern_result["patterns_found"],
                        "categories": pattern_result["categories_triggered"],
                        "language": lang,
                        "method": "pattern_matching_only",
                        "note": "ML analysis unavailable for this language. Result based on keyword patterns.",
                    }

        # Step 4: Run ML model
        model = await self.get_text_model()
        if not model:
            # No ML model — return pattern results only
            if pattern_result["coercion_score"] > 0:
                return {
                    "status": "success",
                    "verdict": "possible_coercion" if pattern_result["coercion_score"] > 0.3 else "safe",
                    "confidence": round(pattern_result["coercion_score"] * 100, 1),
                    "detected_patterns": pattern_result["patterns_found"],
                    "language": lang,
                    "method": "pattern_matching_only",
                }
            return {"error": "Text engine not available"}

        try:
            result = model.predict(analysis_text, device=str(self.device))

            label = result["label"]
            if label == "safe":
                verdict = "safe"
                threat_level = "low"
            elif label == "urgency_manipulation":
                verdict = "urgency_manipulation"
                threat_level = "moderate"
            elif label == "financial_coercion":
                verdict = "financial_coercion"
                threat_level = "high"
            else:
                verdict = "combined_threat"
                threat_level = "critical"

            # Combine ML prediction with pattern matching
            ml_conf = result["confidence"]
            pattern_conf = pattern_result["coercion_score"]

            # If patterns found but ML says safe, boost the coercion signal
            if pattern_conf > 0.5 and verdict == "safe":
                verdict = "possible_coercion"
                threat_level = "moderate"
                ml_conf = max(ml_conf, pattern_conf)

            # If ML says coercion and patterns agree, boost confidence
            if pattern_conf > 0.3 and verdict != "safe":
                ml_conf = min(1.0, ml_conf * 1.15)

            response = {
                "status": "success",
                "verdict": verdict,
                "threat_level": threat_level,
                "confidence": round(ml_conf * 100, 1),
                "detected_patterns": [
                    k for k, v in result["probabilities"].items()
                    if v > 0.1 and k != "safe"
                ],
                "probabilities": {
                    k: round(v * 100, 1)
                    for k, v in result["probabilities"].items()
                },
                "language": lang,
            }

            if pattern_result["patterns_found"]:
                response["keyword_patterns"] = pattern_result["patterns_found"]
                response["keyword_categories"] = pattern_result["categories_triggered"]

            if translation_note:
                response["translation_note"] = translation_note

            return response

        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}

    async def extract_text_embedding(self, text: str) -> Optional[Any]:
        """Extract 768d text embedding for fusion."""
        model = await self.get_text_model()
        if not model:
            return None

        try:
            return model.extract_embedding(text, device=str(self.device))  # (1, 768)
        except Exception as e:
            log.error("Text embedding extraction failed: %s", e)
            return None

    # ─── Image Forensics ───

    async def analyze_media(self, file_path: str) -> Dict[str, Any]:
        """Run image forensics pipeline (ViT-B/16 + frequency + metadata)."""
        detector = await self.get_forensics_model()
        if not detector:
            return {"error": "Image forensics engine not available"}

        return detector.analyze(file_path)

    # ─── Video Analysis ───

    async def analyze_video(self, file_path: str) -> Dict[str, Any]:
        """
        Full video analysis combining:
        1. Video quality assessment (compression, motion blur, resolution)
        2. Image forensics pipeline (frame-level ViT + frequency analysis)
        3. Two-stream deepfake detection (spatial ViT v2 + temporal R3D v2)

        Quality metrics dynamically adjust model weights, verdict thresholds,
        and confidence caps to reduce false positives on compressed phone videos.
        """
        # Step 1: Assess video quality
        quality_metrics = None
        if HAS_ML:
            try:
                qa = VideoQualityAnalyzer()
                quality_metrics = qa.analyze(file_path)
                log.info("Video quality: %s (score=%s, compression=%s, blur=%s)",
                         quality_metrics['quality_tier'], quality_metrics['quality_score'],
                         quality_metrics['compression_score'], quality_metrics['motion_blur_score'])
            except Exception as e:
                log.warning("Quality analysis error: %s", e)

        # Step 2: Run image forensics on video frames
        forensics = await self.get_forensics_model()
        forensics_result = None
        if forensics:
            try:
                forensics_result = forensics.analyze_video(
                    file_path,
                    quality_metrics=quality_metrics,
                )
            except Exception as e:
                log.error("Forensics video analysis error: %s", e)

        # Step 3: Run two-stream deepfake detection
        video_model = await self.get_video_model()
        deepfake_result = None
        if video_model:
            try:
                deepfake_result = video_model.predict(file_path)
            except Exception as e:
                log.error("Two-stream video analysis error: %s", e)

        # Combine results into unified report
        if forensics_result is None and deepfake_result is None:
            return {"error": "No video analysis engines available"}

        report = {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "forensic_checks": [],
            "raw_scores": {},
        }

        # Add quality assessment check
        if quality_metrics and "error" not in quality_metrics:
            qm = quality_metrics
            tier = qm["quality_tier"]
            q_status = "pass" if tier in ("high", "medium") else "warn"
            report["forensic_checks"].append({
                "id": "video_quality",
                "name": "Video Quality Assessment",
                "status": q_status,
                "description": (
                    f"Resolution: {qm['width']}x{qm['height']}, "
                    f"Bitrate: {qm['estimated_bitrate_kbps']:.0f} kbps, "
                    f"Compression: {qm['compression_score']:.0%}, "
                    f"Quality: {tier}"
                ),
            })

            if qm["motion_blur_score"] > 0.4:
                report["forensic_checks"].append({
                    "id": "motion_blur",
                    "name": "Motion Blur Detection",
                    "status": "warn",
                    "description": (
                        f"Significant motion blur detected ({qm['motion_blur_score']:.0%}). "
                        "This may reduce detection accuracy."
                    ),
                })

            report["raw_scores"]["quality_score"] = qm["quality_score"]
            report["raw_scores"]["compression_score"] = qm["compression_score"]
            report["raw_scores"]["motion_blur_score"] = qm["motion_blur_score"]

        # Add forensics checks if available
        if forensics_result and "error" not in forensics_result:
            report["forensic_checks"].extend(forensics_result.get("forensic_checks", []))
            report["raw_scores"].update(forensics_result.get("raw_scores", {}))

        # Add deepfake detection checks if available
        if deepfake_result and "error" not in deepfake_result:
            spatial_prob = deepfake_result["spatial_fake_prob"]
            temporal_prob = deepfake_result["temporal_fake_prob"]
            combined_prob = deepfake_result["fake_probability"]

            report["forensic_checks"].append({
                "id": "spatial_deepfake",
                "name": "Spatial Deepfake Detection (ViT-B/16)",
                "status": "fail" if spatial_prob > 0.6 else "warn" if spatial_prob > 0.4 else "pass",
                "description": (
                    f"Face blending artifact analysis: {spatial_prob * 100:.1f}% deepfake probability"
                    f" ({deepfake_result['num_frames_analyzed']} frames analyzed)"
                ),
            })

            report["forensic_checks"].append({
                "id": "temporal_deepfake",
                "name": "Temporal Deepfake Detection (R3D-18)",
                "status": "fail" if temporal_prob > 0.6 else "warn" if temporal_prob > 0.4 else "pass",
                "description": (
                    f"Temporal consistency analysis: {temporal_prob * 100:.1f}% deepfake probability"
                    f" ({deepfake_result['num_clips_analyzed']} clips analyzed)"
                ),
            })

            report["raw_scores"]["spatial_deepfake"] = spatial_prob
            report["raw_scores"]["temporal_deepfake"] = temporal_prob
            report["raw_scores"]["combined_deepfake"] = combined_prob

        # Determine quality-aware weights and thresholds
        quality_tier = "medium"  # default if quality analysis unavailable
        motion_blur = 0.0
        if quality_metrics and "error" not in quality_metrics:
            quality_tier = quality_metrics["quality_tier"]
            motion_blur = quality_metrics.get("motion_blur_score", 0.0)

        # Quality-aware model weighting
        # Low-quality videos: trust two-stream more (trained on video data with compression)
        # and reduce forensics weight (image-trained ViT misreads compression as deepfake)
        # V2 calibrated thresholds — forensics neural scores are 0.80-0.94 for real content
        QUALITY_CONFIG = {
            "high":     {"deepfake_w": 0.70, "forensics_w": 0.30, "conf_cap": 95.0, "fake_thresh": 0.96, "real_thresh": 0.90},
            "medium":   {"deepfake_w": 0.75, "forensics_w": 0.25, "conf_cap": 92.0, "fake_thresh": 0.96, "real_thresh": 0.90},
            "low":      {"deepfake_w": 0.85, "forensics_w": 0.15, "conf_cap": 85.0, "fake_thresh": 0.97, "real_thresh": 0.88},
            "very_low": {"deepfake_w": 0.90, "forensics_w": 0.10, "conf_cap": 75.0, "fake_thresh": 0.98, "real_thresh": 0.85},
        }
        qc = QUALITY_CONFIG.get(quality_tier, QUALITY_CONFIG["medium"])

        # Determine final verdict using consistency-weighted ensemble
        if deepfake_result and "error" not in deepfake_result:
            spatial_prob = deepfake_result["spatial_fake_prob"]
            temporal_prob = deepfake_result["temporal_fake_prob"]
            twostream_combined = deepfake_result["fake_probability"]

            # --- Signal 1: Image forensics ViT (per-frame) ---
            forensics_neural_avg = 0.5
            forensics_neural_std = 0.5
            if forensics_result and "error" not in forensics_result:
                raw = forensics_result.get("raw_scores", {})
                forensics_neural_avg = raw.get("neural_avg", 0.5)
                forensics_neural_max = raw.get("neural_max", 0.5)
                # Approximate std from avg/max spread (actual std not in report)
                forensics_neural_std = forensics_neural_max - forensics_neural_avg

            # --- Signal 2: Two-stream agreement ---
            # When spatial and temporal strongly disagree, the signal is noisy
            twostream_disagreement = abs(spatial_prob - temporal_prob)

            # --- Consistency-based reliability scoring ---
            # Forensics ViT: low frame-to-frame variance = reliable signal
            # std < 0.05 → very consistent (trust fully)
            # std > 0.20 → inconsistent (model is guessing on some frames)
            forensics_reliability = max(0.2, 1.0 - forensics_neural_std * 4)

            # Two-stream: spatial/temporal agreement = reliable
            # disagreement < 0.15 → in agreement (trust more)
            # disagreement > 0.40 → contradictory (trust less)
            twostream_reliability = max(0.2, 1.0 - twostream_disagreement * 2)

            # Motion blur further reduces temporal reliability
            if motion_blur > 0.4:
                blur_penalty = min(0.5, (motion_blur - 0.4) * 0.83)
                twostream_reliability *= (1.0 - blur_penalty)

            # --- Weighted ensemble based on reliability ---
            total_reliability = forensics_reliability + twostream_reliability
            forensics_weight = forensics_reliability / total_reliability
            twostream_weight = twostream_reliability / total_reliability

            final_score = (
                forensics_neural_avg * forensics_weight
                + twostream_combined * twostream_weight
            )

            # --- Reliability-based confidence damping ---
            # When both models are unreliable (low consistency, high disagreement),
            # pull the final score toward 0.5 (uncertain) to prevent false positives.
            avg_reliability = (forensics_reliability + twostream_reliability) / 2
            if avg_reliability < 0.5:
                damping = avg_reliability / 0.5  # 0 to 1
                final_score = 0.5 + (final_score - 0.5) * damping

            log.info("Ensemble: forensics=%.3f (rel=%.2f), twostream=%.3f (rel=%.2f), avg_rel=%.2f, final=%.4f",
                     forensics_neural_avg, forensics_reliability, twostream_combined,
                     twostream_reliability, avg_reliability, final_score)

            report["raw_scores"]["forensics_reliability"] = round(forensics_reliability, 3)
            report["raw_scores"]["twostream_reliability"] = round(twostream_reliability, 3)
            report["raw_scores"]["avg_reliability"] = round(avg_reliability, 3)
            report["raw_scores"]["final_ensemble_score"] = round(final_score, 4)

            if final_score > qc["fake_thresh"]:
                report["verdict"] = "ai-generated"
                report["confidence"] = round(min(qc["conf_cap"], final_score * 100), 1)
            elif final_score < qc["real_thresh"]:
                report["verdict"] = "authentic"
                report["confidence"] = round(min(qc["conf_cap"], (1 - final_score) * 100), 1)
            else:
                report["verdict"] = "inconclusive"
                report["confidence"] = round(50.0 + abs(final_score - 0.5) * 100, 1)

        elif forensics_result and "error" not in forensics_result:
            # Fallback to forensics-only verdict (cap confidence for low quality)
            report["verdict"] = forensics_result.get("verdict", "inconclusive")
            raw_conf = forensics_result.get("confidence", 0.0)
            report["confidence"] = round(min(qc["conf_cap"], raw_conf), 1)

        # Track which detectors contributed to the verdict
        report["detectors_used"] = []
        if forensics_result and "error" not in forensics_result:
            report["detectors_used"].append("image_forensics_vit")
        if deepfake_result and "error" not in deepfake_result:
            report["detectors_used"].append("two_stream_spatial")
            report["detectors_used"].append("two_stream_temporal")
        if quality_metrics and "error" not in quality_metrics:
            report["detectors_used"].append("video_quality")

        # Add quality disclaimer for low-quality videos
        if quality_tier in ("low", "very_low"):
            report["forensic_checks"].append({
                "id": "quality_disclaimer",
                "name": "Analysis Reliability",
                "status": "warn",
                "description": (
                    f"Video quality is {quality_tier.replace('_', ' ')} — "
                    "compression artifacts and motion blur may affect accuracy. "
                    f"Confidence capped at {qc['conf_cap']:.0f}%."
                ),
            })

        return report

    async def extract_video_embedding(self, file_path: str) -> Optional[Any]:
        """Extract 768d video embedding for fusion."""
        video_model = await self.get_video_model()
        if not video_model:
            return None

        try:
            return video_model.extract_embedding(file_path)  # (1, 768)
        except Exception as e:
            log.error("Video embedding extraction failed: %s", e)
            return None

    # ─── Transcription & Speaker Verification ───

    async def transcribe_audio(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """Transcribe audio to text in real-time."""
        transcriber = await self.get_transcriber()
        if not transcriber:
            return {"text": "", "error": "Transcriber not available"}
        return transcriber.transcribe(audio_data, language)

    async def verify_speaker(self, audio_data: bytes) -> Dict[str, Any]:
        """Verify caller identity against enrolled voice prints."""
        verifier = await self.get_speaker_verifier()
        if not verifier:
            return {"is_verified": False, "error": "Speaker verification not available"}
        return verifier.verify(audio_data)

    async def enroll_voice_print(self, name: str, audio_data: bytes, relationship: str = "unknown") -> Dict[str, Any]:
        """Enroll a family member's voice print."""
        verifier = await self.get_speaker_verifier()
        if not verifier:
            return {"status": "error", "message": "Speaker verification not available"}
        return verifier.enroll(name, audio_data, relationship)

    # ─── Multimodal Fusion ───

    async def analyze_multimodal(
        self,
        audio_data: bytes = None,
        video_path: str = None,
        text: str = None,
    ) -> Dict[str, Any]:
        """
        Full multimodal analysis: run individual modalities, extract embeddings,
        and fuse via cross-attention network for 4-class threat assessment.

        Classes: safe / deepfake / coercion / deepfake+coercion
        """
        # Run individual analyses in parallel
        modality_results = {"audio": None, "video": None, "text": None}
        embeddings = {"audio": None, "video": None, "text": None}

        if audio_data:
            modality_results["audio"] = await self.analyze_audio(audio_data)
            embeddings["audio"] = await self.extract_audio_embedding(audio_data)

        if video_path:
            modality_results["video"] = await self.analyze_video(video_path)
            embeddings["video"] = await self.extract_video_embedding(video_path)

        if text:
            modality_results["text"] = await self.analyze_text(text)
            embeddings["text"] = await self.extract_text_embedding(text)

        # Try cross-attention fusion -- use zero padding for missing modalities
        fusion_result = None
        fusion_model = await self.get_fusion_model()

        n_available = sum(1 for e in embeddings.values() if e is not None)

        if fusion_model and n_available >= 1:
            try:
                # Pad missing modalities with zero embeddings
                # The fusion network can learn to ignore zero-padded modalities
                # through its attention mechanism
                def _pad(emb, dim=768):
                    if emb is not None:
                        return emb.to(self.device)
                    return torch.zeros(1, dim, device=self.device)

                with torch.no_grad():
                    fusion_output = fusion_model(
                        audio_emb=_pad(embeddings["audio"]),
                        video_emb=_pad(embeddings["video"]),
                        text_emb=_pad(embeddings["text"]),
                    )
                    probs = torch.softmax(fusion_output["logits"], dim=-1)[0]
                    pred_idx = probs.argmax().item()

                    fusion_result = {
                        "threat_class": THREAT_LABELS[pred_idx],
                        "confidence": round(probs[pred_idx].item() * 100, 1),
                        "class_probabilities": {
                            THREAT_LABELS[i]: round(probs[i].item() * 100, 1)
                            for i in range(4)
                        },
                        "modalities_used": n_available,
                    }

                    # Discount confidence when fewer modalities are available
                    if n_available < 3:
                        discount = 0.7 if n_available == 2 else 0.5
                        fusion_result["confidence"] = round(
                            fusion_result["confidence"] * discount, 1
                        )
                        fusion_result["partial_input_note"] = (
                            f"Only {n_available}/3 modalities provided. "
                            f"Confidence discounted by {int((1-discount)*100)}%."
                        )
            except Exception as e:
                log.error("Fusion inference error: %s", e)

        # Build response with rule-based fallback if fusion unavailable
        if fusion_result is None:
            fusion_result = self._rule_based_fusion(modality_results)

        return {
            "status": "success",
            "fusion": fusion_result,
            "overall_threat_level": self._threat_level(fusion_result),
            "modality_results": modality_results,
        }

    def _rule_based_fusion(self, modality_results: dict) -> dict:
        """
        Rule-based fusion fallback when the cross-attention network
        is unavailable or not all modalities are present.
        """
        deepfake_score = 0.0
        coercion_score = 0.0
        n_modalities = 0

        # Audio contributes to deepfake detection
        audio = modality_results.get("audio")
        if audio and "error" not in audio:
            n_modalities += 1
            if audio.get("verdict") == "spoof":
                # Audio is spoof -- contribute its confidence as deepfake evidence
                deepfake_score = max(deepfake_score, audio["confidence"] / 100)
            else:
                # Audio is authentic -- this REDUCES deepfake suspicion, not increases it.
                # Only contribute to deepfake_score if confidence is very low (uncertain)
                audio_conf = audio["confidence"] / 100
                if audio_conf < 0.6:
                    # Very uncertain authentic = slight deepfake signal
                    deepfake_score = max(deepfake_score, (1 - audio_conf) * 0.5)

        # Video contributes to deepfake detection
        video = modality_results.get("video")
        if video and "error" not in video:
            n_modalities += 1
            combined_deepfake = video.get("raw_scores", {}).get("combined_deepfake")
            if combined_deepfake is not None:
                deepfake_score = max(deepfake_score, combined_deepfake)
            elif video.get("verdict") == "ai-generated":
                deepfake_score = max(deepfake_score, video["confidence"] / 100)

        # Text contributes to coercion detection
        text = modality_results.get("text")
        if text and "error" not in text:
            n_modalities += 1
            if text.get("verdict") != "safe":
                coercion_score = max(coercion_score, text["confidence"] / 100)

        # Determine threat class
        is_deepfake = deepfake_score > 0.5
        is_coercion = coercion_score > 0.5

        if is_deepfake and is_coercion:
            threat_class = "deepfake_and_coercion"
            confidence = (deepfake_score + coercion_score) / 2
        elif is_deepfake:
            threat_class = "deepfake"
            confidence = deepfake_score
        elif is_coercion:
            threat_class = "coercion"
            confidence = coercion_score
        else:
            threat_class = "safe"
            confidence = 1 - max(deepfake_score, coercion_score)

        return {
            "threat_class": threat_class,
            "confidence": round(confidence * 100, 1),
            "class_probabilities": {
                "safe": round((1 - max(deepfake_score, coercion_score)) * 100, 1),
                "deepfake": round(deepfake_score * 100, 1),
                "coercion": round(coercion_score * 100, 1),
                "deepfake_and_coercion": round(min(deepfake_score, coercion_score) * 100, 1),
            },
            "method": "rule_based",
        }

    @staticmethod
    def _threat_level(fusion_result: dict) -> str:
        threat_class = fusion_result.get("threat_class", "safe")
        confidence = fusion_result.get("confidence", 0)

        if threat_class == "safe":
            return "low"
        elif threat_class == "deepfake_and_coercion":
            return "critical"
        elif confidence > 80:
            return "high"
        elif confidence > 60:
            return "moderate"
        else:
            return "low"


# Global singleton
engine = InferenceEngine()
