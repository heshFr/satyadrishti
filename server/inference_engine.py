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
    FUSION_CKPT, XLS_R_MODEL_PATH,
)

log = logging.getLogger("satyadrishti.engine")

# Try imports but fail gracefully if ml dependencies aren't ready
try:
    import torch
    import numpy as np
    import cv2
    from engine.audio.ast_spoof import ASTSpoofDetector
    from engine.text.coercion_detector import CoercionDetector, COERCION_LABELS
    from engine.image_forensics.detector import ImageForensicsDetector
    from engine.video.two_stream import TwoStreamDetector
    from engine.video.quality_analyzer import VideoQualityAnalyzer
    from engine.video.ai_video_detector import AIVideoDetector
    from engine.video.rppg_analyzer import RPPGAnalyzer
    from engine.video.clip_temporal_drift import CLIPTemporalDriftDetector
    from engine.video.lighting_consistency import LightingConsistencyAnalyzer
    from engine.video.av_sync_analyzer import AudioVisualSyncAnalyzer
    from engine.video.micro_expression_analyzer import MicroExpressionAnalyzer
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

try:
    from engine.audio.tts_artifact_detector import TTSArtifactDetector
    HAS_TTS_DETECTOR = True
except ImportError:
    HAS_TTS_DETECTOR = False

try:
    from engine.audio.codec_detector import CodecDetector
    HAS_CODEC_DETECTOR = True
except ImportError:
    HAS_CODEC_DETECTOR = False

try:
    from engine.audio.voice_clone_detector import VoiceCloneDetector
    HAS_VOICE_CLONE = True
except ImportError:
    HAS_VOICE_CLONE = False

# Phase 8: New audio analyzers
try:
    from engine.audio.spectral_continuity import SpectralContinuityAnalyzer
    HAS_SPECTRAL_CONTINUITY = True
except ImportError:
    HAS_SPECTRAL_CONTINUITY = False

try:
    from engine.audio.phoneme_transition import PhonemeTransitionAnalyzer
    HAS_PHONEME_TRANSITION = True
except ImportError:
    HAS_PHONEME_TRANSITION = False

try:
    from engine.audio.room_acoustics import RoomAcousticsAnalyzer
    HAS_ROOM_ACOUSTICS = True
except ImportError:
    HAS_ROOM_ACOUSTICS = False

# Phase 8: New video analyzers
try:
    from engine.video.face_mesh_analyzer import FaceMeshConsistencyAnalyzer
    HAS_FACE_MESH = True
except ImportError:
    HAS_FACE_MESH = False

try:
    from engine.video.temporal_frequency import TemporalFrequencyAnalyzer
    HAS_TEMPORAL_FREQ = True
except ImportError:
    HAS_TEMPORAL_FREQ = False

try:
    from engine.video.pupil_analyzer import PupilLightReflexAnalyzer
    HAS_PUPIL = True
except ImportError:
    HAS_PUPIL = False

try:
    from engine.video.forgery_localization import ForgeryLocalizer
    HAS_FORGERY_LOCALIZER = True
except ImportError:
    HAS_FORGERY_LOCALIZER = False

# Phase 8: Document forensics
try:
    from engine.document_forensics import DocumentForensicsDetector
    HAS_DOCUMENT_FORENSICS = True
except ImportError:
    HAS_DOCUMENT_FORENSICS = False

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
        self._tts_detector: Optional[Any] = None
        self._codec_detector: Optional[Any] = None
        self._voice_clone_detector: Optional[Any] = None

        # Phase 8: New audio analyzers
        self._spectral_continuity: Optional[Any] = None
        self._phoneme_transition: Optional[Any] = None
        self._room_acoustics: Optional[Any] = None

        # Phase 8: New video analyzers
        self._face_mesh: Optional[Any] = None
        self._temporal_freq: Optional[Any] = None
        self._pupil_analyzer: Optional[Any] = None

        # Phase 8: Document forensics
        self._document_forensics: Optional[Any] = None

        self._loading_locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for thread-safe model loading."""
        if key not in self._loading_locks:
            self._loading_locks[key] = asyncio.Lock()
        return self._loading_locks[key]

    # ─── Lazy Model Loaders ───

    async def get_audio_model(self) -> Optional[Any]:
        """Lazy load Wav2Vec2 deepfake audio detector."""
        async with self._get_lock("audio"):
            if self.models["audio"] is None and HAS_ML:
                log.info("Loading Audio (Wav2Vec2 Deepfake Detector) on %s...", self.device)
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
        """Lazy load image forensics pipeline (pretrained ViT from HuggingFace + frequency + metadata)."""
        async with self._get_lock("forensics"):
            if self.models["forensics"] is None and HAS_ML:
                self.models["forensics"] = ImageForensicsDetector(
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

    def _get_tts_detector(self):
        if self._tts_detector is None and HAS_TTS_DETECTOR:
            self._tts_detector = TTSArtifactDetector()
            if self._tts_detector.is_available:
                log.info("TTSArtifactDetector initialized (Layer 10)")
            else:
                self._tts_detector = None
        return self._tts_detector

    def _get_codec_detector(self):
        if self._codec_detector is None and HAS_CODEC_DETECTOR:
            self._codec_detector = CodecDetector()
            log.info("CodecDetector initialized (Layer 0: Codec Pre-Analysis)")
        return self._codec_detector

    def _get_voice_clone_detector(self):
        if self._voice_clone_detector is None and HAS_VOICE_CLONE:
            self._voice_clone_detector = VoiceCloneDetector(
                speaker_verifier=self.speaker_verifier,
            )
            log.info("VoiceCloneDetector initialized (Post-Ensemble)")
        return self._voice_clone_detector

    # Phase 8: New audio analyzer getters
    def _get_spectral_continuity(self):
        if self._spectral_continuity is None and HAS_SPECTRAL_CONTINUITY:
            self._spectral_continuity = SpectralContinuityAnalyzer()
            log.info("SpectralContinuityAnalyzer initialized (Phase 8)")
        return self._spectral_continuity

    def _get_phoneme_transition(self):
        if self._phoneme_transition is None and HAS_PHONEME_TRANSITION:
            self._phoneme_transition = PhonemeTransitionAnalyzer()
            log.info("PhonemeTransitionAnalyzer initialized (Phase 8)")
        return self._phoneme_transition

    def _get_room_acoustics(self):
        if self._room_acoustics is None and HAS_ROOM_ACOUSTICS:
            self._room_acoustics = RoomAcousticsAnalyzer()
            log.info("RoomAcousticsAnalyzer initialized (Phase 8)")
        return self._room_acoustics

    # Phase 8: New video analyzer getters
    def _get_face_mesh_analyzer(self):
        if self._face_mesh is None and HAS_FACE_MESH:
            self._face_mesh = FaceMeshConsistencyAnalyzer()
            log.info("FaceMeshConsistencyAnalyzer initialized (Phase 8)")
        return self._face_mesh

    def _get_temporal_freq_analyzer(self):
        if self._temporal_freq is None and HAS_TEMPORAL_FREQ:
            self._temporal_freq = TemporalFrequencyAnalyzer()
            log.info("TemporalFrequencyAnalyzer initialized (Phase 8)")
        return self._temporal_freq

    def _get_pupil_analyzer(self):
        if self._pupil_analyzer is None and HAS_PUPIL:
            self._pupil_analyzer = PupilLightReflexAnalyzer()
            log.info("PupilLightReflexAnalyzer initialized (Phase 8)")
        return self._pupil_analyzer

    # Phase 8: Document forensics getter
    def _get_document_forensics(self):
        if self._document_forensics is None and HAS_DOCUMENT_FORENSICS:
            self._document_forensics = DocumentForensicsDetector()
            log.info("DocumentForensicsDetector initialized (Phase 8)")
        return self._document_forensics

    def _get_ensemble_fusion(self):
        if self._ensemble_fusion is None and HAS_ENSEMBLE:
            self._ensemble_fusion = EnsembleFusion()
            log.info("EnsembleFusion initialized")
        return self._ensemble_fusion

    # ─── Audio Analysis (9-Layer Ensemble) ───

    async def analyze_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Full 10-layer audio forensics analysis.

        Layers:
          1. AST (Wav2Vec2) — neural deepfake audio detector (99.7% eval accuracy)
          2. XLS-R 300M SSL backbone — self-supervised representation analysis
          3. Whisper encoder features — log-mel + encoder hidden state analysis
          4. Prosodic Forensics — jitter, shimmer, HNR, F0, speech rate + modern TTS checks
          5. Biological Signals — breathing patterns, periodicity + template reuse detection
          6. Phase & Formant — group delay, instantaneous frequency, formant analysis
          7. Temporal Consistency — speaker embedding stability (single-shot mode)
          8. TTS Artifact Detection — 8-layer modern TTS analysis (spectral bandwidth,
             silence cleanliness, envelope smoothness, sub-band correlation, harmonics,
             micro-pause regularity, onset transients, pitch micro-dynamics)

        All layer scores are fused via calibrated EnsembleFusion with neural dominance,
        corroboration amplification, and biological/TTS artifact veto system.
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
        codec_info = None

        # ── Layer 0: Codec Detection (Pre-Analysis) ──
        codec_detector = self._get_codec_detector()
        if codec_detector and duration >= 0.5:
            try:
                codec_info = await asyncio.to_thread(codec_detector.analyze, waveform, sr)
                codec_name = codec_info.get("codec", "unknown")
                bandwidth = codec_info.get("bandwidth_hz", 0)
                quality_tier = codec_info.get("quality_tier", "unknown")
                codec_conf = codec_info.get("codec_confidence", 0)
                comfort_noise = codec_info.get("comfort_noise_detected", False)
                codec_score = codec_info.get("score", 0)

                if codec_name != "unknown":
                    status = "warn" if codec_score > 0.5 else "info"
                    forensic_checks.append({
                        "id": "codec_detection",
                        "name": "Audio Codec Detection (Pre-Analysis)",
                        "status": status,
                        "description": (
                            f"Detected codec: {codec_name} ({quality_tier}, {bandwidth:.0f}Hz bandwidth, "
                            f"{codec_conf*100:.0f}% confidence). "
                            f"{'Comfort noise detected. ' if comfort_noise else ''}"
                            f"Detection reliability adjusted for codec artifacts."
                        ),
                    })
                    raw_scores["codec"] = codec_name
                    raw_scores["codec_bandwidth_hz"] = bandwidth
                    raw_scores["codec_quality_tier"] = quality_tier
                    raw_scores["codec_reliability_impact"] = round(codec_score, 4)
                    layers_run.append("codec")

                    # Pass reliability adjustments to the ensemble
                    reliability_adj = codec_info.get("reliability_adjustments", {})
                    if reliability_adj:
                        analyzer_outputs["codec"] = {
                            "score": codec_score,
                            "confidence": codec_conf,
                            "reliability_adjustments": reliability_adj,
                        }
            except Exception as e:
                log.warning("Codec detection failed: %s", e)

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
                        f"Wav2Vec2 Neural Detector: {spoof_prob*100:.1f}% synthetic probability. "
                        f"Fine-tuned for modern TTS/voice-cloning detection."
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

        # ── Layers 4-6b: Run CPU-based signal analysis in parallel ──
        # Prosodic, breathing, phase, and formant analyzers are independent
        # and CPU-bound — running them concurrently saves significant time.

        prosodic = self._get_prosodic()
        breathing = self._get_breathing()
        phase = self._get_phase()
        formant = self._get_formant()

        async def _run_prosodic():
            if prosodic and duration >= 0.5:
                try:
                    return await asyncio.to_thread(prosodic.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Prosodic analysis failed: %s", e)
            return None

        async def _run_breathing():
            if breathing and duration >= 1.0:
                try:
                    return await asyncio.to_thread(breathing.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Breathing detection failed: %s", e)
            return None

        async def _run_phase():
            if phase and duration >= 0.1:
                try:
                    return await asyncio.to_thread(phase.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Phase analysis failed: %s", e)
            return None

        async def _run_formant():
            if formant and duration >= 0.1:
                try:
                    return await asyncio.to_thread(formant.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Formant analysis failed: %s", e)
            return None

        p_result, b_result, ph_result, f_result = await asyncio.gather(
            _run_prosodic(), _run_breathing(), _run_phase(), _run_formant(),
        )

        # Format prosodic results
        if p_result:
            analyzer_outputs["prosodic"] = {
                "score": p_result.get("score", 0.5),
                "confidence": p_result.get("confidence", 0.5),
                "anomalies": p_result.get("anomalies", []),
            }
            p_score = p_result.get("score", 0.5)
            p_anomalies = p_result.get("anomalies", [])
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
            anomaly_text = f" Anomalies: {', '.join(str(a) for a in p_anomalies[:3])}." if p_anomalies else ""
            forensic_checks.append({
                "id": "prosodic_forensics",
                "name": "Prosodic Forensics (F0/Jitter/Shimmer/HNR)",
                "status": status,
                "description": f"Prosodic score: {p_score*100:.1f}% synthetic. {'; '.join(detail_parts)}.{anomaly_text}",
            })
            raw_scores["prosodic_score"] = round(p_score, 4)
            layers_run.append("prosodic")
        elif prosodic:
            forensic_checks.append({"id": "prosodic_forensics", "name": "Prosodic Forensics (F0/Jitter/Shimmer/HNR)", "status": "info", "description": "Prosodic analysis unavailable"})

        # Format breathing results
        if b_result:
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
            anomaly_text = f" Anomalies: {', '.join(str(a) for a in b_anomalies[:3])}." if b_anomalies else ""
            forensic_checks.append({
                "id": "breathing_detection",
                "name": "Biological Signals (Breathing Patterns)",
                "status": status,
                "description": f"Breathing score: {b_score*100:.1f}% synthetic. Detected {breath_count} breaths ({breath_rate:.1f}/min). Natural range: 12-20/min.{anomaly_text}",
            })
            raw_scores["breathing_score"] = round(b_score, 4)
            layers_run.append("breathing")
        elif breathing:
            forensic_checks.append({"id": "breathing_detection", "name": "Biological Signals (Breathing Patterns)", "status": "info", "description": "Breathing analysis unavailable"})

        # Format phase results
        if ph_result:
            analyzer_outputs["phase"] = {
                "score": ph_result.get("score", 0.5),
                "confidence": ph_result.get("confidence", 0.5),
                "anomalies": ph_result.get("anomalies", []),
            }
            ph_score = ph_result.get("score", 0.5)
            ph_anomalies = ph_result.get("anomalies", [])
            status = "fail" if ph_score > 0.6 else "warn" if ph_score > 0.4 else "pass"
            anomaly_text = f" Anomalies: {', '.join(str(a) for a in ph_anomalies[:3])}." if ph_anomalies else ""
            forensic_checks.append({
                "id": "phase_forensics",
                "name": "Phase Domain Forensics (Group Delay/IF)",
                "status": status,
                "description": f"Phase score: {ph_score*100:.1f}% synthetic. Analyzes group delay, instantaneous frequency, and phase coherence — vocoders produce patterns incompatible with natural glottal excitation.{anomaly_text}",
            })
            raw_scores["phase_score"] = round(ph_score, 4)
            layers_run.append("phase")
        elif phase:
            forensic_checks.append({"id": "phase_forensics", "name": "Phase Domain Forensics (Group Delay/IF)", "status": "info", "description": "Phase analysis unavailable"})

        # Format formant results
        if f_result:
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
            anomaly_text = f" Anomalies: {', '.join(str(a) for a in f_anomalies[:3])}." if f_anomalies else ""
            forensic_checks.append({
                "id": "formant_analysis",
                "name": "Formant Analysis (LPC/VTL/CV Boundaries)",
                "status": status,
                "description": f"Formant score: {f_score*100:.1f}% synthetic. VTL consistency: {vtl_consistency:.2f}. Deepfake models fail to replicate formant transitions at consonant-vowel boundaries.{anomaly_text}",
            })
            raw_scores["formant_score"] = round(f_score, 4)
            layers_run.append("formant")
        elif formant:
            forensic_checks.append({"id": "formant_analysis", "name": "Formant Analysis (LPC/VTL/CV Boundaries)", "status": "info", "description": "Formant analysis unavailable"})

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

        # ── Layers 8-11: Run TTS artifacts + Phase 8 audio analyzers in parallel ──
        tts_detector = self._get_tts_detector()
        spectral_cont = self._get_spectral_continuity()
        phoneme_trans = self._get_phoneme_transition()
        room_acous = self._get_room_acoustics()

        async def _run_tts():
            if tts_detector and duration >= 0.5:
                try:
                    return await asyncio.to_thread(tts_detector.analyze, waveform, sr)
                except Exception as e:
                    log.warning("TTS artifact detection failed: %s", e)
            return None

        async def _run_spectral_cont():
            if spectral_cont and duration >= 0.5:
                try:
                    return await asyncio.to_thread(spectral_cont.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Spectral continuity analysis failed: %s", e)
            return None

        async def _run_phoneme():
            if phoneme_trans and duration >= 0.5:
                try:
                    return await asyncio.to_thread(phoneme_trans.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Phoneme transition analysis failed: %s", e)
            return None

        async def _run_room():
            if room_acous and duration >= 1.0:
                try:
                    return await asyncio.to_thread(room_acous.analyze, waveform, sr)
                except Exception as e:
                    log.warning("Room acoustics analysis failed: %s", e)
            return None

        tts_result, sc_result, pt_result, ra_result = await asyncio.gather(
            _run_tts(), _run_spectral_cont(), _run_phoneme(), _run_room(),
        )

        # Format TTS results
        if tts_result:
            tts_score = tts_result.get("score", 0.5)
            tts_anomalies = tts_result.get("anomalies", [])
            tts_layer_scores = tts_result.get("layer_scores", {})
            analyzer_outputs["tts_artifacts"] = {"score": tts_score, "confidence": tts_result.get("confidence", 0.5), "anomalies": tts_anomalies}
            status = "fail" if tts_score > 0.6 else "warn" if tts_score > 0.4 else "pass"
            detail_parts = [f"{n}: {s*100:.0f}%" for n, s in list(tts_layer_scores.items())[:4]]
            anomaly_text = f" Anomalies: {', '.join(str(a) for a in tts_anomalies[:4])}." if tts_anomalies else ""
            forensic_checks.append({
                "id": "tts_artifacts",
                "name": "Modern TTS Artifact Detection (ElevenLabs/XTTS/Bark)",
                "status": status,
                "description": f"TTS artifact score: {tts_score*100:.1f}% synthetic. 8-layer analysis: {'; '.join(detail_parts)}. Detects vocoder spectral smoothness, silence cleanliness, onset transients, pitch micro-dynamics, and sub-band correlation patterns unique to neural TTS systems.{anomaly_text}",
            })
            raw_scores["tts_artifact_score"] = round(tts_score, 4)
            if tts_layer_scores:
                raw_scores["tts_layer_scores"] = {k: round(v, 4) for k, v in tts_layer_scores.items()}
            layers_run.append("tts_artifacts")
        elif tts_detector:
            forensic_checks.append({"id": "tts_artifacts", "name": "Modern TTS Artifact Detection (ElevenLabs/XTTS/Bark)", "status": "info", "description": "TTS artifact analysis unavailable"})

        # Format spectral continuity results
        if sc_result:
            analyzer_outputs["spectral_continuity"] = {"score": sc_result["score"], "confidence": sc_result["confidence"], "anomalies": sc_result.get("anomalies", [])}
            status = "fail" if sc_result["score"] > 0.6 else "warn" if sc_result["score"] > 0.4 else "pass"
            forensic_checks.append({
                "id": "spectral_continuity", "name": "Spectral Continuity Analysis", "status": status,
                "description": f"Spectral envelope evolution: {sc_result['score']*100:.1f}% synthetic (flux={sc_result.get('spectral_flux_score', 0):.2f}, tilt={sc_result.get('tilt_consistency_score', 0):.2f})",
            })
            raw_scores["spectral_continuity_score"] = round(sc_result["score"], 4)
            layers_run.append("spectral_continuity")

        # Format phoneme transition results
        if pt_result:
            analyzer_outputs["phoneme_transition"] = {"score": pt_result["score"], "confidence": pt_result["confidence"], "anomalies": pt_result.get("anomalies", [])}
            status = "fail" if pt_result["score"] > 0.6 else "warn" if pt_result["score"] > 0.4 else "pass"
            forensic_checks.append({
                "id": "phoneme_transition", "name": "Phoneme Transition Analysis", "status": status,
                "description": f"Coarticulation patterns: {pt_result['score']*100:.1f}% synthetic (formant={pt_result.get('formant_transition_score', 0):.2f}, duration={pt_result.get('transition_duration_score', 0):.2f})",
            })
            raw_scores["phoneme_transition_score"] = round(pt_result["score"], 4)
            layers_run.append("phoneme_transition")

        # Format room acoustics results
        if ra_result:
            analyzer_outputs["room_acoustics"] = {"score": ra_result["score"], "confidence": ra_result["confidence"], "anomalies": ra_result.get("anomalies", [])}
            status = "fail" if ra_result["score"] > 0.6 else "warn" if ra_result["score"] > 0.4 else "pass"
            forensic_checks.append({
                "id": "room_acoustics", "name": "Room Acoustics Consistency", "status": status,
                "description": f"Room acoustics: {ra_result['score']*100:.1f}% synthetic (reverb={ra_result.get('reverb_consistency_score', 0):.2f}, noise={ra_result.get('noise_floor_score', 0):.2f})",
            })
            raw_scores["room_acoustics_score"] = round(ra_result["score"], 4)
            layers_run.append("room_acoustics")

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
                    "name": f"10-Layer Ensemble Verdict ({len(layers_run)} layers active)",
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

                # ── Post-Ensemble: Voice Clone Detection ──
                clone_result = None
                voice_clone = self._get_voice_clone_detector()
                if voice_clone and voice_clone.is_available and duration >= 1.0:
                    try:
                        clone_result = await asyncio.to_thread(
                            voice_clone.analyze,
                            waveform, sr,
                            spoof_probability=ensemble_prob,
                            spoof_verdict=verdict,
                        )
                        is_clone = clone_result.get("is_clone", False)
                        clone_conf = clone_result.get("clone_confidence", 0)
                        clone_target = clone_result.get("clone_target")
                        emb_anomaly = clone_result.get("embedding_anomaly_score", 0)
                        temporal_cons = clone_result.get("temporal_consistency", 0.5)

                        if is_clone:
                            clone_status = "fail"
                            clone_desc = (
                                f"VOICE CLONING ATTACK DETECTED ({clone_conf*100:.1f}% confidence). "
                                f"{'Target: ' + clone_target + '. ' if clone_target else ''}"
                                f"Embedding anomaly: {emb_anomaly*100:.1f}%, "
                                f"Temporal consistency: {temporal_cons*100:.1f}%."
                            )
                        elif emb_anomaly > 0.5:
                            clone_status = "warn"
                            clone_desc = (
                                f"Speaker embedding anomalies detected (anomaly: {emb_anomaly*100:.1f}%). "
                                f"Temporal consistency: {temporal_cons*100:.1f}%. Possible voice manipulation."
                            )
                        else:
                            clone_status = "pass"
                            clone_desc = (
                                f"No voice cloning indicators. Embedding anomaly: {emb_anomaly*100:.1f}%, "
                                f"Temporal consistency: {temporal_cons*100:.1f}%."
                            )

                        forensic_checks.append({
                            "id": "voice_clone_detection",
                            "name": "Voice Cloning Cross-Reference",
                            "status": clone_status,
                            "description": clone_desc,
                        })
                        raw_scores["voice_clone_detected"] = is_clone
                        raw_scores["voice_clone_confidence"] = round(clone_conf, 4)
                        raw_scores["embedding_anomaly_score"] = round(emb_anomaly, 4)
                        raw_scores["temporal_consistency"] = round(temporal_cons, 4)
                        if clone_target:
                            raw_scores["clone_target"] = clone_target
                        layers_run.append("voice_clone")
                    except Exception as e:
                        log.warning("Voice clone detection failed: %s", e)

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
                        "layers_total": 12,
                        "duration_seconds": round(duration, 2),
                        "biological_veto": fusion_result.get("biological_veto", False),
                        "veto_reason": fusion_result.get("veto_reason"),
                        "codec_info": codec_info,
                        "clone_result": clone_result,
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
        Full video analysis combining 8 detection engines:
        1. Video quality assessment (compression, motion blur, resolution)
        2. Image forensics V3 pipeline (ViT + CLIP + ELA + noise + pixel stats on frames)
        3. Two-stream deepfake detection (spatial ViT v2 + temporal R3D v2)
        4. AI Video Detector (8-layer temporal analysis: flickering, optical flow,
           noise consistency, edge coherence, texture stability, background breathing,
           frequency spectrum, motion naturalness)
        5. rPPG Heartbeat Analyzer (CHROM algorithm, cardiac signal detection)
        6. CLIP Temporal Drift Detector (semantic embedding trajectory analysis)
        7. Lighting/Shadow Physics Verifier (light direction, shadow, color temp)
        8. Audio-Visual Sync Analyzer (lip sync, speech rhythm, energy correlation)
        9. Micro-Expression Timing Analyzer (blink dynamics, bilateral symmetry)
        10. Forgery Localization (manipulation heatmaps: noise, ELA, face boundary)

        The 9-engine ensemble uses reliability-weighted scoring where each engine's
        contribution is scaled by how confident/consistent its own internal signals
        are. This prevents a weak/disagreeing engine from dragging down the ensemble.
        """
        # ─── Step 1: Assess video quality ───
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

        # ─── Step 2: Run image forensics V3 on video frames ───
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

        # ─── Step 3: Run two-stream deepfake detection ───
        video_model = await self.get_video_model()
        deepfake_result = None
        if video_model:
            try:
                deepfake_result = video_model.predict(file_path)
            except Exception as e:
                log.error("Two-stream video analysis error: %s", e)

        # ─── Steps 4-9: Run independent video analyzers in parallel ───
        # These engines are CPU-bound and don't share state, so they can
        # safely run concurrently via asyncio.to_thread for significant speedup.

        async def _run_ai_video():
            try:
                det = AIVideoDetector(sample_frames=32, verbose=False)
                r = det.analyze(file_path)
                log.info("AI Video Detector: %.1f%% AI probability (%d frames, %d layers)",
                         r["ai_probability"] * 100, r.get("frames_analyzed", 0),
                         len(r.get("layer_scores", {})))
                return r
            except Exception as e:
                log.error("AI Video Detector error: %s", e)
                return None

        async def _run_rppg():
            try:
                r = await asyncio.to_thread(RPPGAnalyzer(verbose=False).analyze, file_path, 15.0)
                log.info("rPPG Analyzer: %.1f%% AI probability (has_pulse=%s, SNR=%.1fdB)",
                         r["ai_probability"] * 100, r.get("has_pulse"), r.get("snr_db", 0))
                return r
            except Exception as e:
                log.error("rPPG Analyzer error: %s", e)
                return None

        async def _run_clip_drift():
            try:
                shared_clip = None
                if forensics and hasattr(forensics, 'clip_detector') and forensics.clip_detector:
                    shared_clip = forensics.clip_detector
                det = CLIPTemporalDriftDetector(clip_model=shared_clip, verbose=False)
                r = await asyncio.to_thread(det.analyze, file_path, 40)
                log.info("CLIP Temporal Drift: %.1f%% AI probability", r["ai_probability"] * 100)
                return r
            except Exception as e:
                log.error("CLIP Temporal Drift error: %s", e)
                return None

        async def _run_lighting():
            try:
                r = await asyncio.to_thread(
                    LightingConsistencyAnalyzer(sample_frames=32, verbose=False).analyze, file_path)
                log.info("Lighting Consistency: %.1f%% AI probability", r["ai_probability"] * 100)
                return r
            except Exception as e:
                log.error("Lighting Consistency error: %s", e)
                return None

        async def _run_av_sync():
            try:
                r = await asyncio.to_thread(AudioVisualSyncAnalyzer(verbose=False).analyze, file_path)
                log.info("AV Sync: %.1f%% AI probability", r["ai_probability"] * 100)
                return r
            except Exception as e:
                log.error("AV Sync Analyzer error: %s", e)
                return None

        async def _run_micro_expr():
            try:
                r = await asyncio.to_thread(
                    MicroExpressionAnalyzer(verbose=False).analyze, file_path, 15.0)
                log.info("Micro-Expression: %.1f%% AI probability", r["ai_probability"] * 100)
                return r
            except Exception as e:
                log.error("Micro-Expression Analyzer error: %s", e)
                return None

        # Run all 6 CPU analyzers concurrently
        (ai_video_result, rppg_result, clip_drift_result,
         lighting_result, av_sync_result, micro_expr_result) = await asyncio.gather(
            _run_ai_video(), _run_rppg(), _run_clip_drift(),
            _run_lighting(), _run_av_sync(), _run_micro_expr(),
        )

        # ─── Step 10: Forgery Localization (Heatmaps) ───
        forgery_result = None
        if HAS_FORGERY_LOCALIZER:
            try:
                # Share ViT model from forensics if available
                shared_vit = None
                if forensics and hasattr(forensics, 'neural_detector') and forensics.neural_detector:
                    shared_vit = forensics.neural_detector
                localizer = ForgeryLocalizer(vit_model=shared_vit, verbose=False)

                # Sample a few frames from the video for localization
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sample_indices = np.linspace(0, max(0, total_frames - 1), min(6, total_frames), dtype=int)
                    localization_frames = []

                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frame_result = localizer.analyze_frame(frame, return_heatmap_image=False)
                            localization_frames.append({
                                "frame_idx": int(idx),
                                "score": frame_result.get("score", 0),
                                "face_boundary_score": frame_result.get("face_boundary_score", 0),
                                "attention_concentration": frame_result.get("attention_concentration", 0),
                                "regions": frame_result.get("manipulation_regions", []),
                            })
                    cap.release()

                    if localization_frames:
                        avg_score = np.mean([f["score"] for f in localization_frames])
                        max_score = max(f["score"] for f in localization_frames)
                        forgery_result = {
                            "avg_score": float(avg_score),
                            "max_score": float(max_score),
                            "frames_analyzed": len(localization_frames),
                            "per_frame": localization_frames,
                        }
                        log.info("Forgery Localization: avg=%.1f%%, max=%.1f%% (%d frames)",
                                 avg_score * 100, max_score * 100, len(localization_frames))
            except Exception as e:
                log.error("Forgery Localization error: %s", e)

        # ── Phase 8: Extract frames for new analyzers ──
        frames = None
        if HAS_ML and (HAS_FACE_MESH or HAS_TEMPORAL_FREQ or HAS_PUPIL):
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    n_sample = min(32, total_frames)
                    sample_indices = np.linspace(0, max(0, total_frames - 1), n_sample, dtype=int)
                    frames = []
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frames.append(frame)
                    cap.release()
                    log.info("Phase 8: Extracted %d frames for advanced video analysis", len(frames))
            except Exception as e:
                log.warning("Phase 8: Frame extraction failed: %s", e)

        # ── Phase 8: Run face mesh, temporal frequency, and pupil analyzers in parallel ──
        face_mesh_result = None
        temporal_freq_result = None
        pupil_result = None

        async def _run_face_mesh():
            analyzer = self._get_face_mesh_analyzer()
            if analyzer and frames is not None and len(frames) >= 5:
                try:
                    r = await asyncio.to_thread(analyzer.analyze, frames)
                    log.info("Face Mesh: score=%.1f%%, %d faces",
                             r.get("score", 0) * 100, r.get("n_faces_tracked", 0))
                    return r
                except Exception as e:
                    log.error("Face Mesh error: %s", e)
            return None

        async def _run_temporal_freq():
            analyzer = self._get_temporal_freq_analyzer()
            if analyzer and frames is not None and len(frames) >= 8:
                try:
                    r = await asyncio.to_thread(analyzer.analyze, frames)
                    log.info("Temporal Frequency: score=%.1f%%", r.get("score", 0) * 100)
                    return r
                except Exception as e:
                    log.error("Temporal Frequency error: %s", e)
            return None

        async def _run_pupil():
            analyzer = self._get_pupil_analyzer()
            if analyzer and frames is not None and len(frames) >= 8:
                try:
                    r = await asyncio.to_thread(analyzer.analyze, frames)
                    log.info("Pupil Reflex: score=%.1f%%", r.get("score", 0) * 100)
                    return r
                except Exception as e:
                    log.error("Pupil error: %s", e)
            return None

        face_mesh_result, temporal_freq_result, pupil_result = await asyncio.gather(
            _run_face_mesh(), _run_temporal_freq(), _run_pupil(),
        )

        # Combine results into unified report
        all_results = [forensics_result, deepfake_result, ai_video_result,
                       rppg_result, clip_drift_result, lighting_result,
                       av_sync_result, micro_expr_result,
                       face_mesh_result, temporal_freq_result, pupil_result]
        if all(r is None for r in all_results):
            return {"error": "No video analysis engines available"}

        report = {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "forensic_checks": [],
            "raw_scores": {},
        }

        # ─── Quality Assessment Check ───
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

        # ─── Add Forensics V3 Checks ───
        if forensics_result and "error" not in forensics_result:
            report["forensic_checks"].extend(forensics_result.get("forensic_checks", []))
            report["raw_scores"].update(forensics_result.get("raw_scores", {}))

        # ─── Add Two-Stream Deepfake Checks ───
        if deepfake_result and "error" not in deepfake_result:
            spatial_prob = deepfake_result["spatial_fake_prob"]
            temporal_prob = deepfake_result["temporal_fake_prob"]

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
            report["raw_scores"]["combined_deepfake"] = deepfake_result["fake_probability"]

        # ─── Add AI Video Detector Checks ───
        if ai_video_result and "error" not in ai_video_result:
            report["forensic_checks"].extend(ai_video_result.get("forensic_checks", []))
            report["raw_scores"]["ai_video_probability"] = ai_video_result["ai_probability"]
            report["raw_scores"]["ai_video_layers"] = ai_video_result.get("layer_scores", {})

        # ─── Add rPPG Heartbeat Checks ───
        if rppg_result and "error" not in rppg_result.get("metrics", {}):
            rppg_prob = rppg_result["ai_probability"]
            has_pulse_raw = rppg_result.get("has_pulse")
            # Convert numpy.bool to Python bool for reliable `is` comparisons
            has_pulse = bool(has_pulse_raw) if has_pulse_raw is not None else None
            bpm = rppg_result.get("estimated_bpm")
            snr = rppg_result.get("snr_db", 0)

            if has_pulse is True:
                rppg_status = "pass"
                rppg_desc = f"Cardiac pulse detected (BPM={bpm}, SNR={snr:.1f}dB) — consistent with real person"
            elif has_pulse is False:
                rppg_status = "fail"
                rppg_desc = f"No cardiac pulse detected (SNR={snr:.1f}dB) — AI-generated faces lack blood flow"
            else:
                rppg_status = "warn"
                rppg_desc = f"Cardiac signal inconclusive (SNR={snr:.1f}dB) — insufficient face data"

            report["forensic_checks"].append({
                "id": "rppg_heartbeat",
                "name": "rPPG Cardiac Pulse Analysis (CHROM)",
                "status": rppg_status,
                "description": rppg_desc,
            })
            report["raw_scores"]["rppg_ai_probability"] = rppg_prob
            report["raw_scores"]["rppg_snr_db"] = snr
            if bpm:
                report["raw_scores"]["rppg_estimated_bpm"] = bpm

        # ─── Add CLIP Temporal Drift Checks ───
        if clip_drift_result and "error" not in clip_drift_result.get("metrics", {}):
            clip_d_prob = clip_drift_result["ai_probability"]
            clip_d_layers = clip_drift_result.get("layer_scores", {})

            if clip_d_prob > 0.65:
                cd_status = "fail"
                cd_desc = f"Abnormal semantic drift in CLIP embedding space ({clip_d_prob*100:.1f}% AI probability)"
            elif clip_d_prob > 0.45:
                cd_status = "warn"
                cd_desc = f"Moderate semantic drift detected ({clip_d_prob*100:.1f}% AI probability)"
            else:
                cd_status = "pass"
                cd_desc = f"Natural semantic trajectory in CLIP space ({clip_d_prob*100:.1f}% AI probability)"

            report["forensic_checks"].append({
                "id": "clip_temporal_drift",
                "name": "CLIP Temporal Drift Analysis",
                "status": cd_status,
                "description": cd_desc,
            })

            # Add individual layer checks
            layer_descriptions = {
                "cosine_drift": "Frame-to-frame cosine similarity drift",
                "trajectory_smoothness": "Embedding trajectory smoothness (velocity/acceleration/jerk)",
                "semantic_stability": "Semantic stability zone micro-drift",
                "directional_consistency": "Embedding movement directional consistency",
                "long_range_coherence": "Long-range temporal coherence (tortuosity/reversion)",
            }
            for layer_name, layer_score in clip_d_layers.items():
                layer_status = "fail" if layer_score > 0.65 else "warn" if layer_score > 0.45 else "pass"
                desc = layer_descriptions.get(layer_name, layer_name)
                report["forensic_checks"].append({
                    "id": f"clip_drift_{layer_name}",
                    "name": f"CLIP Drift: {layer_name.replace('_', ' ').title()}",
                    "status": layer_status,
                    "description": f"{desc}: {layer_score*100:.1f}% anomaly score",
                })

            report["raw_scores"]["clip_drift_ai_probability"] = clip_d_prob
            report["raw_scores"]["clip_drift_layers"] = clip_d_layers

        # ─── Add Lighting/Shadow Physics Checks ───
        if lighting_result and "error" not in lighting_result.get("metrics", {}):
            light_prob = lighting_result["ai_probability"]
            light_layers = lighting_result.get("layer_scores", {})

            if light_prob > 0.65:
                lt_status = "fail"
                lt_desc = f"Physics violations in lighting/shadows ({light_prob*100:.1f}% AI probability)"
            elif light_prob > 0.45:
                lt_status = "warn"
                lt_desc = f"Minor lighting inconsistencies detected ({light_prob*100:.1f}% AI probability)"
            else:
                lt_status = "pass"
                lt_desc = f"Lighting/shadow physics consistent ({light_prob*100:.1f}% AI probability)"

            report["forensic_checks"].append({
                "id": "lighting_physics",
                "name": "Lighting & Shadow Physics Verification",
                "status": lt_status,
                "description": lt_desc,
            })

            layer_descriptions = {
                "light_direction": "Light source direction temporal stability",
                "shadow_consistency": "Shadow direction and edge consistency",
                "color_temperature": "Color temperature (LAB) stability across frames",
                "luminance_distribution": "Luminance histogram distribution consistency",
                "specular_highlights": "Specular highlight position/intensity tracking",
            }
            for layer_name, layer_score in light_layers.items():
                layer_status = "fail" if layer_score > 0.65 else "warn" if layer_score > 0.45 else "pass"
                desc = layer_descriptions.get(layer_name, layer_name)
                report["forensic_checks"].append({
                    "id": f"lighting_{layer_name}",
                    "name": f"Lighting: {layer_name.replace('_', ' ').title()}",
                    "status": layer_status,
                    "description": f"{desc}: {layer_score*100:.1f}% anomaly score",
                })

            report["raw_scores"]["lighting_ai_probability"] = light_prob
            report["raw_scores"]["lighting_layers"] = light_layers

        # ─── Add Audio-Visual Sync Checks ───
        if av_sync_result and "error" not in av_sync_result.get("metrics", {}):
            av_prob = av_sync_result["ai_probability"]
            av_layers = av_sync_result.get("layer_scores", {})

            if av_prob > 0.65:
                av_status = "fail"
                av_desc = f"Audio-visual desynchronization detected ({av_prob*100:.1f}% AI probability)"
            elif av_prob > 0.45:
                av_status = "warn"
                av_desc = f"Minor audio-visual sync anomalies ({av_prob*100:.1f}% AI probability)"
            else:
                av_status = "pass"
                av_desc = f"Audio-visual synchronization normal ({av_prob*100:.1f}% AI probability)"

            report["forensic_checks"].append({
                "id": "av_sync",
                "name": "Audio-Visual Synchronization Analysis",
                "status": av_status,
                "description": av_desc,
            })

            layer_descriptions = {
                "cross_correlation": "Audio-lip cross-correlation (lag and peak strength)",
                "energy_correspondence": "Speech energy ↔ lip movement event matching",
                "speech_rhythm": "Speech rhythm alignment with visual lip activity",
                "silence_stillness": "Silence-stillness correspondence (lips still when no speech)",
            }
            for layer_name, layer_score in av_layers.items():
                layer_status = "fail" if layer_score > 0.65 else "warn" if layer_score > 0.45 else "pass"
                desc = layer_descriptions.get(layer_name, layer_name)
                report["forensic_checks"].append({
                    "id": f"av_sync_{layer_name}",
                    "name": f"AV Sync: {layer_name.replace('_', ' ').title()}",
                    "status": layer_status,
                    "description": f"{desc}: {layer_score*100:.1f}% anomaly score",
                })

            report["raw_scores"]["av_sync_ai_probability"] = av_prob
            report["raw_scores"]["av_sync_layers"] = av_layers

        # ─── Add Micro-Expression Timing Checks ───
        if micro_expr_result and "error" not in micro_expr_result.get("metrics", {}):
            me_prob = micro_expr_result["ai_probability"]
            me_layers = micro_expr_result.get("layer_scores", {})

            if me_prob > 0.65:
                me_status = "fail"
                me_desc = f"Abnormal micro-expression timing ({me_prob*100:.1f}% AI probability)"
            elif me_prob > 0.45:
                me_status = "warn"
                me_desc = f"Minor micro-expression anomalies ({me_prob*100:.1f}% AI probability)"
            else:
                me_status = "pass"
                me_desc = f"Natural micro-expression timing ({me_prob*100:.1f}% AI probability)"

            report["forensic_checks"].append({
                "id": "micro_expression",
                "name": "Physiological Micro-Expression Timing",
                "status": me_status,
                "description": me_desc,
            })

            layer_descriptions = {
                "blink_timing": "Blink rate (15-20/min expected) and interval regularity",
                "blink_dynamics": "Blink asymmetry (fast close ~75ms, slow open ~150ms)",
                "blink_rhythm": "Blink interval variability (CV 0.3-0.7 expected)",
                "micro_movements": "Facial micro-movement naturalness between blinks",
                "bilateral_symmetry": "Left-right eye bilateral symmetry analysis",
            }
            for layer_name, layer_score in me_layers.items():
                layer_status = "fail" if layer_score > 0.65 else "warn" if layer_score > 0.45 else "pass"
                desc = layer_descriptions.get(layer_name, layer_name)
                report["forensic_checks"].append({
                    "id": f"micro_expr_{layer_name}",
                    "name": f"MicroExpr: {layer_name.replace('_', ' ').title()}",
                    "status": layer_status,
                    "description": f"{desc}: {layer_score*100:.1f}% anomaly score",
                })

            report["raw_scores"]["micro_expr_ai_probability"] = me_prob
            report["raw_scores"]["micro_expr_layers"] = me_layers

        # ─── Add Forgery Localization Checks ───
        if forgery_result:
            avg_score = forgery_result["avg_score"]
            max_score = forgery_result["max_score"]
            n_frames = forgery_result["frames_analyzed"]

            if max_score > 0.6:
                fl_status = "fail"
                fl_desc = f"Manipulation evidence found in {n_frames} frames (avg: {avg_score*100:.1f}%, peak: {max_score*100:.1f}%)"
            elif max_score > 0.4:
                fl_status = "warn"
                fl_desc = f"Possible manipulation artifacts in {n_frames} frames (avg: {avg_score*100:.1f}%, peak: {max_score*100:.1f}%)"
            else:
                fl_status = "pass"
                fl_desc = f"No localized manipulation detected across {n_frames} sampled frames"

            report["forensic_checks"].append({
                "id": "forgery_localization",
                "name": "Forgery Localization (Noise + ELA + Face Boundary)",
                "status": fl_status,
                "description": fl_desc,
            })
            report["raw_scores"]["forgery_avg_score"] = round(avg_score, 4)
            report["raw_scores"]["forgery_max_score"] = round(max_score, 4)

        # ─── Phase 8: Add new video engine checks ───
        if face_mesh_result and face_mesh_result.get("confidence", 0) > 0:
            fm_score = face_mesh_result["score"]
            fm_status = "fail" if fm_score > 0.6 else "warn" if fm_score > 0.4 else "pass"
            report["forensic_checks"].append({
                "id": "face_mesh_consistency",
                "name": "Face Mesh Geometry Consistency",
                "status": fm_status,
                "description": (
                    f"Facial geometry consistency: {fm_score*100:.1f}% anomaly "
                    f"({face_mesh_result.get('n_faces_tracked', 0)} faces tracked)"
                ),
            })
            report["raw_scores"]["face_mesh_score"] = round(fm_score, 4)

        if temporal_freq_result and temporal_freq_result.get("confidence", 0) > 0:
            tf_score = temporal_freq_result["score"]
            tf_status = "fail" if tf_score > 0.6 else "warn" if tf_score > 0.4 else "pass"
            report["forensic_checks"].append({
                "id": "temporal_frequency",
                "name": "Temporal Frequency Analysis",
                "status": tf_status,
                "description": (
                    f"Temporal frequency anomaly: {tf_score*100:.1f}% "
                    f"(flicker={temporal_freq_result.get('flicker_score', 0):.2f}, "
                    f"noise_corr={temporal_freq_result.get('noise_correlation_score', 0):.2f})"
                ),
            })
            report["raw_scores"]["temporal_freq_score"] = round(tf_score, 4)

        if pupil_result and pupil_result.get("confidence", 0) > 0:
            pr_score = pupil_result["score"]
            pr_status = "fail" if pr_score > 0.6 else "warn" if pr_score > 0.4 else "pass"
            report["forensic_checks"].append({
                "id": "pupil_light_reflex",
                "name": "Pupil Light Reflex Analysis",
                "status": pr_status,
                "description": (
                    f"Pupil behavior anomaly: {pr_score*100:.1f}% "
                    f"(light_response={pupil_result.get('light_response_score', 0):.2f}, "
                    f"symmetry={pupil_result.get('symmetry_score', 0):.2f})"
                ),
            })
            report["raw_scores"]["pupil_reflex_score"] = round(pr_score, 4)

        # ─── 9-Engine Reliability-Weighted Ensemble ───
        quality_tier = "medium"
        motion_blur = 0.0
        if quality_metrics and "error" not in quality_metrics:
            quality_tier = quality_metrics["quality_tier"]
            motion_blur = quality_metrics.get("motion_blur_score", 0.0)

        QUALITY_CONFIG = {
            "high":     {"conf_cap": 95.0, "fake_thresh": 0.55, "real_thresh": 0.35},
            "medium":   {"conf_cap": 92.0, "fake_thresh": 0.56, "real_thresh": 0.36},
            "low":      {"conf_cap": 82.0, "fake_thresh": 0.60, "real_thresh": 0.32},
            "very_low": {"conf_cap": 72.0, "fake_thresh": 0.65, "real_thresh": 0.30},
        }
        qc = QUALITY_CONFIG.get(quality_tier, QUALITY_CONFIG["medium"])

        # Collect all engine scores with reliability weights
        engine_signals = []  # List of (score, weight, reliability, name)

        # Engine 1: Forensics V3 (frame-level ViT + CLIP + ELA + noise + pixel + freq)
        if forensics_result and "error" not in forensics_result:
            raw = forensics_result.get("raw_scores", {})
            forensics_v3_score = raw.get("v3_combined", 0.5)

            # Reliability: based on consistency of neural scores across frames
            neural_avg = raw.get("neural_avg", 0.5)
            neural_max = raw.get("neural_max", 0.5)
            neural_spread = neural_max - neural_avg
            forensics_reliability = max(0.3, 1.0 - neural_spread * 3)

            # CLIP presence boosts forensics reliability significantly
            if "clip_avg" in raw:
                clip_avg = raw["clip_avg"]
                # CLIP and ViT agreement strengthens signal
                if (clip_avg > 0.55 and neural_avg > 0.55) or (clip_avg < 0.45 and neural_avg < 0.45):
                    forensics_reliability = min(1.0, forensics_reliability + 0.2)

            engine_signals.append((forensics_v3_score, 3.0, forensics_reliability, "forensics_v3"))

        # Engine 2: Two-stream deepfake (face-swap specialist)
        # IMPORTANT: This model was trained on face-swap deepfakes (FaceForensics++),
        # NOT on fully AI-generated video. When spatial and temporal strongly disagree
        # (e.g. spatial=7% temporal=60%), it means the model is confused by content
        # that doesn't match its training distribution. In this case, we neutralize
        # its contribution to avoid it dragging down the ensemble with a wrong signal.
        if deepfake_result and "error" not in deepfake_result:
            spatial_prob = deepfake_result["spatial_fake_prob"]
            temporal_prob = deepfake_result["temporal_fake_prob"]
            twostream_score = deepfake_result["fake_probability"]

            disagreement = abs(spatial_prob - temporal_prob)

            # When spatial/temporal disagree strongly (>0.4), the model is confused.
            # Neutralize toward 0.5 to prevent wrong signals from poisoning ensemble.
            if disagreement > 0.4:
                pull_to_neutral = min(0.8, (disagreement - 0.4) * 2)
                twostream_score = twostream_score * (1 - pull_to_neutral) + 0.5 * pull_to_neutral
                log.info("  Two-stream neutralized: disagreement=%.2f, adjusted score=%.4f",
                         disagreement, twostream_score)

            twostream_reliability = max(0.15, 1.0 - disagreement * 2)

            if motion_blur > 0.4:
                blur_penalty = min(0.5, (motion_blur - 0.4) * 0.83)
                twostream_reliability *= (1.0 - blur_penalty)

            # Low base weight: face-swap model shouldn't override AI-gen detection
            engine_signals.append((twostream_score, 1.0, twostream_reliability, "two_stream"))

        # Engine 3: AI Video Detector (8-layer temporal analysis)
        if ai_video_result and "error" not in ai_video_result:
            ai_video_score = ai_video_result["ai_probability"]
            layer_scores = ai_video_result.get("layer_scores", {})

            # Reliability is based on signal strength AND count
            if layer_scores:
                n_ai_layers = sum(1 for s in layer_scores.values() if s > 0.5)
                n_strong = sum(1 for s in layer_scores.values() if s > 0.65)
                n_very_strong = sum(1 for s in layer_scores.values() if s > 0.8)
                n_total = len(layer_scores)

                # Base reliability from agreement ratio
                if n_ai_layers >= n_total * 0.5:
                    # Half or more layers say AI → reliable
                    ai_video_reliability = 0.5 + (n_ai_layers / n_total) * 0.3
                elif n_ai_layers <= n_total * 0.25:
                    # Very few layers say AI → reliably real
                    ai_video_reliability = 0.5 + ((n_total - n_ai_layers) / n_total) * 0.3
                else:
                    ai_video_reliability = 0.35

                # Strong signals boost reliability (even if not all layers agree,
                # 3 very strong signals is significant evidence)
                if n_strong >= 3:
                    ai_video_reliability = min(1.0, ai_video_reliability + 0.15)
                if n_very_strong >= 2:
                    ai_video_reliability = min(1.0, ai_video_reliability + 0.1)
            else:
                ai_video_reliability = 0.4

            engine_signals.append((ai_video_score, 4.0, ai_video_reliability, "ai_video_detector"))

        # Engine 4: rPPG Heartbeat Analysis
        # Biological signal — if no pulse detected, strong evidence of AI generation.
        # If pulse detected, evidence of real person BUT modern AI can simulate pulse-like
        # patterns. Only trust "no pulse" strongly; "has pulse" with moderate confidence.
        if rppg_result and "error" not in rppg_result.get("metrics", {}):
            rppg_score = rppg_result["ai_probability"]
            has_pulse_raw = rppg_result.get("has_pulse")
            has_pulse = bool(has_pulse_raw) if has_pulse_raw is not None else None
            snr = rppg_result.get("snr_db", 0)

            # Reliability depends on signal quality
            if has_pulse is True and snr > 3.0:
                # Pulse found — moderate reliability (AI can fake pulse-like patterns)
                rppg_reliability = min(0.7, 0.4 + snr / 40.0)
            elif has_pulse is False and snr < 0.5:
                # No pulse at all — very reliable indicator of AI
                rppg_reliability = 0.85
            elif has_pulse is None:
                # Couldn't determine — low reliability
                rppg_reliability = 0.2
            else:
                # Ambiguous signal
                rppg_reliability = 0.35

            # rPPG weight: 2.0 when detecting "real" (may be fooled by AI),
            # 2.5 when detecting "AI" (absence of pulse is harder to fake)
            rppg_weight = 2.5 if rppg_score > 0.5 else 1.8
            engine_signals.append((rppg_score, rppg_weight, rppg_reliability, "rppg_heartbeat"))

        # Engine 5: CLIP Temporal Drift
        # Semantic embedding trajectory — AI-generated videos have unnaturally smooth
        # or erratic trajectories in CLIP space compared to natural video.
        if clip_drift_result and "error" not in clip_drift_result.get("metrics", {}):
            clip_d_score = clip_drift_result["ai_probability"]
            clip_d_layers = clip_drift_result.get("layer_scores", {})

            if clip_d_layers:
                n_anomalous = sum(1 for s in clip_d_layers.values() if s > 0.55)
                n_strong = sum(1 for s in clip_d_layers.values() if s > 0.7)
                n_total = len(clip_d_layers)

                # Reliability based on layer agreement
                agreement_ratio = max(n_anomalous, n_total - n_anomalous) / n_total
                clip_d_reliability = 0.3 + agreement_ratio * 0.5
                if n_strong >= 2:
                    clip_d_reliability = min(1.0, clip_d_reliability + 0.15)
            else:
                clip_d_reliability = 0.3

            engine_signals.append((clip_d_score, 2.5, clip_d_reliability, "clip_temporal_drift"))

        # Engine 6: Lighting/Shadow Physics
        # Physics-based — checks for violations of real-world illumination laws.
        # Less reliable on compressed/low-quality video where artifacts confuse signals.
        if lighting_result and "error" not in lighting_result.get("metrics", {}):
            light_score = lighting_result["ai_probability"]
            light_layers = lighting_result.get("layer_scores", {})

            if light_layers:
                n_violation = sum(1 for s in light_layers.values() if s > 0.55)
                n_strong = sum(1 for s in light_layers.values() if s > 0.7)
                n_total = len(light_layers)

                agreement_ratio = max(n_violation, n_total - n_violation) / n_total
                light_reliability = 0.3 + agreement_ratio * 0.4
                if n_strong >= 2:
                    light_reliability = min(1.0, light_reliability + 0.15)

                # Reduce reliability on heavily compressed video
                if quality_tier in ("low", "very_low"):
                    light_reliability *= 0.7
            else:
                light_reliability = 0.3

            engine_signals.append((light_score, 1.5, light_reliability, "lighting_physics"))

        # Engine 7: Audio-Visual Sync
        # Cross-modal — detects desynchronization between lip movements and speech.
        # Only meaningful when video has speech audio; skipped otherwise.
        if av_sync_result and "error" not in av_sync_result.get("metrics", {}):
            av_score = av_sync_result["ai_probability"]
            av_layers = av_sync_result.get("layer_scores", {})

            # Check if it was actually analyzed (not skipped due to no speech)
            av_note = av_sync_result.get("metrics", {}).get("note", "")
            if "skipped" in av_note.lower():
                # AV sync was skipped (no speech) — don't include in ensemble
                pass
            else:
                if av_layers:
                    n_desync = sum(1 for s in av_layers.values() if s > 0.55)
                    n_strong = sum(1 for s in av_layers.values() if s > 0.7)
                    n_total = len(av_layers)

                    agreement_ratio = max(n_desync, n_total - n_desync) / n_total
                    av_reliability = 0.3 + agreement_ratio * 0.5
                    if n_strong >= 2:
                        av_reliability = min(1.0, av_reliability + 0.15)
                else:
                    av_reliability = 0.3

                engine_signals.append((av_score, 2.0, av_reliability, "av_sync"))

        # Engine 8: Micro-Expression Timing
        # Physiological — checks blink dynamics, bilateral symmetry, micro-movements.
        # Requires face detection; reliability depends on face tracking quality.
        if micro_expr_result and "error" not in micro_expr_result.get("metrics", {}):
            me_score = micro_expr_result["ai_probability"]
            me_layers = micro_expr_result.get("layer_scores", {})

            if me_layers:
                n_abnormal = sum(1 for s in me_layers.values() if s > 0.55)
                n_strong = sum(1 for s in me_layers.values() if s > 0.7)
                n_total = len(me_layers)

                agreement_ratio = max(n_abnormal, n_total - n_abnormal) / n_total
                me_reliability = 0.3 + agreement_ratio * 0.45
                if n_strong >= 2:
                    me_reliability = min(1.0, me_reliability + 0.15)
                if n_strong >= 3:
                    me_reliability = min(1.0, me_reliability + 0.1)
            else:
                me_reliability = 0.3

            engine_signals.append((me_score, 1.5, me_reliability, "micro_expression"))

        # Phase 8: Engine 9 — Face Mesh Consistency
        if face_mesh_result and face_mesh_result.get("confidence", 0) > 0:
            fm_score = face_mesh_result["score"]
            fm_reliability = float(np.clip(face_mesh_result.get("confidence", 0.5), 0.2, 1.0))
            engine_signals.append((fm_score, 1.5, fm_reliability, "face_mesh"))

        # Phase 8: Engine 10 — Temporal Frequency
        if temporal_freq_result and temporal_freq_result.get("confidence", 0) > 0:
            tf_score = temporal_freq_result["score"]
            tf_reliability = float(np.clip(temporal_freq_result.get("confidence", 0.5), 0.2, 1.0))
            engine_signals.append((tf_score, 1.5, tf_reliability, "temporal_frequency"))

        # Phase 8: Engine 11 — Pupil Light Reflex
        if pupil_result and pupil_result.get("confidence", 0) > 0:
            pr_score = pupil_result["score"]
            pr_reliability = float(np.clip(pupil_result.get("confidence", 0.5), 0.2, 1.0))
            engine_signals.append((pr_score, 1.5, pr_reliability, "pupil_reflex"))

        if not engine_signals:
            return report

        # Weighted ensemble: score * base_weight * reliability
        total_effective_weight = 0.0
        weighted_score_sum = 0.0

        for score, base_weight, reliability, name in engine_signals:
            effective_weight = base_weight * reliability
            weighted_score_sum += score * effective_weight
            total_effective_weight += effective_weight
            log.info("  Engine %s: score=%.4f, base_w=%.1f, reliability=%.3f, eff_w=%.3f",
                     name, score, base_weight, reliability, effective_weight)

        final_score = weighted_score_sum / total_effective_weight if total_effective_weight > 0 else 0.5

        # ─── AI-Generation Corroboration Amplification ───
        # When AI Video Detector AND CLIP Temporal Drift both strongly indicate AI,
        # these are two independent, complementary methods (temporal physics vs semantic
        # trajectory) arriving at the same conclusion. This deserves extra confidence.
        engine_scores_by_name = {name: score for score, _, _, name in engine_signals}
        ai_vid_score = engine_scores_by_name.get("ai_video_detector", 0.5)
        clip_drift_score = engine_scores_by_name.get("clip_temporal_drift", 0.5)

        if ai_vid_score > 0.6 and clip_drift_score > 0.6 and final_score > 0.45:
            # Both AI-specific engines agree: strong corroboration
            # These are two fundamentally different approaches (pixel physics vs semantic
            # trajectory) so agreement is very strong evidence
            corr_strength = min(ai_vid_score, clip_drift_score) - 0.6
            corr_boost = min(0.15, corr_strength * 1.5)

            # Extra boost if forensics V3 also leans AI (3-way agreement)
            forensics_score = engine_scores_by_name.get("forensics_v3", 0.5)
            if forensics_score > 0.50:
                corr_boost = min(0.20, corr_boost * 1.4)
                log.info("  3-way AI corroboration (forensics=%.3f also AI-leaning)", forensics_score)

            final_score = final_score + corr_boost * (1 - final_score)
            log.info("  AI corroboration boost (ai_vid=%.3f, clip_drift=%.3f): +%.4f -> %.4f",
                     ai_vid_score, clip_drift_score, corr_boost, final_score)

        # Cross-engine agreement bonus: if all engines lean the same direction,
        # boost the signal (corroborating evidence is stronger than any single engine)
        all_scores = [s for s, _, _, _ in engine_signals]
        n_ai = sum(1 for s in all_scores if s > 0.52)
        n_real = sum(1 for s in all_scores if s < 0.48)
        n_engines = len(all_scores)

        if n_engines >= 2:
            ai_ratio = n_ai / n_engines
            real_ratio = n_real / n_engines

            if ai_ratio >= 0.75 and final_score > 0.5:
                # Strong majority or unanimity → AI
                # Scale boost by agreement strength (75% → 0.04, 100% → 0.10)
                agreement_strength = (ai_ratio - 0.75) / 0.25  # 0 at 75%, 1 at 100%
                boost = min(0.10, (0.04 + agreement_strength * 0.06) * (final_score - 0.5) * 4)
                final_score = final_score + boost * (1 - final_score)
                log.info("  Agreement boost (%d/%d engines → AI, ratio=%.0f%%): +%.4f",
                         n_ai, n_engines, ai_ratio * 100, boost)
            elif real_ratio >= 0.75 and final_score < 0.5:
                # Strong majority or unanimity → real
                agreement_strength = (real_ratio - 0.75) / 0.25
                boost = min(0.10, (0.04 + agreement_strength * 0.06) * (0.5 - final_score) * 4)
                final_score = final_score - boost * final_score
                log.info("  Agreement boost (%d/%d engines → real, ratio=%.0f%%): -%.4f",
                         n_real, n_engines, real_ratio * 100, boost)

        # Reliability-based confidence damping when signals are weak
        avg_reliability = np.mean([r for _, _, r, _ in engine_signals])
        if avg_reliability < 0.4:
            damping = avg_reliability / 0.4
            final_score = 0.5 + (final_score - 0.5) * damping
            log.info("  Low reliability damping (avg_rel=%.3f, damping=%.3f)", avg_reliability, damping)

        log.info("  Final ensemble: %.4f (thresh: fake>%.2f, real<%.2f, cap=%.0f%%)",
                 final_score, qc["fake_thresh"], qc["real_thresh"], qc["conf_cap"])

        report["raw_scores"]["final_ensemble_score"] = round(final_score, 4)
        report["raw_scores"]["avg_reliability"] = round(avg_reliability, 3)
        report["raw_scores"]["engine_count"] = n_engines

        if final_score > qc["fake_thresh"]:
            report["verdict"] = "ai-generated"
            report["confidence"] = round(min(qc["conf_cap"], 55.0 + (final_score - qc["fake_thresh"]) * 200), 1)
        elif final_score < qc["real_thresh"]:
            report["verdict"] = "authentic"
            report["confidence"] = round(min(qc["conf_cap"], 55.0 + (qc["real_thresh"] - final_score) * 200), 1)
        else:
            report["verdict"] = "inconclusive"
            report["confidence"] = round(50.0 + abs(final_score - 0.5) * 100, 1)

        # Track which detectors contributed
        report["detectors_used"] = []
        if forensics_result and "error" not in forensics_result:
            report["detectors_used"].append("forensics_v3")
        if deepfake_result and "error" not in deepfake_result:
            report["detectors_used"].append("two_stream")
        if ai_video_result and "error" not in ai_video_result:
            report["detectors_used"].append("ai_video_detector")
        if quality_metrics and "error" not in quality_metrics:
            report["detectors_used"].append("video_quality")
        if rppg_result and "error" not in rppg_result.get("metrics", {}):
            report["detectors_used"].append("rppg_heartbeat")
        if clip_drift_result and "error" not in clip_drift_result.get("metrics", {}):
            report["detectors_used"].append("clip_temporal_drift")
        if lighting_result and "error" not in lighting_result.get("metrics", {}):
            report["detectors_used"].append("lighting_physics")
        if av_sync_result and "error" not in av_sync_result.get("metrics", {}):
            report["detectors_used"].append("av_sync")
        if micro_expr_result and "error" not in micro_expr_result.get("metrics", {}):
            report["detectors_used"].append("micro_expression")

        # Quality disclaimer for low-quality videos
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


# Global singleton — remote client when INFERENCE_URL is set, local engine otherwise
def _create_engine():
    from .config import INFERENCE_URL, INFERENCE_SECRET
    if INFERENCE_URL:
        from .inference_client import RemoteInferenceClient
        log.info("Using REMOTE inference at %s (no local ML models loaded)", INFERENCE_URL)
        return RemoteInferenceClient(INFERENCE_URL, INFERENCE_SECRET)
    if not HAS_ML:
        log.warning("LOCAL mode but ML dependencies missing — stub responses only")
    else:
        log.info("Using LOCAL inference engine (device selection on first request)")
    return InferenceEngine()


engine = _create_engine()
