"""
Module 2 — Visual Artifact & Biometric Detector
=================================================
Two-stream deepfake detector:
  Stream A (Spatial): Vision Transformer (ViT) for boundary blending artifacts
  Stream B (Temporal): I3D/X3D + rPPG for physiological inconsistency detection

Target Datasets: FaceForensics++, DFDC, WildDeepfake

Extended with 6 specialized forensic analyzers:
  - AIVideoDetector: 8-layer pixel/temporal forensics for fully AI-generated video
  - RPPGAnalyzer: Remote photoplethysmography (cardiac pulse detection)
  - CLIPTemporalDriftDetector: CLIP semantic drift across frames
  - LightingConsistencyAnalyzer: Physics-based lighting/shadow consistency
  - AudioVisualSyncAnalyzer: Lip-sync and audio-visual correspondence
  - MicroExpressionAnalyzer: Blink dynamics and facial micro-expression timing
"""

from .ai_video_detector import AIVideoDetector
from .rppg_analyzer import RPPGAnalyzer
from .clip_temporal_drift import CLIPTemporalDriftDetector
from .lighting_consistency import LightingConsistencyAnalyzer
from .av_sync_analyzer import AudioVisualSyncAnalyzer
from .micro_expression_analyzer import MicroExpressionAnalyzer
from .face_mesh_analyzer import FaceMeshConsistencyAnalyzer
from .temporal_frequency import TemporalFrequencyAnalyzer
from .pupil_analyzer import PupilLightReflexAnalyzer

__all__ = [
    "AIVideoDetector",
    "RPPGAnalyzer",
    "CLIPTemporalDriftDetector",
    "LightingConsistencyAnalyzer",
    "AudioVisualSyncAnalyzer",
    "MicroExpressionAnalyzer",
    # Phase 8
    "FaceMeshConsistencyAnalyzer",
    "TemporalFrequencyAnalyzer",
    "PupilLightReflexAnalyzer",
]
