"""
Real-Time Speech-to-Text using faster-whisper
==============================================
Transcribes audio chunks in real-time with automatic language detection.
Supports all major Indian languages.

Model: faster-whisper "small" (461M params, ~1GB VRAM)
  - Real-time factor: ~0.3x on RTX 3050 (3x faster than real-time)
  - Supports 99 languages including Hindi, Marathi, Tamil, Telugu,
    Bengali, Gujarati, Kannada, Malayalam, Punjabi, Urdu
  - Automatic language detection

For lower VRAM (if running alongside other models):
  Use "tiny" (74M params, ~150MB VRAM) — less accurate but fits anywhere
  Or "base" (142M params, ~300MB VRAM) — good balance
"""

import logging
import io
import numpy as np
from typing import Optional, Dict, Any

log = logging.getLogger("satyadrishti.transcriber")

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    log.warning("faster-whisper not installed. Real-time transcription unavailable.")


class RealTimeTranscriber:
    """
    Streaming speech-to-text transcriber.

    Usage:
        transcriber = RealTimeTranscriber(model_size="small")
        result = transcriber.transcribe(audio_bytes)
        print(result["text"], result["language"])
    """

    # Model size -> VRAM usage mapping
    MODEL_VRAM = {
        "tiny": "~150MB",
        "base": "~300MB",
        "small": "~1GB",
        "medium": "~2.5GB",  # Won't fit alongside other models on 4GB
    }

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "float16",
    ):
        """
        Args:
            model_size: "tiny", "base", "small", or "medium"
            device: "cuda", "cpu", or "auto"
            compute_type: "float16" (GPU), "int8" (CPU), or "float32"
        """
        self.model = None
        self.model_size = model_size

        if not HAS_WHISPER:
            log.error("faster-whisper not available")
            return

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use int8 on CPU for speed, float16 on GPU
        if device == "cpu":
            compute_type = "int8"

        try:
            log.info(f"Loading faster-whisper '{model_size}' on {device} ({compute_type})...")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            log.info(f"Transcriber ready. VRAM usage: ~{self.MODEL_VRAM.get(model_size, 'unknown')}")
        except Exception as e:
            log.error(f"Failed to load transcriber: {e}")

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio chunk.

        Args:
            audio_data: WAV audio bytes
            language: Language code (e.g., "hi" for Hindi). None = auto-detect.

        Returns:
            dict with "text", "language", "confidence", "segments"
        """
        if not self.model:
            return {"text": "", "language": "unknown", "error": "Transcriber not loaded"}

        try:
            import soundfile as sf

            # Decode WAV bytes
            waveform, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)

            # Skip very short or silent audio
            if len(waveform) < sr * 0.5:  # Less than 0.5 seconds
                return {"text": "", "language": "unknown", "segments": []}

            rms = np.sqrt(np.mean(waveform ** 2))
            if rms < 0.005:  # Very quiet
                return {"text": "", "language": "unknown", "segments": []}

            # Transcribe
            segments, info = self.model.transcribe(
                waveform,
                beam_size=3,  # Lower beam for speed (default 5)
                language=language,
                vad_filter=True,  # Filter out non-speech
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
            )

            # Collect all segments
            text_parts = []
            segment_list = []
            for segment in segments:
                text_parts.append(segment.text.strip())
                segment_list.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                })

            full_text = " ".join(text_parts)

            return {
                "text": full_text,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "segments": segment_list,
            }

        except Exception as e:
            log.error(f"Transcription failed: {e}")
            return {"text": "", "language": "unknown", "error": str(e)}
