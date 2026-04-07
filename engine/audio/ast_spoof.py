"""
Audio Deepfake Detection — Pretrained Wav2Vec2
===============================================
Uses MelodyMachine/Deepfake-audio-detection-V2 from HuggingFace,
a fine-tuned Wav2Vec2 model for synthetic voice detection (99.7% eval accuracy).

Wav2Vec2 leverages self-supervised learning on 960h of unlabeled speech,
then fine-tuned on deepfake audio datasets for binary classification.
This approach detects subtle artifacts from modern TTS/voice-cloning
systems (ElevenLabs, OpenAI TTS, etc.) that older AST-based models miss.

Replaces the previous MattyB95/AST-VoxCelebSpoof model which was trained
only on older vocoder spoofing methods and could not generalize to
modern neural TTS outputs.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ASTSpoofDetector(nn.Module):
    """
    Wav2Vec2-based audio deepfake detector.

    Replaces the older AST-VoxCelebSpoof model with a Wav2Vec2 model
    fine-tuned specifically for modern deepfake audio detection.
    Class name kept as ASTSpoofDetector for backward compatibility
    with the inference engine.

    Args:
        model_name: HuggingFace model identifier
        embed_dim: Embedding dimension (768 for Wav2Vec2-base)
    """

    MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"

    def __init__(self, model_name: str = None, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self._model_name = model_name or self.MODEL_NAME
        self._loaded = False
        self._fake_idx = None  # Index of the "fake/spoof" class

        if HAS_TRANSFORMERS:
            self._load_model()

    def _load_model(self):
        """Load the pretrained Wav2Vec2 model and feature extractor."""
        self.w2v_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self._model_name,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._model_name
        )
        self._loaded = True

        # Cache label mapping
        self.id2label = self.w2v_model.config.id2label
        self.num_labels = self.w2v_model.config.num_labels

        # Auto-detect which label index is "fake/spoof"
        self._fake_idx = self._find_fake_index()

    def _find_fake_index(self) -> int:
        """Find the label index corresponding to fake/spoof audio."""
        fake_keywords = {"spoof", "fake", "synthetic", "deepfake", "generated", "ai"}
        for idx, label in self.id2label.items():
            idx = int(idx)
            if any(kw in str(label).lower() for kw in fake_keywords):
                return idx
        # Default: assume last class is fake (common convention)
        return self.num_labels - 1

    @property
    def sample_rate(self) -> int:
        """Expected audio sample rate."""
        if self._loaded:
            return self.feature_extractor.sampling_rate
        return 16000

    def preprocess(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Preprocess raw audio waveform for the Wav2Vec2 model.

        Args:
            waveform: Raw audio samples, shape (num_samples,) or (channels, num_samples)
            sample_rate: Audio sample rate in Hz

        Returns:
            Dict with 'input_values' tensor ready for the model
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Install transformers package.")

        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        # Use the model's feature extractor
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            input_values: Preprocessed audio input from feature_extractor
                          Shape: (batch, num_samples)

        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.w2v_model(input_values=input_values)
        return outputs.logits

    def extract_embedding(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate embedding for Cross-Modal Fusion.

        Uses the mean-pooled last hidden state from Wav2Vec2's transformer.

        Args:
            input_values: Preprocessed audio input

        Returns:
            embedding: (batch, 768)
        """
        outputs = self.w2v_model(
            input_values=input_values,
            output_hidden_states=True,
        )
        # Mean pool across time dimension of last hidden state
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, 768)
        embedding = last_hidden.mean(dim=1)  # (batch, 768)
        return embedding

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Full inference pipeline: waveform → prediction.

        Args:
            waveform: Raw audio array
            sample_rate: Sample rate in Hz

        Returns:
            dict with 'label', 'confidence', 'probabilities', 'is_spoof'
        """
        inputs = self.preprocess(waveform, sample_rate)
        device = next(self.parameters()).device
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            logits = self.forward(input_values)
            probs = torch.softmax(logits, dim=-1)

        pred_idx = probs.argmax(dim=-1).item()
        label = self.id2label.get(pred_idx, self.id2label.get(str(pred_idx), str(pred_idx)))

        # Determine if prediction is spoof using auto-detected fake index
        is_spoof = (pred_idx == self._fake_idx)

        return {
            "label": label,
            "confidence": probs[0, pred_idx].item(),
            "probabilities": {
                self.id2label.get(i, self.id2label.get(str(i), str(i))): p.item()
                for i, p in enumerate(probs[0])
            },
            "is_spoof": is_spoof,
        }
