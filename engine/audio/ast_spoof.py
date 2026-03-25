"""
Audio Spoof Detection — Pretrained AST (Audio Spectrogram Transformer)
=======================================================================
Uses MattyB95/AST-VoxCelebSpoof-Synthetic-Voice-Detection from HuggingFace,
a fine-tuned AST model for synthetic voice detection.

The model is based on MIT/ast-finetuned-audioset-10-10-0.4593 and produces
2-class predictions (bonafide/spoof) with 768-dim embeddings for fusion.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ASTSpoofDetector(nn.Module):
    """
    Audio Spectrogram Transformer for synthetic voice detection.

    Wraps the HuggingFace AST model with an interface matching
    the fusion network's expectations.

    Args:
        model_name: HuggingFace model identifier
        embed_dim: Embedding dimension (768 for AST-base)
    """

    MODEL_NAME = "MattyB95/AST-VoxCelebSpoof-Synthetic-Voice-Detection"

    def __init__(self, model_name: str = None, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self._model_name = model_name or self.MODEL_NAME
        self._loaded = False

        if HAS_TRANSFORMERS:
            self._load_model()

    def _load_model(self):
        """Load the pretrained AST model and feature extractor."""
        self.ast_model = ASTForAudioClassification.from_pretrained(
            self._model_name
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._model_name
        )
        self._loaded = True

        # Cache label mapping
        self.id2label = self.ast_model.config.id2label
        self.num_labels = self.ast_model.config.num_labels

    @property
    def sample_rate(self) -> int:
        """Expected audio sample rate."""
        if self._loaded:
            return self.feature_extractor.sampling_rate
        return 16000

    def preprocess(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Preprocess raw audio waveform for the AST model.

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
        )
        return inputs

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            input_values: Preprocessed spectrogram input from feature_extractor
                          Shape: (batch, time_steps, freq_bins)

        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.ast_model(input_values=input_values)
        return outputs.logits

    def extract_embedding(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate embedding for Cross-Modal Fusion.

        Uses the [CLS] token output from the AST transformer.

        Args:
            input_values: Preprocessed spectrogram input

        Returns:
            embedding: (batch, 768)
        """
        outputs = self.ast_model.audio_spectrogram_transformer(input_values)
        # Last hidden state, [CLS] token is at position 0
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Full inference pipeline: waveform → prediction.

        Args:
            waveform: Raw audio array
            sample_rate: Sample rate in Hz

        Returns:
            dict with 'label', 'confidence', 'probabilities'
        """
        inputs = self.preprocess(waveform, sample_rate)
        device = next(self.parameters()).device
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            logits = self.forward(input_values)
            probs = torch.softmax(logits, dim=-1)

        pred_idx = probs.argmax(dim=-1).item()
        return {
            "label": self.id2label.get(pred_idx, str(pred_idx)),
            "confidence": probs[0, pred_idx].item(),
            "probabilities": {
                self.id2label.get(i, str(i)): p.item()
                for i, p in enumerate(probs[0])
            },
            "is_spoof": "spoof" in self.id2label.get(pred_idx, "").lower()
                        or "fake" in self.id2label.get(pred_idx, "").lower()
                        or "synthetic" in self.id2label.get(pred_idx, "").lower(),
        }
