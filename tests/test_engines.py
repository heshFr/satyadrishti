"""
Satya Drishti — ML Engine Unit Tests
======================================
Tests model architectures produce correct output shapes and formats.
Skips tests when ML dependencies or model weights aren't available.
"""

import pytest
import os

# Check if ML dependencies are available
try:
    import torch
    import numpy as np

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Check individual module availability
try:
    from engine.video.spatial_vit import DeepfakeViT

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

try:
    from engine.video.temporal_r3d import TemporalR3D

    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False

try:
    from engine.text.coercion_detector import CoercionDetector, COERCION_LABELS

    HAS_TEXT = True
except ImportError:
    HAS_TEXT = False

try:
    from engine.fusion.cross_attention import MultimodalFusionNetwork

    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False

try:
    from engine.audio.ast_spoof import ASTSpoofDetector

    HAS_AST = True
except ImportError:
    HAS_AST = False


# ─── Spatial ViT-B/16 (Video) ───


@pytest.mark.skipif(not HAS_SPATIAL or not HAS_TORCH, reason="Spatial ViT dependencies missing")
def test_spatial_vit_forward():
    """ViT-B/16 produces correct output shape for face crops."""
    model = DeepfakeViT(pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"
    probs = torch.softmax(logits, dim=1)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


@pytest.mark.skipif(not HAS_SPATIAL or not HAS_TORCH, reason="Spatial ViT dependencies missing")
def test_spatial_vit_embedding():
    """ViT-B/16 extracts 768d embeddings for fusion."""
    model = DeepfakeViT(pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        emb = model.extract_embedding(dummy_input)

    assert emb.shape == (1, 768), f"Expected (1, 768), got {emb.shape}"


# ─── Temporal R3D-18 (Video) ───


@pytest.mark.skipif(not HAS_TEMPORAL or not HAS_TORCH, reason="Temporal R3D dependencies missing")
def test_temporal_r3d_forward():
    """R3D-18 produces correct output shape for 16-frame clips."""
    model = TemporalR3D(pretrained=False)
    model.eval()

    # (batch, channels, temporal_depth, height, width)
    dummy_input = torch.randn(2, 3, 16, 224, 224)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"


@pytest.mark.skipif(not HAS_TEMPORAL or not HAS_TORCH, reason="Temporal R3D dependencies missing")
def test_temporal_r3d_embedding():
    """R3D-18 extracts 512d embeddings for fusion."""
    model = TemporalR3D(pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 16, 224, 224)

    with torch.no_grad():
        emb = model.extract_embedding(dummy_input)

    assert emb.shape == (1, 512), f"Expected (1, 512), got {emb.shape}"


# ─── Text Coercion Detector (DeBERTaV3 + LoRA) ───


@pytest.mark.skipif(not HAS_TEXT or not HAS_TORCH, reason="Text engine dependencies missing")
def test_text_coercion_predict():
    """CoercionDetector.predict() returns dict with label, confidence, probabilities."""
    model = CoercionDetector()

    result = model.predict("URGENT: Your bank account will be closed. Send money now.")

    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["label"] in COERCION_LABELS.values()
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 4


@pytest.mark.skipif(not HAS_TEXT or not HAS_TORCH, reason="Text engine dependencies missing")
def test_text_coercion_embedding():
    """CoercionDetector extracts embeddings for fusion."""
    model = CoercionDetector()

    emb = model.extract_embedding("Test input text.")

    assert isinstance(emb, torch.Tensor)
    assert emb.dim() == 2
    assert emb.shape[0] == 1
    # DeBERTaV3-base has 768d hidden size
    assert emb.shape[1] == 768


# ─── Cross-Attention Fusion Network ───


@pytest.mark.skipif(not HAS_FUSION or not HAS_TORCH, reason="Fusion dependencies missing")
def test_fusion_forward():
    """Fusion network produces correct output shapes."""
    model = MultimodalFusionNetwork(
        audio_embed_dim=768,
        video_embed_dim=768,
        text_embed_dim=768,
        latent_dim=256,
        num_classes=4,
    )
    model.eval()

    a_emb = torch.randn(2, 768)
    v_emb = torch.randn(2, 768)
    t_emb = torch.randn(2, 768)

    with torch.no_grad():
        result = model(a_emb, v_emb, t_emb)

    assert result["logits"].shape == (2, 4)
    assert result["fused_embedding"].shape == (2, 256)
    assert "modality_weights" in result


@pytest.mark.skipif(not HAS_FUSION or not HAS_TORCH, reason="Fusion dependencies missing")
def test_fusion_probabilities():
    """Fusion logits produce valid probability distributions."""
    model = MultimodalFusionNetwork(
        audio_embed_dim=768,
        video_embed_dim=768,
        text_embed_dim=768,
        num_classes=4,
    )
    model.eval()

    with torch.no_grad():
        result = model(torch.randn(1, 768), torch.randn(1, 768), torch.randn(1, 768))
        probs = torch.softmax(result["logits"], dim=-1)

    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)


# ─── AST Audio Spoof Detector ───


@pytest.mark.skipif(not HAS_AST or not HAS_TORCH, reason="AST dependencies missing")
def test_ast_spoof_predict():
    """AST model predicts on synthetic waveform."""
    model = ASTSpoofDetector()
    model.eval()

    # Generate 2s of random audio at 16kHz
    waveform = np.random.randn(32000).astype(np.float32)
    result = model.predict(waveform, sample_rate=16000)

    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert "is_spoof" in result
    assert 0 <= result["confidence"] <= 1


@pytest.mark.skipif(not HAS_AST or not HAS_TORCH, reason="AST dependencies missing")
def test_ast_spoof_embedding():
    """AST model extracts 768d embeddings for fusion."""
    model = ASTSpoofDetector()
    model.eval()

    waveform = np.random.randn(32000).astype(np.float32)
    inputs = model.preprocess(waveform, 16000)
    emb = model.extract_embedding(inputs["input_values"])

    assert emb.shape[1] == 768
