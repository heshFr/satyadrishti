"""
Satya Drishti — ONNX Export & INT8 Quantization
=================================================
Exports trained PyTorch models to ONNX format and applies dynamic
INT8 quantization using ONNX Runtime for edge deployment.

Supported models:
  1. Image Forensics ViT-B/16 (image deepfake detection)
  2. Video Spatial ViT-B/16 (face-level spatial artifact detection)
  3. Video Temporal R3D-18 (temporal consistency analysis)
  4. Audio AST (synthetic voice detection)
  5. Text DeBERTaV3 + LoRA (coercion detection via Optimum — separate)
  6. Cross-Attention Fusion Network (multimodal threat assessment)

Usage:
    python -m scripts.export_onnx                    # Export all
    python -m scripts.export_onnx --model spatial    # Export one model
    python -m scripts.export_onnx --benchmark        # Export + benchmark
"""

import os
import sys
import time
import argparse

try:
    import torch
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Missing dependencies: {e}")
    print("Install: pip install onnx onnxruntime")

# ─── Paths ───

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
EXPORT_DIR = os.path.join(MODEL_DIR, "exported")

# Input checkpoints
FORENSICS_CKPT = os.path.join(MODEL_DIR, "image_forensics", "deepfake_vit_b16.pt")
SPATIAL_CKPT = os.path.join(MODEL_DIR, "video", "vit_spatial_v2_best.pt")
TEMPORAL_CKPT = os.path.join(MODEL_DIR, "video", "r3d_temporal_v2_best.pt")
FUSION_CKPT = os.path.join(MODEL_DIR, "fusion", "fusion_network_best.pt")


def _export_and_quantize(
    model: "torch.nn.Module",
    dummy_input: "torch.Tensor",
    onnx_path: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    opset_version: int = 17,
) -> tuple[str, str]:
    """Export a PyTorch model to ONNX and apply INT8 dynamic quantization."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy TorchScript exporter (dynamo fails on some models)
    )

    # PyTorch 2.10+ may export with external data files (.onnx.data).
    # Consolidate into a single .onnx file for portability & quantization.
    ext_data_path = onnx_path + ".data"
    if os.path.exists(ext_data_path):
        print("  Consolidating external data into single ONNX file...")
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx.save(onnx_model, onnx_path)
        # Remove the leftover .data file
        if os.path.exists(ext_data_path):
            os.remove(ext_data_path)
    else:
        onnx_model = onnx.load(onnx_path)

    # Validate
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX export: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

    # INT8 dynamic quantization
    int8_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    print(f"  INT8 quantized: {int8_path} ({os.path.getsize(int8_path) / 1024 / 1024:.1f} MB)")

    return onnx_path, int8_path


# ─── Model Exporters ───


def export_forensics_vit() -> tuple[str | None, str | None]:
    """Export the image forensics ViT-B/16 (HuggingFace ViTForImageClassification)."""
    print("\n[1/5] Image Forensics ViT-B/16")

    try:
        from transformers import ViTForImageClassification

        pretrained_dir = os.path.join(MODEL_DIR, "image_forensics", "pretrained_vit")
        model_source = pretrained_dir if os.path.isdir(pretrained_dir) else "google/vit-base-patch16-224-in21k"

        model = ViTForImageClassification.from_pretrained(
            model_source, num_labels=2, ignore_mismatched_sizes=True,
        )

        if os.path.exists(FORENSICS_CKPT):
            ckpt = torch.load(FORENSICS_CKPT, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"  Loaded fine-tuned weights from {FORENSICS_CKPT}")
        else:
            print(f"  Warning: No fine-tuned weights at {FORENSICS_CKPT}")

        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        out_path = os.path.join(EXPORT_DIR, "forensics", "forensics_vit.onnx")

        return _export_and_quantize(
            model, dummy, out_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        )
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None


def export_spatial_vit() -> tuple[str | None, str | None]:
    """Export the video spatial ViT-B/16."""
    print("\n[2/5] Video Spatial ViT-B/16")

    from engine.video.spatial_vit import DeepfakeViT

    model = DeepfakeViT(num_classes=2, pretrained=False)

    if os.path.exists(SPATIAL_CKPT):
        ckpt = torch.load(SPATIAL_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded weights from {SPATIAL_CKPT}")
    else:
        print(f"  Warning: No weights at {SPATIAL_CKPT}")

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    out_path = os.path.join(EXPORT_DIR, "video", "spatial_vit.onnx")

    return _export_and_quantize(
        model, dummy, out_path,
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
    )


def export_temporal_r3d() -> tuple[str | None, str | None]:
    """Export the video temporal R3D-18."""
    print("\n[3/5] Video Temporal R3D-18")

    from engine.video.temporal_r3d import TemporalR3D

    model = TemporalR3D(num_classes=2, pretrained=False)

    if os.path.exists(TEMPORAL_CKPT):
        ckpt = torch.load(TEMPORAL_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded weights from {TEMPORAL_CKPT}")
    else:
        print(f"  Warning: No weights at {TEMPORAL_CKPT}")

    model.eval()
    # R3D expects (batch, channels, temporal, height, width)
    dummy = torch.randn(1, 3, 16, 224, 224)
    out_path = os.path.join(EXPORT_DIR, "video", "temporal_r3d.onnx")

    return _export_and_quantize(
        model, dummy, out_path,
        input_names=["clips"],
        output_names=["logits"],
        dynamic_axes={"clips": {0: "batch"}, "logits": {0: "batch"}},
    )


def export_ast_audio() -> tuple[str | None, str | None]:
    """
    Export the AST audio spoof detector.
    Uses the HuggingFace transformers ONNX export path.
    """
    print("\n[4/5] Audio AST Spoof Detector")

    try:
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        model_name = "MattyB95/AST-VoxCelebSpoof-Synthetic-Voice-Detection"
        model = ASTForAudioClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model.eval()

        # Create dummy spectrogram input matching the feature extractor output
        dummy_waveform = np.random.randn(32000).astype(np.float32)
        inputs = feature_extractor(dummy_waveform, sampling_rate=16000, return_tensors="pt")
        dummy_input = inputs["input_values"]

        out_path = os.path.join(EXPORT_DIR, "audio", "ast_spoof.onnx")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={"input_values": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )

        # Consolidate external data if present
        ext_data = out_path + ".data"
        if os.path.exists(ext_data):
            print("  Consolidating external data into single ONNX file...")
            onnx_model = onnx.load(out_path, load_external_data=True)
            onnx.save(onnx_model, out_path)
            if os.path.exists(ext_data):
                os.remove(ext_data)
        else:
            onnx_model = onnx.load(out_path)

        onnx.checker.check_model(onnx_model)
        print(f"  ONNX export: {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")

        int8_path = out_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            model_input=out_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )
        print(f"  INT8 quantized: {int8_path} ({os.path.getsize(int8_path) / 1024 / 1024:.1f} MB)")

        return out_path, int8_path

    except Exception as e:
        print(f"  Failed: {e}")
        return None, None


def export_fusion() -> tuple[str | None, str | None]:
    """Export the cross-attention fusion network."""
    print("\n[5/5] Cross-Attention Fusion Network")

    from engine.fusion.cross_attention import MultimodalFusionNetwork

    model = MultimodalFusionNetwork(
        audio_embed_dim=768,
        video_embed_dim=768,
        text_embed_dim=768,
        latent_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=4,
    )

    if os.path.exists(FUSION_CKPT):
        ckpt = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded weights from {FUSION_CKPT}")
    else:
        print(f"  Warning: No weights at {FUSION_CKPT}")

    model.eval()

    # Fusion takes 3 embedding inputs
    dummy_audio = torch.randn(1, 768)
    dummy_video = torch.randn(1, 768)
    dummy_text = torch.randn(1, 768)

    out_path = os.path.join(EXPORT_DIR, "fusion", "fusion_network.onnx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Wrap the model to return only logits (ONNX needs tensor output)
    class FusionWrapper(torch.nn.Module):
        def __init__(self, fusion_model):
            super().__init__()
            self.model = fusion_model

        def forward(self, audio_emb, video_emb, text_emb):
            result = self.model(audio_emb, video_emb, text_emb)
            return result["logits"]

    wrapper = FusionWrapper(model)
    wrapper.eval()

    return _export_and_quantize(
        wrapper,
        (dummy_audio, dummy_video, dummy_text),
        out_path,
        input_names=["audio_emb", "video_emb", "text_emb"],
        output_names=["logits"],
        dynamic_axes={
            "audio_emb": {0: "batch"},
            "video_emb": {0: "batch"},
            "text_emb": {0: "batch"},
            "logits": {0: "batch"},
        },
    )


# ─── Benchmark ───


def benchmark_model(onnx_path: str, dummy_inputs: dict, num_runs: int = 50):
    """Benchmark ONNX model inference latency on CPU."""
    if not onnx_path or not os.path.exists(onnx_path):
        return

    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        name = os.path.basename(onnx_path)
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  {name}: {size_mb:.1f} MB, SKIPPED (runtime error: {type(e).__name__})")
        return

    # Warmup
    for _ in range(5):
        session.run(None, dummy_inputs)

    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, dummy_inputs)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    name = os.path.basename(onnx_path)
    print(f"  {name}: {size_mb:.1f} MB, {avg_ms:.1f} ms/inference")


# ─── Main ───


EXPORTERS = {
    "forensics": export_forensics_vit,
    "spatial": export_spatial_vit,
    "temporal": export_temporal_r3d,
    "audio": export_ast_audio,
    "fusion": export_fusion,
}


def main():
    if not HAS_DEPS:
        print("Cannot run without PyTorch, ONNX, and ONNX Runtime.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Export Satya Drishti models to ONNX + INT8")
    parser.add_argument("--model", choices=list(EXPORTERS.keys()), help="Export a specific model (default: all)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks after export")
    args = parser.parse_args()

    print("=" * 60)
    print("Satya Drishti — ONNX Export & INT8 Quantization")
    print("=" * 60)

    results = {}

    if args.model:
        name = args.model
        fp32, int8 = EXPORTERS[name]()
        results[name] = (fp32, int8)
    else:
        for name, exporter in EXPORTERS.items():
            try:
                fp32, int8 = exporter()
                results[name] = (fp32, int8)
            except Exception as e:
                print(f"  FAILED: {e}")
                results[name] = (None, None)

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for name, (fp32, int8) in results.items():
        if fp32 and os.path.exists(fp32):
            fp32_mb = os.path.getsize(fp32) / (1024 * 1024)
            int8_mb = os.path.getsize(int8) / (1024 * 1024) if int8 and os.path.exists(int8) else 0
            reduction = (1 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 and int8_mb > 0 else 0
            print(f"  {name:12s}: FP32 {fp32_mb:>7.1f} MB → INT8 {int8_mb:>7.1f} MB ({reduction:.0f}% smaller)")
        else:
            print(f"  {name:12s}: SKIPPED")

    # Benchmarks
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Inference Benchmarks (CPU, 50 runs)")
        print("=" * 60)

        bench_inputs = {
            "forensics": {"pixel_values": np.random.randn(1, 3, 224, 224).astype(np.float32)},
            "spatial": {"frames": np.random.randn(1, 3, 224, 224).astype(np.float32)},
            "temporal": {"clips": np.random.randn(1, 3, 16, 224, 224).astype(np.float32)},
            "fusion": {
                "audio_emb": np.random.randn(1, 768).astype(np.float32),
                "video_emb": np.random.randn(1, 768).astype(np.float32),
                "text_emb": np.random.randn(1, 768).astype(np.float32),
            },
        }

        for name, (fp32, int8) in results.items():
            if name in bench_inputs:
                print(f"\n  {name}:")
                benchmark_model(fp32, bench_inputs[name])
                benchmark_model(int8, bench_inputs[name])

    # Note about text model
    print("\n  Note: DeBERTaV3 text model export is best done via HuggingFace Optimum:")
    print("    optimum-cli export onnx --model models/text/deberta_coercion_lora/best_model/ models/exported/text/")


if __name__ == "__main__":
    main()
