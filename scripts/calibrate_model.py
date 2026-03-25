"""
Satya Drishti — Model Calibration Diagnostic
=============================================
Tests the retrained ViT on a range of synthetic inputs to map
its actual output distribution. Use these numbers to set thresholds.

Run: python scripts/calibrate_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import json
from pathlib import Path


def create_test_images(output_dir: str):
    """Create diverse test images covering the range of inputs the model will see."""
    os.makedirs(output_dir, exist_ok=True)
    images = {}

    # 1. Random noise (should be "real" — no AI generated this)
    noise = np.random.randint(40, 220, (512, 512, 3), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (15, 15), 5)
    path = os.path.join(output_dir, "noise_clean.jpg")
    cv2.imwrite(path, noise, [cv2.IMWRITE_JPEG_QUALITY, 95])
    images["noise_clean"] = path

    # 2. Same noise but WhatsApp-compressed
    path = os.path.join(output_dir, "noise_whatsapp.jpg")
    cv2.imwrite(path, cv2.resize(noise, (800, 800)), [cv2.IMWRITE_JPEG_QUALITY, 65])
    images["noise_whatsapp"] = path

    # 3. Same noise but heavily compressed (Q=30)
    path = os.path.join(output_dir, "noise_heavy_compress.jpg")
    cv2.imwrite(path, noise, [cv2.IMWRITE_JPEG_QUALITY, 30])
    images["noise_heavy_compress"] = path

    # 4. Flat color screenshot
    flat = np.ones((512, 512, 3), dtype=np.uint8) * 200
    cv2.rectangle(flat, (50, 50), (300, 300), (0, 0, 255), -1)
    cv2.putText(flat, "TEST", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    path = os.path.join(output_dir, "screenshot.png")
    cv2.imwrite(path, flat)
    images["screenshot"] = path

    # 5. Smooth gradient (common in AI art)
    gradient = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        gradient[i, :, 0] = int(i / 512 * 255)  # Blue gradient
        gradient[i, :, 1] = int((512 - i) / 512 * 180)  # Green inverse
        gradient[:, i, 2] = int(i / 512 * 200)  # Red gradient
    gradient = cv2.GaussianBlur(gradient, (31, 31), 10)
    path = os.path.join(output_dir, "smooth_gradient.jpg")
    cv2.imwrite(path, gradient, [cv2.IMWRITE_JPEG_QUALITY, 95])
    images["smooth_gradient"] = path

    # 6. Natural-ish photo pattern (textured with realistic noise)
    base = np.random.randint(80, 180, (512, 512, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (7, 7), 2)
    # Add sensor-like noise
    sensor_noise = np.random.normal(0, 8, base.shape).astype(np.float32)
    natural = np.clip(base.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)
    path = os.path.join(output_dir, "natural_texture.jpg")
    cv2.imwrite(path, natural, [cv2.IMWRITE_JPEG_QUALITY, 92])
    images["natural_texture"] = path

    # 7. Same natural but Instagram-compressed
    path = os.path.join(output_dir, "natural_instagram.jpg")
    small = cv2.resize(natural, (1080, 1080))
    cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 72])
    images["natural_instagram"] = path

    return images


def run_calibration():
    from engine.image_forensics.vit_detector import ViTDetector

    print("=" * 60)
    print("SATYA DRISHTI — Model Calibration Diagnostic")
    print("=" * 60)

    detector = ViTDetector(
        pretrained_dir="models/image_forensics/pretrained_vit",
        weights_path="models/image_forensics/deepfake_vit_b16.pt",
    )

    output_dir = "_calibration_test"
    images = create_test_images(output_dir)

    print(f"\nTesting {len(images)} images...\n")
    print(f"{'Image':<25} {'Neural Score':>13} {'TTA Score':>10} {'TTA Std':>8}")
    print("-" * 60)

    results = {}
    for name, path in images.items():
        img = cv2.imread(path)
        if img is None:
            continue

        # Raw single prediction
        raw_score, _ = detector.predict(img)

        # TTA prediction
        tta_score, tta_details = detector.predict_tta(img)
        tta_std = tta_details.get("std_fake_probability", 0)

        results[name] = {
            "raw": round(raw_score, 4),
            "tta": round(tta_score, 4),
            "std": round(tta_std, 4),
        }

        print(f"{name:<25} {raw_score:>13.4f} {tta_score:>10.4f} {tta_std:>8.4f}")

    # Summary statistics
    all_raw = [r["raw"] for r in results.values()]
    all_tta = [r["tta"] for r in results.values()]

    print(f"\n{'='*60}")
    print(f"DISTRIBUTION SUMMARY (for threshold calibration)")
    print(f"{'='*60}")
    print(f"Raw scores:  min={min(all_raw):.4f}  max={max(all_raw):.4f}  mean={np.mean(all_raw):.4f}  std={np.std(all_raw):.4f}")
    print(f"TTA scores:  min={min(all_tta):.4f}  max={max(all_tta):.4f}  mean={np.mean(all_tta):.4f}  std={np.std(all_tta):.4f}")
    print(f"\nIMPORTANT: All test images are REAL (synthetic noise/textures, not AI-generated).")
    print(f"Any score above these ranges on actual content = likely AI-generated.")
    print(f"\nRecommended thresholds based on this distribution:")
    print(f"  FAKE threshold: {max(all_tta) + 0.02:.3f} (above max real score + margin)")
    print(f"  REAL threshold: {min(all_tta) - 0.05:.3f} (below min real score - margin)")
    print(f"  Gray zone: {min(all_tta) - 0.05:.3f} to {max(all_tta) + 0.02:.3f}")

    # Save results for reference
    with open(os.path.join(output_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/calibration_results.json")

    # Cleanup test images
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    print("Test images cleaned up.")


if __name__ == "__main__":
    run_calibration()
