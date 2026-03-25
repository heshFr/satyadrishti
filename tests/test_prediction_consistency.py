"""
Satya Drishti -- Prediction Consistency Tests
=============================================
Verifies that the same input produces the same output across multiple runs.
This catches non-deterministic behavior from face detection, TTA, etc.
"""
import pytest
import os

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from engine.image_forensics.detector import ImageForensicsDetector
    HAS_FORENSICS = True
except ImportError:
    HAS_FORENSICS = False


@pytest.mark.skipif(not HAS_FORENSICS or not HAS_TORCH, reason="Forensics not available")
class TestPredictionConsistency:
    """Verify same input -> same output across multiple runs."""

    def _create_test_image(self, path: str, kind: str = "photo"):
        """Create a synthetic test image."""
        import cv2
        if kind == "photo":
            # Create a realistic-ish photo pattern (gradients + noise)
            img = np.random.RandomState(42).randint(50, 200, (512, 512, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (15, 15), 5)
        else:
            # Create a flat graphic/screenshot
            img = np.ones((512, 512, 3), dtype=np.uint8) * 200
            cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), -1)
            cv2.putText(img, "TEST", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        cv2.imwrite(path, img)
        return path

    def test_same_image_same_verdict(self, tmp_path):
        """Running analysis on the same image twice must produce the same verdict."""
        detector = ImageForensicsDetector()
        img_path = str(tmp_path / "test_photo.jpg")
        self._create_test_image(img_path, "photo")

        results = []
        for _ in range(3):
            result = detector.analyze(img_path)
            results.append(result["verdict"])

        assert len(set(results)) == 1, (
            f"Verdict changed across runs: {results}. "
            "This indicates non-deterministic behavior in the pipeline."
        )

    def test_non_photo_not_flagged_as_deepfake(self, tmp_path):
        """A plain screenshot/graphic should not be flagged as ai-generated
        since the neural model was not trained on this type of content."""
        detector = ImageForensicsDetector()
        img_path = str(tmp_path / "test_graphic.png")
        self._create_test_image(img_path, "graphic")

        result = detector.analyze(img_path)
        # A synthetic graphic should either be authentic or inconclusive,
        # never ai-generated with high confidence (that would be a false positive)
        if result["verdict"] == "ai-generated":
            assert result["confidence"] < 80.0, (
                f"Plain graphic falsely flagged as ai-generated with {result['confidence']}% confidence"
            )
