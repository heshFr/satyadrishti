"""
Uncertainty Quantification Module
===================================
Provides calibrated uncertainty estimation for ensemble detection systems.

Instead of just outputting a score, this module quantifies HOW CERTAIN
the system is about that score. This is critical for:

1. **Flagging borderline cases**: When detectors disagree, the system
   should say "uncertain" rather than giving a wrong confident answer.

2. **Out-of-distribution detection**: When input doesn't match anything
   the system has seen, uncertainty should be high.

3. **Confidence calibration**: Ensuring that "80% confident" means
   the system is correct 80% of the time.

Techniques used:
- Ensemble disagreement (weighted variance)
- Score distribution analysis
- Mutual information between detectors
- Calibration correction (temperature scaling)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """Quantifies detection uncertainty from ensemble outputs."""

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Calibration temperature. >1 reduces confidence,
                <1 increases confidence. Set via calibration on held-out data.
        """
        self.temperature = temperature

    def quantify(self, scores: dict, weights: dict = None) -> dict:
        """
        Compute uncertainty metrics from multiple detector scores.

        Args:
            scores: Dict of detector_name → score (0-1, higher = more AI).
            weights: Dict of detector_name → weight (importance).

        Returns:
            {
                "uncertainty": float (0=certain, 1=maximally uncertain),
                "calibrated_score": float (temperature-scaled),
                "agreement_ratio": float (0=total disagreement, 1=unanimous),
                "entropy": float (information-theoretic uncertainty),
                "is_ood": bool (likely out-of-distribution),
                "confidence_level": str ("high"/"medium"/"low"/"very_low"),
                "recommendation": str,
            }
        """
        if not scores:
            return self._empty_result()

        names = list(scores.keys())
        values = np.array([scores[n] for n in names])

        if weights:
            w = np.array([weights.get(n, 1.0) for n in names])
        else:
            w = np.ones(len(names))
        w = w / (w.sum() + 1e-10)

        # 1. Weighted mean and variance
        weighted_mean = float(np.dot(w, values))
        weighted_var = float(np.dot(w, (values - weighted_mean) ** 2))
        weighted_std = np.sqrt(weighted_var)

        # 2. Agreement analysis
        # What fraction of detectors agree on the verdict?
        ai_count = np.sum(values > 0.5)
        real_count = np.sum(values <= 0.5)
        agreement_ratio = max(ai_count, real_count) / len(values)

        # Weighted agreement
        ai_weight = np.sum(w[values > 0.5])
        real_weight = np.sum(w[values <= 0.5])
        weighted_agreement = max(ai_weight, real_weight)

        # 3. Entropy (information-theoretic uncertainty)
        # Treat each detector as a binary classifier
        # High entropy = high disagreement
        p_ai = np.clip(weighted_mean, 0.01, 0.99)
        binary_entropy = -(p_ai * np.log2(p_ai) + (1 - p_ai) * np.log2(1 - p_ai))

        # 4. Score distribution analysis
        score_range = float(np.ptp(values))
        is_bimodal = self._check_bimodality(values)

        # 5. Out-of-distribution detection
        # OOD heuristic: all scores near 0.5 (no detector has strong opinion)
        max_deviation = float(np.max(np.abs(values - 0.5)))
        is_ood = max_deviation < 0.15 and len(values) >= 3

        # 6. Temperature-scaled calibrated score
        # Convert mean score to logit, apply temperature, convert back
        logit = np.log(np.clip(weighted_mean, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(weighted_mean, 1e-6, 1 - 1e-6)))
        calibrated_logit = logit / self.temperature
        calibrated_score = 1.0 / (1.0 + np.exp(-calibrated_logit))

        # 7. Combined uncertainty metric
        # Weighted combination of multiple uncertainty signals
        uncertainty = (
            0.30 * (1.0 - weighted_agreement)  # Disagreement
            + 0.25 * min(1.0, weighted_std / 0.25)  # Score variance
            + 0.20 * binary_entropy  # Information entropy
            + 0.15 * (1.0 - max_deviation * 2)  # No strong signals
            + 0.10 * (1.0 if is_bimodal else 0.0)  # Bimodal distribution
        )

        # Confidence level
        if uncertainty < 0.15:
            confidence_level = "high"
        elif uncertainty < 0.30:
            confidence_level = "medium"
        elif uncertainty < 0.50:
            confidence_level = "low"
        else:
            confidence_level = "very_low"

        # Recommendation
        if confidence_level == "very_low" or is_ood:
            recommendation = "Manual review recommended — detection system has low confidence"
        elif confidence_level == "low":
            recommendation = "Result should be treated as preliminary — significant detector disagreement"
        elif is_bimodal:
            recommendation = "Detectors are split — some strongly indicate AI, others indicate real"
        else:
            recommendation = "Detection confidence is adequate"

        return {
            "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
            "calibrated_score": float(np.clip(calibrated_score, 0.0, 1.0)),
            "weighted_mean": float(weighted_mean),
            "weighted_std": float(weighted_std),
            "agreement_ratio": float(agreement_ratio),
            "weighted_agreement": float(weighted_agreement),
            "entropy": float(binary_entropy),
            "score_range": float(score_range),
            "is_bimodal": is_bimodal,
            "is_ood": is_ood,
            "max_deviation": float(max_deviation),
            "confidence_level": confidence_level,
            "recommendation": recommendation,
        }

    def calibrate_temperature(self, predicted_scores: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Find optimal temperature for probability calibration.

        Uses the validation set to find T that minimizes negative log-likelihood.

        Args:
            predicted_scores: Array of predicted AI probabilities.
            true_labels: Binary labels (0=real, 1=AI).

        Returns:
            Optimal temperature value.
        """
        best_nll = float('inf')
        best_temp = 1.0

        for temp in np.arange(0.5, 3.0, 0.1):
            # Apply temperature scaling
            logits = np.log(np.clip(predicted_scores, 1e-6, 1 - 1e-6) /
                            (1 - np.clip(predicted_scores, 1e-6, 1 - 1e-6)))
            calibrated = 1.0 / (1.0 + np.exp(-logits / temp))

            # Negative log-likelihood
            nll = -np.mean(
                true_labels * np.log(calibrated + 1e-10) +
                (1 - true_labels) * np.log(1 - calibrated + 1e-10)
            )

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp
        logger.info("Calibrated temperature: %.2f (NLL: %.4f)", best_temp, best_nll)
        return best_temp

    @staticmethod
    def _check_bimodality(scores: np.ndarray) -> bool:
        """
        Check if scores are bimodally distributed (some say AI, some say real).
        This indicates strong disagreement between detectors.
        """
        if len(scores) < 4:
            return False

        # Check if scores cluster around both ends
        low_count = np.sum(scores < 0.35)
        high_count = np.sum(scores > 0.65)
        total = len(scores)

        # Bimodal: at least 25% in each cluster
        return (low_count / total >= 0.25) and (high_count / total >= 0.25)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "uncertainty": 1.0,
            "calibrated_score": 0.5,
            "weighted_mean": 0.5,
            "weighted_std": 0.0,
            "agreement_ratio": 0.0,
            "weighted_agreement": 0.0,
            "entropy": 1.0,
            "score_range": 0.0,
            "is_bimodal": False,
            "is_ood": True,
            "max_deviation": 0.0,
            "confidence_level": "very_low",
            "recommendation": "No detector outputs available",
        }
