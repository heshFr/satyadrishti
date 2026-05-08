"""
Calibrated Ensemble Fusion for Audio Analyzers
================================================
Fuses outputs from all audio analysis modules (AST, RawNet3, SSL,
Whisper features, prosodic, breathing, phase, formant, temporal)
into a single calibrated probability with uncertainty estimation.

Supports two modes:
    1. **Default**: Weighted average with hand-tuned weights
    2. **Trained**: Logistic regression trained on validation data

Missing analyzers are handled gracefully — weights are renormalized
over whichever subset is present.
"""

import json
import logging
import os
from typing import Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# Canonical analyzer ordering (for logistic regression feature vector)
ANALYZER_ORDER = [
    "ast",
    "rawnet3",
    "ssl",
    "whisper_features",
    "prosodic",
    "breathing",
    "phase",
    "formant",
    "temporal",
    "tts_artifacts",
    # Phase 8: New analyzers
    "spectral_continuity",
    "phoneme_transition",
    "room_acoustics",
]

# Default weights — the primary Wav2Vec2 detector (99.7% eval accuracy)
# gets the highest weight. Signal processing layers provide supporting
# evidence but should NOT override a confident neural prediction.
# TTS artifact detector targets ElevenLabs/modern TTS specifically.
# Phase 8: Three new layers add independent detection signals.
DEFAULT_WEIGHTS = {
    "ast": 0.26,           # Primary neural detector (Wav2Vec2) — most reliable
    "rawnet3": 0.03,
    "ssl": 0.12,           # SSL backbone — good discriminator
    "whisper_features": 0.08,
    "prosodic": 0.06,      # Signal processing — supporting evidence only
    "breathing": 0.04,     # Can false-positive on noisy/short audio
    "phase": 0.02,
    "formant": 0.02,
    "temporal": 0.08,
    "tts_artifacts": 0.14, # Modern TTS artifact detector — ElevenLabs/etc.
    # Phase 8: New analyzers (15% total weight across 3 new layers)
    "spectral_continuity": 0.06,  # Spectral envelope evolution analysis
    "phoneme_transition": 0.05,   # Coarticulation pattern analysis
    "room_acoustics": 0.04,       # Room impulse response consistency
}

# Verdict thresholds — calibrated for real-world phone audio (Opus/AMR codec,
# narrowband, codec jitter). Phone calls compress the high-frequency content
# the breathing/prosodic detectors rely on, which inflates false-positive
# rates on real human voices. We err on the side of "do not flag a real
# person as synthetic" because that's the demo-killer failure mode.
SPOOF_THRESHOLD = 0.65
BONAFIDE_THRESHOLD = 0.32
UNCERTAINTY_THRESHOLD = 0.70  # Raised: allow more disagreement before blocking


class EnsembleFusion:
    """
    Calibrated ensemble fusion across all audio analyzers.

    Manages weights and fusion logic. Uses weighted averaging by
    default, with an optional logistic regression trained on
    validation data for calibrated probability output.

    Any subset of analyzers can be present — weights are renormalized
    to sum to 1.0 over available analyzers.
    """

    def __init__(self):
        self._weights = dict(DEFAULT_WEIGHTS)
        self._trained_model: Optional[object] = None
        self._is_trained = False
        logger.info(
            "EnsembleFusion initialized with %d default analyzer weights",
            len(self._weights),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, analyzer_outputs: dict) -> dict:
        """
        Fuse outputs from multiple audio analyzers.

        Args:
            analyzer_outputs: Mapping of analyzer name to its output dict.
                Each value must contain at least ``"score"`` (float, 0–1).
                Optional keys: ``"confidence"`` (float, 0–1),
                ``"anomalies"`` (list[str]).

                Example::

                    {
                        "ast": {"score": 0.8, "confidence": 0.9},
                        "rawnet3": {"score": 0.7, "confidence": 0.85},
                        "prosodic": {"score": 0.6, "confidence": 0.7},
                    }

        Returns:
            Fused result dictionary::

                {
                    "verdict": "bonafide" | "spoof" | "uncertain",
                    "probability": float,
                    "confidence": float,
                    "uncertainty": float,
                    "per_analyzer": { ... },
                    "explanation": list[str],
                }
        """
        if not analyzer_outputs:
            logger.warning("EnsembleFusion.fuse called with no analyzer outputs")
            return self._empty_result()

        # Filter to analyzers that have a valid score
        valid = {}
        for name, output in analyzer_outputs.items():
            if not isinstance(output, dict):
                logger.warning("Skipping analyzer '%s': output is not a dict", name)
                continue
            if "score" not in output:
                logger.warning("Skipping analyzer '%s': missing 'score' key", name)
                continue
            try:
                score = float(output["score"])
                if not (0.0 <= score <= 1.0):
                    logger.warning(
                        "Analyzer '%s' score %.4f outside [0, 1], clamping", name, score
                    )
                    score = float(np.clip(score, 0.0, 1.0))
                valid[name] = {
                    "score": score,
                    "confidence": float(np.clip(output.get("confidence", 0.5), 0.0, 1.0)),
                    "anomalies": output.get("anomalies", []),
                }
            except (ValueError, TypeError) as e:
                logger.warning("Skipping analyzer '%s': %s", name, e)
                continue

        if not valid:
            logger.warning("No valid analyzer outputs after filtering")
            return self._empty_result()

        # Choose fusion strategy
        if self._is_trained and self._trained_model is not None:
            probability = self._fuse_trained(valid)
        else:
            probability = self._fuse_weighted(valid)

        # Compute confidence and uncertainty
        confidence = self._compute_confidence(valid)
        uncertainty = self._compute_uncertainty(valid)

        # Per-analyzer breakdown
        per_analyzer = self._per_analyzer_breakdown(valid, probability)

        # Verdict
        verdict, veto_reason = self._determine_verdict(probability, uncertainty, valid)

        # Explanation
        explanation = self._generate_explanation(valid, per_analyzer, verdict, probability)
        if veto_reason:
            explanation.insert(0, f"BIOLOGICAL VETO: {veto_reason}")

        result = {
            "verdict": verdict,
            "probability": float(np.clip(probability, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
            "per_analyzer": per_analyzer,
            "explanation": explanation,
        }
        
        if veto_reason:
            result["biological_veto"] = True
            result["veto_reason"] = veto_reason

        logger.info(
            "EnsembleFusion result: verdict=%s probability=%.4f confidence=%.4f uncertainty=%.4f (%d analyzers)",
            verdict,
            result["probability"],
            result["confidence"],
            result["uncertainty"],
            len(valid),
        )

        return result

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train a logistic regression on validation data.

        Args:
            X: Feature matrix of shape (n_samples, n_analyzers). Columns
               correspond to ``ANALYZER_ORDER``. Use NaN or -1 for
               missing analyzers.
            y: Binary labels (0 = bonafide, 1 = spoof).

        Raises:
            RuntimeError: If scikit-learn is not installed.
        """
        if not HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required for training. "
                "Install with: pip install scikit-learn"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
            )

        # Replace NaN / -1 with column means (simple imputation)
        X_clean = X.copy()
        for col in range(X_clean.shape[1]):
            mask = np.isnan(X_clean[:, col]) | (X_clean[:, col] < 0)
            if mask.any():
                col_mean = np.nanmean(X_clean[~mask, col]) if (~mask).any() else 0.5
                X_clean[mask, col] = col_mean

        logger.info(
            "Training EnsembleFusion logistic regression on %d samples, %d features",
            X_clean.shape[0],
            X_clean.shape[1],
        )

        # Train calibrated logistic regression
        base_lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )

        if X_clean.shape[0] >= 10:
            # Use calibration with cross-validation if enough data
            n_folds = min(5, X_clean.shape[0] // 2)
            model = CalibratedClassifierCV(
                estimator=base_lr,
                cv=n_folds,
                method="sigmoid",
            )
        else:
            # Not enough data for CV — train base model directly
            model = base_lr

        model.fit(X_clean, y)
        self._trained_model = model
        self._is_trained = True

        # Extract learned weights for interpretability
        if hasattr(model, "estimator") and hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV — get mean coefficients
            coefs = []
            for cc in model.calibrated_classifiers_:
                if hasattr(cc.estimator, "coef_"):
                    coefs.append(cc.estimator.coef_[0])
            if coefs:
                mean_coef = np.mean(coefs, axis=0)
                abs_coef = np.abs(mean_coef)
                if abs_coef.sum() > 0:
                    normalized = abs_coef / abs_coef.sum()
                    for i, name in enumerate(ANALYZER_ORDER):
                        if i < len(normalized):
                            self._weights[name] = float(normalized[i])
        elif hasattr(model, "coef_"):
            abs_coef = np.abs(model.coef_[0])
            if abs_coef.sum() > 0:
                normalized = abs_coef / abs_coef.sum()
                for i, name in enumerate(ANALYZER_ORDER):
                    if i < len(normalized):
                        self._weights[name] = float(normalized[i])

        logger.info("Training complete. Learned weights: %s", self._weights)

    def save(self, path: str):
        """
        Save trained weights and model to disk.

        Args:
            path: File path (JSON for weights; pickle for model if trained).
        """
        data = {
            "weights": self._weights,
            "is_trained": self._is_trained,
            "analyzer_order": ANALYZER_ORDER,
        }

        # Save weights as JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved ensemble weights to %s", path)

        # Save sklearn model alongside if trained
        if self._is_trained and self._trained_model is not None:
            try:
                import pickle

                model_path = path.rsplit(".", 1)[0] + "_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(self._trained_model, f)
                logger.info("Saved trained model to %s", model_path)
            except Exception as e:
                logger.error("Failed to save trained model: %s", e)

    def load(self, path: str):
        """
        Load weights and optionally the trained model from disk.

        Args:
            path: File path to the JSON weights file.
        """
        if not os.path.exists(path):
            logger.warning("Weights file not found: %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "weights" in data:
            self._weights = data["weights"]
            logger.info("Loaded ensemble weights from %s", path)

        # Try to load trained model
        if data.get("is_trained", False):
            model_path = path.rsplit(".", 1)[0] + "_model.pkl"
            if os.path.exists(model_path):
                try:
                    import pickle

                    with open(model_path, "rb") as f:
                        self._trained_model = pickle.load(f)
                    self._is_trained = True
                    logger.info("Loaded trained model from %s", model_path)
                except Exception as e:
                    logger.error("Failed to load trained model: %s", e)
                    self._is_trained = False
            else:
                logger.warning(
                    "Trained model file not found: %s (falling back to weighted average)",
                    model_path,
                )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fuse_weighted(self, valid: dict) -> float:
        """
        Weighted average fusion with confidence-adjusted weights, neural
        dominance, and corroboration amplification.

        Key principles for modern TTS detection:
        1. Neural detectors (AST, SSL) are trusted more than heuristic layers
        2. When neural detectors say "spoof", heuristic "real" signals are dampened
           (modern TTS fools breathing/prosodic/formant detectors)
        3. When neural + TTS artifacts agree, corroboration boost is applied
        4. Heuristic layers saying "spoof" are still trusted (hard to fake physics)
        """
        # Step 1: Compute base weighted average
        weighted_sum = 0.0
        total_weight = 0.0

        for name, info in valid.items():
            base_weight = self._weights.get(name, 0.05)
            effective_weight = base_weight * info["confidence"]
            weighted_sum += info["score"] * effective_weight
            total_weight += effective_weight

        if total_weight < 1e-12:
            return 0.5

        base_probability = weighted_sum / total_weight

        # Step 2: Neural dominance — when AST (primary neural) confidently says
        # "spoof" but heuristic layers say "real", pull toward neural.
        # This handles modern TTS that fools breathing/prosodic/formant.
        NEURAL_LAYERS = {"ast", "ssl", "whisper_features", "tts_artifacts"}
        HEURISTIC_LAYERS = {"prosodic", "breathing", "phase", "formant"}

        neural_scores = []
        neural_weights = []
        heuristic_real_pull = 0.0
        heuristic_total_weight = 0.0

        for name, info in valid.items():
            if name in NEURAL_LAYERS:
                neural_scores.append(info["score"])
                neural_weights.append(self._weights.get(name, 0.05))
            elif name in HEURISTIC_LAYERS:
                w = self._weights.get(name, 0.05)
                heuristic_total_weight += w
                if info["score"] < 0.4:
                    # This heuristic layer is pulling toward "real"
                    heuristic_real_pull += (0.5 - info["score"]) * w

        if neural_scores:
            neural_avg = np.average(neural_scores, weights=neural_weights)

            # When neural average says "spoof" (>0.55) but heuristic layers
            # are pulling toward "real", dampen the heuristic pull
            if neural_avg > 0.55 and heuristic_real_pull > 0:
                # How much the neural detectors are saying "spoof"
                neural_confidence = min(1.0, (neural_avg - 0.55) * 4)
                # Dampen heuristic "real" pull by neural confidence
                dampen_factor = 1.0 - (neural_confidence * 0.7)  # Max 70% dampening
                # Adjust the base probability toward what it would be without
                # heuristic "real" pull
                if heuristic_total_weight > 0:
                    heuristic_pull_effect = heuristic_real_pull / (total_weight + 1e-10)
                    restored = base_probability + heuristic_pull_effect * (1 - dampen_factor)
                    base_probability = restored
                    logger.info(
                        "Neural dominance: neural_avg=%.3f, dampened heuristic 'real' pull by %.0f%% "
                        "(%.4f -> %.4f)",
                        neural_avg, (1 - dampen_factor) * 100,
                        base_probability - heuristic_pull_effect * (1 - dampen_factor),
                        base_probability,
                    )

        # Step 3: Corroboration amplification
        # When multiple independent systems agree on "spoof", boost confidence.
        # This is especially important for modern TTS where individual scores
        # are moderate (0.5-0.7) rather than extreme (>0.9).
        n_spoof_layers = sum(1 for info in valid.values() if info["score"] > 0.52)
        n_total = len(valid)

        if n_total >= 3 and base_probability > 0.45:
            spoof_ratio = n_spoof_layers / n_total
            if spoof_ratio >= 0.5:
                # Majority agreement on spoof — corroboration boost
                boost_strength = (spoof_ratio - 0.5) * 0.3  # 0 at 50%, 0.15 at 100%
                # Also consider if neural + tts_artifacts both agree
                ast_spoof = valid.get("ast", {}).get("score", 0.5) > 0.5
                tts_spoof = valid.get("tts_artifacts", {}).get("score", 0.5) > 0.5
                if ast_spoof and tts_spoof:
                    boost_strength += 0.08  # Neural + physics agree = strong
                base_probability = base_probability + boost_strength * (1 - base_probability)
                logger.info(
                    "Corroboration boost: %d/%d layers say spoof (ratio=%.0f%%), "
                    "boost=%.4f -> %.4f",
                    n_spoof_layers, n_total, spoof_ratio * 100,
                    boost_strength, base_probability,
                )

        return float(np.clip(base_probability, 0.0, 1.0))

    def _fuse_trained(self, valid: dict) -> float:
        """Use trained logistic regression for fusion."""
        # Build feature vector in canonical order
        feature_vec = np.full(len(ANALYZER_ORDER), np.nan)
        for i, name in enumerate(ANALYZER_ORDER):
            if name in valid:
                feature_vec[i] = valid[name]["score"]

        # Impute missing with 0.5 (neutral)
        mask = np.isnan(feature_vec)
        feature_vec[mask] = 0.5

        try:
            proba = self._trained_model.predict_proba(feature_vec.reshape(1, -1))
            # proba[:, 1] is P(spoof)
            return float(proba[0, 1])
        except Exception as e:
            logger.warning(
                "Trained model prediction failed (%s), falling back to weighted average", e
            )
            return self._fuse_weighted(valid)

    def _compute_confidence(self, valid: dict) -> float:
        """
        Overall confidence based on per-analyzer confidences and coverage.

        High confidence requires: multiple analyzers, each with high confidence.
        """
        if not valid:
            return 0.0

        confidences = [info["confidence"] for info in valid.values()]
        mean_conf = float(np.mean(confidences))

        # Coverage bonus: more analyzers → higher confidence
        total_possible = len(DEFAULT_WEIGHTS)
        coverage = len(valid) / total_possible

        # Weighted combination: 70% mean analyzer confidence, 30% coverage
        combined = 0.7 * mean_conf + 0.3 * coverage

        return combined

    def _compute_uncertainty(self, valid: dict) -> float:
        """
        Weight-aware disagreement between analyzers.

        Uses weighted variance so that disagreement among high-weight
        analyzers (AST, SSL) drives uncertainty more than noise from
        low-weight heuristic layers (breathing, phase, formant).
        """
        if len(valid) < 2:
            return 0.5  # single analyzer → moderate uncertainty

        # Build weight-normalised arrays
        names = list(valid.keys())
        scores = np.array([valid[n]["score"] for n in names])
        weights = np.array([self._weights.get(n, 0.05) for n in names])
        weights = weights / weights.sum()

        # Weighted mean (same as fused probability)
        wmean = float(np.dot(weights, scores))

        # Weighted standard deviation
        wvar = float(np.dot(weights, (scores - wmean) ** 2))
        wstd = np.sqrt(wvar)

        # Normalize: wstd of 0.35 → uncertainty 1.0
        uncertainty = min(1.0, wstd / 0.35)

        return float(uncertainty)

    def _per_analyzer_breakdown(self, valid: dict, probability: float) -> dict:
        """Build per-analyzer contribution breakdown."""
        per_analyzer = {}

        # Compute effective weights (normalized over available analyzers)
        available_weights = {
            name: self._weights.get(name, 0.05) for name in valid
        }
        total_weight = sum(available_weights.values())
        if total_weight < 1e-12:
            total_weight = 1.0

        for name, info in valid.items():
            normalized_weight = available_weights[name] / total_weight
            contribution = info["score"] * normalized_weight
            per_analyzer[name] = {
                "score": info["score"],
                "weight": round(normalized_weight, 4),
                "contribution": round(contribution, 4),
                "confidence": info["confidence"],
            }

        return per_analyzer

    @staticmethod
    def _determine_verdict(
        probability: float, uncertainty: float, valid: dict | None = None
    ) -> tuple[str, str | None]:
        """
        Determine verdict from probability, uncertainty, and analyzer data.

        For call protection, we err on the side of caution:
        - High probability spoof should trigger even with analyzer disagreement
        - Neural Override: When the neural model is confident, lower the threshold
        - Biological Veto System: Biological failures immediately trigger a spoof
          alert regardless of neural confidence.
        - TTS Artifact Veto: When TTS artifacts are strong and neural agrees,
          immediate spoof verdict.
        """
        # Biological Veto System: requires very strong signals from BOTH a
        # signal-processing analyzer AND the primary neural model.
        # Thresholds raised because phone-codec compression routinely warps
        # breathing/prosodic/temporal features for real human callers.
        if valid:
            ast_score = valid.get("ast", {}).get("score", 0.5)
            neural_agrees = ast_score > 0.65  # was 0.5 — require stronger neural agreement

            if neural_agrees:
                # Veto now requires near-certain biomarker (0.95+) plus strong neural agreement
                if "breathing" in valid and valid["breathing"]["score"] > 0.95:
                    return "spoof", "Breathing pattern anomaly confirmed by neural model"
                if "prosodic" in valid and valid["prosodic"]["score"] > 0.95:
                    return "spoof", "Vocal cord anomaly confirmed by neural model"
                if "temporal" in valid and valid["temporal"]["score"] > 0.95:
                    return "spoof", "Temporal consistency anomaly confirmed by neural model"

                # TTS Artifact Veto: neural + TTS artifacts both detect spoof
                tts_score = valid.get("tts_artifacts", {}).get("score", 0.0)
                if tts_score > 0.78 and ast_score > 0.70:
                    return "spoof", "TTS artifacts detected and confirmed by neural model"

            # Strong neural + SSL agreement override — tightened for phone audio
            ssl_score = valid.get("ssl", {}).get("score", 0.5)
            if ast_score > 0.78 and ssl_score > 0.70:
                return "spoof", "Neural (AST) and SSL backbone agree on synthetic voice"

        # Very high probability overrides uncertainty entirely (raised to avoid
        # false positives on compressed real-call audio)
        if probability > 0.88:
            return "spoof", None

        # Neural override: when AST is very confident (>0.75), nudge threshold slightly
        effective_threshold = SPOOF_THRESHOLD
        if valid:
            ast_score = valid.get("ast", {}).get("score", 0.5)
            if ast_score > 0.75:
                # Small nudge only when neural detector is highly confident
                effective_threshold = SPOOF_THRESHOLD - 0.02
                logger.info(
                    "Neural override: AST=%.3f, lowering spoof threshold to %.3f",
                    ast_score, effective_threshold,
                )

        if probability > effective_threshold and uncertainty < UNCERTAINTY_THRESHOLD:
            return "spoof", None
        elif probability < BONAFIDE_THRESHOLD:
            return "bonafide", None
        else:
            return "uncertain", None

    def _generate_explanation(
        self,
        valid: dict,
        per_analyzer: dict,
        verdict: str,
        probability: float,
    ) -> list:
        """Generate human-readable explanation listing top contributors."""
        explanation = []

        # Sort analyzers by contribution (descending)
        sorted_analyzers = sorted(
            per_analyzer.items(),
            key=lambda x: x[1]["contribution"],
            reverse=True,
        )

        # Top 3 contributing analyzers
        top_n = min(3, len(sorted_analyzers))
        for name, info in sorted_analyzers[:top_n]:
            score = info["score"]
            weight_pct = info["weight"] * 100

            # Get anomalies if present
            anomalies = valid.get(name, {}).get("anomalies", [])
            anomaly_str = ""
            if anomalies:
                anomaly_str = f" — anomalies: {', '.join(str(a) for a in anomalies[:3])}"

            if score > 0.7:
                level = "HIGH"
            elif score > 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"

            explanation.append(
                f"{name} ({level}, score={score:.2f}, weight={weight_pct:.0f}%){anomaly_str}"
            )

        # Add verdict summary
        if verdict == "spoof":
            explanation.insert(
                0,
                f"Audio classified as SPOOF (P={probability:.2f}): "
                f"{len(valid)} analyzers agree on synthetic indicators.",
            )
        elif verdict == "bonafide":
            explanation.insert(
                0,
                f"Audio classified as BONAFIDE (P={probability:.2f}): "
                f"no significant synthetic indicators detected.",
            )
        else:
            explanation.insert(
                0,
                f"Audio classified as UNCERTAIN (P={probability:.2f}): "
                f"analyzers show mixed signals or insufficient confidence.",
            )

        return explanation

    @staticmethod
    def _empty_result() -> dict:
        """Return a default result when no analyzers are available."""
        return {
            "verdict": "uncertain",
            "probability": 0.5,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "per_analyzer": {},
            "explanation": ["No analyzer outputs available for fusion."],
        }
