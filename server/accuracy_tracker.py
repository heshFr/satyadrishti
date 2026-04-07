"""
Satya Drishti — Accuracy Tracker & Drift Detection
=====================================================
Tracks prediction accuracy over time using user feedback,
detects distribution drift, and provides calibration analytics.

Features:
1. User feedback collection (correct/incorrect with ground truth)
2. Rolling accuracy computation per modality and engine
3. Confidence calibration analysis (is 80% confidence really 80% correct?)
4. Distribution drift detection (has the score distribution shifted?)
5. False positive/negative tracking with root cause analysis
6. Per-check accuracy breakdown (which forensic checks are most reliable?)
7. Accuracy dashboard data for frontend
"""

import logging
import math
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

log = logging.getLogger("satyadrishti.accuracy")

# Maximum entries to keep in memory (rolling window)
MAX_PREDICTIONS = 10000
MAX_FEEDBACK = 5000
# Drift detection window
DRIFT_WINDOW = 500
DRIFT_THRESHOLD = 0.15  # Flag if distribution shift > 15%


class PredictionRecord:
    """A single prediction with optional feedback."""
    __slots__ = (
        "scan_id", "modality", "verdict", "confidence", "raw_scores",
        "forensic_statuses", "timestamp", "feedback", "ground_truth",
        "feedback_time", "details",
    )

    def __init__(
        self,
        scan_id: str,
        modality: str,
        verdict: str,
        confidence: float,
        raw_scores: Dict[str, Any],
        forensic_statuses: Dict[str, str],
        details: Dict[str, Any] = None,
    ):
        self.scan_id = scan_id
        self.modality = modality
        self.verdict = verdict
        self.confidence = confidence
        self.raw_scores = raw_scores
        self.forensic_statuses = forensic_statuses
        self.timestamp = time.time()
        self.details = details or {}
        # Filled in later by user feedback
        self.feedback: Optional[str] = None  # "correct", "incorrect", "unsure"
        self.ground_truth: Optional[str] = None  # actual label if known
        self.feedback_time: Optional[float] = None


class AccuracyTracker:
    """
    Tracks prediction accuracy, calibration, and drift in real-time.
    Thread-safe for concurrent access from multiple request handlers.
    """

    def __init__(self):
        self._predictions: deque = deque(maxlen=MAX_PREDICTIONS)
        self._feedback_queue: deque = deque(maxlen=MAX_FEEDBACK)
        self._lock = threading.Lock()

        # Rolling counters for fast accuracy computation
        self._correct_by_modality = defaultdict(int)
        self._incorrect_by_modality = defaultdict(int)
        self._total_by_modality = defaultdict(int)

        # Confidence calibration bins (10% wide)
        # For each bin, track: total predictions and correct predictions
        self._calibration_bins = defaultdict(lambda: {"total": 0, "correct": 0})

        # Per-check accuracy (which forensic checks are most predictive?)
        self._check_accuracy = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

        # Score distribution tracking for drift detection
        self._score_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=DRIFT_WINDOW * 2))

        # False positive/negative analysis
        self._false_positives: List[Dict] = []
        self._false_negatives: List[Dict] = []

    def record_prediction(
        self,
        scan_id: str,
        modality: str,
        verdict: str,
        confidence: float,
        raw_scores: Dict[str, Any] = None,
        forensic_checks: List[Dict] = None,
        details: Dict[str, Any] = None,
    ):
        """Record a new prediction for tracking."""
        forensic_statuses = {}
        if forensic_checks:
            for check in forensic_checks:
                cid = check.get("id", "unknown")
                status = check.get("status", "unknown")
                forensic_statuses[cid] = status

        record = PredictionRecord(
            scan_id=scan_id,
            modality=modality,
            verdict=verdict,
            confidence=confidence,
            raw_scores=raw_scores or {},
            forensic_statuses=forensic_statuses,
            details=details,
        )

        with self._lock:
            self._predictions.append(record)
            self._total_by_modality[modality] += 1

            # Track score distributions for drift detection
            for key, val in (raw_scores or {}).items():
                if isinstance(val, (int, float)) and 0 <= val <= 1:
                    self._score_windows[f"{modality}:{key}"].append(val)

        return record

    def record_feedback(
        self,
        scan_id: str,
        feedback: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record user feedback on a prediction's accuracy.

        Args:
            scan_id: The scan ID to provide feedback for
            feedback: "correct", "incorrect", or "unsure"
            ground_truth: Actual label if known (e.g. "authentic", "ai-generated", "spoof")

        Returns:
            Summary of the feedback recorded
        """
        with self._lock:
            # Find the prediction
            target = None
            for pred in reversed(self._predictions):
                if pred.scan_id == scan_id:
                    target = pred
                    break

            if target is None:
                return {"error": f"Scan {scan_id} not found in recent predictions"}

            target.feedback = feedback
            target.ground_truth = ground_truth
            target.feedback_time = time.time()

            modality = target.modality

            if feedback == "correct":
                self._correct_by_modality[modality] += 1
                # Update calibration bins
                bin_idx = min(int(target.confidence / 10), 9)
                bin_key = f"{bin_idx * 10}-{(bin_idx + 1) * 10}"
                self._calibration_bins[bin_key]["total"] += 1
                self._calibration_bins[bin_key]["correct"] += 1

                # Update per-check accuracy: checks that agreed with the correct verdict
                for check_id, status in target.forensic_statuses.items():
                    is_threat = target.verdict not in ("authentic", "safe", "bonafide")
                    check_flagged = status == "fail"
                    if is_threat and check_flagged:
                        self._check_accuracy[check_id]["tp"] += 1
                    elif not is_threat and not check_flagged:
                        self._check_accuracy[check_id]["tn"] += 1

            elif feedback == "incorrect":
                self._incorrect_by_modality[modality] += 1
                # Update calibration bins
                bin_idx = min(int(target.confidence / 10), 9)
                bin_key = f"{bin_idx * 10}-{(bin_idx + 1) * 10}"
                self._calibration_bins[bin_key]["total"] += 1
                # Don't increment correct for this bin

                # Track false positive/negative
                is_threat_predicted = target.verdict not in ("authentic", "safe", "bonafide")
                if is_threat_predicted:
                    # Predicted threat but user says incorrect → false positive
                    entry = {
                        "scan_id": scan_id,
                        "modality": modality,
                        "predicted": target.verdict,
                        "confidence": target.confidence,
                        "ground_truth": ground_truth or "authentic",
                        "raw_scores": target.raw_scores,
                        "forensic_statuses": target.forensic_statuses,
                        "timestamp": target.timestamp,
                    }
                    self._false_positives.append(entry)
                    if len(self._false_positives) > 500:
                        self._false_positives = self._false_positives[-500:]

                    # Update per-check: checks that flagged but verdict was wrong
                    for check_id, status in target.forensic_statuses.items():
                        if status == "fail":
                            self._check_accuracy[check_id]["fp"] += 1
                        else:
                            self._check_accuracy[check_id]["tn"] += 1
                else:
                    # Predicted safe but user says incorrect → false negative
                    entry = {
                        "scan_id": scan_id,
                        "modality": modality,
                        "predicted": target.verdict,
                        "confidence": target.confidence,
                        "ground_truth": ground_truth or "ai-generated",
                        "raw_scores": target.raw_scores,
                        "forensic_statuses": target.forensic_statuses,
                        "timestamp": target.timestamp,
                    }
                    self._false_negatives.append(entry)
                    if len(self._false_negatives) > 500:
                        self._false_negatives = self._false_negatives[-500:]

                    for check_id, status in target.forensic_statuses.items():
                        if status != "fail":
                            self._check_accuracy[check_id]["fn"] += 1
                        else:
                            self._check_accuracy[check_id]["tp"] += 1

            self._feedback_queue.append({
                "scan_id": scan_id,
                "feedback": feedback,
                "ground_truth": ground_truth,
                "modality": modality,
                "verdict": target.verdict,
                "confidence": target.confidence,
                "timestamp": time.time(),
            })

        from . import metrics as m
        m.feedback_total.inc(labels={"feedback": feedback, "modality": modality})
        if feedback == "incorrect":
            is_threat = target.verdict not in ("authentic", "safe", "bonafide")
            if is_threat:
                m.false_positive_total.inc(labels={"modality": modality})
            else:
                m.false_negative_total.inc(labels={"modality": modality})

        return {
            "status": "recorded",
            "scan_id": scan_id,
            "feedback": feedback,
            "ground_truth": ground_truth,
        }

    def get_accuracy_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive accuracy report.

        Returns:
            - Overall accuracy per modality
            - Confidence calibration curve data
            - Per-check accuracy ranking
            - Distribution drift alerts
            - False positive/negative analysis
        """
        with self._lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_predictions": len(self._predictions),
                "total_feedback": sum(
                    self._correct_by_modality[m] + self._incorrect_by_modality[m]
                    for m in set(list(self._correct_by_modality.keys()) + list(self._incorrect_by_modality.keys()))
                ),
            }

            # ── 1. Per-modality accuracy ──
            modality_accuracy = {}
            for modality in set(list(self._correct_by_modality.keys()) + list(self._incorrect_by_modality.keys())):
                correct = self._correct_by_modality[modality]
                incorrect = self._incorrect_by_modality[modality]
                total_fb = correct + incorrect
                total_pred = self._total_by_modality[modality]

                modality_accuracy[modality] = {
                    "total_predictions": total_pred,
                    "total_feedback": total_fb,
                    "correct": correct,
                    "incorrect": incorrect,
                    "accuracy": round(correct / total_fb * 100, 2) if total_fb > 0 else None,
                    "feedback_rate": round(total_fb / total_pred * 100, 2) if total_pred > 0 else 0,
                }

            report["modality_accuracy"] = modality_accuracy

            # ── 2. Confidence calibration curve ──
            calibration = {}
            for bin_key, data in sorted(self._calibration_bins.items()):
                if data["total"] > 0:
                    actual_accuracy = round(data["correct"] / data["total"] * 100, 1)
                    calibration[bin_key] = {
                        "predicted_confidence_range": bin_key,
                        "total_samples": data["total"],
                        "actual_accuracy": actual_accuracy,
                        "calibration_error": round(
                            abs(actual_accuracy - float(bin_key.split("-")[0]) - 5), 1
                        ),
                    }
            report["confidence_calibration"] = calibration

            # ── 3. Per-check accuracy ranking ──
            check_rankings = []
            for check_id, counts in self._check_accuracy.items():
                tp = counts["tp"]
                fp = counts["fp"]
                tn = counts["tn"]
                fn = counts["fn"]
                total = tp + fp + tn + fn

                if total == 0:
                    continue

                accuracy = (tp + tn) / total
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                check_rankings.append({
                    "check_id": check_id,
                    "total_evaluations": total,
                    "accuracy": round(accuracy * 100, 1),
                    "precision": round(precision * 100, 1),
                    "recall": round(recall * 100, 1),
                    "f1_score": round(f1 * 100, 1),
                    "true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn,
                })

            check_rankings.sort(key=lambda x: x["f1_score"], reverse=True)
            report["check_accuracy_ranking"] = check_rankings

            # ── 4. Distribution drift detection ──
            drift_alerts = []
            for key, window in self._score_windows.items():
                if len(window) < DRIFT_WINDOW:
                    continue

                scores = list(window)
                mid = len(scores) // 2
                old_half = scores[:mid]
                new_half = scores[mid:]

                old_mean = sum(old_half) / len(old_half)
                new_mean = sum(new_half) / len(new_half)
                shift = abs(new_mean - old_mean)

                if shift > DRIFT_THRESHOLD:
                    drift_alerts.append({
                        "metric": key,
                        "old_mean": round(old_mean, 4),
                        "new_mean": round(new_mean, 4),
                        "shift": round(shift, 4),
                        "severity": "high" if shift > 0.25 else "medium",
                        "direction": "increasing" if new_mean > old_mean else "decreasing",
                        "window_size": len(scores),
                    })

            drift_alerts.sort(key=lambda x: x["shift"], reverse=True)
            report["drift_alerts"] = drift_alerts

            # ── 5. False positive/negative analysis ──
            report["false_positives"] = {
                "count": len(self._false_positives),
                "recent": self._false_positives[-10:],
                "by_modality": self._count_by_key(self._false_positives, "modality"),
            }
            report["false_negatives"] = {
                "count": len(self._false_negatives),
                "recent": self._false_negatives[-10:],
                "by_modality": self._count_by_key(self._false_negatives, "modality"),
            }

            # ── 6. Verdict distribution (recent) ──
            verdict_dist = defaultdict(int)
            recent = list(self._predictions)[-1000:]
            for pred in recent:
                verdict_dist[f"{pred.modality}:{pred.verdict}"] += 1
            report["verdict_distribution"] = dict(verdict_dist)

            # ── 7. Recommendations ──
            report["recommendations"] = self._generate_recommendations(
                modality_accuracy, check_rankings, drift_alerts,
            )

        return report

    def _count_by_key(self, items: List[Dict], key: str) -> Dict[str, int]:
        counts = defaultdict(int)
        for item in items:
            counts[item.get(key, "unknown")] += 1
        return dict(counts)

    def _generate_recommendations(
        self,
        modality_accuracy: Dict,
        check_rankings: List[Dict],
        drift_alerts: List[Dict],
    ) -> List[str]:
        """Generate actionable recommendations based on accuracy data."""
        recs = []

        # Check for low-accuracy modalities
        for modality, data in modality_accuracy.items():
            acc = data.get("accuracy")
            if acc is not None and acc < 90:
                recs.append(
                    f"[ACCURACY] {modality} accuracy is {acc}% — consider retraining "
                    f"or adjusting thresholds. ({data['incorrect']} incorrect out of {data['total_feedback']})"
                )

        # Check for unreliable forensic checks
        for check in check_rankings:
            if check["total_evaluations"] >= 10 and check["f1_score"] < 50:
                recs.append(
                    f"[CHECK] '{check['check_id']}' has low F1={check['f1_score']}% — "
                    f"consider reducing its weight in ensemble or recalibrating thresholds"
                )

        # High false positive checks
        for check in check_rankings:
            if check["false_positives"] > check["true_positives"] and check["total_evaluations"] >= 5:
                recs.append(
                    f"[FP] '{check['check_id']}' has more false positives ({check['false_positives']}) "
                    f"than true positives ({check['true_positives']}) — tighten detection threshold"
                )

        # Distribution drift
        for alert in drift_alerts:
            if alert["severity"] == "high":
                recs.append(
                    f"[DRIFT] {alert['metric']} shifted by {alert['shift']:.3f} "
                    f"({alert['direction']}) — investigate input distribution change"
                )

        # Low feedback rate
        for modality, data in modality_accuracy.items():
            if data["total_predictions"] > 100 and data["feedback_rate"] < 5:
                recs.append(
                    f"[FEEDBACK] Only {data['feedback_rate']}% feedback rate for {modality} — "
                    f"encourage users to rate scan accuracy for better calibration"
                )

        if not recs:
            recs.append("All systems operating within normal parameters.")

        return recs

    def get_calibration_curve(self) -> List[Dict]:
        """Get calibration curve data for plotting."""
        with self._lock:
            curve = []
            for bin_key in sorted(self._calibration_bins.keys()):
                data = self._calibration_bins[bin_key]
                if data["total"] > 0:
                    predicted = float(bin_key.split("-")[0]) + 5  # midpoint
                    actual = data["correct"] / data["total"] * 100
                    curve.append({
                        "predicted_confidence": predicted,
                        "actual_accuracy": round(actual, 1),
                        "sample_count": data["total"],
                    })
            return curve


# Global singleton
accuracy_tracker = AccuracyTracker()
