"""
Satya Drishti — Prometheus Metrics & Observability
====================================================
Exposes /metrics endpoint with detailed performance, accuracy,
and operational metrics for monitoring via Prometheus + Grafana.

Metrics Categories:
1. Request Metrics     — latency, throughput, error rates
2. Engine Metrics      — per-engine inference time, scores, layer activity
3. Accuracy Metrics    — verdict distribution, confidence calibration, feedback
4. Resource Metrics    — active connections, queue depth, memory
5. Detection Metrics   — threat levels, modality breakdown, corroboration rates
"""

import time
import threading
import logging
from collections import defaultdict
from typing import Dict, Any, Optional

log = logging.getLogger("satyadrishti.metrics")


class Histogram:
    """Simple histogram with configurable bucket boundaries."""

    def __init__(self, name: str, help_text: str, buckets: tuple):
        self.name = name
        self.help = help_text
        self.buckets = sorted(buckets)
        self._counts = defaultdict(lambda: defaultdict(int))  # labels -> bucket -> count
        self._sums = defaultdict(float)
        self._totals = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict = None):
        key = _label_key(labels)
        with self._lock:
            self._sums[key] += value
            self._totals[key] += 1
            for b in self.buckets:
                if value <= b:
                    self._counts[key][b] += 1
            self._counts[key][float("inf")] += 1

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key in sorted(self._counts.keys()):
                labels_str = key
                cumulative = 0
                for b in self.buckets:
                    cumulative += self._counts[key].get(b, 0) - (cumulative if b == self.buckets[0] else 0)
                # Re-compute cumulative properly
                cum = 0
                for b in self.buckets:
                    cum += self._counts[key].get(b, 0)
                    le_label = f'{labels_str},le="{b}"' if labels_str else f'le="{b}"'
                    lines.append(f'{self.name}_bucket{{{le_label}}} {cum}')
                inf_label = f'{labels_str},le="+Inf"' if labels_str else 'le="+Inf"'
                lines.append(f'{self.name}_bucket{{{inf_label}}} {self._counts[key][float("inf")]}')
                sum_label = f'{{{labels_str}}}' if labels_str else ''
                lines.append(f'{self.name}_sum{sum_label} {self._sums[key]:.6f}')
                lines.append(f'{self.name}_count{sum_label} {self._totals[key]}')
        return "\n".join(lines)


class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: dict = None):
        key = _label_key(labels)
        with self._lock:
            self._values[key] += amount

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key in sorted(self._values.keys()):
                label_str = f'{{{key}}}' if key else ''
                lines.append(f'{self.name}{label_str} {self._values[key]:.1f}')
        return "\n".join(lines)


class Gauge:
    """Value that can go up and down."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: dict = None):
        key = _label_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: dict = None):
        key = _label_key(labels)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: dict = None):
        key = _label_key(labels)
        with self._lock:
            self._values[key] -= amount

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for key in sorted(self._values.keys()):
                label_str = f'{{{key}}}' if key else ''
                lines.append(f'{self.name}{label_str} {self._values[key]:.4f}')
        return "\n".join(lines)


def _label_key(labels: Optional[dict]) -> str:
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


# ══════════════════════════════════════════════════════════════════
# Global Metric Instances
# ══════════════════════════════════════════════════════════════════

# ── 1. Request Metrics ──
request_latency = Histogram(
    "satya_request_duration_seconds",
    "Request latency by endpoint and method",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

request_total = Counter(
    "satya_requests_total",
    "Total requests by endpoint and status code",
)

request_errors = Counter(
    "satya_request_errors_total",
    "Total request errors by endpoint and error type",
)

# ── 2. Engine Metrics ──
engine_inference_time = Histogram(
    "satya_engine_inference_seconds",
    "Inference time per engine/modality",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

engine_score = Histogram(
    "satya_engine_score",
    "Score distribution per engine (0-1 range)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

engine_layers_active = Histogram(
    "satya_engine_layers_active",
    "Number of active layers per analysis",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
)

# ── 3. Accuracy & Detection Metrics ──
verdict_total = Counter(
    "satya_verdict_total",
    "Total verdicts by verdict type and modality",
)

confidence_distribution = Histogram(
    "satya_confidence_percent",
    "Confidence score distribution by verdict",
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100),
)

feedback_total = Counter(
    "satya_feedback_total",
    "User feedback on accuracy (correct/incorrect/unsure)",
)

false_positive_total = Counter(
    "satya_false_positives_total",
    "Confirmed false positives by modality",
)

false_negative_total = Counter(
    "satya_false_negatives_total",
    "Confirmed false negatives by modality",
)

# ── 4. Resource Metrics ──
active_websockets = Gauge(
    "satya_active_websockets",
    "Currently active WebSocket connections",
)

active_analyses = Gauge(
    "satya_active_analyses",
    "Currently running analysis tasks by modality",
)

model_loaded = Gauge(
    "satya_model_loaded",
    "Whether each model is loaded (1) or not (0)",
)

# ── 5. Threat Metrics ──
threat_level_total = Counter(
    "satya_threat_level_total",
    "Threat level distribution from analyses",
)

corroboration_rate = Histogram(
    "satya_corroboration_rate",
    "Rate of cross-engine agreement (0-1)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

ensemble_uncertainty = Histogram(
    "satya_ensemble_uncertainty",
    "Ensemble fusion uncertainty distribution",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0),
)

# ── 6. Forensic Check Metrics ──
forensic_check_status = Counter(
    "satya_forensic_check_status_total",
    "Forensic check results by check_id and status",
)

biological_veto_total = Counter(
    "satya_biological_veto_total",
    "Times biological veto was triggered",
)

anomaly_count_distribution = Histogram(
    "satya_anomaly_count",
    "Number of anomalies detected per analysis",
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
)


def collect_all() -> str:
    """Collect all metrics in Prometheus exposition format."""
    collectors = [
        request_latency, request_total, request_errors,
        engine_inference_time, engine_score, engine_layers_active,
        verdict_total, confidence_distribution, feedback_total,
        false_positive_total, false_negative_total,
        active_websockets, active_analyses, model_loaded,
        threat_level_total, corroboration_rate, ensemble_uncertainty,
        forensic_check_status, biological_veto_total, anomaly_count_distribution,
    ]
    parts = []
    for c in collectors:
        text = c.collect()
        if text.count("\n") > 2:  # has data beyond HELP/TYPE
            parts.append(text)
    return "\n\n".join(parts) + "\n"


# ══════════════════════════════════════════════════════════════════
# Helper: Record analysis result metrics
# ══════════════════════════════════════════════════════════════════

def record_analysis_result(
    modality: str,
    result: Dict[str, Any],
    latency_seconds: float,
):
    """Extract and record all metrics from an analysis result."""
    verdict = result.get("verdict", "unknown")
    confidence = result.get("confidence", 0.0)
    forensic_checks = result.get("forensic_checks", [])
    raw_scores = result.get("raw_scores", {})
    details = result.get("details", {})

    # Request latency
    engine_inference_time.observe(latency_seconds, {"modality": modality})

    # Verdict distribution
    verdict_total.inc(labels={"verdict": verdict, "modality": modality})

    # Confidence distribution
    confidence_distribution.observe(confidence, {"verdict": verdict, "modality": modality})

    # Forensic check statuses
    fail_count = 0
    for check in forensic_checks:
        check_id = check.get("id", "unknown")
        status = check.get("status", "unknown")
        forensic_check_status.inc(labels={"check_id": check_id, "status": status})
        if status == "fail":
            fail_count += 1

    anomaly_count_distribution.observe(fail_count, {"modality": modality})

    # Audio-specific metrics
    if modality == "audio":
        layers = details.get("layers_active", [])
        engine_layers_active.observe(len(layers), {"modality": "audio"})

        ens_prob = raw_scores.get("ensemble_probability")
        if ens_prob is not None:
            engine_score.observe(ens_prob, {"engine": "audio_ensemble"})

        ens_unc = raw_scores.get("ensemble_uncertainty")
        if ens_unc is not None:
            ensemble_uncertainty.observe(ens_unc, {"modality": "audio"})

        if details.get("biological_veto"):
            biological_veto_total.inc()

        # Per-layer scores
        for key, val in raw_scores.items():
            if key.endswith("_score") and isinstance(val, (int, float)):
                engine_name = key.replace("_score", "")
                engine_score.observe(val, {"engine": engine_name})

    # Video-specific metrics
    elif modality == "video":
        combined = raw_scores.get("combined_deepfake")
        if combined is not None:
            engine_score.observe(combined, {"engine": "video_ensemble"})

        # Per-engine scores
        for key in ("forensics_neural", "deepfake_two_stream", "ai_video_detector",
                     "rppg", "clip_drift", "lighting", "av_sync", "micro_expression"):
            val = raw_scores.get(key)
            if val is not None and isinstance(val, (int, float)):
                engine_score.observe(val, {"engine": key})

    # Image-specific metrics
    elif modality in ("image", "media"):
        for key, val in raw_scores.items():
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                engine_score.observe(val, {"engine": f"image_{key}"})

    # Corroboration: how many engines agree
    if forensic_checks:
        statuses = [c.get("status") for c in forensic_checks if c.get("status") in ("pass", "fail")]
        if statuses:
            majority = max(set(statuses), key=statuses.count)
            agreement = statuses.count(majority) / len(statuses)
            corroboration_rate.observe(agreement, {"modality": modality})
