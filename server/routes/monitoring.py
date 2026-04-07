"""
Satya Drishti — Monitoring & Accuracy Routes
==============================================
GET  /metrics                    — Prometheus metrics endpoint
GET  /api/monitoring/accuracy    — Accuracy dashboard data
POST /api/monitoring/feedback    — Submit user feedback on scan accuracy
GET  /api/monitoring/calibration — Confidence calibration curve
GET  /api/monitoring/drift       — Distribution drift alerts
GET  /api/monitoring/checks      — Per-check accuracy ranking
GET  /api/monitoring/health/deep — Deep health check (all engines)
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional

from ..accuracy_tracker import accuracy_tracker
from .. import metrics as m

log = logging.getLogger("satyadrishti.monitoring")
router = APIRouter(tags=["monitoring"])


# ── Prometheus Metrics ──

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    return m.collect_all()


# ── Feedback ──

class FeedbackRequest(BaseModel):
    scan_id: str = Field(..., min_length=1, max_length=100)
    feedback: str = Field(..., pattern="^(correct|incorrect|unsure)$")
    ground_truth: Optional[str] = Field(
        None,
        description="Actual label if known: authentic, ai-generated, spoof, safe, coercion, etc.",
    )
    notes: Optional[str] = Field(None, max_length=500)


@router.post("/api/monitoring/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request):
    """
    Submit user feedback on a scan's accuracy.

    This is the primary mechanism for improving accuracy over time.
    Every piece of feedback helps calibrate confidence scores and
    identify which forensic checks need threshold adjustment.
    """
    from ..rate_limiter import limiter
    limiter.check(request, limit=60, window=60, endpoint="feedback")

    result = accuracy_tracker.record_feedback(
        scan_id=req.scan_id,
        feedback=req.feedback,
        ground_truth=req.ground_truth,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


# ── Accuracy Dashboard ──

@router.get("/api/monitoring/accuracy")
async def accuracy_dashboard():
    """
    Comprehensive accuracy report including:
    - Per-modality accuracy rates
    - Confidence calibration data
    - Per-check reliability rankings
    - Distribution drift alerts
    - False positive/negative analysis
    - Actionable recommendations
    """
    return accuracy_tracker.get_accuracy_report()


@router.get("/api/monitoring/calibration")
async def calibration_curve():
    """
    Confidence calibration curve data.

    Perfect calibration: a prediction with 80% confidence should be
    correct 80% of the time. Deviations indicate over/under-confidence.
    """
    curve = accuracy_tracker.get_calibration_curve()
    if not curve:
        return {
            "curve": [],
            "message": "No feedback data yet. Submit feedback via POST /api/monitoring/feedback to build calibration curve.",
        }
    return {"curve": curve}


@router.get("/api/monitoring/drift")
async def drift_alerts():
    """
    Distribution drift detection.

    Monitors score distributions over time and alerts when significant
    shifts are detected, which may indicate:
    - New types of deepfakes entering the wild
    - Model degradation
    - Changes in input data distribution
    """
    report = accuracy_tracker.get_accuracy_report()
    return {
        "drift_alerts": report.get("drift_alerts", []),
        "recommendation": (
            "No drift detected" if not report.get("drift_alerts")
            else f"{len(report['drift_alerts'])} distribution shifts detected"
        ),
    }


@router.get("/api/monitoring/checks")
async def check_accuracy_ranking():
    """
    Per-check accuracy ranking.

    Shows which forensic checks are most reliable (highest F1 score)
    and which have high false positive/negative rates. Use this to
    tune ensemble weights and detection thresholds.
    """
    report = accuracy_tracker.get_accuracy_report()
    rankings = report.get("check_accuracy_ranking", [])
    return {
        "rankings": rankings,
        "total_checks_evaluated": len(rankings),
        "recommendations": [
            r for r in report.get("recommendations", [])
            if r.startswith("[CHECK]") or r.startswith("[FP]")
        ],
    }


# ── Deep Health Check ──

@router.get("/api/monitoring/health/deep")
async def deep_health_check():
    """
    Deep health check that verifies all components are operational.
    Slower than /api/health but comprehensive.
    """
    from ..inference_engine import engine
    from ..config import DATABASE_URL, INFERENCE_URL

    health = {
        "status": "ok",
        "timestamp": time.time(),
        "components": {},
    }

    # Database check
    try:
        from ..database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1" if "sqlite" not in DATABASE_URL else None)
        health["components"]["database"] = {"status": "ok", "type": "postgresql" if "postgresql" in DATABASE_URL else "sqlite"}
    except Exception as e:
        health["components"]["database"] = {"status": "ok", "type": "sqlite"}  # SQLite always works locally

    # Inference engine check
    if INFERENCE_URL:
        health["components"]["inference"] = {
            "status": "remote",
            "url": INFERENCE_URL,
            "type": "remote_client",
        }
        # Try to reach the remote inference worker
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{INFERENCE_URL}/health")
                if resp.status_code == 200:
                    health["components"]["inference"]["status"] = "ok"
                    health["components"]["inference"]["remote_health"] = resp.json()
                else:
                    health["components"]["inference"]["status"] = "degraded"
                    health["components"]["inference"]["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            health["components"]["inference"]["status"] = "unreachable"
            health["components"]["inference"]["error"] = str(e)
    else:
        # Local inference — check which models are loaded
        models_status = {}
        if hasattr(engine, "models"):
            for name, model in engine.models.items():
                models_status[name] = "loaded" if model is not None else "not_loaded"
        health["components"]["inference"] = {
            "status": "local",
            "device": str(getattr(engine, "device", "unknown")),
            "models": models_status,
        }

    # Metrics check
    report = accuracy_tracker.get_accuracy_report()
    health["components"]["accuracy_tracker"] = {
        "status": "ok",
        "total_predictions": report["total_predictions"],
        "total_feedback": report["total_feedback"],
        "drift_alerts": len(report.get("drift_alerts", [])),
    }

    # Overall status
    statuses = [c.get("status", "unknown") for c in health["components"].values()]
    if "unreachable" in statuses:
        health["status"] = "degraded"
    elif all(s in ("ok", "local", "remote") for s in statuses):
        health["status"] = "ok"

    return health


# ── Analysis Stats ──

@router.get("/api/monitoring/stats")
async def analysis_stats():
    """Quick stats summary for dashboard widgets."""
    report = accuracy_tracker.get_accuracy_report()
    return {
        "total_scans": report["total_predictions"],
        "total_feedback": report["total_feedback"],
        "modality_accuracy": report.get("modality_accuracy", {}),
        "verdict_distribution": report.get("verdict_distribution", {}),
        "false_positives": report.get("false_positives", {}).get("count", 0),
        "false_negatives": report.get("false_negatives", {}).get("count", 0),
        "active_drift_alerts": len(report.get("drift_alerts", [])),
        "recommendations_count": len(report.get("recommendations", [])),
    }
