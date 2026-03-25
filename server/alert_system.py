"""
Satya Drishti — Family Alert System
=====================================
Sends alerts to designated family members or authorities when
threats are detected during call protection.

Channels:
- In-app push notification (via WebSocket to connected family devices)
- SMS alert (via Twilio or similar — requires API key)
- Email alert (via SMTP)

Privacy-first: Only sends alert level and recommendation,
never the actual call audio or transcript.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

log = logging.getLogger("satyadrishti.alerts")


class AlertSystem:
    """Manages threat alerts for family protection."""

    def __init__(self):
        # Connected family member WebSocket sessions
        self.family_connections: Dict[str, list] = {}  # user_id -> [websockets]
        self.alert_history: List[Dict] = []

    def register_family_device(self, user_id: str, websocket):
        """Register a family member's device for receiving alerts."""
        if user_id not in self.family_connections:
            self.family_connections[user_id] = []
        self.family_connections[user_id].append(websocket)
        log.info(f"Family device registered for user {user_id}")

    def unregister_family_device(self, user_id: str, websocket):
        """Unregister a family member's device."""
        if user_id in self.family_connections:
            self.family_connections[user_id] = [
                ws for ws in self.family_connections[user_id]
                if ws != websocket
            ]

    async def send_alert(
        self,
        user_id: str,
        alert_level: str,
        message: str,
        details: Dict = None,
    ):
        """
        Send a threat alert to all registered family devices.

        Privacy: We send the alert LEVEL and recommendation,
        but NEVER the call audio or transcript content.
        """
        alert = {
            "type": "family_alert",
            "timestamp": datetime.utcnow().isoformat(),
            "alert_level": alert_level,
            "message": message,
            "user_id": user_id,
        }

        # Add non-sensitive details
        if details:
            alert["deepfake_detected"] = details.get("deepfake_detected", False)
            alert["coercion_detected"] = details.get("coercion_detected", False)
            alert["recommendation"] = details.get("recommendation", "")

        self.alert_history.append(alert)

        # Send to all connected family devices
        connections = self.family_connections.get(user_id, [])
        for ws in connections:
            try:
                await ws.send_json(alert)
            except Exception:
                pass  # Device disconnected

        log.info(f"Alert sent for user {user_id}: {alert_level}")

        return alert


# Global singleton
alert_system = AlertSystem()
