"""
Conversation Context Analyzer
================================
Detects coercion and manipulation patterns across MULTI-TURN conversations,
not just individual messages. This is critical because:

1. Coercion is rarely a single message — it's a TRAJECTORY
2. Scammers follow predictable escalation patterns
3. Single-message detection misses gradual manipulation
4. The most dangerous attacks build trust before exploiting it

COERCION ESCALATION PATTERNS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real scam calls follow these stages:

Stage 1: APPROACH (Trust Building)
  - Impersonation: "I'm calling from SBI bank / police station"
  - Familiarity: "Is this [victim's name]? Your [relative] gave your number"
  - Authority: "This is Inspector Sharma from cyber crime division"

Stage 2: HOOK (Problem Introduction)
  - Urgency: "Your account has been flagged for suspicious activity"
  - Fear: "A case has been filed against your Aadhaar number"
  - Opportunity: "You've won a lottery / government scheme benefit"

Stage 3: PRESSURE (Emotional Manipulation)
  - Time pressure: "You must act in the next 30 minutes"
  - Isolation: "Don't tell anyone or the case will get worse"
  - Consequences: "Your account will be frozen / you'll be arrested"

Stage 4: EXTRACTION (The Ask)
  - Financial: "Transfer ₹X to this account for verification"
  - Information: "Share your OTP / Aadhaar / bank details"
  - Action: "Install this app / click this link"

Stage 5: REINFORCEMENT (Keeping Control)
  - Repeated urgency: "Have you done it yet?"
  - Emotional guilt: "Your family will suffer if you don't act"
  - Threats: "Police will come to your house"

DETECTION METHOD:
━━━━━━━━━━━━━━━━━
1. Per-message scoring (using coercion detector)
2. Stage classification (which stage is the conversation at?)
3. Escalation tracking (is urgency/pressure increasing over time?)
4. Topic trajectory (normal → financial → threats = red flag)
5. Power dynamic analysis (who is controlling the conversation?)
6. Information extraction attempts (requests for OTP, bank details, etc.)

OUTPUT:
━━━━━━━
  {
      "conversation_threat_level": float (0-1),
      "current_stage": "approach" | "hook" | "pressure" | "extraction" | "reinforcement",
      "escalation_rate": float (how fast threat is increasing),
      "stages_detected": [ ... ],
      "topic_trajectory": [ ... ],
      "information_extraction_attempts": [ ... ],
      "recommendation": str,
      "alert_level": "safe" | "caution" | "warning" | "danger" | "critical",
  }
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


# ── Stage Detection Patterns ──
# Each pattern maps to a conversation stage and contributes to threat scoring

STAGE_PATTERNS = {
    "approach": {
        "authority_impersonation": [
            r"(?:i am|this is|calling from)\s+(?:police|bank|sbi|rbi|income tax|cyber|officer|inspector|manager)",
            r"(?:government|ministry|department)\s+(?:of|from)",
            r"(?:your|the)\s+(?:case|complaint|account)\s+(?:number|id|reference)",
            r"(?:verification|security)\s+(?:team|department|division)",
        ],
        "familiarity_claim": [
            r"(?:your|the)\s+(?:son|daughter|husband|wife|father|mother|brother|sister|relative|friend)\s+(?:gave|told|said|sent)",
            r"(?:i know|we know)\s+(?:your|about you)",
            r"(?:remember me|you know me|we met)",
        ],
    },
    "hook": {
        "problem_introduction": [
            r"(?:suspicious|illegal|unauthorized|fraudulent)\s+(?:activity|transaction|access|login)",
            r"(?:your|the)\s+(?:account|aadhaar|pan|card)\s+(?:has been|is|was)\s+(?:compromised|blocked|flagged|hacked|misused)",
            r"(?:case|complaint|fir)\s+(?:filed|registered|lodged)\s+(?:against|on|for)",
            r"(?:warrant|arrest|summon)\s+(?:issued|pending|against)",
        ],
        "opportunity_bait": [
            r"(?:won|winner|selected|eligible|entitled)\s+(?:a|the|for)\s+(?:prize|lottery|reward|benefit|subsidy|cashback)",
            r"(?:government|pm|cm)\s+(?:scheme|yojana|benefit|subsidy)",
            r"(?:free|bonus|extra|additional)\s+(?:money|cash|amount|reward)",
        ],
    },
    "pressure": {
        "time_urgency": [
            r"(?:within|in the next|before)\s+(?:\d+)\s+(?:minutes?|hours?|days?)",
            r"(?:immediately|urgently|right now|at once|asap|right away)",
            r"(?:deadline|expires?|last chance|final warning|limited time)",
            r"(?:today only|today itself|before end of day)",
        ],
        "isolation_tactics": [
            r"(?:don't|do not|never)\s+(?:tell|inform|share|discuss|mention)\s+(?:anyone|anybody|family|police|bank)",
            r"(?:keep this|this is)\s+(?:confidential|secret|private|between us)",
            r"(?:if you tell|telling)\s+(?:anyone|others)\s+(?:will|can)\s+(?:make|cause|worsen)",
        ],
        "consequence_threats": [
            r"(?:will be|shall be|going to be)\s+(?:arrested|jailed|prosecuted|fined|blacklisted|blocked)",
            r"(?:your|the)\s+(?:family|children|home|property)\s+(?:will|shall)\s+(?:suffer|lose|be affected)",
            r"(?:legal|criminal|civil)\s+(?:action|proceedings|case)\s+(?:will|shall)",
            r"(?:police|enforcement|authorities)\s+(?:will come|are coming|on the way)",
        ],
    },
    "extraction": {
        "financial_request": [
            r"(?:transfer|send|pay|deposit)\s+(?:₹|rs\.?|rupees?|inr)\s*\d+",
            r"(?:transfer|send|pay)\s+(?:to|into)\s+(?:this|the|following)\s+(?:account|upi|number)",
            r"(?:upi|gpay|phonepe|paytm|neft|rtgs|imps)\s+(?:id|number|transfer)",
            r"(?:processing|verification|security|refundable)\s+(?:fee|charge|deposit|amount)",
        ],
        "information_request": [
            r"(?:share|tell|give|provide|send)\s+(?:your|the)\s+(?:otp|pin|password|cvv|card number|account number|aadhaar|pan)",
            r"(?:what is|tell me)\s+(?:your|the)\s+(?:otp|pin|password|cvv|card|account|aadhaar)",
            r"(?:enter|type|input)\s+(?:your|the)\s+(?:otp|pin|password|details)",
            r"(?:verification|security)\s+(?:code|number|otp|pin)",
        ],
        "action_request": [
            r"(?:install|download|open)\s+(?:this|the|an)\s+(?:app|application|software|anydesk|teamviewer|quick support)",
            r"(?:click|tap|open)\s+(?:this|the|on)\s+(?:link|url|website)",
            r"(?:scan|use)\s+(?:this|the)\s+(?:qr code|barcode)",
        ],
    },
    "reinforcement": {
        "repeated_pressure": [
            r"(?:have you|did you|are you)\s+(?:done|completed|finished|transferred|sent|shared)",
            r"(?:hurry|faster|quickly|we're waiting|time is running out|still waiting)",
            r"(?:why|what|how)\s+(?:haven't you|are you not|is it not|still not)",
        ],
        "emotional_guilt": [
            r"(?:your family|your children|your parents)\s+(?:will|are going to)\s+(?:suffer|pay|lose)",
            r"(?:it's your|this is your)\s+(?:fault|responsibility|problem)",
            r"(?:you|yourself)\s+(?:caused|brought|created)\s+(?:this|the problem)",
        ],
    },
}

# Sensitive information patterns (for extraction tracking)
SENSITIVE_INFO_PATTERNS = {
    "otp": r"\b(?:otp|one[\s-]?time[\s-]?(?:password|pin|code))\b",
    "bank_details": r"\b(?:account\s*(?:number|no)|ifsc|bank\s*(?:name|details)|branch)\b",
    "card_details": r"\b(?:card\s*(?:number|no)|cvv|expiry|credit\s*card|debit\s*card)\b",
    "identity": r"\b(?:aadhaar|aadhar|pan\s*(?:card|number)|passport|voter\s*id)\b",
    "credentials": r"\b(?:password|pin|login|username|mpin|upi\s*pin)\b",
    "money_transfer": r"\b(?:₹|rs\.?\s*)\d{2,}|(?:transfer|send|pay)\s+\d{2,}\b",
}

# Alert level thresholds
ALERT_THRESHOLDS = {
    "safe": (0.0, 0.15),
    "caution": (0.15, 0.35),
    "warning": (0.35, 0.55),
    "danger": (0.55, 0.75),
    "critical": (0.75, 1.0),
}


class ConversationAnalyzer:
    """
    Tracks and analyzes multi-turn conversations for coercion patterns.

    Maintains conversation state across messages for a single session.
    Create one instance per conversation/call.
    """

    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: Maximum number of messages to retain in history.
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.stage_history: List[str] = []
        self.threat_trajectory: List[float] = []
        self.info_extraction_attempts: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self._compiled_patterns: Dict[str, Dict[str, List[re.Pattern]]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regex patterns for efficiency."""
        for stage, categories in STAGE_PATTERNS.items():
            self._compiled_patterns[stage] = {}
            for category, patterns in categories.items():
                self._compiled_patterns[stage][category] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

        self._sensitive_compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in SENSITIVE_INFO_PATTERNS.items()
        }

    def add_message(
        self,
        text: str,
        speaker: str = "caller",
        per_message_score: float = 0.0,
        per_message_label: str = "safe",
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to the conversation and return updated analysis.

        Args:
            text: The message text
            speaker: "caller" or "victim" (who is speaking)
            per_message_score: Coercion score from single-message detector (0-1)
            per_message_label: Coercion label from single-message detector
            timestamp: Unix timestamp (default: current time)

        Returns:
            Full conversation analysis result.
        """
        if timestamp is None:
            timestamp = time.time()

        # Classify this message into conversation stage
        stage_scores = self._classify_stage(text)
        primary_stage = max(stage_scores, key=stage_scores.get) if stage_scores else "unknown"
        primary_stage_score = stage_scores.get(primary_stage, 0.0)

        # Detect information extraction attempts
        info_attempts = self._detect_info_extraction(text)

        # Compute per-message threat score (combining ML and pattern matching)
        pattern_score = self._compute_pattern_score(text, stage_scores)
        message_threat = max(per_message_score, pattern_score)

        # Store message
        message_entry = {
            "text": text[:500],  # cap length for memory
            "speaker": speaker,
            "timestamp": timestamp,
            "ml_score": per_message_score,
            "ml_label": per_message_label,
            "pattern_score": pattern_score,
            "message_threat": message_threat,
            "stage": primary_stage,
            "stage_score": primary_stage_score,
            "stage_scores": stage_scores,
            "info_attempts": info_attempts,
        }

        self.history.append(message_entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self.stage_history.append(primary_stage)
        self.threat_trajectory.append(message_threat)

        if info_attempts:
            for attempt in info_attempts:
                self.info_extraction_attempts.append({
                    "type": attempt,
                    "message_index": len(self.history) - 1,
                    "timestamp": timestamp,
                })

        # Compute conversation-level analysis
        return self._compute_conversation_analysis()

    def _classify_stage(self, text: str) -> Dict[str, float]:
        """Classify text into conversation stages by pattern matching."""
        stage_scores = {}

        for stage, categories in self._compiled_patterns.items():
            matches = 0
            total_patterns = 0
            for category, patterns in categories.items():
                for pattern in patterns:
                    total_patterns += 1
                    if pattern.search(text):
                        matches += 1

            if total_patterns > 0:
                stage_scores[stage] = matches / total_patterns
            else:
                stage_scores[stage] = 0.0

        return stage_scores

    def _detect_info_extraction(self, text: str) -> List[str]:
        """Detect attempts to extract sensitive information."""
        attempts = []
        for info_type, pattern in self._sensitive_compiled.items():
            if pattern.search(text):
                attempts.append(info_type)
        return attempts

    def _compute_pattern_score(self, text: str, stage_scores: Dict[str, float]) -> float:
        """
        Compute threat score from pattern matching.

        Extraction and pressure patterns score highest.
        """
        weights = {
            "approach": 0.15,
            "hook": 0.25,
            "pressure": 0.55,
            "extraction": 0.80,
            "reinforcement": 0.65,
        }

        if not stage_scores:
            return 0.0

        weighted_score = sum(
            stage_scores.get(stage, 0.0) * weight
            for stage, weight in weights.items()
        )

        return min(1.0, weighted_score)

    def _compute_conversation_analysis(self) -> Dict[str, Any]:
        """Compute full conversation-level threat analysis."""

        n_messages = len(self.history)

        if n_messages == 0:
            return self._empty_result()

        # 1. Current conversation stage (most recent dominant stage)
        recent_stages = self.stage_history[-5:]  # last 5 messages
        stage_counts = {}
        for s in recent_stages:
            stage_counts[s] = stage_counts.get(s, 0) + 1
        current_stage = max(stage_counts, key=stage_counts.get) if stage_counts else "unknown"

        # 2. Stages detected throughout conversation
        all_stages = list(set(self.stage_history))

        # 3. Escalation rate (slope of threat trajectory)
        if len(self.threat_trajectory) >= 3:
            # Linear regression on threat scores
            x = np.arange(len(self.threat_trajectory), dtype=float)
            y = np.array(self.threat_trajectory)
            # Simple slope calculation
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                    (n * np.sum(x**2) - np.sum(x)**2 + 1e-10)
            escalation_rate = float(np.clip(slope * 10, -1, 1))  # scale to [-1, 1]
        else:
            escalation_rate = 0.0

        # 4. Topic trajectory
        topic_trajectory = []
        for i, msg in enumerate(self.history[-10:]):
            topic_trajectory.append({
                "index": i,
                "stage": msg["stage"],
                "threat": round(msg["message_threat"], 3),
            })

        # 5. Conversation threat level
        # Combines: current threat, escalation, stage progression, extraction attempts
        threat_components = []

        # a. Current threat (weighted recent messages)
        recent_threats = [m["message_threat"] for m in self.history[-5:]]
        current_threat = float(np.mean(recent_threats)) if recent_threats else 0.0
        threat_components.append(("current_threat", current_threat, 0.30))

        # b. Peak threat (maximum threat seen)
        peak_threat = float(np.max(self.threat_trajectory)) if self.threat_trajectory else 0.0
        threat_components.append(("peak_threat", peak_threat, 0.20))

        # c. Escalation (positive escalation = increasing danger)
        escalation_contribution = max(0.0, escalation_rate) * 0.5
        threat_components.append(("escalation", escalation_contribution, 0.15))

        # d. Stage progression (reaching extraction stage is very bad)
        stage_danger = {
            "approach": 0.1,
            "hook": 0.3,
            "pressure": 0.6,
            "extraction": 0.9,
            "reinforcement": 0.8,
            "unknown": 0.0,
        }
        current_stage_danger = stage_danger.get(current_stage, 0.0)
        threat_components.append(("stage_danger", current_stage_danger, 0.20))

        # e. Information extraction attempts
        n_info_attempts = len(self.info_extraction_attempts)
        info_danger = min(1.0, n_info_attempts * 0.25)
        threat_components.append(("info_extraction", info_danger, 0.15))

        # Weighted average
        conversation_threat = sum(
            score * weight for _, score, weight in threat_components
        )
        conversation_threat = float(np.clip(conversation_threat, 0, 1))

        # 6. Alert level
        alert_level = "safe"
        for level, (low, high) in ALERT_THRESHOLDS.items():
            if low <= conversation_threat < high:
                alert_level = level
                break
        if conversation_threat >= 0.75:
            alert_level = "critical"

        # 7. Recommendation
        recommendation = self._generate_recommendation(
            alert_level, current_stage, n_info_attempts, conversation_threat
        )

        # 8. Power dynamic (who is controlling the conversation?)
        caller_threats = [
            m["message_threat"] for m in self.history if m["speaker"] == "caller"
        ]
        victim_threats = [
            m["message_threat"] for m in self.history if m["speaker"] == "victim"
        ]
        if caller_threats:
            power_imbalance = float(np.mean(caller_threats)) - float(
                np.mean(victim_threats) if victim_threats else 0
            )
        else:
            power_imbalance = 0.0

        duration = time.time() - self.start_time

        return {
            "conversation_threat_level": round(conversation_threat, 4),
            "current_stage": current_stage,
            "escalation_rate": round(escalation_rate, 4),
            "stages_detected": all_stages,
            "topic_trajectory": topic_trajectory,
            "information_extraction_attempts": [
                {"type": a["type"], "message_index": a["message_index"]}
                for a in self.info_extraction_attempts[-10:]
            ],
            "power_imbalance": round(power_imbalance, 4),
            "recommendation": recommendation,
            "alert_level": alert_level,
            "threat_components": {
                name: round(score, 4) for name, score, _ in threat_components
            },
            "statistics": {
                "total_messages": n_messages,
                "caller_messages": sum(1 for m in self.history if m["speaker"] == "caller"),
                "victim_messages": sum(1 for m in self.history if m["speaker"] == "victim"),
                "duration_seconds": round(duration, 1),
                "peak_threat": round(peak_threat, 4),
                "info_extraction_count": n_info_attempts,
            },
            "score": round(conversation_threat, 4),
            "confidence": round(min(1.0, n_messages / 5.0) * 0.7 + 0.3, 4),
        }

    def _generate_recommendation(
        self,
        alert_level: str,
        current_stage: str,
        n_info_attempts: int,
        threat_level: float,
    ) -> str:
        """Generate actionable recommendation based on analysis."""
        if alert_level == "safe":
            return "No coercion indicators detected. Conversation appears normal."

        if alert_level == "caution":
            return (
                "Mild pressure language detected. Monitor the conversation. "
                "Do not share personal information unless you initiated the contact."
            )

        if alert_level == "warning":
            if current_stage == "pressure":
                return (
                    "WARNING: Pressure tactics detected! The caller is using urgency and threats. "
                    "Do NOT share OTP, bank details, or make any transfers. "
                    "Real banks and police NEVER ask for OTP or UPI PIN over phone."
                )
            return (
                "WARNING: Suspicious conversation pattern detected. "
                "Verify the caller's identity independently before sharing any information."
            )

        if alert_level == "danger":
            if n_info_attempts > 0:
                return (
                    "DANGER: Active attempt to extract sensitive information! "
                    "This appears to be a scam call. Hang up immediately. "
                    "Report to Cyber Crime helpline 1930."
                )
            return (
                "DANGER: High-risk coercion pattern detected. "
                "The caller is likely a scammer. Do NOT transfer money or share details. "
                "Hang up and call your bank directly from the number on your card."
            )

        # critical
        return (
            "CRITICAL ALERT: This is almost certainly a scam/fraud call. "
            "Multiple coercion tactics, information extraction attempts, and "
            "pressure patterns detected. HANG UP NOW. "
            "Call Cyber Crime helpline 1930 or local police. "
            "Do NOT call back the same number."
        )

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "conversation_threat_level": 0.0,
            "current_stage": "unknown",
            "escalation_rate": 0.0,
            "stages_detected": [],
            "topic_trajectory": [],
            "information_extraction_attempts": [],
            "power_imbalance": 0.0,
            "recommendation": "No messages analyzed yet.",
            "alert_level": "safe",
            "threat_components": {},
            "statistics": {
                "total_messages": 0,
                "caller_messages": 0,
                "victim_messages": 0,
                "duration_seconds": 0.0,
                "peak_threat": 0.0,
                "info_extraction_count": 0,
            },
            "score": 0.0,
            "confidence": 0.0,
        }

    def reset(self):
        """Reset conversation state for a new conversation."""
        self.history.clear()
        self.stage_history.clear()
        self.threat_trajectory.clear()
        self.info_extraction_attempts.clear()
        self.start_time = time.time()
