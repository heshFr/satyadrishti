"""
Sentiment Trajectory Analyzer
================================
Tracks emotional manipulation patterns across conversation turns.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
Emotional manipulation in scam calls follows specific trajectories:

1. FEAR ESCALATION:
   neutral → mild concern → anxiety → fear → panic
   - Scammer gradually increases threat level
   - Each message adds more urgency/consequences
   - Goal: victim too scared to think rationally

2. EMOTIONAL SEESAW:
   positive → negative → positive → negative
   - Alternating between hope and fear
   - "You've won a prize" → "but you'll lose it if you don't act now"
   - Creates emotional dependency on the scammer

3. GUILT/SHAME TRAJECTORY:
   neutral → blame → guilt → compliance
   - "This happened because of YOUR negligence"
   - "Your family will suffer because of you"
   - Goal: victim complies to alleviate guilt

4. AUTHORITY PRESSURE:
   respect → submission → compliance
   - Leverage authority figures (police, bank, government)
   - Victim increasingly submissive to perceived authority

DETECTION METHOD:
━━━━━━━━━━━━━━━━━
Uses a lightweight lexicon-based sentiment analyzer (no external model needed)
combined with domain-specific emotion patterns for Indian scam contexts.

Tracks:
  - Valence (positive/negative) per message
  - Arousal (calm/excited) per message
  - Dominance (submissive/dominant) per message
  - Emotion category (fear/anger/sadness/joy/neutral)
  - Trajectory analysis (slope, volatility, pattern matching)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


# ── Emotion Lexicons ──
# Domain-specific for Indian scam/coercion context
# Each word maps to (valence, arousal, dominance) in range [-1, 1]

EMOTION_LEXICON = {
    # Fear / Threat words (negative valence, high arousal)
    "arrested": (-0.8, 0.9, -0.7),
    "arrest": (-0.8, 0.9, -0.7),
    "jail": (-0.9, 0.8, -0.8),
    "prison": (-0.9, 0.8, -0.8),
    "police": (-0.3, 0.7, -0.5),
    "court": (-0.4, 0.6, -0.4),
    "legal": (-0.3, 0.5, -0.3),
    "lawsuit": (-0.6, 0.7, -0.5),
    "warrant": (-0.7, 0.8, -0.7),
    "fraud": (-0.7, 0.7, -0.5),
    "illegal": (-0.7, 0.7, -0.5),
    "crime": (-0.8, 0.8, -0.6),
    "criminal": (-0.8, 0.8, -0.6),
    "penalty": (-0.6, 0.6, -0.5),
    "fine": (-0.4, 0.5, -0.4),
    "blocked": (-0.5, 0.6, -0.5),
    "frozen": (-0.5, 0.6, -0.5),
    "suspended": (-0.5, 0.6, -0.5),
    "terminated": (-0.6, 0.6, -0.6),
    "hacked": (-0.7, 0.8, -0.6),
    "compromised": (-0.6, 0.7, -0.5),
    "danger": (-0.8, 0.9, -0.7),
    "dangerous": (-0.8, 0.9, -0.7),
    "threat": (-0.7, 0.8, -0.6),
    "threatening": (-0.7, 0.8, -0.6),
    "risk": (-0.5, 0.6, -0.4),

    # Urgency words (negative valence, very high arousal)
    "immediately": (-0.3, 0.9, -0.3),
    "urgently": (-0.4, 0.9, -0.4),
    "urgent": (-0.4, 0.9, -0.4),
    "emergency": (-0.6, 0.95, -0.5),
    "hurry": (-0.3, 0.8, -0.3),
    "quickly": (-0.2, 0.7, -0.2),
    "deadline": (-0.4, 0.7, -0.4),
    "expire": (-0.5, 0.7, -0.5),
    "last": (-0.3, 0.6, -0.3),

    # Financial words (contextually negative in scam)
    "transfer": (-0.2, 0.5, -0.3),
    "payment": (-0.2, 0.4, -0.2),
    "money": (-0.1, 0.4, -0.2),
    "amount": (-0.1, 0.3, -0.1),
    "account": (-0.1, 0.3, -0.1),
    "bank": (0.0, 0.3, -0.1),
    "otp": (-0.3, 0.6, -0.5),
    "pin": (-0.3, 0.6, -0.5),
    "password": (-0.3, 0.6, -0.5),

    # Guilt/shame words
    "fault": (-0.6, 0.5, -0.6),
    "blame": (-0.6, 0.6, -0.6),
    "responsible": (-0.3, 0.4, -0.4),
    "negligent": (-0.5, 0.5, -0.5),
    "careless": (-0.5, 0.5, -0.5),
    "suffer": (-0.7, 0.7, -0.6),
    "consequence": (-0.5, 0.5, -0.4),

    # Isolation words
    "secret": (-0.3, 0.5, -0.4),
    "confidential": (-0.2, 0.4, -0.3),
    "private": (-0.1, 0.3, -0.2),

    # Positive / reward words (used in hook stage)
    "congratulations": (0.8, 0.7, 0.3),
    "winner": (0.7, 0.7, 0.3),
    "prize": (0.6, 0.6, 0.2),
    "reward": (0.6, 0.5, 0.2),
    "bonus": (0.5, 0.5, 0.2),
    "benefit": (0.4, 0.4, 0.1),
    "eligible": (0.3, 0.4, 0.1),
    "selected": (0.4, 0.5, 0.2),
    "lucky": (0.5, 0.5, 0.2),
    "free": (0.4, 0.4, 0.1),
    "cashback": (0.4, 0.5, 0.2),

    # Neutral / professional
    "verify": (0.0, 0.3, 0.0),
    "confirm": (0.0, 0.3, 0.0),
    "process": (0.0, 0.2, 0.0),
    "update": (0.0, 0.2, 0.0),
    "information": (0.0, 0.2, 0.0),

    # Hindi/Hinglish (common in Indian scam calls)
    "paisa": (-0.2, 0.5, -0.3),
    "khatarnak": (-0.8, 0.9, -0.7),  # dangerous
    "giraftar": (-0.8, 0.9, -0.7),   # arrest
    "jaldi": (-0.3, 0.8, -0.3),      # quickly
    "turant": (-0.4, 0.9, -0.4),     # immediately
    "khatre": (-0.7, 0.8, -0.6),     # danger
    "galti": (-0.5, 0.5, -0.5),      # mistake/fault
}

# Emotion categories derived from VAD values
EMOTION_CATEGORIES = {
    "fear": {"valence": (-1, -0.3), "arousal": (0.5, 1.0)},
    "anger": {"valence": (-1, -0.3), "arousal": (0.3, 1.0)},
    "sadness": {"valence": (-1, -0.2), "arousal": (-1, 0.3)},
    "anxiety": {"valence": (-0.5, 0.0), "arousal": (0.4, 1.0)},
    "joy": {"valence": (0.3, 1.0), "arousal": (0.0, 1.0)},
    "neutral": {"valence": (-0.2, 0.2), "arousal": (-0.5, 0.5)},
}


class SentimentTrajectory:
    """
    Tracks emotional sentiment across conversation turns and detects
    manipulation trajectories.

    Lightweight: uses lexicon-based analysis, no ML model required.
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of recent messages for trajectory analysis.
        """
        self.window_size = window_size
        self.valence_history: List[float] = []
        self.arousal_history: List[float] = []
        self.dominance_history: List[float] = []
        self.emotion_history: List[str] = []

    def analyze_message(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single message and update trajectory.

        Args:
            text: Message text

        Returns:
            Dict with per-message sentiment and trajectory analysis.
        """
        # Per-message sentiment
        valence, arousal, dominance, matched_words = self._compute_vad(text)
        emotion = self._classify_emotion(valence, arousal)

        self.valence_history.append(valence)
        self.arousal_history.append(arousal)
        self.dominance_history.append(dominance)
        self.emotion_history.append(emotion)

        # Trajectory analysis
        trajectory = self._analyze_trajectory()

        # Manipulation score from trajectory
        manipulation_score = self._compute_manipulation_score(trajectory)

        return {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "emotion": emotion,
            "matched_words": matched_words[:10],
            "trajectory": trajectory,
            "manipulation_score": round(manipulation_score, 4),
            "score": round(manipulation_score, 4),
            "confidence": round(min(1.0, len(self.valence_history) / 5.0) * 0.6 + 0.2, 4),
        }

    def _compute_vad(self, text: str) -> Tuple[float, float, float, List[str]]:
        """
        Compute Valence-Arousal-Dominance from lexicon matching.
        """
        words = re.findall(r'\b[a-zA-Z\u0900-\u097F]+\b', text.lower())

        valences = []
        arousals = []
        dominances = []
        matched = []

        for word in words:
            if word in EMOTION_LEXICON:
                v, a, d = EMOTION_LEXICON[word]
                valences.append(v)
                arousals.append(a)
                dominances.append(d)
                matched.append(word)

        if not valences:
            return 0.0, 0.0, 0.0, []

        # Weight more extreme values slightly higher
        weights = [abs(v) + 0.5 for v in valences]
        total_weight = sum(weights)

        valence = sum(v * w for v, w in zip(valences, weights)) / total_weight
        arousal = sum(a * w for a, w in zip(arousals, weights)) / total_weight
        dominance = sum(d * w for d, w in zip(dominances, weights)) / total_weight

        return float(valence), float(arousal), float(dominance), matched

    def _classify_emotion(self, valence: float, arousal: float) -> str:
        """Classify emotion from VAD values."""
        best_emotion = "neutral"
        best_fit = -1

        for emotion, ranges in EMOTION_CATEGORIES.items():
            v_low, v_high = ranges["valence"]
            a_low, a_high = ranges["arousal"]

            if v_low <= valence <= v_high and a_low <= arousal <= a_high:
                # Distance from category center
                v_center = (v_low + v_high) / 2
                a_center = (a_low + a_high) / 2
                fit = 1.0 - np.sqrt((valence - v_center)**2 + (arousal - a_center)**2)
                if fit > best_fit:
                    best_fit = fit
                    best_emotion = emotion

        return best_emotion

    def _analyze_trajectory(self) -> Dict[str, Any]:
        """
        Analyze emotional trajectory over recent messages.
        """
        n = len(self.valence_history)
        window = min(n, self.window_size)

        if window < 2:
            return {
                "valence_slope": 0.0,
                "arousal_slope": 0.0,
                "dominance_slope": 0.0,
                "volatility": 0.0,
                "pattern": "insufficient_data",
                "emotion_sequence": self.emotion_history[-window:],
            }

        recent_v = self.valence_history[-window:]
        recent_a = self.arousal_history[-window:]
        recent_d = self.dominance_history[-window:]

        # Slopes (direction of emotional change)
        x = np.arange(window, dtype=float)

        def simple_slope(y_vals):
            y = np.array(y_vals)
            n = len(y)
            return float((n * np.sum(x[:n] * y) - np.sum(x[:n]) * np.sum(y)) /
                        (n * np.sum(x[:n]**2) - np.sum(x[:n])**2 + 1e-10))

        valence_slope = simple_slope(recent_v)
        arousal_slope = simple_slope(recent_a)
        dominance_slope = simple_slope(recent_d)

        # Volatility (how much emotion swings)
        valence_volatility = float(np.std(np.diff(recent_v))) if len(recent_v) > 1 else 0
        arousal_volatility = float(np.std(np.diff(recent_a))) if len(recent_a) > 1 else 0
        volatility = (valence_volatility + arousal_volatility) / 2.0

        # Pattern detection
        pattern = self._detect_pattern(recent_v, recent_a, recent_d)

        # Emotion sequence
        emotion_seq = self.emotion_history[-window:]

        return {
            "valence_slope": round(valence_slope, 4),
            "arousal_slope": round(arousal_slope, 4),
            "dominance_slope": round(dominance_slope, 4),
            "volatility": round(volatility, 4),
            "valence_volatility": round(valence_volatility, 4),
            "arousal_volatility": round(arousal_volatility, 4),
            "pattern": pattern,
            "emotion_sequence": emotion_seq,
            "valence_mean": round(float(np.mean(recent_v)), 4),
            "arousal_mean": round(float(np.mean(recent_a)), 4),
            "dominance_mean": round(float(np.mean(recent_d)), 4),
        }

    def _detect_pattern(
        self,
        valence: List[float],
        arousal: List[float],
        dominance: List[float],
    ) -> str:
        """
        Detect manipulation trajectory patterns.
        """
        if len(valence) < 3:
            return "insufficient_data"

        v = np.array(valence)
        a = np.array(arousal)
        d = np.array(dominance)

        # Pattern 1: Fear Escalation (decreasing valence + increasing arousal)
        v_decreasing = np.all(np.diff(v[-4:]) < 0.05) if len(v) >= 4 else False
        a_increasing = np.all(np.diff(a[-4:]) > -0.05) if len(a) >= 4 else False
        if v_decreasing and a_increasing and np.mean(v[-3:]) < -0.3:
            return "fear_escalation"

        # Pattern 2: Emotional Seesaw (high volatility with alternating signs)
        if len(v) >= 4:
            diffs = np.diff(v[-5:])
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes >= 2 and np.std(v[-5:]) > 0.3:
                return "emotional_seesaw"

        # Pattern 3: Guilt/Shame (decreasing dominance)
        if len(d) >= 3 and np.mean(d[-3:]) < -0.3 and d[-1] < d[0]:
            return "guilt_trajectory"

        # Pattern 4: Authority Pressure (decreasing dominance + negative valence)
        if np.mean(d[-3:]) < -0.3 and np.mean(v[-3:]) < -0.2:
            return "authority_pressure"

        # Pattern 5: Sustained negative (consistently negative valence)
        if np.mean(v) < -0.3 and np.mean(a) > 0.3:
            return "sustained_negative"

        # Pattern 6: Positive to negative shift (started positive, now negative)
        if len(v) >= 4 and np.mean(v[:2]) > 0.1 and np.mean(v[-2:]) < -0.2:
            return "positive_to_negative"

        return "normal"

    def _compute_manipulation_score(self, trajectory: Dict[str, Any]) -> float:
        """
        Compute manipulation probability from trajectory analysis.
        """
        scores = []

        # Negative valence slope = increasing negativity
        v_slope = trajectory.get("valence_slope", 0.0)
        if v_slope < -0.05:
            scores.append(min(1.0, abs(v_slope) * 5))

        # Increasing arousal = escalating pressure
        a_slope = trajectory.get("arousal_slope", 0.0)
        if a_slope > 0.03:
            scores.append(min(1.0, a_slope * 5))

        # Decreasing dominance = victim losing power
        d_slope = trajectory.get("dominance_slope", 0.0)
        if d_slope < -0.03:
            scores.append(min(1.0, abs(d_slope) * 5))

        # High volatility = emotional manipulation
        volatility = trajectory.get("volatility", 0.0)
        if volatility > 0.2:
            scores.append(min(1.0, volatility * 2))

        # Pattern-specific scores
        pattern_scores = {
            "fear_escalation": 0.8,
            "emotional_seesaw": 0.7,
            "guilt_trajectory": 0.7,
            "authority_pressure": 0.6,
            "sustained_negative": 0.5,
            "positive_to_negative": 0.5,
            "normal": 0.0,
            "insufficient_data": 0.0,
        }
        pattern = trajectory.get("pattern", "normal")
        pattern_score = pattern_scores.get(pattern, 0.0)
        if pattern_score > 0:
            scores.append(pattern_score)

        # Mean negative valence
        v_mean = trajectory.get("valence_mean", 0.0)
        if v_mean < -0.3:
            scores.append(min(1.0, abs(v_mean)))

        if not scores:
            return 0.0

        return float(np.clip(np.max(scores) * 0.6 + np.mean(scores) * 0.4, 0, 1))

    def reset(self):
        """Reset trajectory for a new conversation."""
        self.valence_history.clear()
        self.arousal_history.clear()
        self.dominance_history.clear()
        self.emotion_history.clear()
