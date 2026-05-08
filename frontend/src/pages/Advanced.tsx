import Layout from "@/components/Layout";
import MaterialIcon from "@/components/MaterialIcon";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useRef, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart,
  Tooltip,
} from "recharts";
import { WS_BASE } from "@/lib/api";

// ─── Types ─────────────────────────────────────────────────────────────

interface AudioDataPoint {
  time: number;
  low: number;
  mid: number;
  high: number;
}

interface ConfidenceScore {
  label: string;
  score: number;
  verdict: string;
  color: string;
  icon: string;
}

interface TranscriptLine {
  time: string;
  text: string;
  flagged: boolean;
  stage?: string;
}

interface SentimentPoint {
  time: number;
  fear: number;
  urgency: number;
  trust: number;
  label: string;
}

type ConversationStage = "idle" | "approach" | "hook" | "pressure" | "extraction";
type ThreatLevel = "safe" | "caution" | "elevated" | "critical";

// ─── Constants ─────────────────────────────────────────────────────────

const STAGE_CONFIG: Record<ConversationStage, { label: string; color: string; icon: string; desc: string }> = {
  idle: { label: "IDLE", color: "text-on-surface-variant", icon: "hourglass_empty", desc: "Waiting for call data" },
  approach: { label: "APPROACH", color: "text-secondary", icon: "waving_hand", desc: "Caller establishing rapport" },
  hook: { label: "HOOK", color: "text-primary", icon: "phishing", desc: "Emotional manipulation detected" },
  pressure: { label: "PRESSURE", color: "text-warning", icon: "priority_high", desc: "Urgency/fear tactics active" },
  extraction: { label: "EXTRACTION", color: "text-error", icon: "dangerous", desc: "Attempting data/money extraction" },
};

const THREAT_CONFIG: Record<ThreatLevel, { label: string; color: string; bg: string; ring: string; pct: number }> = {
  safe: { label: "SAFE", color: "text-secondary", bg: "bg-secondary", ring: "ring-secondary/30", pct: 10 },
  caution: { label: "CAUTION", color: "text-primary", bg: "bg-primary", ring: "ring-primary/30", pct: 40 },
  elevated: { label: "ELEVATED", color: "text-warning", bg: "bg-amber-500", ring: "ring-amber-500/30", pct: 70 },
  critical: { label: "CRITICAL", color: "text-error", bg: "bg-error", ring: "ring-error/30", pct: 95 },
};

// ─── Component ─────────────────────────────────────────────────────────

const Advanced = () => {
  const [connected, setConnected] = useState(false);
  const [callActive, setCallActive] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [stage, setStage] = useState<ConversationStage>("idle");
  const [threatLevel, setThreatLevel] = useState<ThreatLevel>("safe");
  const [audioData, setAudioData] = useState<AudioDataPoint[]>(
    Array.from({ length: 60 }, (_, i) => ({ time: i, low: 50, mid: 45, high: 40 })),
  );
  const [sentimentData, setSentimentData] = useState<SentimentPoint[]>(
    Array.from({ length: 30 }, (_, i) => ({ time: i, fear: 10, urgency: 10, trust: 80, label: "" })),
  );
  const [scores, setScores] = useState<ConfidenceScore[]>([
    { label: "Audio Deepfake", score: 0, verdict: "WAITING", color: "text-on-surface-variant", icon: "graphic_eq" },
    { label: "Voice Clone", score: 0, verdict: "WAITING", color: "text-on-surface-variant", icon: "record_voice_over" },
    { label: "Text Coercion", score: 0, verdict: "WAITING", color: "text-on-surface-variant", icon: "text_snippet" },
    { label: "Sentiment Risk", score: 0, verdict: "WAITING", color: "text-on-surface-variant", icon: "psychology" },
    { label: "Overall Threat", score: 0, verdict: "WAITING", color: "text-on-surface-variant", icon: "shield" },
  ]);
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [alertSent, setAlertSent] = useState(false);
  const [showEndConfirm, setShowEndConfirm] = useState(false);
  const counterRef = useRef(0);
  const callTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  // Call duration timer
  useEffect(() => {
    if (callActive) {
      callTimerRef.current = setInterval(() => setCallDuration((d) => d + 1), 1000);
    } else {
      if (callTimerRef.current) clearInterval(callTimerRef.current);
    }
    return () => { if (callTimerRef.current) clearInterval(callTimerRef.current); };
  }, [callActive]);

  const fmtDuration = (s: number) => {
    const m = Math.floor(s / 60);
    return `${m}:${String(s % 60).padStart(2, "0")}`;
  };

  const deriveThreat = useCallback((audioConf: number, textConf: number, isSpoof: boolean, isThreat: boolean): ThreatLevel => {
    const maxScore = Math.max(audioConf, textConf);
    if ((isSpoof && audioConf > 70) || (isThreat && textConf > 70) || maxScore > 80) return "critical";
    if (isSpoof || isThreat || maxScore > 50) return "elevated";
    if (maxScore > 25) return "caution";
    return "safe";
  }, []);

  const deriveStage = useCallback((textVerdict: string, textConf: number): ConversationStage => {
    if (textVerdict === "combined_threat" && textConf > 60) return "extraction";
    if (textVerdict === "financial_fraud" && textConf > 50) return "extraction";
    if ((textVerdict === "urgency_pressure" || textVerdict === "combined_threat") && textConf > 40) return "pressure";
    if (textVerdict !== "safe" && textConf > 25) return "hook";
    if (textConf > 10) return "approach";
    return "idle";
  }, []);

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/live`);
    let pingInterval: ReturnType<typeof setInterval>;
    let audioInterval: ReturnType<typeof setInterval>;

    ws.onopen = () => {
      setConnected(true);
      setCallActive(true);
      pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
      }, 30000);
      audioInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "audio", data: "" }));
      }, 3000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "analysis_result") handleResult(data);
      } catch { /* ignore */ }
    };

    ws.onerror = () => setConnected(false);
    ws.onclose = () => {
      setConnected(false);
      setCallActive(false);
      clearInterval(pingInterval);
      clearInterval(audioInterval);
    };

    return () => {
      clearInterval(pingInterval);
      clearInterval(audioInterval);
      ws.close();
    };
  }, []);

  const handleResult = (data: Record<string, unknown>) => {
    counterRef.current++;
    const t = counterRef.current;
    const now = new Date();
    const timeStr = `${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;

    if (data.modality === "audio") {
      const confidence = (data.confidence as number) ?? 0;
      const status = data.status as string;
      const isSpoof = status === "danger";

      setAudioData((prev) => {
        const next = [...prev.slice(1)];
        next.push({
          time: t,
          low: Math.sin(t * 0.3) * 30 + 50 + (isSpoof ? 20 : 0),
          mid: Math.cos(t * 0.2) * 25 + 45 + Math.random() * 10,
          high: isSpoof ? 60 + Math.random() * 20 : 30 + Math.random() * 10,
        });
        return next;
      });

      setScores((prev) => prev.map((s) => {
        if (s.label === "Audio Deepfake") {
          return { ...s, score: confidence, verdict: isSpoof ? "SPOOF" : "BONAFIDE", color: isSpoof ? "text-error" : "text-secondary" };
        }
        if (s.label === "Voice Clone") {
          const cloneScore = isSpoof ? Math.min(confidence + 10, 100) : Math.max(confidence - 20, 0);
          return { ...s, score: cloneScore, verdict: cloneScore > 60 ? "DETECTED" : "CLEAR", color: cloneScore > 60 ? "text-error" : "text-secondary" };
        }
        return s;
      }));

      if (isSpoof && confidence > 50) {
        setTranscript((prev) => [...prev.slice(-50), { time: timeStr, text: `Voice synthesis detected (${confidence.toFixed(1)}%)`, flagged: true, stage: "audio" }]);
      }
    }

    if (data.modality === "text") {
      const confidence = (data.confidence as number) ?? 0;
      const verdict = (data.verdict as string) ?? "safe";
      const isThreat = verdict !== "safe";

      const newStage = deriveStage(verdict, confidence);
      setStage(newStage);

      // Update sentiment
      setSentimentData((prev) => {
        const next = [...prev.slice(1)];
        next.push({
          time: t,
          fear: isThreat ? Math.min(confidence * 0.8, 90) : Math.max(10, (prev[prev.length - 1]?.fear ?? 10) - 5),
          urgency: verdict.includes("urgency") ? Math.min(confidence, 95) : Math.max(10, (prev[prev.length - 1]?.urgency ?? 10) - 3),
          trust: isThreat ? Math.max(10, 80 - confidence * 0.6) : Math.min(90, (prev[prev.length - 1]?.trust ?? 80) + 2),
          label: isThreat ? verdict.replace(/_/g, " ") : "",
        });
        return next;
      });

      setScores((prev) => {
        const audioScore = prev.find((p) => p.label === "Audio Deepfake")?.score ?? 0;
        const isSpoof = prev.find((p) => p.label === "Audio Deepfake")?.verdict === "SPOOF";
        const threat = deriveThreat(audioScore, confidence, isSpoof, isThreat);
        setThreatLevel(threat);
        const threatPct = THREAT_CONFIG[threat].pct;

        return prev.map((s) => {
          if (s.label === "Text Coercion") {
            return { ...s, score: confidence, verdict: isThreat ? verdict.toUpperCase().replace(/_/g, " ") : "SAFE", color: isThreat ? "text-error" : "text-secondary" };
          }
          if (s.label === "Sentiment Risk") {
            const sentRisk = isThreat ? Math.min(confidence * 0.9, 95) : 15;
            return { ...s, score: sentRisk, verdict: sentRisk > 60 ? "HIGH RISK" : sentRisk > 30 ? "MODERATE" : "LOW", color: sentRisk > 60 ? "text-error" : sentRisk > 30 ? "text-primary" : "text-secondary" };
          }
          if (s.label === "Overall Threat") {
            return { ...s, score: threatPct, verdict: THREAT_CONFIG[threat].label, color: THREAT_CONFIG[threat].color };
          }
          return s;
        });
      });

      if (isThreat) {
        setTranscript((prev) => [...prev.slice(-50), { time: timeStr, text: `[${verdict.replace(/_/g, " ")}] detected (${confidence.toFixed(1)}%)`, flagged: true, stage: newStage }]);
      }
    }
  };

  const handleEndCall = () => {
    setShowEndConfirm(false);
    setCallActive(false);
    setTranscript((prev) => [...prev, { time: fmtDuration(callDuration), text: "Call terminated by user", flagged: false }]);
  };

  const handleFamilyAlert = () => {
    setAlertSent(true);
    setTranscript((prev) => [...prev, {
      time: new Date().toLocaleTimeString([], { minute: "2-digit", second: "2-digit" }),
      text: "Family alert notification sent",
      flagged: false,
    }]);
    setTimeout(() => setAlertSent(false), 5000);
  };

  const tc = THREAT_CONFIG[threatLevel];
  const sc = STAGE_CONFIG[stage];
  const threatCircumference = 2 * Math.PI * 54;
  const threatDashoffset = threatCircumference * (1 - tc.pct / 100);

  return (
    <Layout systemStatus="monitoring">
      <div className="pt-32 pb-20 px-4 md:px-8 max-w-[1600px] mx-auto">

        {/* ─── Header ──────────────────────────────────────────── */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div className="flex items-center gap-4">
            <Link to="/settings" className="p-2 rounded-full hover:bg-on-surface/[0.06] text-on-surface-variant transition-colors cursor-pointer">
              <MaterialIcon icon="arrow_back" size={20} />
            </Link>
            <div>
              <h1 className="font-headline text-4xl md:text-5xl font-black tracking-tighter text-on-surface">
                Call Protection<span className="text-primary">.</span>
              </h1>
              <p className="text-sm text-on-surface-variant mt-1">Real-time deepfake & coercion monitoring</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${connected ? "bg-secondary/10 border-secondary/20" : "bg-error/10 border-error/20"}`}>
              <div className={`h-2 w-2 rounded-full ${connected ? "bg-secondary animate-pulse" : "bg-error"}`} />
              <span className={`text-xs font-bold uppercase tracking-widest ${connected ? "text-secondary" : "text-error"}`}>
                {connected ? "LIVE" : "OFFLINE"}
              </span>
            </div>
            {/* Call Duration */}
            {callActive && (
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-surface-container-high border border-outline-variant/20">
                <div className="w-2 h-2 rounded-full bg-error animate-pulse" />
                <span className="text-sm font-mono font-bold text-on-surface">{fmtDuration(callDuration)}</span>
              </div>
            )}
          </div>
        </div>

        {/* ─── Top Row: Threat Meter + Stage + Quick Actions ──── */}
        <div className="grid grid-cols-12 gap-6 mb-6">

          {/* Threat Meter — Large Gauge */}
          <div className="col-span-12 md:col-span-4 xl:col-span-3">
            <div className={`relative p-6 rounded-2xl bg-surface-container-low border border-outline-variant/10 flex flex-col items-center gap-4 overflow-hidden`}>
              <div className={`absolute inset-0 opacity-10 ${tc.bg}`} style={{ filter: "blur(60px)" }} />
              <p className="text-[12px] font-bold text-outline uppercase tracking-widest z-10">Threat Assessment</p>
              <div className="relative w-32 h-32 z-10">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="54" fill="none" stroke="currentColor" strokeWidth="6" className="text-surface-container-high" />
                  <circle cx="60" cy="60" r="54" fill="none" stroke="currentColor" strokeWidth="6"
                    strokeDasharray={threatCircumference} strokeDashoffset={threatDashoffset} strokeLinecap="round"
                    className={tc.color} style={{ transition: "stroke-dashoffset 0.8s ease" }} />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <MaterialIcon icon={threatLevel === "critical" ? "dangerous" : threatLevel === "elevated" ? "warning" : "shield"} size={28} className={tc.color} />
                  <span className={`text-lg font-headline font-black mt-1 ${tc.color}`}>{tc.pct}%</span>
                </div>
              </div>
              <span className={`text-sm font-headline font-extrabold uppercase tracking-widest z-10 ${tc.color}`}>{tc.label}</span>
            </div>
          </div>

          {/* Conversation Stage Indicator */}
          <div className="col-span-12 md:col-span-8 xl:col-span-5">
            <div className="p-6 rounded-2xl bg-surface-container-low border border-outline-variant/10 h-full">
              <p className="text-[12px] font-bold text-outline uppercase tracking-widest mb-4">Conversation Stage</p>
              <div className="flex items-center gap-3 mb-4">
                {(["approach", "hook", "pressure", "extraction"] as ConversationStage[]).map((s, i) => {
                  const cfg = STAGE_CONFIG[s];
                  const isActive = s === stage;
                  const isPast = ["approach", "hook", "pressure", "extraction"].indexOf(stage) > i;
                  return (
                    <div key={s} className="flex-1 flex flex-col items-center gap-2">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-500 ${
                        isActive ? `${cfg.color === "text-error" ? "bg-error/20 ring-4 ring-error/20" : cfg.color === "text-warning" ? "bg-amber-500/20 ring-4 ring-amber-500/20" : cfg.color === "text-primary" ? "bg-primary/20 ring-4 ring-primary/20" : "bg-secondary/20 ring-4 ring-secondary/20"} scale-110` :
                        isPast ? "bg-surface-container-high opacity-60" : "bg-surface-container-high/50 opacity-30"
                      }`}>
                        <MaterialIcon icon={cfg.icon} size={20} className={isActive || isPast ? cfg.color : "text-outline"} />
                      </div>
                      <span className={`text-[11px] font-black uppercase tracking-widest ${isActive ? cfg.color : "text-outline"}`}>{cfg.label}</span>
                    </div>
                  );
                })}
              </div>
              <div className={`p-3 rounded-xl border ${
                stage === "idle" ? "bg-surface-container-high/50 border-outline-variant/10" :
                stage === "extraction" ? "bg-error/5 border-error/20" :
                stage === "pressure" ? "bg-amber-500/5 border-amber-500/20" :
                "bg-primary/5 border-primary/20"
              }`}>
                <div className="flex items-center gap-2">
                  <MaterialIcon icon={sc.icon} size={16} className={sc.color} />
                  <span className={`text-xs font-bold ${sc.color}`}>{sc.desc}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions Panel */}
          <div className="col-span-12 xl:col-span-4">
            <div className="p-6 rounded-2xl bg-surface-container-low border border-outline-variant/10 h-full flex flex-col gap-4">
              <p className="text-[12px] font-bold text-outline uppercase tracking-widest">Quick Actions</p>

              {/* End Call Button */}
              <AnimatePresence>
                {showEndConfirm ? (
                  <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
                    className="p-4 rounded-xl bg-error/10 border border-error/30 space-y-3">
                    <p className="text-sm text-error font-bold">End this call?</p>
                    <div className="flex gap-3">
                      <button onClick={handleEndCall} className="flex-1 py-2 bg-error text-white rounded-lg text-xs font-bold uppercase tracking-widest cursor-pointer hover:bg-error/90 transition-colors">
                        End Call
                      </button>
                      <button onClick={() => setShowEndConfirm(false)} className="flex-1 py-2 bg-surface-container-high text-on-surface-variant rounded-lg text-xs font-bold uppercase tracking-widest cursor-pointer hover:bg-surface-container-highest transition-colors">
                        Cancel
                      </button>
                    </div>
                  </motion.div>
                ) : (
                  <button onClick={() => setShowEndConfirm(true)}
                    className="w-full py-3 bg-error/10 hover:bg-error/20 border border-error/30 rounded-xl text-error font-bold text-sm uppercase tracking-widest flex items-center justify-center gap-2 cursor-pointer transition-colors">
                    <MaterialIcon icon="call_end" size={20} /> End Call & Report
                  </button>
                )}
              </AnimatePresence>

              {/* Family Alert */}
              <button onClick={handleFamilyAlert} disabled={alertSent}
                className={`w-full py-3 rounded-xl font-bold text-sm uppercase tracking-widest flex items-center justify-center gap-2 cursor-pointer transition-all ${
                  alertSent ? "bg-secondary/10 border border-secondary/30 text-secondary" : "bg-primary/10 hover:bg-primary/20 border border-primary/30 text-primary"
                }`}>
                <MaterialIcon icon={alertSent ? "check_circle" : "family_restroom"} size={20} />
                {alertSent ? "Alert Sent" : "Alert Family Member"}
              </button>

              {/* Report to Authorities */}
              <button className="w-full py-3 bg-surface-container-high hover:bg-surface-container-highest border border-outline-variant/20 rounded-xl text-on-surface-variant font-bold text-sm uppercase tracking-widest flex items-center justify-center gap-2 cursor-pointer transition-colors">
                <MaterialIcon icon="local_police" size={20} /> Report to Cyber Cell
              </button>
            </div>
          </div>
        </div>

        {/* ─── Middle Row: Audio Viz + Sentiment Timeline ──── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">

          {/* Audio Frequency Analysis */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <MaterialIcon icon="graphic_eq" size={18} className="text-primary" />
                <h3 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Audio Frequency Spectrum</h3>
              </div>
              <div className="flex gap-3 text-[12px] font-mono">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-secondary" />Low</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#06B6D4]" />Mid</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-error" />High</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={audioData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1F2E" />
                <XAxis dataKey="time" hide />
                <YAxis hide />
                <Area type="monotone" dataKey="low" stroke="#10B981" fill="#10B981" fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
                <Area type="monotone" dataKey="mid" stroke="#06B6D4" fill="#06B6D4" fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
                <Area type="monotone" dataKey="high" stroke="#EF5350" fill="#EF5350" fillOpacity={0.05} strokeWidth={1.5} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Sentiment Emotion Timeline */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <MaterialIcon icon="psychology" size={18} className="text-primary" />
                <h3 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Sentiment Timeline</h3>
              </div>
              <div className="flex gap-3 text-[12px] font-mono">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-error" />Fear</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500" />Urgency</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-secondary" />Trust</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={sentimentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1F2E" />
                <XAxis dataKey="time" hide />
                <YAxis hide domain={[0, 100]} />
                <Tooltip content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0]?.payload as SentimentPoint;
                  return d?.label ? (
                    <div className="bg-surface-container-highest border border-outline-variant/20 rounded-lg px-3 py-2 text-xs">
                      <span className="text-error font-bold">{d.label}</span>
                    </div>
                  ) : null;
                }} />
                <Area type="monotone" dataKey="fear" stroke="#EF5350" fill="#EF5350" fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
                <Area type="monotone" dataKey="urgency" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.08} strokeWidth={1.5} isAnimationActive={false} />
                <Area type="monotone" dataKey="trust" stroke="#10B981" fill="#10B981" fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* ─── Bottom Row: Module Scores + Event Log ──────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Module Confidence Scores */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
            className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-6">
            <div className="flex items-center gap-2 mb-6">
              <MaterialIcon icon="hub" size={18} className="text-primary" />
              <h3 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Detection Modules</h3>
            </div>
            <div className="space-y-5">
              {scores.map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <MaterialIcon icon={item.icon} size={16} className={item.color !== "text-on-surface-variant" ? item.color : "text-outline"} />
                      <span className="text-sm font-headline font-bold text-on-surface">{item.label}</span>
                    </div>
                    <span className={`text-[12px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full border ${
                      item.color === "text-error" ? "border-error/20 bg-error/10 text-error" :
                      item.color === "text-primary" ? "border-primary/20 bg-primary/10 text-primary" :
                      item.color === "text-secondary" ? "border-secondary/20 bg-secondary/10 text-secondary" :
                      item.color === "text-warning" ? "border-amber-500/20 bg-amber-500/10 text-amber-500" :
                      "border-outline-variant/10 bg-surface-container-high text-outline"
                    }`}>{item.verdict}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-surface-container-high rounded-full overflow-hidden">
                      <motion.div animate={{ width: `${item.score}%` }} transition={{ duration: 0.5 }}
                        className={`h-full rounded-full ${
                          item.color === "text-error" ? "bg-error" : item.color === "text-primary" ? "bg-primary" :
                          item.color === "text-secondary" ? "bg-secondary" : item.color === "text-warning" ? "bg-amber-500" : "bg-on-surface-variant/30"
                        }`} />
                    </div>
                    <span className="text-xs font-mono font-bold text-on-surface-variant w-10 text-right">{item.score.toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Live Event Log */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <MaterialIcon icon="terminal" size={18} className="text-primary" />
                <h3 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Live Event Log</h3>
              </div>
              <span className="text-[12px] font-mono text-outline">{transcript.length} events</span>
            </div>
            <div className="space-y-2 max-h-[320px] overflow-y-auto pr-2 scrollbar-thin">
              {transcript.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <MaterialIcon icon="hourglass_empty" size={32} className="text-outline/30 mb-3" />
                  <p className="text-sm text-on-surface-variant/50">Waiting for analysis events...</p>
                </div>
              ) : (
                transcript.map((line, i) => (
                  <div key={i} className={`flex items-start gap-3 py-2 px-3 rounded-lg text-sm transition-colors ${
                    line.flagged ? "bg-error/5 border-l-2 border-l-error" : "hover:bg-surface-container-high/40"
                  }`}>
                    <span className="text-on-surface-variant font-mono text-xs shrink-0 mt-0.5">{line.time}</span>
                    {line.stage && (
                      <span className={`text-[11px] font-black uppercase tracking-wider px-1.5 py-0.5 rounded shrink-0 mt-0.5 ${
                        line.stage === "extraction" ? "bg-error/10 text-error" :
                        line.stage === "pressure" ? "bg-amber-500/10 text-amber-500" :
                        line.stage === "audio" ? "bg-primary/10 text-primary" :
                        "bg-surface-container-high text-outline"
                      }`}>{line.stage}</span>
                    )}
                    <span className={line.flagged ? "text-error font-medium" : "text-on-surface"}>{line.text}</span>
                  </div>
                ))
              )}
              <div ref={transcriptEndRef} />
            </div>
          </motion.div>
        </div>
      </div>
    </Layout>
  );
};

export default Advanced;
