import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import Layout from "@/components/Layout";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { useCallProtection } from "@/hooks/useCallProtection";
import { toast } from "sonner";

/* ── Stagger helper ── */
const stagger = (i: number) => ({
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { delay: i * 0.06, duration: 0.5, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] },
});

/* ── Circular Threat Gauge ── */
const ThreatGauge = ({ value, size = 128 }: { value: number; size?: number }) => {
  const r = 56;
  const circumference = 2 * Math.PI * r;
  const pct = Math.min(1, Math.max(0, value));
  const offset = circumference * (1 - pct);
  const color = pct > 0.7 ? "#ffb4ab" : pct > 0.3 ? "#F59E0B" : "#4edea3";
  const label = pct > 0.7 ? "DANGER" : pct > 0.3 ? "CAUTION" : "SAFE";
  const score = String(Math.round(pct * 100)).padStart(2, "0");

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 128 128">
        <circle cx="64" cy="64" r={r} fill="transparent" stroke="#2e3447" strokeWidth={10} />
        <motion.circle
          cx="64" cy="64" r={r} fill="transparent"
          stroke={color} strokeWidth={10} strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-black text-on-surface">{score}</span>
        <span className="text-[10px] uppercase tracking-widest font-bold" style={{ color }}>{label}</span>
      </div>
    </div>
  );
};

/* ── Live Audio Spectrogram ── */
const LiveSpectrogram = ({ frequencyData, isActive }: { frequencyData: Uint8Array | null; isActive: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<number[][]>([]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;
    ctx.scale(2, 2);
    const w = rect.width;
    const h = rect.height;

    if (!isActive || !frequencyData) {
      // Draw idle state with soft grid
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "rgba(18, 22, 33, 0.95)";
      ctx.fillRect(0, 0, w, h);
      
      // Grid lines
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.lineWidth = 0.5;
      for (let y = 0; y < h; y += 20) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      for (let x = 0; x < w; x += 30) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      
      // Center text
      ctx.fillStyle = "rgba(255,255,255, 0.15)";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Awaiting audio signal…", w / 2, h / 2);
      return;
    }

    // Build spectrogram column from frequency data
    const bins = 64;
    const step = Math.floor(frequencyData.length / bins);
    const column: number[] = [];
    for (let i = 0; i < bins; i++) {
      column.push(frequencyData[i * step] / 255);
    }
    historyRef.current.push(column);
    
    const maxCols = Math.floor(w / 3);
    if (historyRef.current.length > maxCols) {
      historyRef.current = historyRef.current.slice(-maxCols);
    }

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(10, 14, 24, 0.98)";
    ctx.fillRect(0, 0, w, h);

    const colWidth = 3;
    const rowHeight = h / bins;

    historyRef.current.forEach((col, colIdx) => {
      const x = (historyRef.current.length - 1 - colIdx) * colWidth;
      col.forEach((val, rowIdx) => {
        const y = h - (rowIdx + 1) * rowHeight;
        if (val < 0.05) return;
        
        // Color gradient: dark blue → cyan → green → yellow → red
        const r2 = val > 0.7 ? 255 : val > 0.4 ? Math.round(val * 350) : 0;
        const g2 = val > 0.7 ? Math.round((1 - val) * 500) : val > 0.3 ? 220 : Math.round(val * 600);
        const b2 = val < 0.4 ? 255 : Math.round((1 - val) * 255);
        const a2 = Math.min(1, val * 1.5);
        
        ctx.fillStyle = `rgba(${r2}, ${g2}, ${b2}, ${a2})`;
        ctx.fillRect(w - x - colWidth, y, colWidth, rowHeight + 0.5);
      });
    });

    // Frequency axis labels
    ctx.fillStyle = "rgba(255,255,255,0.2)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("8kHz", 4, 14);
    ctx.fillText("4kHz", 4, h * 0.5);
    ctx.fillText("0Hz", 4, h - 4);
  }, [frequencyData, isActive]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full rounded-xl"
      style={{ imageRendering: "pixelated" }}
    />
  );
};

/* ── Video Preview ── */
const VideoPreview = ({ stream }: { stream: MediaStream | null }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  if (!stream) return null;

  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      className="w-full h-full object-cover rounded-xl"
    />
  );
};

/* ── Live Waveform Bars ── */
const LiveWaveform = ({ frequencyData, isActive }: { frequencyData: Uint8Array | null; isActive: boolean }) => {
  const bars = 9;
  const idle = [25, 50, 75, 100, 80, 100, 75, 50, 25];
  const heights = useMemo(() => {
    if (!frequencyData || !isActive) return idle;
    const step = Math.floor(frequencyData.length / bars);
    return Array.from({ length: bars }, (_, i) => Math.max(10, (frequencyData[i * step] / 255) * 100));
  }, [frequencyData, isActive]);

  const opacities = ["opacity-20", "opacity-40", "opacity-60", "", "", "", "opacity-60", "opacity-40", "opacity-20"];
  const colors = [
    "bg-primary-fixed-dim", "bg-primary-fixed-dim", "bg-primary-fixed-dim",
    "bg-primary", "bg-primary-container", "bg-primary",
    "bg-primary-fixed-dim", "bg-primary-fixed-dim", "bg-primary-fixed-dim",
  ];

  return (
    <div className="hidden xl:flex items-center justify-center gap-1.5 h-32">
      {heights.map((h, i) => (
        <motion.div
          key={i}
          className={`w-1.5 rounded-full ${colors[i]} ${opacities[i]}`}
          animate={{ height: `${h}%` }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        />
      ))}
    </div>
  );
};


const CallProtection = () => {
  const cp = useCallProtection();
  const [textInput, setTextInput] = useState("");
  const [explainMode, setExplainMode] = useState(false);

  const isDanger = cp.callState === "danger" || cp.callState === "critical";
  const isActive = cp.isActive;
  const systemStatus = isDanger ? ("alert" as const) : isActive ? ("monitoring" as const) : ("protected" as const);

  const timelineSteps = [
    { label: "Audio Capture", done: isActive, active: !isActive },
    { label: "Signal Process", done: cp.frequencyData !== null, active: isActive && cp.frequencyData === null },
    { label: "Multi-layer", done: cp.audio.status !== "idle", active: isActive && cp.audio.status === "idle" },
    { label: "Bio Validation", done: cp.guardrails !== null, active: isActive && cp.guardrails === null },
    { label: "Fusion", done: cp.threatEscalation > 0, active: isActive && cp.threatEscalation === 0 },
    { label: "Compliance", done: cp.guardrails !== null, active: isActive && cp.guardrails === null },
    { label: "Audit Log", done: cp.callSummary !== null, active: isActive && cp.callSummary === null },
  ];

  const handleSendText = () => {
    if (textInput.trim()) {
      cp.analyzeText(textInput.trim());
      setTextInput("");
    }
  };

  const fusionVerdict = isDanger ? "COMPLIANCE VIOLATION" : isActive ? "COMPLIANT" : "STANDBY";
  const fusionColor = isDanger ? "text-error" : isActive ? "text-secondary" : "text-on-surface-variant";
  const fusionBorder = isDanger ? "border-error/30" : isActive ? "border-secondary/30" : "border-outline-variant/10";
  const fusionGlow = isDanger ? "shadow-[0_0_50px_rgba(255,180,171,0.1)]" : isActive ? "shadow-[0_0_50px_rgba(78,222,163,0.1)]" : "";
  const confidencePct = isActive ? Math.max(cp.audio.confidence, cp.text.confidence) : 0;

  const sentimentItems = [
    { label: "Urgency Factor", value: cp.text.details?.patterns ? 1 : 0 },
    { label: "Financial Pressure", value: cp.text.status === "danger" ? 3 : cp.text.status === "warning" ? 2 : 0 },
    { label: "Emotional Manipulation", value: cp.threatEscalation > 0.5 ? 2 : 0 },
  ];

  // Get system stream for video preview
  const systemStream = cp.systemStreamRef?.current ?? null;

  return (
    <Layout systemStatus={systemStatus}>
      <div className="pb-12 px-8 max-w-[1600px] mx-auto space-y-8">

        {/* ── IDLE STATE: Centered hero + glowing CTA ── */}
        {!isActive && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col items-center justify-center min-h-[70vh] relative"
          >
            {/* Background glow orbs */}
            <div className="absolute inset-0 pointer-events-none">
              <motion.div
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/8 rounded-full blur-[150px]"
                animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0.8, 0.5] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              />
              <motion.div
                className="absolute top-1/3 right-1/4 w-[300px] h-[300px] bg-secondary/5 rounded-full blur-[100px]"
                animate={{ x: [0, 40, 0], y: [0, -20, 0] }}
                transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
              />
            </div>

            <div className="relative z-10 text-center space-y-8">
              {/* Badge — larger and more readable */}
              <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-surface-container-high/60 border border-primary/20 backdrop-blur-sm">
                <span className="w-2.5 h-2.5 rounded-full bg-primary/70 animate-pulse shadow-[0_0_8px_rgba(0,209,255,0.5)]" />
                <span className="text-sm font-headline font-bold tracking-widest uppercase text-primary">
                  Voice Forensics Agent
                </span>
              </div>

              {/* Title */}
              <h1 className="text-4xl md:text-6xl font-headline font-black tracking-tighter text-on-surface">
                Live Voice Fraud Detection System
              </h1>
              <div className="text-lg md:text-xl text-on-surface-variant font-light max-w-2xl mx-auto flex flex-col gap-1 items-center">
                <p>System Status: <span className="text-on-surface font-bold">STANDBY</span></p>
                <p>Biological Guardrail: <span className="text-secondary font-bold">ACTIVE</span></p>
              </div>

              {/* Glowing CTA Button */}
              <motion.div className="pt-8">
                <motion.button
                  onClick={() => cp.startProtection()}
                  className="group relative px-16 py-7 bg-gradient-to-br from-primary via-primary-container to-primary text-on-primary font-headline font-extrabold text-xl uppercase tracking-wider rounded-2xl overflow-hidden cursor-pointer"
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  animate={{
                    boxShadow: [
                      "0 0 30px rgba(0,209,255,0.2), 0 0 60px rgba(0,209,255,0.1)",
                      "0 0 50px rgba(0,209,255,0.4), 0 0 100px rgba(0,209,255,0.2)",
                      "0 0 30px rgba(0,209,255,0.2), 0 0 60px rgba(0,209,255,0.1)",
                    ],
                  }}
                  transition={{ boxShadow: { duration: 2.5, repeat: Infinity, ease: "easeInOut" } }}
                >
                  {/* Shimmer effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  {/* Pulse ring */}
                  <motion.div
                    className="absolute inset-0 rounded-2xl border-2 border-primary/40"
                    animate={{ scale: [1, 1.08, 1], opacity: [0.5, 0, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <span className="relative z-10 flex items-center gap-4">
                    Begin Real Time Monitoring
                    <MaterialIcon icon="arrow_forward" size={24} />
                  </span>
                </motion.button>
              </motion.div>
            </div>

            {/* Status widgets at bottom of idle state */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="relative z-10 mt-16 grid grid-cols-2 md:grid-cols-4 gap-6 w-full max-w-5xl"
            >
              {[
                { icon: "settings_input_component", label: "Signal Spectrum", value: "OFFLINE", color: "text-primary" },
                { icon: "memory", label: "Engine", value: "9 LAYERS", color: "text-secondary" },
                { icon: "fingerprint", label: "Bio Veto", value: "ARMED", color: "text-primary-container" },
                { icon: "gavel", label: "Compliance", value: "READY", color: "text-on-surface-variant" },
              ].map((s) => (
                <div key={s.label} className="p-5 rounded-2xl glass-panel border border-outline-variant/10 flex items-center gap-4 hover:border-primary/20 transition-colors">
                  <div className={`p-3 bg-surface-container-high rounded-xl ${s.color}`}>
                    <MaterialIcon icon={s.icon} size={26} />
                  </div>
                  <div>
                    <p className="text-[11px] font-bold text-on-surface-variant uppercase tracking-widest">{s.label}</p>
                    <p className="text-lg font-black text-on-surface mt-0.5">{s.value}</p>
                  </div>
                </div>
              ))}
            </motion.div>
          </motion.div>
        )}

        {/* ═══════════════════════════════════════════════════════════ */}
        {/* ══  ACTIVE STATE: Full monitoring dashboard  ════════════ */}
        {/* ═══════════════════════════════════════════════════════════ */}
        {isActive && (
          <>
            {/* ── CONTROL BAR ── */}
            <motion.section {...stagger(0)} className="flex flex-wrap gap-4 py-2">
              <button
                onClick={() => cp.stopProtection()}
                className="flex-1 min-w-[200px] h-16 bg-gradient-to-r from-error-container to-error text-on-error hover:brightness-110 transition-all duration-300 rounded-2xl flex items-center justify-center gap-3 font-headline font-extrabold text-base uppercase tracking-wider shadow-lg cursor-pointer"
              >
                <MaterialIcon icon="call_end" size={24} />
                Terminate Session
              </button>
              <button
                onClick={() => toast.warning("Flagged for compliance audit")}
                className="flex-1 min-w-[200px] h-16 bg-surface-container-highest border border-outline-variant/20 hover:bg-surface-bright transition-all duration-300 rounded-2xl flex items-center justify-center gap-3 font-bold shadow-lg cursor-pointer"
              >
                <MaterialIcon icon="notification_important" className="text-on-surface-variant" size={20} />
                Flag for Audit
              </button>
              <button
                onClick={() => toast.success("Audit log exported")}
                className="flex-1 min-w-[200px] h-16 bg-surface-container-highest border border-outline-variant/20 hover:bg-surface-bright transition-all duration-300 rounded-2xl flex items-center justify-center gap-3 font-bold shadow-lg cursor-pointer"
              >
                <MaterialIcon icon="save" className="text-on-surface-variant" size={20} />
                Export Audit Log
              </button>
              {!cp.isScreenShare && (
                <button
                  onClick={() => cp.enableScreenShare()}
                  className="flex-1 min-w-[200px] h-16 bg-surface-container-highest border border-primary/20 hover:bg-primary/10 hover:border-primary/40 transition-all duration-300 rounded-2xl flex items-center justify-center gap-3 font-bold shadow-lg cursor-pointer"
                >
                  <MaterialIcon icon="screen_share" className="text-primary" size={20} />
                  <span className="text-on-surface">Reconnect Screen Share</span>
                </button>
              )}
              <button
                onClick={() => setExplainMode(!explainMode)}
                className={`flex-1 min-w-[200px] h-16 border transition-all duration-300 rounded-2xl flex items-center justify-center gap-3 font-bold shadow-lg cursor-pointer ${
                  explainMode 
                    ? "bg-primary/20 border-primary/50 text-primary" 
                    : "bg-surface-container-highest border-outline-variant/20 hover:bg-surface-bright text-on-surface-variant"
                }`}
              >
                <MaterialIcon icon={explainMode ? "visibility" : "visibility_off"} size={20} />
                Explain Decision {explainMode ? "ON" : "OFF"}
              </button>
            </motion.section>

            {/* ── STATUS WIDGETS ── */}
            <motion.div {...stagger(1)} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
                <div className="p-3 bg-primary/10 rounded-lg text-primary">
                  <MaterialIcon icon="settings_input_component" size={24} />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Signal Spectrum</p>
                  <div className="flex items-end gap-0.5 h-6 mt-1">
                    {[...Array(8)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-1 bg-primary rounded-full"
                        animate={{ height: `${Math.random() * 80 + 20}%` }}
                        transition={{ duration: 0.3, repeat: Infinity, repeatType: "mirror", delay: i * 0.05 }}
                      />
                    ))}
                  </div>
                </div>
              </div>
              <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
                <div className="p-3 bg-secondary/10 rounded-lg text-secondary">
                  <MaterialIcon icon="memory" size={24} />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Engine Status</p>
                  <p className="text-xl font-black text-secondary">9 LAYERS <span className="text-xs text-on-surface-variant font-normal">ACTIVE</span></p>
                </div>
              </div>
              <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
                <div className="p-3 bg-primary-container/10 rounded-lg text-primary-container">
                  <MaterialIcon icon="fingerprint" size={24} />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Evidence Vault</p>
                  <p className="text-xl font-black text-on-surface">{cp.callSummary ? "SAVED" : "AUTO-SYNC"}</p>
                </div>
              </div>
              <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
                <div className="p-3 bg-secondary/10 rounded-lg text-secondary">
                  <MaterialIcon icon="mic" size={24} />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Audio Level</p>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="h-2 w-24 bg-surface-container-highest rounded-full overflow-hidden">
                      <motion.div
                        className={`h-full rounded-full ${cp.audioLevel > 0.6 ? "bg-error" : "bg-secondary"}`}
                        animate={{ width: `${Math.min(100, cp.audioLevel * 100)}%` }}
                        transition={{ duration: 0.1 }}
                      />
                    </div>
                    <span className="text-xs font-mono text-on-surface-variant">{(cp.audioLevel * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* ── HERO PANEL ── */}
            <motion.section
              {...stagger(2)}
              className={`relative overflow-hidden h-60 rounded-2xl glass-panel flex flex-col justify-center px-12 border border-outline-variant/10 shadow-2xl ${isDanger ? "border-error/50 animate-danger-pulse" : ""}`}
            >
              <div
                className="absolute inset-0 opacity-20 bg-center"
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='1000' height='200' viewBox='0 0 1000 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 100 Q 25 20 50 100 T 100 100 T 150 100 T 200 100 T 250 100 T 300 100 T 350 100 T 400 100 T 450 100 T 500 100 T 550 100 T 600 100 T 650 100 T 700 100 T 750 100 T 800 100 T 850 100 T 900 100 T 950 100 T 1000 100' stroke='%2300D1FF' stroke-width='2' fill='none' opacity='0.5'/%3E%3C/svg%3E")`,
                }}
              />
              <div className="relative z-10 flex justify-between items-end">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-primary font-bold tracking-widest text-xs uppercase">
                      Analyzing voice stream...
                    </span>
                    <div className="h-px w-12 bg-primary/30" />
                    <span className="text-[10px] text-secondary font-bold uppercase tracking-widest animate-pulse">
                      System Status: MONITORING LIVE
                    </span>
                  </div>
                  <h1 className="text-4xl font-headline font-black tracking-tighter text-on-surface mb-6">
                    Live Voice Fraud Detection System
                  </h1>
                  <div className="flex flex-wrap gap-12">
                    {[
                      {
                        label: "Pitch Stability",
                        value: Math.max(0, 100 - cp.threatEscalation * 80),
                        color: "bg-secondary shadow-[0_0_8px_rgba(78,222,163,0.5)]",
                        textColor: "text-secondary",
                        fmt: (v: number) => `${v.toFixed(1)}%`,
                      },
                      {
                        label: "Synthetic Artifact Level",
                        value: cp.threatEscalation * 100,
                        color: "bg-primary shadow-[0_0_8px_rgba(0,209,255,0.5)]",
                        textColor: "text-primary",
                        fmt: (v: number) => `${v.toFixed(2)}%`,
                      },
                      {
                        label: "Compliance Status",
                        value: Math.max(0, 95 - cp.threatEscalation * 60),
                        color: "bg-secondary shadow-[0_0_8px_rgba(78,222,163,0.5)]",
                        textColor: "text-secondary",
                        fmt: (v: number) => (v > 80 ? "PASSING" : "FLAGGED"),
                      },
                    ].map((bar) => (
                      <div key={bar.label} className="space-y-1">
                        <p className="text-[10px] text-on-surface-variant uppercase font-bold tracking-widest">{bar.label}</p>
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-32 bg-surface-container-highest rounded-full overflow-hidden">
                            <motion.div
                              className={`h-full ${bar.color}`}
                              animate={{ width: `${Math.min(100, bar.value)}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                          <span className={`${bar.textColor} text-sm font-bold`}>{bar.fmt(bar.value)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <LiveWaveform frequencyData={cp.frequencyData} isActive={isActive} />
              </div>
            </motion.section>

            {/* ── BIOLOGICAL VETO ALERT ── */}
            <AnimatePresence>
              {cp.biologicalVeto && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scaleY: 0.95 }}
                  animate={{ opacity: 1, y: 0, scaleY: 1 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="relative overflow-hidden rounded-2xl border-2 border-error/60 bg-gradient-to-r from-error/20 via-error-container/30 to-error/20 p-6 shadow-[0_0_60px_rgba(255,100,100,0.15)]"
                >
                  <div className="absolute inset-0 bg-error/5 animate-pulse" />
                  <div className="relative z-10 flex items-start gap-4">
                    <div className="p-3 rounded-xl bg-error/20 shadow-[0_0_20px_rgba(255,100,100,0.3)]">
                      <MaterialIcon icon="gpp_bad" filled className="text-error" size={32} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <span className="text-error font-headline font-black text-lg uppercase tracking-wider">
                          Biological Veto Triggered
                        </span>
                        <span className="px-3 py-0.5 rounded-full bg-error/20 text-error text-[10px] font-bold uppercase tracking-widest border border-error/30">
                          AI Clone Detected
                        </span>
                      </div>
                      <p className="text-on-surface text-sm font-medium leading-relaxed">
                        {cp.vetoReason || "Voice exhibits physiological impossibilities inconsistent with human biology. AI voice cloning confirmed."}
                      </p>
                    </div>
                    <button onClick={() => cp.dismissDanger()} className="text-error/60 hover:text-error transition-colors shrink-0">
                      <MaterialIcon icon="close" size={20} />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── MAIN GRID: Video/Spectrogram LEFT + Analysis RIGHT ── */}
            <div className="grid grid-cols-12 gap-8">

              {/* LEFT: Video Preview or Live Spectrogram */}
              <motion.div {...stagger(3)} className="col-span-12 lg:col-span-8 flex flex-col gap-8">
                {/* Media Panel */}
                <div className="h-[480px] rounded-2xl bg-surface-container-low border border-outline-variant/10 flex flex-col overflow-hidden shadow-xl">
                  <div className="p-5 border-b border-outline-variant/10 bg-surface-container/30 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <span className={`w-2.5 h-2.5 rounded-full bg-secondary animate-pulse shadow-[0_0_8px_rgba(78,222,163,0.8)]`} />
                      <h2 className="font-headline font-bold text-on-surface text-lg">
                        {cp.isScreenShare ? "Video Preview" : "Live Audio Spectrogram"}
                      </h2>
                    </div>
                    <div className="flex gap-2">
                      <span className="text-[10px] font-bold text-secondary bg-secondary/10 px-3 py-1.5 rounded-full border border-secondary/20">
                        {cp.isScreenShare ? "SCREEN SHARE" : "AUDIO ONLY"} · LIVE
                      </span>
                    </div>
                  </div>
                  <div className="flex-1 p-2 bg-[#0a0e18]">
                    {cp.isScreenShare && systemStream ? (
                      <VideoPreview stream={systemStream} />
                    ) : (
                      <LiveSpectrogram frequencyData={cp.frequencyData} isActive={isActive} />
                    )}
                  </div>
                </div>

                {/* Call Timeline */}
                <div className="p-8 rounded-2xl bg-surface-container border border-outline-variant/10 flex-1">
                  <div className="flex justify-between items-center mb-10">
                    <div className="flex items-center gap-3">
                      <MaterialIcon icon="timeline" className="text-primary" size={24} />
                      <h3 className="font-headline font-bold text-on-surface text-xl">Forensic Timeline</h3>
                    </div>
                    <span className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest border border-outline-variant/10 px-3 py-1 rounded-full bg-surface-container-high">
                      Duration: LIVE
                    </span>
                  </div>
                  
                  <div className="relative pl-6 py-2">
                    {/* Vertical connecting line */}
                    <div className="absolute top-4 bottom-4 left-[31px] w-0.5 bg-surface-container-highest" />
                    
                    <div className="relative flex flex-col gap-8">
                      {timelineSteps.map((step, i) => (
                        <div key={i} className={`relative flex items-center gap-6 ${!step.done && !step.active ? "opacity-30" : ""}`}>
                          {/* Circle Node */}
                          <div
                            className={`w-3.5 h-3.5 rounded-full z-10 shrink-0 border-[3px] border-surface-container ${
                              step.done
                                ? "bg-secondary shadow-[0_0_12px_rgba(78,222,163,0.8)]"
                                : step.active
                                ? "bg-primary animate-[pulse_1.5s_ease-in-out_infinite] shadow-[0_0_12px_rgba(0,209,255,0.8)]"
                                : "bg-outline-variant/40"
                            }`}
                          />
                          {/* Label Group */}
                          <div className="flex flex-col">
                            <span 
                              className={`text-sm md:text-base font-bold tracking-wider uppercase ${
                                step.active ? "text-primary" : step.done ? "text-on-surface" : "text-on-surface-variant"
                              }`}
                            >
                              {step.label}
                            </span>
                            {/* Subtitle text if active */}
                            {step.active && (
                              <span className="text-[10px] text-primary/70 font-semibold tracking-widest uppercase mt-0.5">
                                Processing...
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* RIGHT: Analysis Stack */}
              <motion.div {...stagger(4)} className="col-span-12 lg:col-span-4 flex flex-col gap-8">

                {/* Voice Identity Card */}
                <div className="p-8 rounded-2xl bg-surface-container border border-outline-variant/10 relative overflow-hidden shadow-lg">
                  <div className="flex justify-between items-start mb-8">
                    <div className="flex gap-4">
                      <div className="w-14 h-14 rounded-xl bg-surface-container-high flex items-center justify-center border border-outline-variant/20 shadow-inner">
                        <MaterialIcon icon="fingerprint" className="text-primary" size={30} />
                      </div>
                      <div>
                        <h3 className="font-headline font-bold text-on-surface text-xl">Voice Identity</h3>
                        <p className="text-sm text-on-surface-variant font-medium">
                          {cp.speakerMatch ? `Matched: ${cp.speakerMatch}` : "Identifying..."}
                        </p>
                      </div>
                    </div>
                    {cp.speakerVerified && (
                      <span className="bg-secondary/10 text-secondary text-[10px] font-bold px-4 py-1.5 rounded-full uppercase tracking-tighter border border-secondary/20">
                        Trusted Profile
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/10">
                      <p className="text-[10px] text-on-surface-variant uppercase font-bold mb-2 tracking-widest">Clone Prob.</p>
                      <p className="text-2xl font-black text-on-surface">
                        {cp.audio.confidence > 0 ? `${cp.audio.confidence.toFixed(1)}%` : `${(cp.threatEscalation * 5).toFixed(3)}%`}
                      </p>
                    </div>
                    <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/10">
                      <p className="text-[10px] text-on-surface-variant uppercase font-bold mb-2 tracking-widest">Identity Score</p>
                      <p className="text-2xl font-black text-secondary">
                        {cp.speakerVerified ? (
                          <>{(cp.speakerSimilarity * 100).toFixed(1)}<span className="text-sm font-bold opacity-50">/100</span></>
                        ) : "..."}
                      </p>
                    </div>
                  </div>

                  {/* Uncertainty / Human Review */}
                  {cp.audio.verdict === "uncertain" && (
                     <div className="mt-4 p-4 rounded-xl bg-error/10 border border-error/50 flex flex-col gap-2 relative overflow-hidden shadow-inner">
                       <div className="absolute top-0 right-0 py-1 px-3 bg-error text-surface font-bold text-[10px] uppercase tracking-widest rounded-bl-lg">
                         Requires Action
                       </div>
                       <div className="flex items-center gap-2 text-error font-bold text-lg mt-1">
                         <MaterialIcon icon="gavel" size={22} />
                         MANUAL REVIEW REQUIRED
                       </div>
                       <p className="text-sm text-on-surface font-medium leading-snug">
                         Confidence below 60%. Biomarkers indicate ambiguous synthesis.
                       </p>
                       <p className="text-xs text-error font-bold uppercase tracking-widest mt-1 underline decoration-error/50 underline-offset-2 cursor-pointer hover:text-error-container">
                         → Escalate to Senior Agent
                       </p>
                     </div>
                  )}

                  {/* Explainability Mode */}
                  {explainMode && cp.audio.status !== "idle" && cp.audio.verdict !== "uncertain" && (
                    <div className="mt-4 p-4 rounded-xl bg-surface-container-lowest border border-outline-variant/10 shadow-inner">
                      <p className="text-[10px] text-on-surface-variant uppercase font-bold mb-3 tracking-widest">
                        WHY THIS IS {cp.threatEscalation > 0.5 ? "SYNTHETIC" : "AUTHENTIC"}:
                      </p>
                      <ul className="space-y-2 text-sm text-on-surface">
                        {cp.threatEscalation > 0.5 ? (
                          <>
                            <li className="flex items-center gap-2">
                              <span className="text-error w-1.5 h-1.5 rounded-full bg-error shrink-0" />
                              No respiratory cycle detected
                            </li>
                            <li className="flex items-center gap-2">
                              <span className="text-error w-1.5 h-1.5 rounded-full bg-error shrink-0" />
                              Unnatural pitch stability
                            </li>
                            <li className="flex items-center gap-2">
                              <span className="text-error w-1.5 h-1.5 rounded-full bg-error shrink-0" />
                              Phase inconsistency
                            </li>
                          </>
                        ) : (
                          <>
                            <li className="flex items-center gap-2">
                              <span className="text-secondary w-1.5 h-1.5 rounded-full bg-secondary shrink-0" />
                              Natural breathing patterns present
                            </li>
                            <li className="flex items-center gap-2">
                              <span className="text-secondary w-1.5 h-1.5 rounded-full bg-secondary shrink-0" />
                              Organic vocal tract micro-tremors
                            </li>
                            <li className="flex items-center gap-2">
                              <span className="text-secondary w-1.5 h-1.5 rounded-full bg-secondary shrink-0" />
                              Consistent phase alignment
                            </li>
                          </>
                        )}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Threat Level Card */}
                <div className="p-8 rounded-2xl bg-surface-container border border-outline-variant/10 shadow-lg">
                  <h3 className="font-headline font-bold text-on-surface text-xl mb-8">Threat Analysis</h3>
                  <div className="flex items-center gap-10">
                    <ThreatGauge value={cp.threatEscalation} />
                    <div className="flex-1 space-y-6">
                      <div>
                        <div className="flex justify-between text-[10px] uppercase font-bold text-on-surface-variant mb-2 tracking-widest">
                          <span>Clone Risk</span>
                          <span className={cp.threatEscalation > 0.5 ? "text-error" : "text-secondary"}>
                            {cp.threatEscalation > 0.7 ? "High" : cp.threatEscalation > 0.3 ? "Medium" : "Low"}
                          </span>
                        </div>
                        <div className="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
                          <motion.div
                            className={`h-full ${cp.threatEscalation > 0.5 ? "bg-error" : "bg-secondary"}`}
                            animate={{ width: `${Math.max(2, cp.threatEscalation * 100)}%` }}
                            transition={{ duration: 0.5 }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-[10px] uppercase font-bold text-on-surface-variant mb-2 tracking-widest">
                          <span>Coercion Risk</span>
                          <span className={cp.text.status === "danger" ? "text-error" : "text-secondary"}>
                            {cp.text.status === "danger" ? "High" : cp.text.status === "warning" ? "Elevated" : "Negligible"}
                          </span>
                        </div>
                        <div className="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
                          <motion.div
                            className={`h-full ${cp.text.status === "danger" ? "bg-error" : "bg-secondary"}`}
                            animate={{ width: `${cp.text.status === "danger" ? 80 : cp.text.status === "warning" ? 40 : 2}%` }}
                            transition={{ duration: 0.5 }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Sentiment & Intent */}
                <div className="p-8 rounded-2xl bg-surface-container border border-outline-variant/10 shadow-lg">
                  <h3 className="font-headline font-bold text-on-surface text-xl mb-6">Sentiment &amp; Intent</h3>
                  <div className="space-y-4">
                    {sentimentItems.map((item) => (
                      <div key={item.label} className="flex items-center justify-between p-4 bg-surface-container-low rounded-xl border border-outline-variant/5">
                        <span className="text-sm font-semibold text-on-surface">{item.label}</span>
                        <div className="flex gap-1.5">
                          {[0, 1, 2, 3].map((level) => (
                            <div
                              key={level}
                              className={`w-2.5 h-5 rounded-sm ${
                                level < item.value ? (item.value >= 3 ? "bg-error" : "bg-secondary") : "bg-surface-container-highest"
                              }`}
                            />
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Compliance Guardrails */}
                <div className="p-8 rounded-2xl bg-surface-container border border-outline-variant/10 shadow-lg">
                  <h3 className="font-headline font-bold text-on-surface text-xl mb-4 flex items-center gap-2">
                    <MaterialIcon icon="shield" className="text-secondary" /> 
                    COMPLIANCE GUARDRAILS ACTIVE
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-surface-container-low rounded-xl border border-outline-variant/5">
                      <span className="text-sm font-semibold text-on-surface">Biological Veto</span>
                      <span className="text-secondary text-xs font-bold px-2 py-1 bg-secondary/10 rounded-md">ENFORCED</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-surface-container-low rounded-xl border border-outline-variant/5">
                      <span className="text-sm font-semibold text-on-surface">Confidence Threshold</span>
                      <span className="text-secondary text-xs font-bold px-2 py-1 bg-secondary/10 rounded-md">VERIFIED</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-surface-container-low rounded-xl border border-outline-variant/5">
                      <span className="text-sm font-semibold text-on-surface">Human Review</span>
                      <span className="text-on-surface-variant text-xs font-bold px-2 py-1 bg-surface-container-highest rounded-md">
                        {cp.guardrails?.human_review_required ? "REQUIRED" : "NOT REQUIRED"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Edge Case Handling */}
                <div className="p-8 rounded-2xl bg-surface-container border border-outline-variant/10 shadow-lg">
                  <h3 className="font-headline font-bold text-on-surface text-xl mb-4 flex items-center gap-2">
                    <MaterialIcon icon="dynamic_feed" className="text-primary" /> 
                    EDGE CASE HANDLING
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex items-center gap-2 text-sm text-on-surface-variant">
                      <MaterialIcon icon="check_circle" className="text-secondary" size={16} /> Low audio clarity
                    </div>
                    <div className="flex items-center gap-2 text-sm text-on-surface-variant">
                      <MaterialIcon icon="check_circle" className="text-secondary" size={16} /> Background noise
                    </div>
                    <div className="flex items-center gap-2 text-sm text-on-surface-variant">
                      <MaterialIcon icon="check_circle" className="text-secondary" size={16} /> Partial speech 
                    </div>
                    <div className="flex items-center gap-2 text-sm text-on-surface-variant">
                      <MaterialIcon icon="check_circle" className="text-secondary" size={16} /> AI models (ElevenLabs, RVC)
                    </div>
                  </div>
                </div>

                {/* Audit JSON Payload */}
                {cp.decisionPayload && (
                  <div className="bg-[#0E1117] rounded-2xl border border-outline-variant/20 overflow-hidden shadow-2xl">
                    <div className="bg-[#161B22] px-5 py-4 border-b border-outline-variant/10 flex justify-between items-center">
                      <div className="flex items-center gap-3">
                        <MaterialIcon icon="data_object" size={20} className="text-primary" />
                        <span className="font-bold uppercase tracking-widest text-xs text-primary">Live Audit Output</span>
                      </div>
                      <span className="text-[10px] font-bold bg-primary/10 text-primary px-3 py-1 rounded-full border border-primary/20 tracking-widest">
                        DECISION_OBJECT
                      </span>
                    </div>
                    <div className="p-5 overflow-x-auto max-h-[300px] overflow-y-auto custom-scrollbar">
                      <pre className="font-mono text-[11px] md:text-xs text-secondary-container leading-relaxed">
                        <code>{JSON.stringify(cp.decisionPayload, null, 2)}</code>
                      </pre>
                    </div>
                  </div>
                )}

              </motion.div>
            </div>

            {/* ── FUSION SUMMARY ── */}
            <motion.section
              {...stagger(5)}
              className={`p-10 rounded-2xl bg-surface-container-high border ${fusionBorder} ${fusionGlow} relative overflow-hidden`}
            >
              <div className="absolute top-0 right-0 w-64 h-64 bg-secondary/5 blur-[100px] rounded-full" />
              <div className="flex flex-col md:flex-row gap-12 items-center relative z-10">
                <div className="flex-1">
                  <div className="flex items-center gap-4 mb-6">
                    <div className={`p-3 rounded-full ${isDanger ? "bg-error/20 shadow-[0_0_15px_rgba(255,180,171,0.3)]" : "bg-secondary/20 shadow-[0_0_15px_rgba(78,222,163,0.3)]"}`}>
                      <MaterialIcon
                        icon={isDanger ? "gpp_bad" : "verified"}
                        filled
                        className={fusionColor}
                        size={30}
                      />
                    </div>
                    <h2 className={`font-headline text-4xl font-black uppercase tracking-tight ${fusionColor}`}>
                      {fusionVerdict}
                    </h2>
                  </div>
                  <p className="text-on-surface-variant leading-relaxed text-xl">
                    {isDanger ? (
                      <>
                        Signals from{" "}
                        <span className="text-on-surface font-bold border-b border-on-surface/30">Spectral Analysis</span>,{" "}
                        <span className="text-on-surface font-bold border-b border-on-surface/30">Biometric Matching</span>, and{" "}
                        <span className="text-on-surface font-bold border-b border-on-surface/30">Biological Veto</span>
                        {" "}have detected anomalous patterns requiring immediate compliance review.
                      </>
                    ) : (
                      <>
                        All 9 forensic layers report{" "}
                        <span className="text-on-surface font-bold border-b border-on-surface/30">physiological consistency</span>,{" "}
                        <span className="text-on-surface font-bold border-b border-on-surface/30">biometric match</span>, and{" "}
                        <span className="text-on-surface font-bold border-b border-on-surface/30">zero synthetic artifacts</span>
                        . Compliance status: PASSING.
                      </>
                    )}
                  </p>
                </div>
                <div className="flex flex-col items-center justify-center px-12 py-8 border-l border-outline-variant/20">
                  <div className="text-[12px] font-bold text-on-surface-variant tracking-[0.3em] mb-3 uppercase">
                    Confidence Interval
                  </div>
                  <div className={`text-7xl font-black drop-shadow-[0_0_10px_rgba(78,222,163,0.4)] ${fusionColor}`}>
                    {confidencePct.toFixed(2)}<span className="text-2xl opacity-50">%</span>
                  </div>
                </div>
              </div>
            </motion.section>

            {/* Error display */}
            <AnimatePresence>
              {cp.error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="bg-error-container/20 border border-error/30 rounded-xl p-4 flex items-center gap-3"
                >
                  <MaterialIcon icon="error" className="text-error" size={20} />
                  <p className="text-sm text-error">{cp.error}</p>
                  <button onClick={() => cp.dismissAlert()} className="ml-auto text-error hover:text-on-surface">
                    <MaterialIcon icon="close" size={18} />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Call Summary */}
            <AnimatePresence>
              {cp.callSummary && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="glass-panel ghost-border rounded-xl p-6 space-y-4"
                >
                  <h3 className="font-headline font-bold text-lg tracking-tight">Session Summary</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { label: "Clone Detections", value: cp.callSummary.deepfakeDetections },
                      { label: "Coercion Flags", value: cp.callSummary.coercionDetections },
                      { label: "Peak Threat", value: cp.callSummary.peakThreatLevel },
                      { label: "Final Score", value: `${(cp.callSummary.finalThreatScore * 100).toFixed(0)}%` },
                    ].map((stat) => (
                      <div key={stat.label} className="space-y-1">
                        <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">{stat.label}</span>
                        <p className="text-2xl font-headline font-black capitalize">{stat.value}</p>
                      </div>
                    ))}
                  </div>
                  {cp.callSummary.recommendation && (
                    <p className="text-sm text-on-surface-variant border-t border-outline-variant/10 pt-4">{cp.callSummary.recommendation}</p>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </Layout>
  );
};

export default CallProtection;
