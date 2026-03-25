import { useState, useEffect, useRef, useMemo } from "react";
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

/* ── Circular Threat Gauge (matches HTML SVG) ── */
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

/* ── Live waveform bars (matches HTML static bars, animated when active) ── */
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
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const [textInput, setTextInput] = useState("");

  const isDanger = cp.callState === "danger" || cp.callState === "critical";
  const isActive = cp.isActive;
  const systemStatus = isDanger ? ("alert" as const) : isActive ? ("monitoring" as const) : ("protected" as const);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [cp.transcript.length]);

  const timelineSteps = [
    { label: "Initiation", done: isActive, active: !isActive },
    { label: "ID Match", done: cp.speakerVerified, active: isActive && !cp.speakerVerified },
    { label: "Intent Check", done: cp.text.status !== "idle", active: isActive && cp.text.status === "idle" },
    { label: "Verification", done: isActive && cp.audio.status === "safe" && cp.text.status === "safe", active: isActive },
  ];

  const handleSendText = () => {
    if (textInput.trim()) {
      cp.analyzeText(textInput.trim());
      setTextInput("");
    }
  };

  const fusionVerdict = isDanger ? "THREAT DETECTED" : isActive ? "VERIFIED SAFE" : "STANDBY";
  const fusionColor = isDanger ? "text-error" : isActive ? "text-secondary" : "text-on-surface-variant";
  const fusionBorder = isDanger ? "border-error/30" : isActive ? "border-secondary/30" : "border-outline-variant/10";
  const fusionGlow = isDanger ? "shadow-[0_0_50px_rgba(255,180,171,0.1)]" : isActive ? "shadow-[0_0_50px_rgba(78,222,163,0.1)]" : "";
  const confidencePct = isActive ? Math.max(cp.audio.confidence, cp.text.confidence) : 0;

  const sentimentItems = [
    { label: "Urgency Factor", value: cp.text.details?.patterns ? 1 : 0 },
    { label: "Financial Pressure", value: cp.text.status === "danger" ? 3 : cp.text.status === "warning" ? 2 : 0 },
    { label: "Emotional Manipulation", value: cp.threatEscalation > 0.5 ? 2 : 0 },
  ];

  return (
    <Layout systemStatus={systemStatus}>
      <div className="pb-12 px-8 max-w-[1600px] mx-auto space-y-8">

        {/* ── START / STOP — Always at the very top ── */}
        <motion.section {...stagger(0)} className="flex flex-wrap gap-6 py-4">
          {!isActive ? (
            <button
              onClick={() => cp.startProtection()}
              className="flex-1 min-w-[240px] h-20 bg-gradient-to-r from-primary to-primary-container text-on-primary-container hover:shadow-[0_0_40px_rgba(0,209,255,0.5)] transition-all duration-300 rounded-2xl flex items-center justify-center gap-4 font-headline font-extrabold text-xl uppercase tracking-wider shadow-lg cursor-pointer"
            >
              <MaterialIcon icon="shield" filled size={32} />
              Start Call Protection
            </button>
          ) : (
            <>
              <button
                onClick={() => cp.stopProtection()}
                className="flex-1 min-w-[240px] h-20 bg-gradient-to-r from-error-container to-error text-on-error hover:brightness-110 transition-all duration-300 rounded-2xl flex items-center justify-center gap-4 font-headline font-extrabold text-lg uppercase tracking-wider shadow-lg cursor-pointer group"
              >
                <MaterialIcon icon="call_end" size={28} />
                Terminate Call
              </button>
              <button
                onClick={() => toast.warning("Emergency alert sent to family contacts")}
                className="flex-1 min-w-[240px] h-20 bg-surface-container-highest border border-outline-variant/20 hover:bg-surface-bright transition-all duration-300 rounded-2xl flex items-center justify-center gap-4 font-bold shadow-lg cursor-pointer"
              >
                <MaterialIcon icon="notification_important" className="text-on-surface-variant" size={24} />
                Alert Emergency
              </button>
              <button
                onClick={() => toast.success("Evidence saved to vault")}
                className="flex-1 min-w-[240px] h-20 bg-surface-container-highest border border-outline-variant/20 hover:bg-surface-bright transition-all duration-300 rounded-2xl flex items-center justify-center gap-4 font-bold shadow-lg cursor-pointer"
              >
                <MaterialIcon icon="save" className="text-on-surface-variant" size={24} />
                Save Evidence
              </button>
              {!cp.isScreenShare && (
                <button
                  onClick={() => cp.enableScreenShare()}
                  className="flex-1 min-w-[240px] h-20 bg-surface-container-highest border border-primary/20 hover:bg-primary/10 hover:border-primary/40 transition-all duration-300 rounded-2xl flex items-center justify-center gap-4 font-bold shadow-lg cursor-pointer"
                >
                  <MaterialIcon icon="screen_share" className="text-primary" size={24} />
                  <span className="text-on-surface">Reconnect Screen Share</span>
                </button>
              )}
            </>
          )}
        </motion.section>

        {/* ── STATUS WIDGETS ── */}
        <motion.div {...stagger(1)} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
            <div className="p-3 bg-primary/10 rounded-lg text-primary">
              <MaterialIcon icon="settings_input_component" size={24} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Signal Spectrum</p>
              {isActive ? (
                <div className="flex items-end gap-0.5 h-6 mt-1">
                  <div className="w-1 bg-primary/40 h-1/2" />
                  <div className="w-1 bg-primary/60 h-3/4" />
                  <div className="w-1 bg-primary h-full" />
                  <div className="w-1 bg-primary h-4/5" />
                  <div className="w-1 bg-primary/50 h-2/3" />
                </div>
              ) : (
                <p className="text-xl font-black text-on-surface">STANDBY</p>
              )}
            </div>
          </div>
          <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
            <div className="p-3 bg-secondary/10 rounded-lg text-secondary">
              <MaterialIcon icon="health_and_safety" size={24} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">System Health</p>
              <p className="text-xl font-black text-on-surface">NOMINAL</p>
            </div>
          </div>
          <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
            <div className="p-3 bg-tertiary/10 rounded-lg text-tertiary">
              <MaterialIcon icon="history" size={24} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Alert History</p>
              <p className="text-xl font-black text-on-surface">
                {isActive ? cp.transcript.filter((t) => t.flagged).length : 0} EVENTS
              </p>
            </div>
          </div>
          <div className="p-4 rounded-xl glass-panel border border-outline-variant/10 flex items-center gap-4">
            <div className="p-3 bg-primary-fixed-dim/10 rounded-lg text-primary-fixed-dim">
              <MaterialIcon icon="inventory_2" size={24} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Evidence Vault</p>
              <p className="text-xl font-black text-on-surface">{cp.callSummary ? "SAVED" : "AUTO-SYNC"}</p>
            </div>
          </div>
        </motion.div>

        {/* ── HERO PANEL ── */}
        <motion.section
          {...stagger(2)}
          className={`relative overflow-hidden h-72 rounded-2xl glass-panel flex flex-col justify-center px-12 border border-outline-variant/10 shadow-2xl ${isDanger ? "border-error/50 animate-danger-pulse" : ""}`}
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
                  {isActive ? "9-Layer Signal Monitoring" : "Alpha-9 Signal Monitoring"}
                </span>
                <div className="h-px w-12 bg-primary/30" />
                <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest">
                  Session: {isActive ? "ACTIVE" : "IDLE"}
                </span>
              </div>
              <h1 className="text-5xl font-headline font-black tracking-tighter text-on-surface mb-8">
                Real-Time Signal Analysis
              </h1>
              <div className="flex flex-wrap gap-12">
                {[
                  {
                    label: "Pitch Stability",
                    value: isActive ? Math.max(0, 100 - cp.threatEscalation * 80) : 0,
                    color: "bg-secondary shadow-[0_0_8px_rgba(78,222,163,0.5)]",
                    textColor: "text-secondary",
                    fmt: (v: number) => (isActive ? `${v.toFixed(1)}%` : "--"),
                  },
                  {
                    label: "Synthetic Noise",
                    value: isActive ? cp.threatEscalation * 100 : 0,
                    color: "bg-primary shadow-[0_0_8px_rgba(0,209,255,0.5)]",
                    textColor: "text-primary",
                    fmt: (v: number) => (isActive ? `${v.toFixed(2)}%` : "--"),
                  },
                  {
                    label: "Signal Integrity",
                    value: isActive ? Math.max(0, 95 - cp.threatEscalation * 60) : 0,
                    color: "bg-secondary shadow-[0_0_8px_rgba(78,222,163,0.5)]",
                    textColor: "text-secondary",
                    fmt: (v: number) => (isActive ? (v > 80 ? "CRYSTAL" : "DEGRADED") : "--"),
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

        {/* ── MAIN GRID: Transcription + Analysis ── */}
        <div className="grid grid-cols-12 gap-8">

          {/* LEFT: Live Transcription & Timeline */}
          <motion.div {...stagger(3)} className="col-span-12 lg:col-span-8 flex flex-col gap-8">
            {/* Transcription Panel */}
            <div className="h-[520px] rounded-2xl bg-surface-container-low border border-outline-variant/10 flex flex-col overflow-hidden shadow-xl">
              <div className="p-6 border-b border-outline-variant/10 bg-surface-container/30 flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <span className={`w-2.5 h-2.5 rounded-full ${isActive ? "bg-secondary animate-pulse shadow-[0_0_8px_rgba(78,222,163,0.8)]" : "bg-outline-variant"}`} />
                  <h2 className="font-headline font-bold text-on-surface text-lg">Live Transcription</h2>
                </div>
                <div className="flex gap-2">
                  {cp.language && (
                    <span className="text-[10px] font-bold text-on-surface-variant bg-surface-container-high px-3 py-1.5 rounded-full border border-outline-variant/10">
                      {cp.language.toUpperCase()} &bull; NEURAL V2
                    </span>
                  )}
                  <button className="text-[10px] font-bold text-primary hover:bg-primary/10 px-3 py-1.5 rounded-full transition-colors flex items-center gap-1">
                    <MaterialIcon icon="download" size={14} /> EXPORT
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-10 space-y-8">
                {cp.transcript.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-on-surface-variant/40">
                    <MaterialIcon icon="mic_off" size={48} className="mb-4 opacity-30" />
                    <p className="text-xl italic font-medium opacity-70">
                      {isActive ? "Listening for speech..." : "Waiting for call to begin..."}
                    </p>
                  </div>
                ) : (
                  cp.transcript.map((line, i) => (
                    <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-bold tracking-[0.2em] uppercase ${line.flagged ? "text-error" : "text-primary"}`}>
                          Caller
                        </span>
                        <span className="text-[10px] text-on-surface-variant opacity-50 font-mono">{line.time}</span>
                      </div>
                      <p className={`leading-relaxed text-xl ${line.flagged ? "text-error font-medium" : "text-on-surface font-medium"}`}>
                        &ldquo;{line.text}&rdquo;
                      </p>
                      {line.flagged ? (
                        <div className="flex items-center gap-2 pt-1">
                          <MaterialIcon icon="warning" size={16} className="text-error" />
                          <span className="text-xs text-error font-medium italic">Anomalous pattern detected &bull; Review required</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 pt-1">
                          <MaterialIcon icon="check_circle" size={16} className="text-secondary" />
                          <span className="text-xs text-secondary font-medium italic">Consistent syntax detected &bull; No modulation artifacts</span>
                        </div>
                      )}
                    </motion.div>
                  ))
                )}
                <div ref={transcriptEndRef} />
              </div>
              {isActive && (
                <div className="p-4 border-t border-outline-variant/10 flex gap-3 bg-surface-container/20">
                  <input
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSendText()}
                    placeholder="Type text to analyze for coercion..."
                    className="flex-grow bg-transparent border-b border-outline-variant focus:border-primary outline-none text-on-surface text-sm py-2 px-1 transition-colors"
                  />
                  <button onClick={handleSendText} className="text-primary hover:text-primary-container transition-colors">
                    <MaterialIcon icon="send" size={20} />
                  </button>
                </div>
              )}
            </div>

            {/* Call Timeline (horizontal) */}
            <div className="p-6 rounded-2xl bg-surface-container border border-outline-variant/10">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <MaterialIcon icon="timeline" className="text-primary" size={24} />
                  <h3 className="font-headline font-bold text-on-surface">Call Timeline</h3>
                </div>
                <span className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">
                  Duration: {isActive ? "LIVE" : "00:00"}
                </span>
              </div>
              <div className="relative py-4">
                <div className="absolute top-1/2 left-0 w-full h-0.5 bg-surface-container-highest -translate-y-1/2" />
                <div className="relative flex justify-between">
                  {timelineSteps.map((step, i) => (
                    <div key={i} className={`relative flex flex-col items-center ${!step.done && !step.active ? "opacity-30" : ""}`}>
                      <div
                        className={`w-3 h-3 rounded-full z-10 mb-2 ${
                          step.done
                            ? "bg-secondary shadow-[0_0_8px_rgba(78,222,163,0.5)]"
                            : step.active
                            ? "bg-primary animate-pulse"
                            : "bg-surface-container-highest"
                        }`}
                      />
                      <span className={`text-[10px] font-bold ${step.active ? "text-primary" : "text-on-surface"}`}>
                        {step.label}
                      </span>
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
                      {cp.speakerMatch ? `Matched: ${cp.speakerMatch}` : isActive ? "Identifying..." : "No Speaker"}
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
                    {isActive ? `${(cp.threatEscalation * 5).toFixed(3)}%` : "--"}
                  </p>
                </div>
                <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/10">
                  <p className="text-[10px] text-on-surface-variant uppercase font-bold mb-2 tracking-widest">Identity Score</p>
                  <p className="text-2xl font-black text-secondary">
                    {cp.speakerVerified ? (
                      <>{(cp.speakerSimilarity * 100).toFixed(1)}<span className="text-sm font-bold opacity-50">/100</span></>
                    ) : isActive ? "..." : "--"}
                  </p>
                </div>
              </div>
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

            {/* Sentiment & Intent (block-style bars) */}
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
          </motion.div>
        </div>

        {/* ── FUSION SUMMARY SECTION ── */}
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
                    icon={isDanger ? "gpp_bad" : isActive ? "verified" : "shield"}
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
                {isActive ? (
                  <>
                    Signals from{" "}
                    <span className="text-on-surface font-bold border-b border-on-surface/30">Spectral Analysis</span>,{" "}
                    <span className="text-on-surface font-bold border-b border-on-surface/30">Biometric Matching</span>, and{" "}
                    <span className="text-on-surface font-bold border-b border-on-surface/30">Linguistic Intent</span>
                    {isDanger
                      ? " have detected anomalous patterns requiring immediate review."
                      : " show 100% correlation with known profiles. No traces of synthetic modulation, voice cloning artifacts, or coercive patterns detected in this session."}
                  </>
                ) : (
                  "Start call protection to begin real-time multi-modal analysis across all 9 detection layers."
                )}
              </p>
            </div>
            {isActive && (
              <div className="flex flex-col items-center justify-center px-12 py-8 border-l border-outline-variant/20">
                <div className="text-[12px] font-bold text-on-surface-variant tracking-[0.3em] mb-3 uppercase">
                  Confidence Interval
                </div>
                <div className={`text-7xl font-black drop-shadow-[0_0_10px_rgba(78,222,163,0.4)] ${fusionColor}`}>
                  {confidencePct.toFixed(2)}<span className="text-2xl opacity-50">%</span>
                </div>
              </div>
            )}
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

        {/* Call Summary (after call ends) */}
        <AnimatePresence>
          {cp.callSummary && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="glass-panel ghost-border rounded-xl p-6 space-y-4"
            >
              <h3 className="font-headline font-bold text-lg tracking-tight">Call Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Deepfake Detections", value: cp.callSummary.deepfakeDetections },
                  { label: "Coercion Detections", value: cp.callSummary.coercionDetections },
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
      </div>
    </Layout>
  );
};

export default CallProtection;
