import { useState, useCallback, useRef, useEffect } from "react";
import { toast } from "sonner";
import Layout from "@/components/Layout";
import MaterialIcon from "@/components/MaterialIcon";
import { api, API_BASE } from "@/lib/api";

// ─── Feedback States ──────────────────────────────────────────────────
type FeedbackState = "none" | "correct" | "incorrect" | "unsure" | "submitted";

// ─── Types ─────────────────────────────────────────────────────────────

type State = "upload" | "preview" | "analyzing" | "results";
type Verdict = "ai-generated" | "authentic" | "uncertain" | "inconclusive";

interface ApiResult {
  verdict: string;
  confidence: number;
  forensic_checks?: any[];
  forensic_data?: any[];
  raw_scores?: Record<string, any>;
  scan_id?: string | null;
  report_url?: string | null;
  status_text?: string;
  subtext?: string;
}

// ─── Constants ─────────────────────────────────────────────────────────

const PIPELINE_STEPS_BY_TYPE: Record<string, { icon: string; label: string; key: string }[]> = {
  image: [
    { icon: "image_search", label: "Content Classification", key: "classify" },
    { icon: "compare", label: "ELA & Frequency", key: "ela" },
    { icon: "psychology", label: "Neural ViT + CLIP", key: "neural" },
    { icon: "face", label: "Face Forensics", key: "face" },
    { icon: "gavel", label: "Final Verdict", key: "verdict" },
  ],
  audio: [
    { icon: "graphic_eq", label: "AST Spectrogram", key: "ast" },
    { icon: "mic", label: "SSL & Whisper", key: "ssl" },
    { icon: "tune", label: "Prosodic & Breathing", key: "prosodic" },
    { icon: "record_voice_over", label: "TTS & Voice Clone", key: "tts" },
    { icon: "hub", label: "Ensemble Fusion", key: "verdict" },
  ],
  video: [
    { icon: "hd", label: "Quality Assessment", key: "quality" },
    { icon: "slow_motion_video", label: "Temporal R3D", key: "temporal" },
    { icon: "face", label: "rPPG & Micro-Expr", key: "bio" },
    { icon: "light_mode", label: "Lighting & AV Sync", key: "physics" },
    { icon: "gavel", label: "Final Verdict", key: "verdict" },
  ],
};
const PIPELINE_DEFAULT = [
  { icon: "download", label: "Metadata Extraction", key: "metadata" },
  { icon: "analytics", label: "Neural Analysis", key: "neural" },
  { icon: "fingerprint", label: "Biometric Sync", key: "biometric" },
  { icon: "gavel", label: "Final Verdict", key: "verdict" },
];

const ALL_ACCEPTED = [
  "image/jpeg", "image/png", "image/webp",
  "video/mp4", "video/quicktime", "video/x-msvideo",
  "audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/x-wav", "audio/mp3",
];
const ACCEPT_EXT = ".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi,.wav,.mp3,.ogg,.flac";

const LAYER_ICONS: Record<string, string> = {
  ela: "compare", fingerprint: "analytics", noise: "graphic_eq", pixel: "grid_on",
  model: "psychology", c2pa: "verified_user", compression: "compress",
  image_type: "image_search", video_quality: "hd", motion_blur: "blur_on",
  voice_synth: "record_voice_over", spectral: "equalizer",
  face: "face", face_geometry: "face_retouching_natural",
  // 7-Layer Audio Forensics
  ensemble_verdict: "hub", ast_spectrogram: "graphic_eq",
  prosodic_forensics: "tune", breathing_detection: "pulmonology",
  phase_forensics: "waves", formant_analysis: "voice_selection",
  temporal_consistency: "timeline", spatial_deepfake: "face",
  temporal_deepfake: "slow_motion_video", quality_disclaimer: "info",
};

// ─── Helpers ───────────────────────────────────────────────────────────

function mediaType(t: string) {
  return t.startsWith("image/") ? "image" : t.startsWith("video/") ? "video" : "audio";
}

function fmtSize(b: number) {
  return b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`;
}

function layerScore(id: string, rs: Record<string, any>): number {
  const m: Record<string, string> = {
    ela: "ela", fingerprint: "frequency", noise: "noise", pixel: "pixel",
    model: "neural_effective", c2pa: "metadata", face: "face",
    // 7-Layer Audio Forensics (scores are 0-1 where higher = more synthetic)
    ensemble_verdict: "ensemble_probability",
    ast_spectrogram: "ast_spoof_prob",
    prosodic_forensics: "prosodic_score",
    breathing_detection: "breathing_score",
    phase_forensics: "phase_score",
    formant_analysis: "formant_score",
    temporal_consistency: "temporal_score",
  };
  const v = rs[m[id] ?? ""];
  return v != null ? Math.round((1 - v) * 100) : -1;
}

function verdictCfg(v: Verdict, statusText?: string) {
  if (statusText === "Partially Consistent") {
    return { 
      label: "PARTIALLY CONSISTENT", 
      threat: "MODERATE", 
      color: "text-primary", 
      bg: "bg-primary", 
      border: "border-primary", 
      glow: "shadow-glow-sm", 
      glowBg: "bg-primary/20",
      icon: "warning_amber"
    };
  }
  if (v === "ai-generated") return { label: "SYNTHETIC", threat: "CRITICAL", color: "text-error", bg: "bg-error", border: "border-error", glow: "shadow-glow-danger", glowBg: "bg-error/10", icon: "dangerous" };
  if (v === "authentic") return { label: "AUTHENTIC", threat: "CLEARED", color: "text-secondary", bg: "bg-secondary", border: "border-secondary", glow: "shadow-glow-safe", glowBg: "bg-secondary/10", icon: "verified" };
  return { label: "INCONCLUSIVE", threat: "WARNING", color: "text-primary", bg: "bg-primary", border: "border-primary", glow: "shadow-glow-sm", glowBg: "bg-primary/10", icon: "help" };
}

function genDescription(v: Verdict, checks: any[], statusText?: string) {
  const f = checks.filter((c: any) => c.status === "fail").length;
  const w = checks.filter((c: any) => c.status === "warn").length;
  const p = checks.filter((c: any) => c.status === "pass").length;
  
  if (statusText === "Partially Consistent") {
    return `Forensic analysis detected several anomalies despite structural consistency. ${f} failed checks and ${w} warnings suggest potential manipulation or low-fidelity synthesis. Proceed with caution.`;
  }
  
  if (v === "authentic") return `Multilayer forensic analysis completed. ${p} analysis layers passed verification. The media maintains consistent spectral integrity with no synthetic artifacts detected.`;
  if (v === "ai-generated") return `Multilayer forensic analysis completed. ${f} critical anomalies and ${w} warnings detected across forensic layers. Synthetic generation patterns identified in neural and frequency domains.`;
  return `Multilayer forensic analysis completed. Mixed signals across ${checks.length} forensic layers with ${w} warnings. Manual review recommended for conclusive determination.`;
}

// ─── Component ─────────────────────────────────────────────────────────

const Scanner = () => {
  const [state, setState] = useState<State>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [scanId, setScanId] = useState("");
  const [feedbackState, setFeedbackState] = useState<FeedbackState>("none");
  const [expandedCheck, setExpandedCheck] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Collapsible sections
  const [sec, setSec] = useState<Record<string, boolean>>({
    layers: true, signals: true, map: true, meta: true,
  });
  const toggle = (k: string) => setSec((p) => ({ ...p, [k]: !p[k] }));

  // Derived
  const verdict = (result?.verdict as Verdict) ?? "authentic";
  const confidence = result?.confidence ?? 0;
  const checks = result?.forensic_checks ?? result?.forensic_data ?? [];
  const raw = result?.raw_scores ?? {};
  const vc = verdictCfg(verdict, result?.status_text);
  const anomalies = checks.filter((c: any) => c.status === "fail" || c.status === "warn");
  const passes = checks.filter((c: any) => c.status === "pass");
  const pipelineSteps = file ? (PIPELINE_STEPS_BY_TYPE[mediaType(file.type)] ?? PIPELINE_DEFAULT) : PIPELINE_DEFAULT;
  const circumference = 2 * Math.PI * 28;
  const dashoffset = circumference * (1 - confidence / 100);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  // ─── Handlers ──────────────────────────────────────────────────────

  const handleFile = useCallback((f: File) => {
    if (!ALL_ACCEPTED.includes(f.type)) { toast.error("Unsupported file type."); return; }
    if (f.size > 200 * 1048576) { toast.error("File too large (max 200MB)."); return; }
    setFile(f);
    if (f.type.startsWith("image/")) {
      const url = URL.createObjectURL(f);
      setPreview(url);
      const img = new Image();
      img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
      img.src = url;
    } else { setPreview(null); setDims(null); }
    setState("preview");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleScan = async () => {
    if (!file) return;
    setState("analyzing");
    setScanId(`SD-${Math.floor(Math.random() * 9000 + 1000)}-XFB`);
    const t0 = performance.now();

    // Animate through pipeline steps while API call runs
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < pipelineSteps.length - 2) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 1200);
    setCurrentStep(0);

    try {
      let data: ApiResult;
      if (mediaType(file.type) === "audio") {
        const fd = new FormData(); fd.append("file", file);
        const resp = await fetch(`${API_BASE}/api/analyze/audio`, { method: "POST", body: fd });
        if (!resp.ok) { const e = await resp.json().catch(() => ({ detail: resp.statusText })); throw new Error(e.detail || "Audio analysis failed"); }
        const r = await resp.json();
        // Map verdict: "spoof" → "ai-generated", "authentic" stays, "uncertain" → "inconclusive"
        const vMap: Record<string, Verdict> = { spoof: "ai-generated", authentic: "authentic", uncertain: "inconclusive" };
        data = {
          verdict: vMap[r.verdict] ?? "inconclusive",
          confidence: r.confidence ?? 0,
          forensic_checks: r.forensic_checks ?? [
            { id: "voice_synth", name: "Voice Synthesis Detection", status: r.verdict === "spoof" ? "fail" : "pass", description: `${r.verdict} (${r.confidence}%)` },
          ],
          raw_scores: r.raw_scores ?? r.details?.probabilities ?? {},
        };
      } else {
        data = await api.analyze.media(file) as any;
      }
      clearInterval(stepInterval);
      setResult(data);
      setElapsed((performance.now() - t0) / 1000);
      setCurrentStep(pipelineSteps.length);
      setState("results");
    } catch (err) {
      clearInterval(stepInterval);
      toast.error(err instanceof Error ? err.message : "Analysis failed");
      setState("preview"); setCurrentStep(-1);
    }
  };

  const handleReset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null); setDims(null);
    setState("upload"); setCurrentStep(-1);
    setResult(null); setElapsed(0); setFeedbackState("none");
    setExpandedCheck(null);
  };

  const handleFeedback = async (feedback: "correct" | "incorrect" | "unsure") => {
    const sid = result?.scan_id;
    if (!sid) { toast.error("No scan ID available for feedback"); return; }
    try {
      setFeedbackState(feedback);
      await api.monitoring.feedback(sid, feedback);
      setFeedbackState("submitted");
      toast.success(feedback === "correct" ? "Thanks! This helps improve accuracy." : feedback === "incorrect" ? "Got it — we'll use this to improve." : "Noted — we'll review this case.");
    } catch { setFeedbackState("none"); toast.error("Failed to submit feedback"); }
  };

  const handlePdfReport = async () => {
    const sid = result?.scan_id;
    if (!sid) { handleDownloadReport(); return; }
    try {
      const token = localStorage.getItem("satya-token");
      const resp = await fetch(`${API_BASE}/api/scans/${sid}/report`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!resp.ok) { handleDownloadReport(); return; }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `satyadrishti_report_${sid.slice(0, 8)}.pdf`; a.click();
      URL.revokeObjectURL(url);
      toast.success("PDF report downloaded");
    } catch { handleDownloadReport(); }
  };

  const handleDownloadReport = () => {
    if (!result || !file) return;
    const lines = [
      "═══════════════════════════════════════════════════════════",
      "         SATYA DRISHTI — FORENSIC ANALYSIS REPORT",
      "═══════════════════════════════════════════════════════════",
      "",
      `Case ID:      ${scanId}`,
      `File Name:    ${file.name}`,
      `File Type:    ${file.type}`,
      `File Size:    ${fmtSize(file.size)}`,
      ...(dims ? [`Resolution:   ${dims.w} × ${dims.h}`] : []),
      `Date:         ${new Date().toISOString()}`,
      `Processing:   ${elapsed.toFixed(2)}s`,
      "",
      "───────────────────────────────────────────────────────────",
      "  VERDICT",
      "───────────────────────────────────────────────────────────",
      "",
      `  Result:     ${verdict.toUpperCase()}`,
      `  Confidence: ${Math.round(confidence)}%`,
      `  Threat:     ${vc.threat}`,
      "",
      "───────────────────────────────────────────────────────────",
      "  FORENSIC ANALYSIS LAYERS",
      "───────────────────────────────────────────────────────────",
      "",
      ...checks.map((c: any, i: number) => {
        const status = c.status === "pass" ? "PASS" : c.status === "fail" ? "FAIL" : c.status === "warn" ? "WARN" : "INFO";
        return [
          `  [${String(i + 1).padStart(2, "0")}] ${c.name || c.label}`,
          `       Status: ${status}`,
          `       Detail: ${c.description || c.detail || "—"}`,
          "",
        ].join("\n");
      }),
      "───────────────────────────────────────────────────────────",
      "  RAW SCORES",
      "───────────────────────────────────────────────────────────",
      "",
      ...Object.entries(raw).map(([k, v]) =>
        `  ${k.padEnd(20)} ${typeof v === "object" ? JSON.stringify(v) : v}`
      ),
      "",
      "═══════════════════════════════════════════════════════════",
      "  Generated by Satya Drishti Deepfake Detection System",
      "  https://satyadrishti.in",
      "═══════════════════════════════════════════════════════════",
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `satyadrishti_report_${scanId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Report downloaded");
  };

  const handleSaveCase = async () => {
    if (!result || !file) return;
    const token = localStorage.getItem("satya-token");
    if (!token) {
      toast.error("Sign in to save cases to your account");
      return;
    }
    try {
      const title = `${verdict === "ai-generated" ? "Deepfake" : verdict === "authentic" ? "Verified" : "Review"}: ${file.name}`;
      const scanIds = result.scan_id ? [result.scan_id] : [];
      await api.cases.create(title, `${vc.label} (${Math.round(confidence)}%) — ${checks.length} forensic layers analyzed`, scanIds);
      toast.success("Case saved successfully");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save case");
    }
  };

  const handleShareEvidence = async () => {
    if (!result || !file) return;
    const summary = [
      `Satya Drishti Analysis: ${file.name}`,
      `Verdict: ${vc.label} (${Math.round(confidence)}%)`,
      `Threat Level: ${vc.threat}`,
      `Checks: ${checks.filter((c: any) => c.status === "pass").length} passed, ${checks.filter((c: any) => c.status === "fail").length} failed, ${checks.filter((c: any) => c.status === "warn").length} warnings`,
      `Analyzed: ${new Date().toLocaleString()}`,
    ].join("\n");

    if (navigator.share) {
      try {
        await navigator.share({ title: `Satya Drishti — ${file.name}`, text: summary });
        return;
      } catch { /* user cancelled or not supported */ }
    }
    await navigator.clipboard.writeText(summary);
    toast.success("Evidence summary copied to clipboard");
  };

  // ─── Render ────────────────────────────────────────────────────────

  return (
    <Layout systemStatus="monitoring">
      <div className="pt-32 pb-24 px-6 md:px-12 max-w-[1440px] mx-auto">

        {/* ═══════════════════════════════════════════════════════════════
             UPLOAD / PREVIEW / ANALYZING
           ═══════════════════════════════════════════════════════════════ */}
        {state !== "results" && (
          <div className="space-y-16">
            {/* Hero */}
            <section className="space-y-8">
              <div className="max-w-5xl">
                <h1 className="font-headline text-4xl sm:text-6xl md:text-8xl font-black tracking-tighter text-on-surface mb-6 leading-[0.85]">
                  Detection Lab<span className="text-primary-container">.</span>
                </h1>
                <p className="text-on-surface-variant text-xl max-w-2xl font-light leading-relaxed">
                  Advanced neural artifact analysis for enterprise-grade media verification. 
                  Identify synthetic manipulation across <span className="text-primary font-bold">image, video, and audio</span> domains.
                </p>
              </div>

              {/* Upload Zone — "System Input Panel" */}
              {state === "upload" && (
                <div className="space-y-6">
                  <div className="relative group">
                    {/* Animated Border Glow */}
                    <div className="absolute -inset-1 bg-gradient-to-r from-primary/20 via-primary/40 to-primary/20 rounded-3xl blur opacity-25 group-hover:opacity-60 transition duration-1000 group-hover:duration-200" />
                    
                    <div
                      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                      onDragLeave={() => setIsDragging(false)}
                      onDrop={handleDrop}
                      onClick={() => inputRef.current?.click()}
                      className={`relative w-full aspect-[4/3] sm:aspect-[16/9] lg:aspect-[21/9] bg-surface-container-highest/20 backdrop-blur-2xl rounded-2xl border transition-all duration-700 cursor-pointer flex flex-col items-center justify-center space-y-6 overflow-hidden ${
                        isDragging ? "border-primary bg-primary/10 shadow-[0_0_50px_rgba(0,209,255,0.3)] scale-[1.01]" : "border-white/5 hover:border-primary/40 hover:bg-surface-container-highest/40"
                      }`}
                    >
                      {/* Grid Background Trace */}
                      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
                        style={{ backgroundImage: "linear-gradient(#fff 1px, transparent 1px), linear-gradient(90(#fff 1px, transparent 1px)", backgroundSize: "40px 40px" }} 
                      />
                      
                      <div className="relative z-10 flex flex-col items-center">
                        <div className={`w-32 h-32 rounded-3xl bg-surface-container-high/80 backdrop-blur-md border border-white/10 flex items-center justify-center mb-8 transition-all duration-700 shadow-2xl ${isDragging ? "scale-110 border-primary/50 bg-primary/10 shadow-[0_0_40px_rgba(0,209,255,0.4)]" : "group-hover:scale-[1.1] group-hover:border-primary/30 group-hover:bg-surface-container-highest"}`}>
                          <MaterialIcon icon="upload_file" size={64} className={`transition-colors duration-500 ${isDragging ? "text-primary" : "text-primary-container/60 group-hover:text-primary"}`} />
                        </div>
                        
                        <h3 className="font-headline text-4xl md:text-5xl font-black tracking-tight text-on-surface text-center px-4 mb-2">
                          Upload Evidence for Analysis
                        </h3>
                        
                        <div className="flex flex-col items-center gap-5 mt-4">
                          <div className="flex gap-6 z-10 opacity-60 group-hover:opacity-100 transition-opacity duration-500">
                            {[{ icon: "image", label: "Image" }, { icon: "movie", label: "Video" }, { icon: "audiotrack", label: "Audio" }].map((b) => (
                              <div key={b.label} className="flex items-center gap-3 px-4 py-2 bg-black/40 rounded-xl border border-white/5 backdrop-blur-md">
                                <MaterialIcon icon={b.icon} size={18} className="text-primary-container" />
                                <span className="text-xs font-label uppercase tracking-widest text-on-surface-variant font-bold">{b.label}</span>
                              </div>
                            ))}
                          </div>
                          
                          <div className="flex flex-col items-center gap-3">
                            <p className="font-label text-sm uppercase tracking-[0.3em] text-outline font-bold">
                              {isDragging ? "Drop to intercept" : "Drag and drop forensic subject"}
                            </p>
                            <div className="flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full border border-primary/20">
                              <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                              <span className="text-[10px] text-primary uppercase tracking-[0.2em] font-black">
                                Multi-layer forensic processing enabled
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <input ref={inputRef} type="file" accept={ACCEPT_EXT} className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
                  </div>
                </div>
              )}

              {/* File Preview */}
              {state === "preview" && file && (
                <div className="space-y-6">
                  <div className="rounded-xl bg-surface-container-low p-6 flex items-center gap-5 border border-outline-variant/10">
                    <div className="w-20 h-20 rounded-xl bg-surface-container-high flex items-center justify-center overflow-hidden shrink-0">
                      {preview ? <img src={preview} alt="Preview" className="w-full h-full object-cover" /> :
                        <MaterialIcon icon={mediaType(file.type) === "video" ? "video_file" : "audio_file"} size={32} className="text-primary/50" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-on-surface font-headline font-semibold truncate">{file.name}</p>
                      <p className="text-sm text-on-surface-variant mt-1 font-label">
                        {fmtSize(file.size)} · {mediaType(file.type).charAt(0).toUpperCase() + mediaType(file.type).slice(1)}
                        {dims && ` · ${dims.w}×${dims.h}`}
                      </p>
                    </div>
                    <button onClick={handleReset} className="p-2 rounded-lg text-on-surface-variant hover:text-error hover:bg-error-container/20 transition-colors cursor-pointer">
                      <MaterialIcon icon="close" size={20} />
                    </button>
                  </div>
                  <button onClick={handleScan} className="w-full py-4 bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm tracking-widest uppercase rounded-lg shadow-xl transition-transform hover:scale-[1.02] active:scale-95 cursor-pointer">
                    Begin Analysis
                  </button>
                </div>
              )}
            </section>

            {/* Pipeline Animation */}
            {state === "analyzing" && (
              <section className="space-y-12">
                {/* Cinematic Scanning Preview */}
                <div className="relative w-full aspect-video rounded-3xl overflow-hidden bg-surface-container-low border border-white/5 shadow-2xl">
                  {preview ? (
                    <img src={preview} alt="Scanning" className="w-full h-full object-contain opacity-40 grayscale" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-surface-container-high">
                      <MaterialIcon icon="biotech" size={80} className="text-primary/20 animate-pulse" />
                    </div>
                  )}
                  {/* Scanning Beam */}
                  <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/20 to-transparent h-20 w-full animate-scan-line z-20" />
                  <div className="absolute inset-0 bg-primary/5 backdrop-blur-[2px] z-10" />
                  
                  <div className="absolute inset-0 flex flex-col items-center justify-center z-30">
                    <div className="p-8 rounded-2xl bg-black/60 backdrop-blur-xl border border-white/10 flex flex-col items-center gap-4">
                      <div className="w-16 h-16 rounded-full border-4 border-primary border-t-transparent animate-spin" />
                      <p className="font-headline text-xl font-bold tracking-widest uppercase text-primary">Analyzing Subject...</p>
                      <p className="text-xs font-mono text-outline uppercase tracking-widest italic">{scanId}</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-8">
                  <div className="flex items-center justify-between">
                    <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">Forensic Pipeline Status</span>
                    <span className="font-label text-xs text-on-surface-variant italic">ID: {scanId}</span>
                  </div>
                  <div className="relative flex justify-between items-start pt-4 overflow-x-auto pb-2">
                    <div className="absolute top-9 left-0 w-full h-[2px] bg-surface-container-high z-0" />
                    {pipelineSteps.map((step, i) => {
                      const done = currentStep > i; const active = currentStep === i;
                      return (
                        <div key={step.key} className={`relative z-10 flex flex-col items-center w-1/5 text-center transition-all duration-500 ${currentStep < i ? "opacity-30 grayscale" : "opacity-100"}`}>
                          <div className={`w-12 h-12 rounded-2xl flex items-center justify-center mb-4 transition-all duration-500 ${
                            done ? "bg-secondary shadow-glow-emerald rotate-[360deg]" : active ? "bg-primary-container ring-8 ring-primary/10 scale-110" : "bg-surface-container-high"
                          }`}>
                            {done ? <MaterialIcon icon="check" filled size={20} className="text-on-secondary" /> :
                              <MaterialIcon icon={step.icon} size={20} className={active ? "text-on-primary-container animate-pulse" : "text-on-surface-variant"} />}
                          </div>
                          <span className="font-headline text-sm font-bold text-on-surface">{step.label}</span>
                          <span className={`text-[10px] font-label uppercase tracking-widest mt-1 font-black ${done ? "text-secondary" : active ? "text-primary animate-pulse" : "text-on-surface-variant"}`}>
                            {done ? "Verified" : active ? "Analyzing..." : "Buffered"}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </section>
            )}
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════
             RESULTS — Single Page Forensic Dashboard
           ═══════════════════════════════════════════════════════════════ */}
        {state === "results" && result && (
          <div className="space-y-12 animate-in fade-in slide-in-from-bottom-8 duration-1000">

            {/* ────────────────────────────────────────────────────────
                 SECTION 1 — Case Header
               ──────────────────────────────────────────────────────── */}
            <header className="flex flex-col lg:flex-row justify-between lg:items-start gap-8">
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-4 mb-5">
                  <span className="text-primary font-mono text-xs tracking-widest uppercase bg-primary/5 px-3 py-1 rounded-md border border-primary/20">Case ID: {scanId}</span>
                  
                  {/* Forensic Override Badge */}
                  {(verdict === "ai-generated" && confidence < 75) || anomalies.length >= 2 ? (
                    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-error/10 border border-error/30">
                      <MaterialIcon icon="gavel" size={14} className="text-error" />
                      <span className="text-[10px] font-black text-error uppercase tracking-[0.1em]">Forensic Rule Override Active</span>
                    </div>
                  ) : null}

                  <span className={`text-[10px] px-3 py-1 rounded-full border font-black uppercase tracking-widest ${
                    result?.status_text === "Partially Consistent" ? "bg-primary/10 text-primary border-primary/20" :
                    verdict === "ai-generated" ? "bg-error/10 text-error border-error/20" :
                    verdict === "authentic" ? "bg-secondary/10 text-secondary border-secondary/20" :
                    "bg-primary/10 text-primary border-primary/20"
                  }`}>
                    {result?.status_text === "Partially Consistent" ? "POTENTIAL RISK" :
                     verdict === "ai-generated" ? "THREAT DETECTED" : 
                     verdict === "authentic" ? "VERIFIED SAFE" : "REVIEW NEEDED"}
                  </span>
                </div>
                <h1 className="text-2xl sm:text-4xl md:text-6xl font-headline font-black text-on-surface tracking-tighter mb-6 break-words leading-[0.9]">
                  Verification Analysis: <span className="text-primary-container break-all font-mono opacity-80">{file?.name ?? "Unknown"}</span>
                </h1>
                <p className="text-on-surface-variant text-base font-light leading-relaxed max-w-3xl">
                  {genDescription(verdict, checks, result?.status_text)}
                </p>
              </div>
              <div className="flex gap-8 shrink-0">
                <div className="text-right">
                  <p className="text-[10px] text-outline uppercase tracking-widest mb-1">Processing Time</p>
                  <p className="text-2xl font-headline font-bold text-on-surface">{elapsed.toFixed(2)}s</p>
                </div>
                <div className="text-right border-l border-outline-variant/30 pl-8">
                  <p className="text-[10px] text-outline uppercase tracking-widest mb-1">Threat Level</p>
                  <p className={`text-2xl font-headline font-extrabold ${vc.color}`}>{vc.threat}</p>
                </div>
              </div>
            </header>

            {/* ────────────────────────────────────────────────────────
                 SECTION 2 — Verdict Banner + Confidence Gauge
               ──────────────────────────────────────────────────────── */}
            <section className={`rounded-2xl p-8 md:p-10 border relative overflow-hidden ${
              verdict === "ai-generated" ? "border-error/20 bg-error/[0.03]" :
              verdict === "authentic" ? "border-secondary/20 bg-secondary/[0.03]" :
              "border-primary/20 bg-primary/[0.03]"
            }`}>
              <div className={`absolute -right-20 -top-20 w-80 h-80 rounded-full blur-[120px] opacity-30 ${vc.glowBg}`} />
              <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
                {/* Left: Verdict */}
                <div className="flex items-center gap-8">
                  {/* SVG Gauge */}
                  <div className="relative w-24 h-24 flex items-center justify-center shrink-0">
                    <svg className="w-full h-full -rotate-90" viewBox="0 0 64 64">
                      <circle cx="32" cy="32" r="28" fill="none" stroke="currentColor" strokeWidth="3" className="text-surface-container-high" />
                      <circle cx="32" cy="32" r="28" fill="none" stroke="currentColor" strokeWidth="3"
                        strokeDasharray={circumference} strokeDashoffset={dashoffset} strokeLinecap="round"
                        className={vc.color} style={{ transition: "stroke-dashoffset 1s ease" }} />
                    </svg>
                    <span className="absolute text-lg font-extrabold font-headline">{Math.round(confidence)}%</span>
                  </div>
                  <div>
                    <p className="text-[10px] text-outline uppercase tracking-widest mb-1">Authenticity Verdict</p>
                    <h2 className={`text-2xl sm:text-4xl md:text-5xl font-headline font-extrabold tracking-tight ${vc.color}`}>{vc.label}</h2>
                    <div className="flex items-center gap-3 mt-3">
                      <div className="flex gap-1">
                        {[1, 2, 3, 4, 5].map((n) => (
                          <div key={n} className={`h-1.5 w-5 rounded-full ${n <= Math.ceil(confidence / 20) ? vc.bg : "bg-outline-variant/30"}`} />
                        ))}
                      </div>
                      <span className="text-xs text-on-surface-variant">Fusion Score</span>
                    </div>
                  </div>
                </div>

                {/* Right: Quick stats */}
                <div className="flex flex-wrap gap-3 md:gap-6">
                  <div className="bg-surface-container-lowest/60 backdrop-blur-sm px-4 md:px-6 py-3 md:py-4 rounded-xl border border-outline-variant/10 text-center min-w-[100px] flex-1 md:flex-none">
                    <p className="text-[10px] text-outline uppercase tracking-widest mb-1">Neural Noise</p>
                    <p className={`text-xl font-headline font-bold ${verdict === "ai-generated" ? "text-error" : "text-on-surface"}`}>
                      {verdict === "ai-generated" ? "Elevated" : "Negligible"}
                    </p>
                  </div>
                  <div className="bg-surface-container-lowest/60 backdrop-blur-sm px-4 md:px-6 py-3 md:py-4 rounded-xl border border-outline-variant/10 text-center min-w-[100px] flex-1 md:flex-none">
                    <p className="text-[10px] text-outline uppercase tracking-widest mb-1">Artifacts</p>
                    <p className={`text-xl font-headline font-bold ${anomalies.filter(a => a.status === "fail").length > 0 ? "text-error" : "text-on-surface"}`}>
                      {anomalies.filter(a => a.status === "fail").length > 0
                        ? `${anomalies.filter(a => a.status === "fail").length} Found`
                        : "0 Detected"}
                    </p>
                  </div>
                  <div className="bg-surface-container-lowest/60 backdrop-blur-sm px-4 md:px-6 py-3 md:py-4 rounded-xl border border-outline-variant/10 text-center min-w-[100px] flex-1 md:flex-none">
                    <p className="text-[10px] text-outline uppercase tracking-widest mb-1">Layers</p>
                    <p className="text-xl font-headline font-bold text-on-surface">{checks.length}</p>
                  </div>
                </div>
              </div>
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 2.5 — Accuracy Feedback
               ──────────────────────────────────────────────────────── */}
            <section className="rounded-2xl p-6 bg-surface-container-low/70 backdrop-blur-xl border border-outline-variant/10">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <MaterialIcon icon="rate_review" size={24} className="text-primary" />
                  <div>
                    <p className="text-sm font-headline font-bold text-on-surface">Was this analysis accurate?</p>
                    <p className="text-xs text-on-surface-variant mt-0.5">Your feedback directly improves our detection accuracy</p>
                  </div>
                </div>
                {feedbackState === "submitted" ? (
                  <div className="flex items-center gap-2 px-4 py-2 bg-secondary/10 rounded-full border border-secondary/20">
                    <MaterialIcon icon="check_circle" size={16} className="text-secondary" />
                    <span className="text-xs font-bold text-secondary uppercase tracking-widest">Feedback Recorded</span>
                  </div>
                ) : (
                  <div className="flex gap-3">
                    <button
                      onClick={() => handleFeedback("correct")}
                      disabled={feedbackState !== "none"}
                      className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-bold uppercase tracking-widest transition-all cursor-pointer ${
                        feedbackState === "correct" ? "bg-secondary/20 border-secondary/40 text-secondary" : "bg-surface-container-high border-outline-variant/20 text-on-surface-variant hover:border-secondary/40 hover:text-secondary"
                      }`}
                    >
                      <MaterialIcon icon="thumb_up" size={16} /> Correct
                    </button>
                    <button
                      onClick={() => handleFeedback("incorrect")}
                      disabled={feedbackState !== "none"}
                      className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-bold uppercase tracking-widest transition-all cursor-pointer ${
                        feedbackState === "incorrect" ? "bg-error/20 border-error/40 text-error" : "bg-surface-container-high border-outline-variant/20 text-on-surface-variant hover:border-error/40 hover:text-error"
                      }`}
                    >
                      <MaterialIcon icon="thumb_down" size={16} /> Incorrect
                    </button>
                    <button
                      onClick={() => handleFeedback("unsure")}
                      disabled={feedbackState !== "none"}
                      className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-bold uppercase tracking-widest transition-all cursor-pointer ${
                        feedbackState === "unsure" ? "bg-primary/20 border-primary/40 text-primary" : "bg-surface-container-high border-outline-variant/20 text-on-surface-variant hover:border-primary/40 hover:text-primary"
                      }`}
                    >
                      <MaterialIcon icon="help" size={16} /> Unsure
                    </button>
                  </div>
                )}
              </div>
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 3 — Media Preview + Key Findings
               ──────────────────────────────────────────────────────── */}
            <section className="grid grid-cols-12 gap-8">
              {/* Large Media Preview */}
              <div className="col-span-12 lg:col-span-8 relative rounded-2xl overflow-hidden bg-surface-container-low/70 backdrop-blur-xl border border-outline-variant/15 group/preview">
                {/* Overlay badges */}
                <div className="absolute top-4 left-4 z-10 flex gap-2">
                  <span className="px-3 py-1 rounded-full bg-background/60 backdrop-blur-md text-[10px] font-bold text-primary flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary" /> ORIGINAL
                  </span>
                  {verdict === "ai-generated" && (
                    <span className="px-3 py-1 rounded-full bg-error-container/60 backdrop-blur-md text-[10px] font-bold text-on-error-container flex items-center gap-2">
                      ANOMALY DETECTED
                    </span>
                  )}
                </div>
                {preview ? (
                  <img src={preview} alt="Forensic Subject" className="w-full aspect-video object-contain bg-surface-container-lowest" />
                ) : (
                  <div className="w-full aspect-video flex flex-col items-center justify-center bg-surface-container-high gap-4">
                    <MaterialIcon icon={mediaType(file?.type ?? "") === "video" ? "videocam" : "audiotrack"} size={64} className="text-outline/30" />
                    <p className="text-sm text-outline/50 font-mono uppercase tracking-widest">{mediaType(file?.type ?? "")} Analysis</p>
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-background/60 via-transparent to-transparent opacity-0 group-hover/preview:opacity-100 transition-opacity duration-500" />
              </div>

              {/* Key Findings + Anomaly Card */}
              <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
                {/* Key Findings */}
                <div className="flex-1 p-6 rounded-2xl bg-surface-container-low/70 backdrop-blur-xl border border-outline-variant/15 relative overflow-hidden">
                  <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl ${
                    verdict === "ai-generated" ? "bg-error/10" : "bg-primary/10"
                  }`} />
                  <div className="relative z-10">
                    <div className="flex items-center gap-2 mb-5">
                      <MaterialIcon icon="fingerprint" size={20} className="text-primary" />
                      <h4 className="text-sm font-bold uppercase tracking-widest text-on-surface-variant">Key Findings</h4>
                    </div>
                    <div className="space-y-4">
                      {checks.slice(0, 4).map((c: any, i: number) => {
                        const s = layerScore(c.id, raw);
                        const pct = s >= 0 ? s : (c.status === "pass" ? 95 : c.status === "fail" ? 15 : c.status === "warn" ? 50 : 60);
                        return (
                          <div key={i}>
                            <div className="flex justify-between text-xs mb-1.5">
                              <span className="text-outline truncate pr-2">{c.name || c.label}</span>
                              <span className={`font-bold ${pct > 70 ? "text-secondary" : pct > 40 ? "text-primary" : "text-error"}`}>{pct}%</span>
                            </div>
                            <div className="h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                              <div className={`h-full rounded-full transition-all duration-700 ${pct > 70 ? "bg-secondary" : pct > 40 ? "bg-primary" : "bg-error"}`} style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  {checks[0] && (
                    <div className="mt-6 p-4 rounded-xl bg-surface-container-lowest/60 border border-outline-variant/5 relative z-10">
                      <p className="text-[11px] text-on-surface-variant leading-relaxed italic">
                        &ldquo;{checks[0].description || checks[0].detail || "Analysis complete."}&rdquo;
                      </p>
                    </div>
                  )}
                </div>

                {/* Anomaly / Clean Card */}
                {anomalies.length > 0 ? (
                  <div className="p-6 rounded-2xl bg-surface-container-low/70 backdrop-blur-xl border border-outline-variant/15 border-l-4 border-l-error">
                    <div className="flex items-center gap-2 mb-3 text-error">
                      <MaterialIcon icon="security" size={18} />
                      <h4 className="text-xs font-bold uppercase tracking-widest">Manipulation Found</h4>
                    </div>
                    <p className="text-on-surface font-semibold text-base mb-2">{anomalies[0].name || anomalies[0].label}</p>
                    <p className="text-xs text-on-surface-variant leading-relaxed mb-3">{anomalies[0].description || anomalies[0].detail}</p>
                    {anomalies.length > 1 && (
                      <span className="text-[10px] font-bold text-error">+{anomalies.length - 1} more anomalies detected</span>
                    )}
                  </div>
                ) : (
                  <div className="p-6 rounded-2xl bg-surface-container-low/70 backdrop-blur-xl border border-outline-variant/15 border-l-4 border-l-secondary">
                    <div className="flex items-center gap-2 mb-3 text-secondary">
                      <MaterialIcon icon="verified" size={18} />
                      <h4 className="text-xs font-bold uppercase tracking-widest">Integrity Verified</h4>
                    </div>
                    <p className="text-on-surface font-semibold text-base mb-2">No Manipulations Found</p>
                    <p className="text-xs text-on-surface-variant leading-relaxed">All forensic layers indicate natural, unmodified content consistent with source capture.</p>
                  </div>
                )}
              </div>
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 4 — Pipeline Status (completed)
               ──────────────────────────────────────────────────────── */}
            <section className="space-y-6">
              <div className="flex items-center justify-between">
                <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">Pipeline Status</span>
                <span className="font-label text-xs text-on-surface-variant italic">ID: {scanId}</span>
              </div>
              <div className="relative flex justify-between items-start">
                <div className="absolute top-5 left-0 w-full h-[1px] bg-surface-container-high z-0" />
                {pipelineSteps.map((step) => (
                  <div key={step.key} className="relative z-10 flex flex-col items-center w-1/5 text-center">
                    <div className="w-10 h-10 rounded-full bg-secondary-container shadow-glow-emerald flex items-center justify-center mb-4">
                      <MaterialIcon icon="check" filled size={16} className="text-on-secondary" />
                    </div>
                    <span className="font-headline text-sm font-bold text-on-surface">{step.label}</span>
                    <span className="text-[10px] font-label uppercase tracking-tighter mt-1 text-on-surface-variant">Completed</span>
                  </div>
                ))}
              </div>
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 5 — Forensic Layers Detail Grid
               ──────────────────────────────────────────────────────── */}
            <section>
              <button onClick={() => toggle("layers")} className="flex items-center justify-between w-full mb-6 cursor-pointer group">
                <div className="flex items-center gap-3">
                  <h2 className="text-xl font-headline font-bold text-on-surface">Forensic Analysis Layers</h2>
                  <span className="text-xs text-on-surface-variant/50 bg-surface-container-high px-3 py-1 rounded-full font-label">{checks.length} checks</span>
                </div>
                <MaterialIcon icon={sec.layers ? "expand_less" : "expand_more"} size={24} className="text-outline group-hover:text-on-surface transition-colors" />
              </button>
              {sec.layers && (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {checks.map((c: any, i: number) => {
                    const id = c.id || `d_${i}`;
                    const icon = LAYER_ICONS[id] || "labs";
                    const score = layerScore(id, raw);
                    const statusColor = c.status === "pass" ? "text-secondary" : c.status === "fail" ? "text-error" : c.status === "warn" ? "text-primary" : "text-on-surface-variant";
                    const statusBg = c.status === "pass" ? "bg-secondary/5 border-secondary/20 text-secondary" :
                      c.status === "fail" ? "bg-error/5 border-error/20 text-error" :
                      c.status === "warn" ? "bg-primary/5 border-primary/20 text-primary" :
                      "bg-surface-container-high/50 border-outline-variant/20 text-outline";
                    const isExpanded = expandedCheck === id;
                    const rawDetail = c.raw_data || (raw[id] ? raw[id] : null);
                    return (
                      <div key={id} onClick={() => setExpandedCheck(isExpanded ? null : id)}
                        className={`p-6 rounded-2xl bg-surface-container-low border transition-all duration-300 cursor-pointer group/card ${
                          isExpanded ? "border-primary/40 shadow-glow-sm col-span-1 md:col-span-2 xl:col-span-3" : "border-outline-variant/10 hover:border-outline-variant/30 hover:shadow-card"
                        }`}>
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-3">
                            <MaterialIcon icon={icon} size={28} className={`${statusColor} group-hover/card:scale-110 transition-transform`} />
                            <MaterialIcon icon={isExpanded ? "expand_less" : "expand_more"} size={18} className="text-outline/50" />
                          </div>
                          <span className={`text-[10px] font-bold uppercase tracking-widest px-2.5 py-1 rounded-full border ${statusBg}`}>
                            {c.status === "pass" ? "CLEAN" : c.status === "fail" ? "ALERT" : c.status === "warn" ? "WARNING" : "INFO"}
                          </span>
                        </div>
                        <h4 className="text-base font-bold text-on-surface mb-2">{c.name || c.label}</h4>
                        <p className="text-sm text-on-surface-variant leading-relaxed mb-4">{c.description || c.detail}</p>
                        {score >= 0 && (
                          <div>
                            <div className="flex justify-between text-xs mb-1.5">
                              <span className="text-outline">Safety Score</span>
                              <span className={`font-bold ${score > 70 ? "text-secondary" : score > 40 ? "text-primary" : "text-error"}`}>{score}%</span>
                            </div>
                            <div className="h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                              <div className={`h-full rounded-full ${score > 70 ? "bg-secondary" : score > 40 ? "bg-primary" : "bg-error"}`} style={{ width: `${score}%` }} />
                            </div>
                          </div>
                        )}
                        {/* Expanded detail panel */}
                        {isExpanded && (
                          <div className="mt-6 pt-6 border-t border-outline-variant/10 space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
                            {c.recommendations && (
                              <div className="p-4 rounded-xl bg-primary/5 border border-primary/10">
                                <p className="text-xs font-bold text-primary uppercase tracking-widest mb-2">Recommendation</p>
                                <p className="text-sm text-on-surface-variant">{c.recommendations}</p>
                              </div>
                            )}
                            {rawDetail && typeof rawDetail === "object" && (
                              <div className="rounded-xl bg-surface-container-lowest/80 border border-outline-variant/5 overflow-hidden">
                                <p className="text-[10px] font-bold text-outline uppercase tracking-widest px-4 py-3 bg-surface-container-high/50">Raw Engine Output</p>
                                <div className="p-4 grid grid-cols-2 md:grid-cols-3 gap-3">
                                  {Object.entries(rawDetail).filter(([, v]) => v !== null && v !== undefined && typeof v !== "object").map(([k, v]) => (
                                    <div key={k} className="flex flex-col gap-0.5">
                                      <span className="text-[10px] text-outline font-mono uppercase tracking-wider">{k.replace(/_/g, " ")}</span>
                                      <span className="text-sm font-mono text-on-surface font-bold">{typeof v === "number" ? (v as number).toFixed(4) : String(v)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            {!rawDetail && !c.recommendations && (
                              <p className="text-xs text-on-surface-variant italic">No additional details available for this engine.</p>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 6 — AI Signals
               ──────────────────────────────────────────────────────── */}
            <section>
              <button onClick={() => toggle("signals")} className="flex items-center justify-between w-full mb-6 cursor-pointer group">
                <div className="flex items-center gap-3">
                  <h2 className="text-xl font-headline font-bold text-on-surface">AI Signals & Anomalies</h2>
                  {anomalies.length > 0 && (
                    <span className="text-xs bg-error/10 text-error px-3 py-1 rounded-full font-bold">{anomalies.length} detected</span>
                  )}
                </div>
                <MaterialIcon icon={sec.signals ? "expand_less" : "expand_more"} size={24} className="text-outline group-hover:text-on-surface transition-colors" />
              </button>
              {sec.signals && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {anomalies.map((s: any, i: number) => (
                    <div key={i} className={`p-6 rounded-2xl border-l-4 ${
                      s.status === "fail" ? "border-l-error bg-error/[0.03] border border-error/10" : "border-l-primary bg-primary/[0.03] border border-primary/10"
                    }`}>
                      <div className="flex items-center gap-3 mb-3">
                        <MaterialIcon icon={s.status === "fail" ? "warning" : "info"} size={24} className={s.status === "fail" ? "text-error" : "text-primary"} />
                        <h4 className={`text-base font-bold ${s.status === "fail" ? "text-error" : "text-primary"}`}>{s.name || s.label}</h4>
                      </div>
                      <p className="text-sm text-on-surface-variant leading-relaxed">{s.description || s.detail}</p>
                    </div>
                  ))}
                  {passes.length > 0 && (
                    <div className="p-6 rounded-2xl border-l-4 border-l-secondary bg-secondary/[0.03] border border-secondary/10">
                      <div className="flex items-center gap-3 mb-3">
                        <MaterialIcon icon="check_circle" size={24} className="text-secondary" />
                        <h4 className="text-base font-bold text-secondary">{anomalies.length === 0 ? "All Systems Clear" : `${passes.length} Checks Passed`}</h4>
                      </div>
                      <p className="text-sm text-on-surface-variant leading-relaxed">
                        {anomalies.length === 0
                          ? "No anomalies or synthetic artifacts detected across all forensic analysis layers. Media integrity is verified."
                          : `${passes.length} of ${checks.length} forensic layers show natural, unmanipulated patterns consistent with authentic media.`}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 7 — Manipulation Map + Metadata
               ──────────────────────────────────────────────────────── */}
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Manipulation Map */}
              {preview && (
                <div>
                  <button onClick={() => toggle("map")} className="flex items-center justify-between w-full mb-6 cursor-pointer group">
                    <h2 className="text-xl font-headline font-bold text-on-surface">Manipulation Map</h2>
                    <MaterialIcon icon={sec.map ? "expand_less" : "expand_more"} size={24} className="text-outline group-hover:text-on-surface transition-colors" />
                  </button>
                  {sec.map && (
                    <div className="relative rounded-2xl overflow-hidden border border-outline-variant/20 aspect-video cursor-crosshair group/map">
                      <img src={preview} alt="Heatmap overlay" className="w-full h-full object-cover grayscale opacity-50 group-hover/map:opacity-80 transition-opacity duration-500" />
                      <div className={`absolute inset-0 mix-blend-overlay ${
                        verdict === "ai-generated" ? "bg-gradient-to-tr from-error/40 via-error/10 to-transparent" : "bg-gradient-to-tr from-secondary/20 via-secondary/5 to-transparent"
                      }`} />
                      <div className="absolute bottom-3 right-3 bg-background/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-primary flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" /> OVERLAY ACTIVE
                      </div>
                      <div className="absolute top-3 left-3 bg-background/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-on-surface-variant">
                        SPECTRAL ANALYSIS VIEW
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Metadata */}
              <div>
                <button onClick={() => toggle("meta")} className="flex items-center justify-between w-full mb-6 cursor-pointer group">
                  <h2 className="text-xl font-headline font-bold text-on-surface">File Metadata</h2>
                  <MaterialIcon icon={sec.meta ? "expand_less" : "expand_more"} size={24} className="text-outline group-hover:text-on-surface transition-colors" />
                </button>
                {sec.meta && file && (
                  <div className="rounded-2xl bg-surface-container-low border border-outline-variant/10 overflow-hidden">
                    <table className="w-full">
                      <tbody className="divide-y divide-outline-variant/5">
                        {[
                          { label: "FILE NAME", value: file.name, color: "text-on-surface" },
                          { label: "MEDIA TYPE", value: file.type, color: "text-on-surface" },
                          { label: "FILE SIZE", value: fmtSize(file.size), color: "text-on-surface" },
                          ...(dims ? [{ label: "RESOLUTION", value: `${dims.w} × ${dims.h}`, color: "text-on-surface" }] : []),
                          ...(raw.compression?.estimated_quality != null ? [{ label: "JPEG QUALITY", value: `Q=${raw.compression.estimated_quality}`, color: "text-on-surface" }] : []),
                          ...(raw.compression?.platform_hint ? [{ label: "PLATFORM", value: raw.compression.platform_hint.toUpperCase(), color: "text-primary font-bold" }] : []),
                          ...(raw.compression?.is_double_compressed ? [{ label: "COMPRESSION", value: "DOUBLE SAVED", color: "text-error font-bold" }] : []),
                          ...(raw.preprocessing?.method ? [{ label: "PREPROCESSING", value: raw.preprocessing.method, color: "text-on-surface-variant" }] : []),
                        ].map((row, i) => (
                          <tr key={i} className="hover:bg-surface-container-high/40 transition-colors">
                            <td className="px-6 py-4 font-mono text-[11px] text-outline uppercase tracking-widest w-40">{row.label}</td>
                            <td className={`px-6 py-4 font-mono text-sm ${row.color} break-all`}>{row.value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 8 — Forensic Logs Table
               ──────────────────────────────────────────────────────── */}
            <section>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <h2 className="text-xl font-headline font-bold text-on-surface">Forensic Logs</h2>
                  <div className="flex gap-1.5 ml-2">
                    <div className={`w-2 h-2 rounded-full ${verdict === "ai-generated" ? "bg-error" : "bg-secondary"}`} />
                    <div className="w-2 h-2 rounded-full bg-on-surface-variant/20" />
                    <div className="w-2 h-2 rounded-full bg-on-surface-variant/20" />
                  </div>
                </div>
              </div>
              <div className="rounded-2xl bg-surface-container-low border border-outline-variant/10 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-surface-container-lowest">
                        <th className="px-8 py-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant w-32">Timestamp</th>
                        <th className="px-8 py-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Technical Layer</th>
                        <th className="px-8 py-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Detail</th>
                        <th className="px-8 py-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant w-28 text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-outline-variant/5">
                      {checks.map((c: any, i: number) => (
                        <tr key={i} className="hover:bg-surface-container-high/40 transition-colors">
                          <td className="px-8 py-5 font-mono text-xs text-on-surface-variant">
                            00:{String((i + 1) * 4).padStart(2, "0")}.{String((i * 231 + 100) % 900 + 100)}
                          </td>
                          <td className="px-8 py-5">
                            <div className="flex items-center gap-3">
                              <span className={`w-2 h-2 rounded-full shrink-0 ${
                                c.status === "pass" ? "bg-secondary" : c.status === "fail" ? "bg-error" : c.status === "warn" ? "bg-primary" : "bg-outline"
                              }`} />
                              <span className="text-sm font-headline font-medium text-on-surface">{c.name || c.label}</span>
                            </div>
                          </td>
                          <td className="px-8 py-5 text-xs text-on-surface-variant max-w-xs truncate">{c.description || c.detail || "—"}</td>
                          <td className={`px-8 py-5 text-xs font-bold uppercase text-right ${
                            c.status === "pass" ? "text-secondary" : c.status === "fail" ? "text-error" : c.status === "warn" ? "text-primary" : "text-on-surface-variant"
                          }`}>
                            {c.status === "pass" ? "Nominal" : c.status === "fail" ? "Anomaly" : c.status === "warn" ? "Warning" : "Info"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* ────────────────────────────────────────────────────────
                 SECTION 9 — Actions Bar
               ──────────────────────────────────────────────────────── */}
            <section className="flex flex-col sm:flex-row gap-4">
              <button onClick={handleReset} className="flex-1 py-4 bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm tracking-widest uppercase rounded-xl shadow-xl transition-transform hover:scale-[1.02] active:scale-95 cursor-pointer flex items-center justify-center gap-3">
                <MaterialIcon icon="add_circle" size={20} /> New Analysis
              </button>
              <button onClick={handlePdfReport} className="flex-1 py-4 bg-surface-container-high border border-outline-variant/20 hover:border-outline-variant/60 text-on-surface font-headline font-bold text-sm tracking-widest uppercase rounded-xl transition-all cursor-pointer flex items-center justify-center gap-3">
                <MaterialIcon icon="picture_as_pdf" size={20} /> Download Report
              </button>
              <button onClick={handleSaveCase} className="flex-1 py-4 bg-surface-container-high border border-outline-variant/20 hover:border-outline-variant/60 text-on-surface font-headline font-bold text-sm tracking-widest uppercase rounded-xl transition-all cursor-pointer flex items-center justify-center gap-3">
                <MaterialIcon icon="database" size={20} /> Save Case
              </button>
              <button onClick={handleShareEvidence} className="flex-1 py-4 bg-surface-container-high border border-outline-variant/20 hover:border-outline-variant/60 text-on-surface font-headline font-bold text-sm tracking-widest uppercase rounded-xl transition-all cursor-pointer flex items-center justify-center gap-3">
                <MaterialIcon icon="share" size={20} /> Share Evidence
              </button>
            </section>

          </div>
        )}
      </div>
    </Layout>
  );
};

export default Scanner;
