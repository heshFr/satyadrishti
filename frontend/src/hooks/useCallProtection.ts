import { useState, useRef, useCallback, useEffect } from "react";

export interface ModalityResult {
  status: "idle" | "safe" | "warning" | "danger";
  confidence: number;
  verdict: string;
  details?: Record<string, unknown>;
}

export interface CallSummary {
  deepfakeDetections: number;
  coercionDetections: number;
  peakThreatLevel: string;
  finalThreatScore: number;
  recommendation: string;
}

export interface TranscriptLine {
  time: string;
  text: string;
  flagged: boolean;
  verdict?: string;
}

export interface CallProtectionState {
  isActive: boolean;
  isConnected: boolean;
  isMicOn: boolean;
  isSystemAudio: boolean;
  isScreenShare: boolean;
  callState: "idle" | "safe" | "warning" | "danger" | "critical";
  audio: ModalityResult;
  text: ModalityResult;
  transcript: TranscriptLine[];
  audioLevel: number;
  frequencyData: Uint8Array | null;
  waveformData: Uint8Array | null;
  error: string | null;
  threatEscalation: number;
  alertLevel: "safe" | "warning" | "danger" | "critical";
  alertMessage: string | null;
  callSummary: CallSummary | null;
  language: string | null;
  transcriptText: string;
  transcriptLanguage: string;
  speakerMatch: string | null;
  speakerVerified: boolean;
  speakerSimilarity: number;
  biologicalVeto: boolean;
  vetoReason: string | null;
  guardrails: { biological_veto: boolean; confidence_threshold_check: boolean; human_review_required: boolean } | null;
  edgeCaseHandling: {
    low_signal: string;
    noise_detected: boolean;
    confidence_adjusted: boolean;
  } | null;
  decisionPayload: {
    decision: string;
    confidence: number;
    guardrail_triggered: boolean;
    guardrail_type: string;
    explanation: {
      primary_reason: string;
      supporting_layers: string[];
    };
    audit: any;
  } | null;
}

const DEFAULT_MODALITY: ModalityResult = {
  status: "idle",
  confidence: 0,
  verdict: "Standing By",
};

const INITIAL_STATE: CallProtectionState = {
  isActive: false,
  isConnected: false,
  isMicOn: false,
  isSystemAudio: false,
  isScreenShare: false,
  callState: "idle",
  audio: { ...DEFAULT_MODALITY },
  text: { ...DEFAULT_MODALITY },
  transcript: [],
  audioLevel: 0,
  frequencyData: null,
  waveformData: null,
  error: null,
  threatEscalation: 0,
  alertLevel: "safe",
  alertMessage: null,
  callSummary: null,
  language: null,
  transcriptText: "",
  transcriptLanguage: "",
  speakerMatch: null,
  speakerVerified: false,
  speakerSimilarity: 0,
  biologicalVeto: false,
  vetoReason: null,
  guardrails: null,
  edgeCaseHandling: null,
  decisionPayload: null,
};

import { WS_BASE } from "@/lib/api";

/**
 * Encode raw PCM float32 samples into a WAV file as base64.
 * Produces a format that the backend's soundfile.read() can decode natively.
 */
function encodeWAV(samples: Float32Array, sampleRate: number): string {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);   // PCM format
  view.setUint16(22, 1, true);   // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);   // block align
  view.setUint16(34, 16, true);  // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Real-time call protection hook.
 * Captures system audio + microphone via AudioWorklet, sends WAV chunks to backend,
 * provides real frequency/waveform data for visualization.
 */
export function useCallProtection() {
  const [state, setState] = useState<CallProtectionState>({ ...INITIAL_STATE });

  const wsRef = useRef<WebSocket | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const systemStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const levelAnimRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const screenVideoRef = useRef<MediaStreamTrack | null>(null);
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Guard to prevent concurrent startProtection calls
  const startingRef = useRef(false);

  const cleanup = useCallback(() => {
    // Reset the starting guard
    startingRef.current = false;

    if (levelAnimRef.current) {
      cancelAnimationFrame(levelAnimRef.current);
      levelAnimRef.current = null;
    }
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (systemStreamRef.current) {
      systemStreamRef.current.getTracks().forEach((t) => t.stop());
      systemStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    audioBufferRef.current = [];
    screenVideoRef.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  // Real-time frequency/waveform data pump (requestAnimationFrame)
  const startVisualization = useCallback((analyser: AnalyserNode) => {
    const freqData = new Uint8Array(analyser.frequencyBinCount);
    const waveData = new Uint8Array(analyser.frequencyBinCount);

    const pump = () => {
      analyser.getByteFrequencyData(freqData);
      analyser.getByteTimeDomainData(waveData);

      // Compute RMS level
      let sum = 0;
      for (let i = 0; i < waveData.length; i++) {
        const v = (waveData[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / waveData.length);

      setState((prev) => ({
        ...prev,
        audioLevel: Math.min(1, rms * 3),
        frequencyData: new Uint8Array(freqData),
        waveformData: new Uint8Array(waveData),
      }));

      levelAnimRef.current = requestAnimationFrame(pump);
    };

    pump();
  }, []);

  // Handle WebSocket messages
  const handleWSMessage = useCallback((data: Record<string, unknown>) => {
    setState((prev) => {
      const next = { ...prev };

      if (data.type === "call_started") {
        return next;
      }

      if (data.type === "analysis_result") {
        // Update threat escalation from server
        if (typeof data.threat_escalation === "number") {
          next.threatEscalation = data.threat_escalation;
        }

        if (data.modality === "audio") {
          const confidence = (data.confidence as number) ?? 0;
          const isSynthetic = (data.is_synthetic as boolean) ?? false;
          const biologicalVeto = (data.biological_veto as boolean) ?? false;
          const vetoReason = (data.veto_reason as string) ?? null;

          // If biological veto triggered, force critical state
          if (biologicalVeto) {
            next.biologicalVeto = true;
            next.vetoReason = vetoReason;
            next.audio = {
              status: "danger",
              confidence: Math.max(confidence, 95),
              verdict: vetoReason || "Biological Impossibility Detected",
            };
            next.callState = "critical";

            // Add veto to transcript
            next.transcript = [
              ...prev.transcript,
              {
                time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
                text: `🚨 BIOLOGICAL VETO: ${vetoReason || "AI voice clone detected — physiological impossibility"}`,
                flagged: true,
                verdict: "biological_veto",
              },
            ].slice(-20);
          } else {
            next.audio = {
              status: isSynthetic ? "danger" : (data.status_text === "Partially Consistent" ? "warning" : "safe"),
              confidence,
              verdict: data.status_text === "Partially Consistent" 
                ? "Partially Consistent"
                : (isSynthetic
                  ? `Synthetic Voice (${confidence.toFixed(1)}%)`
                  : `Verified (${confidence.toFixed(1)}%)`),
            };

            if (isSynthetic && confidence > 60) {
              next.callState = "danger";
            }
          }

          // Update guardrails if provided
          if (data.guardrails) {
            next.guardrails = data.guardrails as any;
          }

          if (data.edge_case_handling) {
            next.edgeCaseHandling = data.edge_case_handling as any;
          }

          if (data.decision_payload) {
            next.decisionPayload = data.decision_payload as any;
          }
        }

        if (data.modality === "text") {
          const confidence = (data.confidence as number) ?? 0;
          const verdict = (data.verdict as string) ?? "safe";
          const patterns = (data.detected_patterns as string[]) ?? [];
          const language = (data.language as string) ?? null;
          const isThreat = verdict !== "safe";

          next.text = {
            status: isThreat ? (confidence > 70 ? "danger" : "warning") : "safe",
            confidence,
            verdict: verdict === "safe"
              ? "Normal"
              : verdict.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
            details: { patterns },
          };

          next.language = language;

          // Add to transcript if threat detected
          if (isThreat) {
            const now = new Date();
            next.transcript = [
              ...prev.transcript,
              {
                time: now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
                text: `Coercion detected: ${verdict.replace(/_/g, " ")}`,
                flagged: true,
                verdict,
              },
            ].slice(-20);
          }
        }

        if (data.modality === "video") {
          const verdict = data.verdict as string;
          const confidence = (data.confidence as number) ?? 0;

          if (verdict === "ai-generated" && confidence > 70) {
            next.callState = "danger";
            next.transcript = [
              ...prev.transcript,
              {
                time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
                text: `Video deepfake detected (${confidence.toFixed(0)}% confidence)`,
                flagged: true,
                verdict: "deepfake_video",
              },
            ].slice(-20);
          }
        }
      }

      if (data.type === "transcript") {
        next.transcriptText = data.text as string;
        next.transcriptLanguage = data.language as string;

        // Add to transcript history
        next.transcript = [
          ...prev.transcript,
          {
            time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
            text: data.text as string,
            flagged: false,
          },
        ].slice(-50);
      }

      if (data.type === "speaker_verification") {
        next.speakerMatch = (data.best_match as string) || null;
        next.speakerVerified = (data.is_verified as boolean) || false;
        next.speakerSimilarity = (data.similarity as number) || 0;
      }

      if (data.type === "threat_alert") {
        const level = data.level as "warning" | "danger" | "critical";
        const message = (data.message as string) ?? null;

        next.alertLevel = level;
        next.alertMessage = message;
        next.callState = level === "critical" ? "critical" : level === "danger" ? "danger" : "warning";

        if (typeof data.threat_score === "number") {
          next.threatEscalation = data.threat_score;
        }

        // Add alert to transcript
        next.transcript = [
          ...prev.transcript,
          {
            time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
            text: message ?? `Threat alert: ${level}`,
            flagged: true,
            verdict: level,
          },
        ].slice(-20);
      }

      if (data.type === "pong") {
        return next;
      }

      return next;
    });
  }, []);

  // Handle call summary (separate to avoid closure issues)
  const handleCallSummary = useCallback((data: Record<string, unknown>) => {
    const summary: CallSummary = {
      deepfakeDetections: (data.deepfake_detections as number) ?? 0,
      coercionDetections: (data.coercion_detections as number) ?? 0,
      peakThreatLevel: (data.peak_threat_level as string) ?? "safe",
      finalThreatScore: (data.final_threat_score as number) ?? 0,
      recommendation: (data.recommendation as string) ?? "",
    };
    setState((prev) => ({ ...prev, callSummary: summary }));
  }, []);

  // Enable screen share (system audio + video deepfake detection).
  // Called separately so the browser popup doesn't confuse users.
  const enableScreenShare = useCallback(async () => {
    try {
      const systemStream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: true,
      });
      systemStreamRef.current = systemStream;

      const audioTracks = systemStream.getAudioTracks();
      const hasSystemAudio = audioTracks.length > 0;

      const videoTracks = systemStream.getVideoTracks();
      const hasScreenVideo = videoTracks.length > 0;

      // Merge system audio into existing audio context
      if (hasSystemAudio && audioContextRef.current && analyserRef.current) {
        const systemSource = audioContextRef.current.createMediaStreamSource(systemStream);
        systemSource.connect(analyserRef.current);
      }

      if (hasScreenVideo) {
        screenVideoRef.current = videoTracks[0];
        videoTracks[0].onended = () => {
          setState((prev) => ({ ...prev, isScreenShare: false, isSystemAudio: false }));
          screenVideoRef.current = null;
          if (systemStreamRef.current) {
            systemStreamRef.current.getTracks().forEach((t) => t.stop());
            systemStreamRef.current = null;
          }
        };

        // Start sending screen frames for video deepfake detection
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");

          frameIntervalRef.current = setInterval(async () => {
            if (!screenVideoRef.current || !ctx) return;
            if (!ws || ws.readyState !== WebSocket.OPEN) return;

            try {
              const track = screenVideoRef.current;
              if (typeof (window as any).ImageCapture !== "undefined") {
                const capture = new (window as any).ImageCapture(track);
                const bitmap = await capture.grabFrame() as ImageBitmap;
                canvas.width = Math.min(bitmap.width, 640);
                canvas.height = Math.round(bitmap.height * (canvas.width / bitmap.width));
                ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                bitmap.close();

                canvas.toBlob((blob) => {
                  if (blob && ws.readyState === WebSocket.OPEN) {
                    const reader = new FileReader();
                    reader.onload = () => {
                      const base64 = (reader.result as string).split(",")[1];
                      ws.send(JSON.stringify({ type: "video_frame", data: base64 }));
                    };
                    reader.readAsDataURL(blob);
                  }
                }, "image/jpeg", 0.7);
              }
            } catch {
              // Frame capture failed — skip
            }
          }, 5000);
        }
      }

      setState((prev) => ({
        ...prev,
        isSystemAudio: hasSystemAudio,
        isScreenShare: hasScreenVideo,
      }));
    } catch {
      // User cancelled — no-op, mic-only protection continues
    }
  }, []);

  const startProtection = useCallback(async () => {
    // ── CRITICAL: Re-entry guard ──
    // Prevents stacked getDisplayMedia sessions when clicking multiple times
    if (startingRef.current || state.isActive) {
      console.warn("[CallProtection] Already starting or active — ignoring duplicate click");
      return;
    }
    startingRef.current = true;

    // Clean up any previous session before starting fresh
    cleanup();

    setState((prev) => ({ ...prev, error: null, callSummary: null, biologicalVeto: false, vetoReason: null }));

    try {
      // ── Step 1: Capture microphone ──
      let micStream: MediaStream;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000,
          },
        });
        micStreamRef.current = micStream;
      } catch {
        throw new Error("Microphone access denied. Please allow microphone access to use call protection.");
      }

      // ── Step 2: Capture system audio via screen share ──
      // Required to listen to the other caller's voice. If the user
      // cancels this dialog the entire flow aborts cleanly.
      let systemStream: MediaStream;
      let hasSystemAudio = false;
      let hasScreenVideo = false;

      try {
        systemStream = await navigator.mediaDevices.getDisplayMedia({
          audio: true,
          video: true,
        });
        systemStreamRef.current = systemStream;

        const audioTracks = systemStream.getAudioTracks();
        hasSystemAudio = audioTracks.length > 0;

        const videoTracks = systemStream.getVideoTracks();
        hasScreenVideo = videoTracks.length > 0;
        if (hasScreenVideo) {
          screenVideoRef.current = videoTracks[0];
          videoTracks[0].onended = () => {
            setState((prev) => ({ ...prev, isScreenShare: false }));
            screenVideoRef.current = null;
          };
        }
      } catch {
        // User cancelled screen share — abort: stop mic and do NOT start protection
        micStream.getTracks().forEach((t) => t.stop());
        micStreamRef.current = null;
        startingRef.current = false;
        return;
      }

      // ── Step 3: Set up audio processing ──
      const sampleRate = 16000;
      const audioCtx = new AudioContext({ sampleRate });
      audioContextRef.current = audioCtx;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const micSource = audioCtx.createMediaStreamSource(micStream);

      // Merge mic + system audio if available
      if (hasSystemAudio && systemStream) {
        const systemSource = audioCtx.createMediaStreamSource(systemStream);
        const merger = audioCtx.createChannelMerger(2);
        micSource.connect(merger, 0, 0);
        systemSource.connect(merger, 0, 1);
        merger.connect(analyser);
      } else {
        micSource.connect(analyser);
      }

      // ── Step 4: Record audio via AudioWorklet (replaces deprecated ScriptProcessorNode) ──
      try {
        await audioCtx.audioWorklet.addModule("/pcm-processor.js");
        const workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");
        workletNodeRef.current = workletNode;

        workletNode.port.onmessage = (e: MessageEvent) => {
          if (e.data?.type === "pcm" && e.data.samples) {
            audioBufferRef.current.push(new Float32Array(e.data.samples));
          }
        };

        analyser.connect(workletNode);
        workletNode.connect(audioCtx.destination);
      } catch (workletErr) {
        // Fallback to ScriptProcessorNode if AudioWorklet fails
        console.warn("[CallProtection] AudioWorklet failed, falling back to ScriptProcessorNode:", workletErr);
        const bufferSize = 4096;
        const recorder = audioCtx.createScriptProcessor(bufferSize, 1, 1);

        recorder.onaudioprocess = (e: AudioProcessingEvent) => {
          const channelData = e.inputBuffer.getChannelData(0);
          audioBufferRef.current.push(new Float32Array(channelData));
        };

        analyser.connect(recorder);
        recorder.connect(audioCtx.destination);
      }

      // ── Step 5: Start visualization ──
      startVisualization(analyser);

      // ── Step 6: Connect WebSocket ──
      const ws = new WebSocket(`${WS_BASE}/ws/live`);
      wsRef.current = ws;

      ws.onopen = () => {
        // Release the starting guard now that we're fully connected
        startingRef.current = false;

        setState((prev) => ({
          ...prev,
          isActive: true,
          isConnected: true,
          isMicOn: true,
          isSystemAudio: hasSystemAudio,
          isScreenShare: hasScreenVideo,
          callState: "safe",
          audio: { status: "safe", confidence: 0, verdict: "Monitoring" },
          text: { status: "safe", confidence: 0, verdict: "Monitoring" },
          threatEscalation: 0,
          alertLevel: "safe",
          alertMessage: null,
          callSummary: null,
          language: null,
          biologicalVeto: false,
          vetoReason: null,
          guardrails: null,
          edgeCaseHandling: null,
        }));

        ws.send(JSON.stringify({ type: "call_start" }));

        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "call_summary") {
            handleCallSummary(data);
          } else {
            handleWSMessage(data);
          }
        } catch {
          // Ignore parse errors
        }
      };

      ws.onerror = () => {
        setState((prev) => ({
          ...prev,
          error: "Connection error. Make sure the backend server is running.",
          isConnected: false,
        }));
      };

      ws.onclose = () => {
        setState((prev) => ({ ...prev, isConnected: false }));
      };

      // ── Step 7: Send audio chunks every 5 seconds as WAV ──
      // (Increased from 3s to 5s to give biological analyzers enough data)
      sendIntervalRef.current = setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN) return;
        if (audioBufferRef.current.length === 0) return;

        // Concatenate all buffered samples
        const totalLength = audioBufferRef.current.reduce((sum, buf) => sum + buf.length, 0);
        const combined = new Float32Array(totalLength);
        let offset = 0;
        for (const buf of audioBufferRef.current) {
          combined.set(buf, offset);
          offset += buf.length;
        }
        audioBufferRef.current = [];

        // Skip silence (lowered threshold to catch faint tab audio)
        const rms = Math.sqrt(combined.reduce((sum, v) => sum + v * v, 0) / combined.length);
        if (rms < 0.005) return;

        // Encode as WAV and send
        const wavBase64 = encodeWAV(combined, sampleRate);
        ws.send(JSON.stringify({ type: "audio", data: wavBase64 }));
      }, 5000);

      // ── Step 8: Send screen frames for video deepfake detection (every 5s) ──
      if (hasScreenVideo && screenVideoRef.current) {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        frameIntervalRef.current = setInterval(async () => {
          if (!screenVideoRef.current || !ctx) return;
          if (ws.readyState !== WebSocket.OPEN) return;

          try {
            const track = screenVideoRef.current;
            if (typeof (window as any).ImageCapture !== "undefined") {
              // @ts-ignore — ImageCapture API (Chrome only)
              const capture = new (window as any).ImageCapture(track);
              const bitmap = await capture.grabFrame() as ImageBitmap;
              canvas.width = Math.min(bitmap.width, 640);
              canvas.height = Math.round(bitmap.height * (canvas.width / bitmap.width));
              ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
              bitmap.close();

              canvas.toBlob((blob) => {
                if (blob && ws.readyState === WebSocket.OPEN) {
                  const reader = new FileReader();
                  reader.onload = () => {
                    const base64 = (reader.result as string).split(",")[1];
                    ws.send(JSON.stringify({ type: "video_frame", data: base64 }));
                  };
                  reader.readAsDataURL(blob);
                }
              }, "image/jpeg", 0.7);
            }
          } catch {
            // Frame capture failed — skip
          }
        }, 5000);
      }

    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start call protection";
      setState((prev) => ({ ...prev, error: message }));
      cleanup();
    }
  }, [cleanup, startVisualization, handleWSMessage, handleCallSummary, state.isActive]);

  const stopProtection = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "call_end" }));

      // Give server a brief window to send back call_summary
      const summaryTimeout = setTimeout(() => {
        cleanup();
        setState((prev) => ({
          ...prev,
          isActive: false,
          isConnected: false,
          isMicOn: false,
          isSystemAudio: false,
          isScreenShare: false,
          callState: "idle",
          audio: { ...DEFAULT_MODALITY },
          text: { ...DEFAULT_MODALITY },
          audioLevel: 0,
          frequencyData: null,
          waveformData: null,
          error: null,
          biologicalVeto: false,
          vetoReason: null,
          guardrails: null,
          edgeCaseHandling: null,
        }));
      }, 1500);

      const origOnMessage = ws.onmessage;
      ws.onmessage = (event) => {
        if (origOnMessage) origOnMessage.call(ws, event);
        try {
          const data = JSON.parse(event.data);
          if (data.type === "call_summary") {
            clearTimeout(summaryTimeout);
            cleanup();
            setState((prev) => ({
              ...prev,
              isActive: false,
              isConnected: false,
              isMicOn: false,
              isSystemAudio: false,
              isScreenShare: false,
              callState: "idle",
              audio: { ...DEFAULT_MODALITY },
              text: { ...DEFAULT_MODALITY },
              audioLevel: 0,
              frequencyData: null,
              waveformData: null,
              error: null,
              biologicalVeto: false,
              vetoReason: null,
              guardrails: null,
              edgeCaseHandling: null,
            }));
          }
        } catch {
          // Ignore
        }
      };
    } else {
      cleanup();
      setState({ ...INITIAL_STATE });
    }
  }, [cleanup]);

  const analyzeText = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && text.trim()) {
      wsRef.current.send(JSON.stringify({ type: "text", data: text }));
    }
  }, []);

  const dismissDanger = useCallback(() => {
    setState((prev) => ({ ...prev, callState: "safe", alertLevel: "safe", alertMessage: null, threatEscalation: 0, biologicalVeto: false, vetoReason: null }));
  }, []);

  const dismissAlert = useCallback(() => {
    setState((prev) => ({ ...prev, alertLevel: "safe", alertMessage: null, callState: "safe" }));
  }, []);

  return {
    ...state,
    startProtection,
    stopProtection,
    enableScreenShare,
    analyzeText,
    dismissDanger,
    dismissAlert,
    systemStreamRef,
  };
}
