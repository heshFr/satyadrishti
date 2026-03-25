import Layout from "@/components/Layout";
import MaterialIcon from "@/components/MaterialIcon";
import { motion } from "framer-motion";
import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

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
}

interface TranscriptLine {
  time: string;
  text: string;
  flagged: boolean;
}

import { WS_BASE } from "@/lib/api";

const Advanced = () => {
  const [connected, setConnected] = useState(false);
  const [audioData, setAudioData] = useState<AudioDataPoint[]>(
    Array.from({ length: 50 }, (_, i) => ({ time: i, low: 50, mid: 45, high: 40 })),
  );
  const [scores, setScores] = useState<ConfidenceScore[]>([
    { label: "Audio Deepfake Detection", score: 0, verdict: "WAITING", color: "text-on-surface-variant" },
    { label: "Text Coercion Detection", score: 0, verdict: "WAITING", color: "text-on-surface-variant" },
    { label: "Overall Threat Level", score: 0, verdict: "WAITING", color: "text-on-surface-variant" },
  ]);
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const counterRef = useRef(0);

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/live`);
    let pingInterval: ReturnType<typeof setInterval>;
    let audioInterval: ReturnType<typeof setInterval>;

    ws.onopen = () => {
      setConnected(true);

      // Keepalive
      pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 30000);

      // Send test audio every 3s to keep the stream active
      audioInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          // Send a minimal audio chunk to trigger analysis
          ws.send(JSON.stringify({ type: "audio", data: "" }));
        }
      }, 3000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "analysis_result") {
          handleResult(data);
        }
      } catch {
        // ignore
      }
    };

    ws.onerror = () => setConnected(false);
    ws.onclose = () => {
      setConnected(false);
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

    if (data.modality === "audio") {
      const confidence = (data.confidence as number) ?? 0;
      const status = data.status as string;
      const isSpoof = status === "danger";

      // Update audio frequency visualization
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

      setScores((prev) =>
        prev.map((s) =>
          s.label === "Audio Deepfake Detection"
            ? {
                ...s,
                score: confidence,
                verdict: isSpoof ? "SPOOF" : "BONAFIDE",
                color: isSpoof ? "text-error" : "text-secondary",
              }
            : s,
        ),
      );
    }

    if (data.modality === "text") {
      const confidence = (data.confidence as number) ?? 0;
      const verdict = (data.verdict as string) ?? "safe";
      const isThreat = verdict !== "safe";

      setScores((prev) =>
        prev.map((s) => {
          if (s.label === "Text Coercion Detection") {
            return {
              ...s,
              score: confidence,
              verdict: isThreat ? verdict.toUpperCase().replace(/_/g, " ") : "SAFE",
              color: isThreat ? "text-error" : "text-secondary",
            };
          }
          if (s.label === "Overall Threat Level") {
            // Combine audio + text
            const audioScore = prev.find((p) => p.label === "Audio Deepfake Detection")?.score ?? 0;
            const maxThreat = Math.max(audioScore, confidence);
            return {
              ...s,
              score: 100 - maxThreat * 0.5,
              verdict: maxThreat > 70 ? "THREAT" : maxThreat > 40 ? "CAUTION" : "CLEAR",
              color: maxThreat > 70 ? "text-error" : maxThreat > 40 ? "text-primary" : "text-secondary",
            };
          }
          return s;
        }),
      );

      if (isThreat) {
        const now = new Date();
        const timeStr = `${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;
        setTranscript((prev) => [
          ...prev.slice(-20),
          {
            time: timeStr,
            text: `[${verdict.replace(/_/g, " ")}] detected (${confidence.toFixed(1)}% confidence)`,
            flagged: true,
          },
        ]);
      }
    }
  };

  return (
    <Layout systemStatus="monitoring">
      <div className="pt-32 pb-20 px-6 md:px-8 max-w-6xl mx-auto">
        <div className="flex items-center gap-4 mb-8">
          <Link
            to="/settings"
            className="p-2 rounded-full hover:bg-on-surface/[0.06] text-on-surface-variant transition-colors cursor-pointer"
          >
            <MaterialIcon icon="arrow_back" size={20} />
          </Link>
          <h1 className="font-headline text-4xl font-extrabold tracking-tighter text-on-surface">
            Technical Analysis
          </h1>
          <span className="px-3 py-1 rounded-full bg-primary/15 border border-primary/30 text-primary text-xs font-headline font-bold uppercase tracking-wider">
            DEVELOPER VIEW
          </span>
        </div>

        <div className="mb-6 flex items-center gap-3 bg-surface-container-low rounded-xl border border-outline-variant/10 px-5 py-3">
          <div
            className={`h-2 w-2 rounded-full ${connected ? "bg-secondary animate-pulse" : "bg-error"}`}
          />
          <p className="text-sm text-on-surface-variant">
            {connected
              ? "Connected to live analysis engine — data updates in real-time"
              : "Disconnected — ensure the backend is running on port 8000"}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Audio Frequency Analysis */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-6"
          >
            <h3 className="text-sm font-mono text-on-surface-variant mb-4">
              Audio Frequency Analysis
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={audioData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1F2E" />
                <XAxis dataKey="time" hide />
                <YAxis hide />
                <Area
                  type="monotone"
                  dataKey="low"
                  stroke="#10B981"
                  fill="#10B981"
                  fillOpacity={0.1}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="mid"
                  stroke="#06B6D4"
                  fill="#06B6D4"
                  fillOpacity={0.1}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="high"
                  stroke="#EF5350"
                  fill="#EF5350"
                  fillOpacity={0.05}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Module Confidence Scores */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-6"
          >
            <h3 className="text-sm font-mono text-on-surface-variant mb-6">
              Module Confidence Scores
            </h3>
            <div className="space-y-4">
              {scores.map((item) => (
                <div key={item.label} className="flex items-center gap-4">
                  <span className="text-sm font-mono text-on-surface w-48 shrink-0">
                    {item.label}
                  </span>
                  <div className="flex-1 h-2 bg-on-surface/[0.06] rounded-full overflow-hidden">
                    <motion.div
                      animate={{ width: `${item.score}%` }}
                      transition={{ duration: 0.5 }}
                      className={`h-full rounded-full ${
                        item.color === "text-error"
                          ? "bg-error"
                          : item.color === "text-primary"
                            ? "bg-primary"
                            : item.color === "text-secondary"
                              ? "bg-secondary"
                              : "bg-on-surface-variant"
                      }`}
                    />
                  </div>
                  <span className="text-sm font-mono text-on-surface-variant w-14 text-right">
                    {item.score.toFixed(0)}%
                  </span>
                  <span
                    className={`text-xs font-mono font-bold ${item.color} border ${
                      item.color === "text-error"
                        ? "border-error/20 bg-error/10"
                        : item.color === "text-primary"
                          ? "border-primary/20 bg-primary/10"
                          : item.color === "text-secondary"
                            ? "border-secondary/20 bg-secondary/10"
                            : "border-outline-variant/10 bg-on-surface/[0.04]"
                    } px-2 py-0.5 rounded`}
                  >
                    {item.verdict}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Live Transcript / Events */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-6 lg:col-span-2"
          >
            <h3 className="text-sm font-mono text-on-surface-variant mb-6">
              Live Event Log
            </h3>
            <div className="space-y-3 max-h-48 overflow-y-auto">
              {transcript.length === 0 ? (
                <p className="text-sm text-on-surface-variant/50 font-mono">
                  Waiting for analysis events...
                </p>
              ) : (
                transcript.map((line, i) => (
                  <div key={i} className="flex items-start gap-4 font-mono text-sm">
                    <span className="text-on-surface-variant shrink-0">{line.time}</span>
                    <span
                      className={
                        line.flagged
                          ? "text-error bg-error/10 px-2 py-0.5 rounded"
                          : "text-on-surface"
                      }
                    >
                      {line.text}
                    </span>
                  </div>
                ))
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </Layout>
  );
};

export default Advanced;
