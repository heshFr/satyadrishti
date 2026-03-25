import { Mic, Video, MessageCircle } from "lucide-react";
import { motion } from "framer-motion";
import type { CallState } from "./CallStatusCard";
import type { ModalityResult } from "@/hooks/useCallProtection";

interface CompanionCardsProps {
  callState: CallState;
  audioResult?: ModalityResult;
  textResult?: ModalityResult;
}

interface StatusConfig {
  status: string;
  color: string;
  dot: string;
  glow: string;
}

// Map expanded call states to the 3 base states for fallback lookup
const toBaseState = (s: CallState): "idle" | "safe" | "danger" =>
  s === "warning" || s === "danger" || s === "critical" ? "danger" : s === "safe" ? "safe" : "idle";

const CompanionCards = ({ callState, audioResult, textResult }: CompanionCardsProps) => {
  const baseState = toBaseState(callState);

  const getConfig = (
    result: ModalityResult | undefined,
    fallback: { idle: StatusConfig; safe: StatusConfig; danger: StatusConfig },
  ): StatusConfig => {
    if (result && result.status !== "idle") {
      const statusMap: Record<string, StatusConfig> = {
        safe: { status: result.verdict || "Verified", color: "text-neon-green", dot: "bg-neon-green", glow: "shadow-[0_0_8px_rgba(16,185,129,0.4)]" },
        warning: { status: result.verdict || "Suspicious", color: "text-warning", dot: "bg-warning", glow: "shadow-[0_0_8px_rgba(255,167,38,0.4)]" },
        danger: { status: result.verdict || "Threat Detected", color: "text-danger", dot: "bg-danger", glow: "shadow-[0_0_8px_rgba(239,83,80,0.4)]" },
      };
      return statusMap[result.status] || fallback[baseState];
    }
    return fallback[baseState];
  };

  const noGlow = "shadow-none";

  const checks = [
    {
      icon: Mic,
      label: "Voice Check",
      config: getConfig(audioResult, {
        idle: { status: "Standing By", color: "text-muted-foreground", dot: "bg-muted-foreground", glow: noGlow },
        safe: { status: "Verified", color: "text-neon-green", dot: "bg-neon-green", glow: "shadow-[0_0_8px_rgba(16,185,129,0.3)]" },
        danger: { status: "Suspicious", color: "text-danger", dot: "bg-danger", glow: "shadow-[0_0_8px_rgba(239,83,80,0.3)]" },
      }),
      confidence: audioResult?.confidence,
    },
    {
      icon: Video,
      label: "Video Check",
      config: {
        idle: { status: "Standing By", color: "text-muted-foreground", dot: "bg-muted-foreground", glow: noGlow },
        safe: { status: "Not Active", color: "text-muted-foreground", dot: "bg-muted-foreground", glow: noGlow },
        danger: { status: "Not Active", color: "text-muted-foreground", dot: "bg-muted-foreground", glow: noGlow },
      }[baseState],
      confidence: undefined,
    },
    {
      icon: MessageCircle,
      label: "Conversation",
      config: getConfig(textResult, {
        idle: { status: "Standing By", color: "text-muted-foreground", dot: "bg-muted-foreground", glow: noGlow },
        safe: { status: "Normal", color: "text-neon-green", dot: "bg-neon-green", glow: "shadow-[0_0_8px_rgba(16,185,129,0.3)]" },
        danger: { status: "Unusual", color: "text-danger", dot: "bg-danger", glow: "shadow-[0_0_8px_rgba(239,83,80,0.3)]" },
      }),
      confidence: textResult?.confidence,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
      {checks.map((check, i) => {
        const Icon = check.icon;

        return (
          <motion.div
            key={check.label}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="glass-card rounded-2xl p-5 glass-card-hover"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="bg-gradient-to-br from-primary to-accent p-[1px] rounded-xl">
                  <div className="p-2.5 rounded-xl bg-card">
                    <Icon className="h-5 w-5 text-primary/80" />
                  </div>
                </div>
                <span className="text-sm font-medium text-foreground/60">
                  {check.label}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <div className={`h-2 w-2 rounded-full ${check.config.dot} ${check.config.glow}`} />
                <span
                  className={`text-xs font-display font-bold ${check.config.color} uppercase tracking-wider`}
                >
                  {check.config.status}
                </span>
              </div>
            </div>

            {/* Confidence bar */}
            {check.confidence !== undefined && check.confidence > 0 && (
              <div className="mt-4 flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(check.confidence, 100)}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className={`h-full rounded-full ${
                      check.config.color === "text-danger"
                        ? "bg-danger"
                        : check.config.color === "text-warning"
                          ? "bg-warning"
                          : "bg-neon-green"
                    }`}
                  />
                </div>
                <span className="text-xs font-mono text-muted-foreground/70">
                  {check.confidence.toFixed(0)}%
                </span>
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
};

export default CompanionCards;
