import { useRef, useEffect } from "react";
import { motion } from "framer-motion";

interface AudioWaveformProps {
  /** Uint8Array from AnalyserNode.getByteTimeDomainData() */
  waveformData: Uint8Array | null;
  /** Uint8Array from AnalyserNode.getByteFrequencyData() */
  frequencyData: Uint8Array | null;
  /** 0-1 RMS audio level */
  audioLevel: number;
  /** Visual mode */
  mode?: "waveform" | "frequency" | "both";
  /** Height of the visualization */
  height?: number;
  /** Color scheme */
  color?: "safe" | "danger" | "primary";
  /** Whether audio is actively being captured */
  isActive?: boolean;
}

const COLOR_MAP = {
  safe: { stroke: "#66BB6A", fill: "rgba(102, 187, 106, 0.1)", glow: "rgba(102, 187, 106, 0.3)" },
  danger: { stroke: "#EF5350", fill: "rgba(239, 83, 80, 0.1)", glow: "rgba(239, 83, 80, 0.3)" },
  primary: { stroke: "#10B981", fill: "rgba(16, 185, 129, 0.1)", glow: "rgba(16, 185, 129, 0.3)" },
};

const AudioWaveform = ({
  waveformData,
  frequencyData,
  audioLevel,
  mode = "both",
  height = 120,
  color = "safe",
  isActive = false,
}: AudioWaveformProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const colors = COLOR_MAP[color];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!isActive || (!waveformData && !frequencyData)) {
      // Idle state — dashed center line
      ctx.beginPath();
      ctx.strokeStyle = `${colors.stroke}40`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }

    // Draw frequency bars (bottom layer)
    if (frequencyData && (mode === "frequency" || mode === "both")) {
      const barCount = 64;
      const barWidth = w / barCount;
      const step = Math.floor(frequencyData.length / barCount);

      for (let i = 0; i < barCount; i++) {
        const value = frequencyData[i * step] / 255;
        const barHeight = value * h * 0.4;

        const gradient = ctx.createLinearGradient(0, h, 0, h - barHeight);
        gradient.addColorStop(0, `${colors.stroke}10`);
        gradient.addColorStop(1, `${colors.stroke}50`);

        ctx.fillStyle = gradient;
        ctx.fillRect(
          i * barWidth + 1,
          h - barHeight,
          barWidth - 2,
          barHeight
        );
      }
    }

    // Draw waveform (centered, top layer)
    if (waveformData && (mode === "waveform" || mode === "both")) {
      ctx.beginPath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = colors.stroke;
      ctx.shadowColor = colors.glow;
      ctx.shadowBlur = 8;

      const sliceWidth = w / waveformData.length;
      let x = 0;

      for (let i = 0; i < waveformData.length; i++) {
        const v = waveformData[i] / 128.0;
        const y = (v * h) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.stroke();
      ctx.shadowBlur = 0;

      // Fill under waveform
      ctx.lineTo(w, h / 2);
      ctx.lineTo(0, h / 2);
      ctx.closePath();
      ctx.fillStyle = colors.fill;
      ctx.fill();
    }
  }, [waveformData, frequencyData, isActive, mode, colors, height]);

  // Resize canvas to container
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
    });

    observer.observe(canvas.parentElement!);
    return () => observer.disconnect();
  }, [height]);

  return (
    <div className="relative w-full rounded-lg overflow-hidden border-2 border-white/[0.08]"
      style={{ height, boxShadow: "4px 4px 0px rgba(0,0,0,0.2)" }}
    >
      {/* Background grid — neo-brutal tech feel */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px)`,
          backgroundSize: "20px 20px",
        }}
      />

      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height }}
      />

      {/* Level indicator bar at bottom */}
      <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-white/[0.04]">
        <motion.div
          className="h-full"
          style={{ backgroundColor: colors.stroke }}
          animate={{ width: `${Math.min(100, audioLevel * 100)}%` }}
          transition={{ duration: 0.1 }}
        />
      </div>

      {/* Status badge */}
      <div className="absolute top-2 right-2">
        {isActive ? (
          <span className="flex items-center gap-1.5 text-[9px] font-black uppercase tracking-widest px-2 py-1 rounded-md bg-black/50 border border-white/10"
            style={{ color: colors.stroke }}
          >
            <span className="w-1.5 h-1.5 rounded-sm animate-pulse" style={{ backgroundColor: colors.stroke }} />
            LIVE
          </span>
        ) : (
          <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground/30 px-2 py-1 font-mono">
            STANDBY
          </span>
        )}
      </div>
    </div>
  );
};

export default AudioWaveform;
