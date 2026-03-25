import { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { TranscriptLine } from "@/hooks/useCallProtection";

interface TranscriptPanelProps {
  transcript: TranscriptLine[];
}

const TranscriptPanel = ({ transcript }: TranscriptPanelProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  if (transcript.length === 0) return null;

  return (
    <div
      ref={scrollRef}
      className="max-h-48 overflow-y-auto space-y-2 mt-3 border-t border-border pt-3"
    >
      <AnimatePresence>
        {transcript.map((line, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-start gap-3 text-sm"
          >
            <span className="text-muted-foreground font-mono text-xs shrink-0 pt-0.5">
              {line.time}
            </span>
            <span
              className={
                line.flagged
                  ? "text-danger bg-danger/10 px-2 py-0.5 rounded"
                  : "text-white/80"
              }
            >
              {line.text}
            </span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default TranscriptPanel;
