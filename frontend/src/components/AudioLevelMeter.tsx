import { motion } from "framer-motion";

interface AudioLevelMeterProps {
  level: number; // 0-1
}

const AudioLevelMeter = ({ level }: AudioLevelMeterProps) => {
  const bars = 8;
  const activeCount = Math.ceil(level * bars);

  return (
    <div className="flex items-end gap-0.5 h-4">
      {Array.from({ length: bars }, (_, i) => {
        const isActive = i < activeCount;
        const height = 4 + (i / bars) * 12; // 4px to 16px
        const color =
          i < bars * 0.5
            ? "bg-safe"
            : i < bars * 0.75
              ? "bg-warning"
              : "bg-danger";

        return (
          <motion.div
            key={i}
            className={`w-1 rounded-full ${isActive ? color : "bg-muted/40"}`}
            animate={{ height: isActive ? height : 3 }}
            transition={{ duration: 0.1 }}
          />
        );
      })}
    </div>
  );
};

export default AudioLevelMeter;
