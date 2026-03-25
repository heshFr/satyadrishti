import type { CallState } from "./CallStatusCard";
import { Phone, PhoneOff, AlertTriangle } from "lucide-react";

interface DemoControlsProps {
  currentState: CallState;
  onChange: (state: CallState) => void;
}

const DemoControls = ({ currentState, onChange }: DemoControlsProps) => {
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-3 rounded-2xl bg-card border border-border shadow-2xl z-50">
      <span className="text-xs text-muted-foreground mr-2 font-mono">
        Demo:
      </span>
      <button
        onClick={() => onChange("idle")}
        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 ${
          currentState === "idle"
            ? "bg-primary text-white"
            : "bg-transparent border border-border text-muted-foreground hover:text-white"
        }`}
      >
        <PhoneOff className="w-3 h-3" />
        Idle
      </button>
      <button
        onClick={() => onChange("safe")}
        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 ${
          currentState === "safe"
            ? "bg-primary text-white"
            : "bg-transparent border border-border text-muted-foreground hover:text-white"
        }`}
      >
        <Phone className="w-3 h-3" />
        Safe Call
      </button>
      <button
        onClick={() => onChange("danger")}
        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 ${
          currentState === "danger"
            ? "bg-primary text-white"
            : "bg-transparent border border-border text-muted-foreground hover:text-white"
        }`}
      >
        <AlertTriangle className="w-3 h-3" />
        Danger
      </button>
    </div>
  );
};

export default DemoControls;
