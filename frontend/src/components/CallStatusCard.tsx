import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, PhoneOff, UserCheck } from "lucide-react";
import ShieldBreathing from "./ShieldBreathing";

export type CallState = "idle" | "safe" | "warning" | "danger" | "critical";

interface CallStatusCardProps {
  state: CallState;
  onHangUp?: () => void;
  onDismiss?: () => void;
}

const CallStatusCard = ({
  state,
  onHangUp,
  onDismiss,
}: CallStatusCardProps) => {
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const timeStr = now.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <AnimatePresence mode="wait">
      {state === "danger" || state === "warning" || state === "critical" ? (
        <motion.div
          key="danger"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="relative rounded-3xl border-2 border-danger glass-card p-12 text-center animate-danger-pulse overflow-hidden"
        >
          {/* Danger flash overlay */}
          <div className="absolute inset-0 bg-danger/[0.03] pointer-events-none animate-danger-flash" />

          {/* Corner glows */}
          <div className="absolute -top-20 -left-20 w-40 h-40 bg-danger/20 blur-[80px] rounded-full" />
          <div className="absolute -bottom-20 -right-20 w-40 h-40 bg-danger/20 blur-[80px] rounded-full" />

          <div className="relative z-10 space-y-8">
            <motion.div
              className="flex justify-center"
              animate={{ scale: [1, 1.08, 1] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            >
              <div className="relative">
                <div className="absolute inset-0 -m-4 rounded-full bg-danger/20 blur-xl" />
                <div className="rounded-full bg-danger/20 p-6 border border-danger/30 shadow-glow-danger">
                  <AlertTriangle className="h-16 w-16 text-danger drop-shadow-[0_0_20px_rgba(239,83,80,0.6)]" />
                </div>
              </div>
            </motion.div>

            <div className="space-y-4">
              <motion.h2
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-4xl font-bold tracking-tight text-white uppercase tracking-[0.15em]"
              >
                Warning: Suspicious Call
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="text-xl text-muted-foreground mx-auto max-w-lg"
              >
                Our AI has detected voice-cloning patterns and deepfake
                characteristics. This call is likely fraudulent.
              </motion.p>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="flex flex-col sm:flex-row gap-4 justify-center pt-8"
            >
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onHangUp}
                className="h-16 px-12 text-xl font-bold rounded-2xl bg-danger text-white hover:bg-red-600 hover:shadow-[0_0_40px_rgba(239,83,80,0.4)] transition-all flex items-center justify-center gap-3"
              >
                <PhoneOff className="h-6 w-6" />
                Hang Up Now
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onDismiss}
                className="h-16 px-12 text-xl font-medium rounded-2xl glass border border-white/10 text-white hover:bg-white/[0.06] transition-all flex items-center justify-center"
              >
                I Trust Them
              </motion.button>
            </motion.div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          key="safe-idle"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="space-y-12 text-center py-8"
        >
          <div className="space-y-2">
            <p className="text-primary/40 font-medium tracking-widest uppercase text-sm">
              {dateStr}
            </p>
            <p className="text-5xl font-light text-white/90">{timeStr}</p>
          </div>

          <div className="flex justify-center py-8">
            <ShieldBreathing status={state} />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-center gap-3">
              <div className="relative">
                <div
                  className={`h-2.5 w-2.5 rounded-full ${state === "safe" ? "bg-safe" : "bg-primary/30"}`}
                />
                {state === "safe" && (
                  <div className="absolute inset-0 h-2.5 w-2.5 rounded-full bg-safe animate-ping opacity-30" />
                )}
              </div>
              <h2 className="text-3xl font-semibold text-white tracking-tight">
                {state === "safe" ? "Call Protected" : "System Ready"}
              </h2>
            </div>
            <p className="text-lg text-muted-foreground/60 max-w-sm mx-auto leading-relaxed">
              {state === "safe"
                ? "Satya Drishti is actively monitoring this call for deepfakes and AI voice cloning."
                : "All defensive layers are active. Your communications are being secured."}
            </p>
          </div>

          <AnimatePresence>
            {state === "safe" && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex justify-center pt-4"
              >
                <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-safe/10 border border-safe/20 text-safe text-sm font-semibold shadow-glow-safe">
                  <UserCheck className="h-4 w-4" />
                  Verified Identity
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default CallStatusCard;
