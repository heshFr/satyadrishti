import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { ScanFace, Search, Activity, Database, Brain, Check, Loader2 } from "lucide-react";

const stepKeys = [
  { key: "faceDetection", icon: ScanFace },
  { key: "artifactScanning", icon: Search },
  { key: "frequencyAnalysis", icon: Activity },
  { key: "metadataExtraction", icon: Database },
  { key: "aiPatternMatching", icon: Brain },
];

interface AnalysisProgressProps {
  onComplete: () => void;
}

const AnalysisProgress = ({ onComplete }: AnalysisProgressProps) => {
  const { t } = useTranslation();
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    if (activeStep < stepKeys.length) {
      const timer = setTimeout(() => setActiveStep((s) => s + 1), 600);
      return () => clearTimeout(timer);
    } else {
      const timer = setTimeout(onComplete, 400);
      return () => clearTimeout(timer);
    }
  }, [activeStep, onComplete]);

  return (
    <div className="space-y-6">
      <div className="space-y-3">
        {stepKeys.map((step, i) => {
          const done = i < activeStep;
          const active = i === activeStep;
          const Icon = step.icon;

          return (
            <motion.div
              key={step.key}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className="flex items-center gap-4"
            >
              <div
                className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-all duration-300 ${
                  done
                    ? "bg-neon-green/15 text-neon-green shadow-[0_0_12px_rgba(16,185,129,0.2)]"
                    : active
                      ? "bg-primary/15 text-primary shadow-[0_0_12px_rgba(6,182,212,0.3)]"
                      : "bg-white/[0.03] text-muted-foreground/50"
                }`}
              >
                <AnimatePresence mode="wait">
                  {done ? (
                    <motion.div
                      key="check"
                      initial={{ scale: 0, rotate: -90 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      <Check className="w-4 h-4" />
                    </motion.div>
                  ) : active ? (
                    <motion.div
                      key="loading"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Loader2 className="w-4 h-4" />
                    </motion.div>
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </AnimatePresence>
              </div>

              <div className="flex-1">
                <p className={`text-sm font-medium transition-colors duration-300 ${
                  done ? "text-neon-green" : active ? "text-foreground" : "text-muted-foreground/50"
                }`}>
                  {t(`scanner.steps.${step.key}`)}
                </p>
                <div className="mt-2 h-1.5 w-full rounded-full bg-white/[0.04] overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${done ? "bg-neon-green" : active ? "bg-primary" : "bg-transparent"}`}
                    initial={{ width: "0%" }}
                    animate={{ width: done || active ? "100%" : "0%" }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                  {active && (
                    <motion.div
                      className="h-full -mt-1.5 rounded-full bg-gradient-to-r from-transparent via-primary/50 to-transparent"
                      animate={{ x: ["-100%", "200%"] }}
                      transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                      style={{ width: "50%" }}
                    />
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="flex items-center justify-center gap-2">
        <motion.div
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="w-1.5 h-1.5 rounded-full bg-primary"
        />
        <p className="text-sm text-muted-foreground">{t("scanner.analyzing")}</p>
      </div>
    </div>
  );
};

export default AnalysisProgress;
