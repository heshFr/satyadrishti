import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import MaterialIcon from "./MaterialIcon";

export interface ForensicItem {
  label: string;
  status: "pass" | "fail" | "warn" | "info";
  detail: string;
}

interface ForensicDetailsProps {
  items: ForensicItem[];
}

const statusConfig: Record<string, { icon: string; color: string; bg: string; border: string }> = {
  pass: { icon: "check_circle", color: "text-secondary", bg: "bg-secondary-container/10", border: "border-secondary/10" },
  fail: { icon: "cancel", color: "text-error", bg: "bg-error-container/10", border: "border-error/10" },
  warn: { icon: "warning", color: "text-primary", bg: "bg-primary/10", border: "border-primary/10" },
  info: { icon: "info", color: "text-on-surface-variant", bg: "bg-surface-container-high/30", border: "border-outline-variant/10" },
};

const ForensicDetails = ({ items }: ForensicDetailsProps) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-xl bg-surface-container-low overflow-hidden border border-outline-variant/10">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-6 py-5 text-left hover:bg-surface-container-high/40 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-headline font-semibold text-on-surface">
            {t("scanner.forensicEvidence")}
          </span>
          <span className="text-xs text-on-surface-variant/50 bg-surface-container-high px-2 py-0.5 rounded-full font-label">
            {items.length} checks
          </span>
        </div>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <MaterialIcon icon="expand_more" size={20} className="text-on-surface-variant" />
        </motion.div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-5 space-y-2">
              {items.map((item, i) => {
                const cfg = statusConfig[item.status];
                return (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className={`flex items-start gap-3 rounded-lg ${cfg.bg} border ${cfg.border} px-4 py-3.5`}
                  >
                    <MaterialIcon icon={cfg.icon} size={16} className={`mt-0.5 shrink-0 ${cfg.color}`} />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-on-surface">{item.label}</p>
                      <p className="text-xs text-on-surface-variant/70 mt-0.5 leading-relaxed">{item.detail}</p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ForensicDetails;
