import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { toast } from "sonner";
import MaterialIcon from "./MaterialIcon";
import ForensicDetails, { type ForensicItem } from "./ForensicDetails";
import GuidanceCard from "./GuidanceCard";
import { api } from "@/lib/api";

export type Verdict = "ai-generated" | "authentic" | "uncertain" | "inconclusive";

interface ScanVerdictProps {
  verdict: Verdict;
  confidence: number;
  forensicItems?: ForensicItem[];
  onScanAnother: () => void;
  scanId?: string | null;
}

const verdictConfig = {
  "ai-generated": {
    icon: "gpp_bad",
    titleKey: "scanner.verdictAiGenerated",
    messageKey: "scanner.verdictAiMessage",
    border: "border-error/40",
    iconColor: "text-error",
    barColor: "bg-error",
    bg: "from-error/[0.08] to-error/[0.02]",
  },
  authentic: {
    icon: "verified_user",
    titleKey: "scanner.verdictAuthentic",
    messageKey: "scanner.verdictAuthenticMessage",
    border: "border-secondary/40",
    iconColor: "text-secondary",
    barColor: "bg-secondary",
    bg: "from-secondary/[0.08] to-secondary/[0.02]",
  },
  uncertain: {
    icon: "help",
    titleKey: "scanner.verdictUncertain",
    messageKey: "scanner.verdictUncertainMessage",
    border: "border-primary/40",
    iconColor: "text-primary",
    barColor: "bg-primary",
    bg: "from-primary/[0.08] to-primary/[0.02]",
  },
  inconclusive: {
    icon: "help",
    titleKey: "scanner.verdictUncertain",
    messageKey: "scanner.verdictUncertainMessage",
    border: "border-primary/40",
    iconColor: "text-primary",
    barColor: "bg-primary",
    bg: "from-primary/[0.08] to-primary/[0.02]",
  },
};

const ScanVerdict = ({ verdict, confidence, forensicItems, onScanAnother, scanId }: ScanVerdictProps) => {
  const { t } = useTranslation();
  const cfg = verdictConfig[verdict];

  const handleDownload = () => {
    if (scanId) {
      const token = localStorage.getItem("satya-token");
      const url = api.scans.reportUrl(scanId);
      fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
        .then((res) => res.blob())
        .then((blob) => {
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = blobUrl;
          a.download = `satyadrishti_report_${scanId.slice(0, 8)}.pdf`;
          a.click();
          URL.revokeObjectURL(blobUrl);
        })
        .catch(() => {
          toast.error("Failed to download report. Please try again.");
        });
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, type: "spring", stiffness: 150 }}
      className="space-y-5"
    >
      <div className={`relative rounded-xl border-2 ${cfg.border} bg-gradient-to-br ${cfg.bg} p-8 space-y-6 overflow-hidden`}>
        <div className={`absolute -top-20 -right-20 w-40 h-40 rounded-full ${cfg.barColor}/10 blur-[80px]`} />

        <div className="flex items-center gap-4 relative z-10">
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          >
            <MaterialIcon icon={cfg.icon} filled size={40} className={cfg.iconColor} />
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="text-2xl font-headline font-bold text-on-surface tracking-tight"
          >
            {t(cfg.titleKey)}
          </motion.h2>
        </div>

        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
          className="text-on-surface-variant leading-relaxed relative z-10"
        >
          {t(cfg.messageKey)}
        </motion.p>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
          className="space-y-2 relative z-10"
        >
          <div className="flex items-center justify-between text-sm">
            <span className="text-on-surface-variant">{t("scanner.confidenceLabel")}</span>
            <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}
              className="font-headline font-bold text-on-surface text-lg"
            >
              {confidence}%
            </motion.span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-surface-container-highest overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${cfg.barColor}`}
              initial={{ width: 0 }}
              animate={{ width: `${confidence}%` }}
              transition={{ delay: 0.5, duration: 1, type: "spring", damping: 15 }}
            />
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
          className="flex items-center gap-3 pt-2 relative z-10"
        >
          <button
            onClick={handleDownload}
            disabled={!scanId}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-outline-variant/20 text-on-surface text-sm font-headline font-bold hover:border-primary/30 transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <MaterialIcon icon="download" size={16} /> {t("common.downloadReport")}
          </button>
          <button
            onClick={onScanAnother}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-on-surface-variant text-sm font-headline font-bold hover:bg-surface-container-high transition-all cursor-pointer"
          >
            <MaterialIcon icon="refresh" size={16} /> {t("common.scanAnother")}
          </button>
        </motion.div>
      </div>

      {forensicItems && forensicItems.length > 0 && <ForensicDetails items={forensicItems} />}
      {verdict === "ai-generated" && <GuidanceCard />}
    </motion.div>
  );
};

export default ScanVerdict;
