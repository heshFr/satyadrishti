import { useTranslation } from "react-i18next";
import { Heart, Download, Phone, Users, ShieldCheck } from "lucide-react";
import { motion } from "framer-motion";

const GuidanceCard = () => {
  const { t } = useTranslation();

  const steps = [
    { icon: Download, text: t("guidance.step1") },
    { icon: Phone, text: t("guidance.step2") },
    { icon: Users, text: t("guidance.step3") },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="rounded-2xl glass-card p-6 space-y-5 border-primary/10 relative overflow-hidden"
    >
      {/* Subtle background glow */}
      <div className="absolute -top-10 -right-10 w-32 h-32 bg-primary/5 blur-[60px] rounded-full" />

      <div className="flex items-center gap-3 relative z-10">
        <div className="bg-gradient-to-br from-primary to-accent p-[1px] rounded-lg">
          <div className="w-8 h-8 rounded-lg bg-card flex items-center justify-center">
            <Heart className="w-4 h-4 text-primary" />
          </div>
        </div>
        <h3 className="font-display font-semibold text-foreground">{t("guidance.title")}</h3>
      </div>

      <div className="space-y-3 relative z-10">
        {steps.map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -5 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 + i * 0.1 }}
            className="flex items-start gap-3"
          >
            <div className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center shrink-0 mt-0.5">
              <step.icon className="w-3.5 h-3.5 text-primary/70" />
            </div>
            <p className="text-sm text-foreground/70 leading-relaxed">{step.text}</p>
          </motion.div>
        ))}
      </div>

      <div className="flex items-start gap-3 rounded-xl bg-neon-green/[0.04] border border-neon-green/10 p-4 relative z-10">
        <ShieldCheck className="w-5 h-5 text-neon-green shrink-0 mt-0.5" />
        <p className="text-sm text-foreground/70 leading-relaxed">{t("guidance.reassurance")}</p>
      </div>
    </motion.div>
  );
};

export default GuidanceCard;
