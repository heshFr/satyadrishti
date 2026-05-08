import { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import MaterialIcon from "@/components/MaterialIcon";
import TopBar from "@/components/TopBar";

const UPI_ID = "8976053362@fam";
const UPI_PAYEE_NAME = "Hetesh Sandeep Vichare";
const UPI_QR_PATH = "/upi-qr.png";

const Donate = () => {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const copyUpi = async () => {
    try {
      await navigator.clipboard.writeText(UPI_ID);
      setCopied(true);
      toast.success(t("donate.copied"));
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error(t("common.error"));
    }
  };

  const upiDeepLink = `upi://pay?pa=${encodeURIComponent(UPI_ID)}&pn=${encodeURIComponent(UPI_PAYEE_NAME)}&cu=INR`;

  return (
    <div className="min-h-screen bg-background text-on-surface overflow-x-hidden">
      <TopBar systemStatus="protected" />

      {/* Hero */}
      <section className="relative pt-40 pb-20 px-6 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-8"
        >
          <span className="inline-block px-4 py-1.5 bg-primary/10 border border-primary/20 rounded-full text-primary font-mono text-[12px] font-black uppercase tracking-[0.3em]">
            {t("donate.badge")}
          </span>
          <h1 className="text-6xl md:text-8xl font-black font-headline tracking-tighter leading-[0.85] uppercase">
            {t("donate.heroLine1")} <br />
            <span className="text-primary-container">{t("donate.heroLine2")}</span>
          </h1>
          <p className="max-w-3xl text-xl md:text-2xl text-on-surface-variant font-light leading-relaxed">
            {t("donate.heroDescription")}
          </p>
        </motion.div>
      </section>

      {/* Why support — 3 cards */}
      <section className="py-20 px-6 max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          { title: t("donate.reasonComputeTitle"), icon: "memory", desc: t("donate.reasonComputeDesc") },
          { title: t("donate.reasonOpenTitle"), icon: "code", desc: t("donate.reasonOpenDesc") },
          { title: t("donate.reasonFreeTitle"), icon: "volunteer_activism", desc: t("donate.reasonFreeDesc") },
        ].map((item, i) => (
          <motion.div
            key={item.title}
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.1, duration: 0.5 }}
            className="p-10 rounded-[2.5rem] bg-surface-container-low/30 border border-white/5 backdrop-blur-xl hover:bg-surface-container-low/50 transition-all group"
          >
            <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-8 group-hover:bg-primary/20 group-hover:border-primary/40 transition-all">
              <MaterialIcon icon={item.icon} size={32} className="text-on-surface-variant group-hover:text-primary" />
            </div>
            <h3 className="text-3xl font-headline font-black mb-4 uppercase tracking-tight">{item.title}</h3>
            <p className="text-on-surface-variant text-lg font-light leading-snug">{item.desc}</p>
          </motion.div>
        ))}
      </section>

      {/* Give — two-column */}
      <section className="py-32 bg-surface-container-lowest/50 backdrop-blur-3xl border-y border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-20 space-y-4">
            <span className="inline-block px-4 py-1.5 bg-primary/10 border border-primary/20 rounded-full text-primary font-mono text-[12px] font-black uppercase tracking-[0.3em]">
              {t("donate.giveBadge")}
            </span>
            <h2 className="text-5xl md:text-6xl font-headline font-black uppercase tracking-tighter">
              {t("donate.giveTitle")}
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* India / UPI */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="p-10 rounded-[2.5rem] bg-surface-container-low/40 border border-primary/20 backdrop-blur-xl"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-3xl leading-none">🇮🇳</span>
                <span className="font-mono text-[12px] font-black uppercase tracking-[0.3em] text-primary">
                  {t("donate.indiaTag")}
                </span>
              </div>
              <h3 className="text-4xl font-headline font-black mb-2 uppercase tracking-tight">
                {t("donate.indiaTitle")}
              </h3>
              <p className="text-on-surface-variant mb-8 font-light">
                {t("donate.indiaDesc")}
              </p>

              <div className="bg-white p-6 rounded-3xl mb-6 flex items-center justify-center">
                <img
                  src={UPI_QR_PATH}
                  alt={t("donate.qrAlt")}
                  className="w-full max-w-[280px] h-auto aspect-square object-contain"
                  loading="lazy"
                />
              </div>

              <div className="space-y-3 mb-6">
                <div className="text-[12px] font-mono uppercase tracking-[0.3em] text-on-surface-variant/70">
                  {t("donate.upiIdLabel")}
                </div>
                <button
                  onClick={copyUpi}
                  className="w-full flex items-center justify-between gap-3 px-5 py-4 bg-surface-container-high/60 hover:bg-surface-container-high border border-outline-variant/20 hover:border-primary/40 rounded-2xl transition-all group cursor-pointer"
                >
                  <span className="font-mono text-base md:text-lg text-on-surface truncate">{UPI_ID}</span>
                  <span className="flex items-center gap-2 text-primary text-sm font-headline font-semibold uppercase tracking-wider shrink-0">
                    <MaterialIcon icon={copied ? "check" : "content_copy"} size={18} />
                    {copied ? t("donate.copied") : t("donate.copy")}
                  </span>
                </button>
                <div className="text-[12px] font-mono uppercase tracking-[0.3em] text-on-surface-variant/70 pt-2">
                  {t("donate.payeeLabel")}
                </div>
                <div className="px-5 py-4 bg-surface-container-high/60 border border-outline-variant/20 rounded-2xl font-headline font-semibold text-on-surface">
                  {UPI_PAYEE_NAME}
                </div>
              </div>

              <a
                href={upiDeepLink}
                className="block sm:hidden w-full text-center px-6 py-4 bg-primary text-on-primary rounded-2xl font-headline font-black uppercase tracking-wider hover:bg-primary/90 transition-colors mb-6"
              >
                {t("donate.payInApp")}
              </a>

              <div className="text-[12px] font-mono uppercase tracking-[0.3em] text-on-surface-variant/70 mb-3">
                {t("donate.supportedApps")}
              </div>
              <div className="flex flex-wrap gap-2">
                {["PhonePe", "Google Pay", "Paytm", "BHIM", "Amazon Pay"].map((app) => (
                  <span
                    key={app}
                    className="px-3 py-1.5 bg-surface-container-high/40 border border-outline-variant/20 rounded-full text-xs font-mono text-on-surface-variant"
                  >
                    {app}
                  </span>
                ))}
              </div>
            </motion.div>

            {/* International — coming soon */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="p-10 rounded-[2.5rem] bg-surface-container-low/20 border border-dashed border-outline-variant/30 backdrop-blur-xl flex flex-col"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-3xl leading-none">🌍</span>
                <span className="font-mono text-[12px] font-black uppercase tracking-[0.3em] text-on-surface-variant">
                  {t("donate.intlTag")}
                </span>
              </div>
              <h3 className="text-4xl font-headline font-black mb-2 uppercase tracking-tight">
                {t("donate.intlTitle")}
              </h3>
              <p className="text-on-surface-variant mb-8 font-light">
                {t("donate.intlDesc")}
              </p>

              <div className="flex-1 flex flex-col items-center justify-center text-center py-12 px-6 rounded-3xl bg-surface-container-high/20 border border-outline-variant/10">
                <div className="w-20 h-20 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mb-6">
                  <MaterialIcon icon="schedule" size={40} className="text-on-surface-variant" />
                </div>
                <div className="font-mono text-[12px] font-black uppercase tracking-[0.3em] text-on-surface-variant/70 mb-3">
                  {t("donate.comingSoonTag")}
                </div>
                <h4 className="text-2xl font-headline font-black mb-3 uppercase tracking-tight">
                  {t("donate.intlComingSoonTitle")}
                </h4>
                <p className="text-on-surface-variant font-light max-w-sm mb-8">
                  {t("donate.intlComingSoonDesc")}
                </p>
                <Link
                  to="/contact"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-surface-container-high/60 hover:bg-surface-container-high border border-outline-variant/20 hover:border-primary/40 rounded-2xl font-headline font-semibold uppercase tracking-wider transition-all"
                >
                  <MaterialIcon icon="mail" size={18} />
                  {t("donate.intlContactCta")}
                </Link>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Transparency */}
      <section className="py-32 px-6">
        <div className="max-w-4xl mx-auto space-y-8 text-center">
          <span className="inline-block px-4 py-1.5 bg-primary/10 border border-primary/20 rounded-full text-primary font-mono text-[12px] font-black uppercase tracking-[0.3em]">
            {t("donate.transparencyTag")}
          </span>
          <h2 className="text-4xl md:text-5xl font-headline font-black uppercase tracking-tighter">
            {t("donate.transparencyTitle")}
          </h2>
          <p className="text-xl text-on-surface-variant font-light leading-relaxed">
            {t("donate.transparencyBody")}
          </p>
          <Link
            to="/contact"
            className="inline-flex items-center gap-2 text-primary hover:text-primary-container font-headline font-semibold uppercase tracking-wider transition-colors"
          >
            {t("donate.transparencyContact")}
            <MaterialIcon icon="arrow_forward" size={20} />
          </Link>
        </div>
      </section>

      <footer className="py-20 text-center text-outline text-[12px] font-mono uppercase tracking-[0.5em] opacity-30">
        Satya Drishti • {t("donate.footerTagline")}
      </footer>
    </div>
  );
};

export default Donate;
