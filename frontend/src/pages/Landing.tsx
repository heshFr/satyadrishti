import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, useInView, AnimatePresence } from "framer-motion";
import { useAuth } from "@/contexts/AuthContext";
import Footer from "@/components/Footer";
import MaterialIcon from "@/components/MaterialIcon";
import { useRef, useState, useEffect } from "react";
import LandingNav from "@/components/LandingNav";
import LanguageDropdown from "@/components/LanguageDropdown";
import TopBar from "@/components/TopBar";

/* ── Stagger animations ── */
const stagger = {
  container: { animate: { transition: { staggerChildren: 0.12 } } },
  item: {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] as const } },
  },
};

/* ── Animated counter ── */
const AnimatedCounter = ({ target, suffix = "+" }: { target: number; suffix?: string }) => {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!isInView) return;
    let start = 0;
    const duration = 2000;
    const stepTime = 16;
    const steps = duration / stepTime;
    const increment = target / steps;
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, stepTime);
    return () => clearInterval(timer);
  }, [isInView, target]);

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
};

const Landing = () => {
  const { t } = useTranslation();
  const { isAuthenticated } = useAuth();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <div className="min-h-screen bg-surface text-on-surface font-body selection:bg-primary/30 selection:text-primary-container">
      {/* ── TopNavBar (Landing variant) ── */}
      <TopBar systemStatus="protected" />

      <main className="pt-32 relative">
        {/* Global continuous background */}
        <div className="fixed inset-0 pointer-events-none z-0">
          <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-primary/[0.03] rounded-full blur-[200px]" />
          <div className="absolute bottom-[30%] left-0 w-[600px] h-[600px] bg-secondary/[0.02] rounded-full blur-[150px]" />
        </div>

        {/* ── Hero Section ── */}
        <section
          className="relative px-4 sm:px-8 md:px-12 py-16 md:py-24 overflow-hidden min-h-[90vh] flex items-center"
        >
          {/* BACKGROUND IMAGE WITH ZOOM */}
          <div 
            className="absolute inset-0 bg-cover bg-center animate-subtle-zoom z-0"
            style={{ backgroundImage: "url('/LandingBackground.png')" }}
          />

          {/* CINEMATIC OVERLAY */}
          <div className="absolute inset-0 bg-gradient-to-r from-surface via-surface/90 to-transparent z-10"></div>
          <div className="absolute inset-0 bg-gradient-to-t from-surface via-transparent to-transparent z-10 opacity-60"></div>

          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center gap-10 md:gap-16 relative z-20 w-full">
            <motion.div
              className="w-full md:w-1/2 space-y-8"
              variants={stagger.container}
              initial="initial"
              animate="animate"
            >
              <motion.div variants={stagger.item} className="inline-flex items-center gap-2 px-4 py-2 rounded-[0.75rem] bg-surface-container-high border border-outline-variant/15">
                <span className="w-2 h-2 rounded-full bg-secondary shadow-[0_0_10px_rgba(78,222,163,0.5)]" />
                <span className="text-xs font-label tracking-widest uppercase text-on-surface-variant">{t("landing.badge")}</span>
              </motion.div>

              <motion.h1 variants={stagger.item} className="text-4xl sm:text-5xl md:text-7xl font-headline font-extrabold tracking-tighter leading-none text-on-surface">
                <span className="text-5xl sm:text-7xl md:text-[8.5rem] leading-[0.9] break-words">{t("landing.heroTitle")}</span><br />
                <span className="text-primary-container text-xl sm:text-2xl md:text-4xl font-medium tracking-wide mt-4 block">{t("landing.heroSubtitle")}</span>
              </motion.h1>

              <motion.p variants={stagger.item} className="text-lg sm:text-xl md:text-2xl text-on-surface-variant font-body leading-relaxed max-w-4xl font-light">
                {t("landing.heroDescription")} <strong className="text-on-surface font-headline font-bold">{t("landing.heroDescBold")}</strong> {t("landing.heroDescEnd")}
              </motion.p>

              <motion.div variants={stagger.item} className="flex flex-wrap gap-4 sm:gap-6 pt-6 md:pt-8">
                <Link
                  to="/hub"
                  className="group relative px-6 sm:px-10 py-4 sm:py-5 bg-gradient-to-br from-primary to-primary-container text-on-primary font-headline font-black uppercase tracking-widest rounded-2xl shadow-[0_0_40px_rgba(0,209,255,0.3)] hover:shadow-[0_0_60px_rgba(0,209,255,0.5)] hover:-translate-y-1 transition-all text-base sm:text-lg overflow-hidden active:scale-95"
                >
                  <div className="absolute inset-0 bg-white/20 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                  <span className="relative z-10 flex items-center gap-3">
                    {t("common.getProtected")}
                    <MaterialIcon icon="arrow_forward" />
                  </span>
                </Link>

                <Link
                  to="/live-demo"
                  className="group px-6 sm:px-10 py-4 sm:py-5 bg-white/5 border border-white/10 text-on-surface font-headline font-black uppercase tracking-widest rounded-2xl hover:bg-white/10 hover:border-white/20 hover:-translate-y-1 transition-all text-base sm:text-lg flex items-center gap-3 active:scale-95"
                >
                  <MaterialIcon icon="play_circle" size={28} className="text-secondary group-hover:scale-110 transition-transform" />
                  {t("common.watchLiveDemo")}
                </Link>
              </motion.div>
            </motion.div>

            {/* Hero visual — 3-layer logo with orbiting rings */}
            <motion.div
              className="w-full md:w-1/2 relative flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            >
              <div className="relative w-64 h-64 sm:w-80 sm:h-80 md:w-[480px] md:h-[480px] lg:w-[480px] lg:h-[480px] flex items-center justify-center">
                
                {/* AMBIENT GLOW */}
                <div className="absolute inset-[-40px] bg-primary/10 rounded-full blur-[100px] animate-pulse" />

                {/* ORBITAL RINGS */}
                <div className="absolute inset-[-30px] rounded-full border border-primary/20 animate-spin-slow opacity-40">
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 w-4 h-4 bg-primary rounded-full blur-[5px]" />
                </div>
                <div className="absolute inset-[-60px] rounded-full border border-secondary/10 animate-spin-slow-reverse opacity-30">
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-secondary rounded-full blur-[4px]" />
                </div>

                {/* SCANNING LINE CONTAINER */}
                <div className="absolute inset-0 overflow-hidden rounded-2xl z-10 pointer-events-none">
                  <div className="scan-line"></div>
                </div>

                {/* LOGO */}
                <img
                  src="/logo.png"
                  alt="Satya Drishti Logo"
                  className="relative z-7 w-full h-full object-contain rounded-2xl drop-shadow-[0_0_60px_rgba(0,209,255,0.5)] animate-float"
                />

                {/* Particles */}
                <div className="absolute w-2 h-2 bg-cyan-400 rounded-full top-20 left-10 animate-ping opacity-60 z-30"></div>
                <div className="absolute w-2 h-2 bg-emerald-400 rounded-full bottom-20 right-10 animate-ping delay-500 opacity-60 z-30"></div>
              </div>
            </motion.div>
          </div>

          {/* Background gradient */}
          <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-primary/5 to-transparent blur-3xl pointer-events-none" />
        </section>

        {/* ── Gradient transition ── */}
        <div className="h-32 section-transition relative z-10" />

        {/* ── Stats Section ── */}
        <section className="px-4 sm:px-8 md:px-12 py-16 md:py-24 relative z-10">
          <div className="absolute inset-0 section-glow-cyan pointer-events-none" />
          <motion.div 
            className="max-w-7xl mx-auto"
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, margin: "-100px" }}
            variants={stagger.container}
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 md:gap-20">
              <motion.div variants={stagger.item} className="space-y-4 border-l border-outline-variant/15 pl-6 md:pl-8">
                <div className="text-5xl sm:text-6xl md:text-7xl font-headline font-black text-on-surface tracking-tighter break-words">
                  ₹<AnimatedCounter target={1947} suffix="Cr" />
                </div>
                <p className="text-on-surface-variant uppercase tracking-widest text-sm md:text-base font-label">{t("landing.statFraud")}</p>
              </motion.div>
              <motion.div variants={stagger.item} className="space-y-4 border-l border-outline-variant/15 pl-6 md:pl-8">
                <div className="text-5xl sm:text-6xl md:text-7xl font-headline font-black text-secondary tracking-tighter break-words">
                  <AnimatedCounter target={500} suffix="%" />
                </div>
                <p className="text-on-surface-variant uppercase tracking-widest text-sm md:text-base font-label">{t("landing.statVoiceClone")}</p>
              </motion.div>
              <motion.div variants={stagger.item} className="space-y-4 border-l border-outline-variant/15 pl-6 md:pl-8">
                <div className="text-5xl sm:text-6xl md:text-7xl font-headline font-black text-primary-container tracking-tighter break-words">
                  <AnimatedCounter target={9} suffix="" />
                </div>
                <p className="text-on-surface-variant uppercase tracking-widest text-sm md:text-base font-label">{t("landing.statLayers")}</p>
              </motion.div>
            </div>
          </motion.div>
        </section>

        {/* ── Gradient transition ── */}
        <div className="h-24 section-transition relative z-10" />

        {/* ── Features Bento Grid ── */}
        <section id="features" className="px-4 sm:px-8 md:px-12 py-16 md:py-24 relative z-10">
          <div className="absolute inset-0 section-glow-emerald pointer-events-none" />
          <div className="max-w-7xl mx-auto space-y-10 md:space-y-16">
            <div className="text-center space-y-4">
              <h2 className="text-3xl sm:text-4xl md:text-6xl font-headline font-extrabold text-on-surface">{t("landing.featuresTitle")}</h2>
              <p className="text-base sm:text-lg md:text-2xl text-on-surface-variant max-w-3xl mx-auto font-light">{t("landing.featuresSubtitle")}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-8">
              {/* Voice Cloning Detection — Large */}
              <div className="md:col-span-8 bg-surface-container-low/60 backdrop-blur-sm p-6 sm:p-8 md:p-12 rounded-3xl group relative overflow-hidden border border-outline-variant/10 hover:border-primary/15 transition-all duration-500">
                <div className="relative z-10 space-y-4 md:space-y-6">
                  <MaterialIcon icon="record_voice_over" filled size={48} className="text-secondary" />
                  <h3 className="text-2xl sm:text-3xl md:text-5xl font-headline font-bold text-on-surface">{t("landing.featureVoiceTitle")}</h3>
                  <p className="text-base md:text-xl text-on-surface-variant max-w-2xl font-light">{t("landing.featureVoiceDesc")}</p>
                </div>
                <div className="absolute -bottom-20 -right-20 w-80 h-80 bg-secondary/10 rounded-full blur-[100px] group-hover:bg-secondary/20 transition-all" />
              </div>

              {/* Visual Verifier */}
              <div className="md:col-span-4 bg-surface-container/60 backdrop-blur-sm p-6 sm:p-8 md:p-12 rounded-3xl border border-outline-variant/10 hover:border-primary/15 flex flex-col justify-between transition-all duration-500">
                <MaterialIcon icon="visibility" size={48} className="text-primary-container" />
                <div className="space-y-3 mt-6 md:mt-8">
                  <h3 className="text-2xl sm:text-3xl lg:text-4xl font-headline font-bold text-on-surface">{t("landing.featureMediaTitle")}</h3>
                  <p className="text-base lg:text-lg text-on-surface-variant font-light">{t("landing.featureMediaDesc")}</p>
                </div>
              </div>

              {/* Digital Signature */}
              <div className="md:col-span-4 bg-surface-container-high/60 backdrop-blur-sm p-6 sm:p-8 md:p-12 rounded-3xl border border-outline-variant/10 hover:border-primary/15 transition-all duration-500">
                <MaterialIcon icon="verified_user" filled size={48} className="text-primary" />
                <div className="mt-6 md:mt-8 space-y-3">
                  <h3 className="text-2xl sm:text-3xl lg:text-4xl font-headline font-bold text-on-surface">{t("landing.featureBiometricTitle")}</h3>
                  <p className="text-base lg:text-lg text-on-surface-variant font-light">{t("landing.featureBiometricDesc")}</p>
                </div>
              </div>

              {/* Live Scanning — Large */}
              <div className="md:col-span-8 bg-surface-container-low/60 backdrop-blur-sm p-6 sm:p-8 md:p-12 rounded-3xl relative overflow-hidden group border border-outline-variant/10 hover:border-primary/15 transition-all duration-500">
                <div className="relative z-10 space-y-4 md:space-y-6">
                  <h3 className="text-2xl sm:text-3xl md:text-5xl font-headline font-bold text-on-surface">{t("landing.featureMonitorTitle")}</h3>
                  <p className="text-base md:text-xl text-on-surface-variant max-w-2xl font-light">{t("landing.featureMonitorDesc")}</p>
                  <div className="flex flex-wrap gap-2 sm:gap-4">
                    <span className="px-3 sm:px-4 py-2 bg-secondary/10 text-secondary rounded-[0.75rem] text-xs font-bold uppercase tracking-widest">{t("landing.tagGuardrailed")}</span>
                    <span className="px-3 sm:px-4 py-2 bg-primary/10 text-primary rounded-[0.75rem] text-xs font-bold uppercase tracking-widest">{t("landing.tagNineLayers")}</span>
                    <span className="px-3 sm:px-4 py-2 bg-error/10 text-error rounded-[0.75rem] text-xs font-bold uppercase tracking-widest">{t("landing.tagAuditable")}</span>
                  </div>
                </div>
                <div className="absolute -bottom-20 -right-20 w-80 h-80 bg-primary/10 rounded-full blur-[100px] group-hover:bg-primary/20 transition-all" />
              </div>
            </div>
          </div>
        </section>

        {/* ── Why Trust Satya Drishti ── */}
        <section className="px-4 sm:px-8 md:px-12 py-20 md:py-32 relative">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/3 to-transparent pointer-events-none" />
          <div className="max-w-7xl mx-auto relative z-10">
            <div className="text-center space-y-4 md:space-y-6 mb-12 md:mb-20">
              <h2 className="text-3xl sm:text-4xl md:text-6xl font-headline font-extrabold text-on-surface">{t("landing.whyTitle")}</h2>
              <p className="text-on-surface-variant text-base sm:text-lg md:text-2xl max-w-3xl mx-auto font-light">{t("landing.whySubtitle")}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-8">
              {[
                { icon: "gavel", title: t("landing.whyGuardrailsTitle"), desc: t("landing.whyGuardrailsDesc") },
                { icon: "speed", title: t("landing.whyDetectionTitle"), desc: t("landing.whyDetectionDesc") },
                { icon: "description", title: t("landing.whyAuditsTitle"), desc: t("landing.whyAuditsDesc") },
              ].map((item) => (
                <div key={item.title} className="p-6 sm:p-8 md:p-10 rounded-3xl bg-surface-container-low/60 backdrop-blur-sm border border-outline-variant/10 space-y-4 md:space-y-5 group hover:bg-surface-container-high/40 transition-all duration-300">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                    <MaterialIcon icon={item.icon} filled size={28} className="text-primary" />
                  </div>
                  <h3 className="text-2xl md:text-3xl font-headline font-bold text-on-surface">{item.title}</h3>
                  <p className="text-base md:text-lg text-on-surface-variant leading-relaxed font-light">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Final CTA ── */}
        <section className="px-4 sm:px-8 md:px-12 py-24 md:py-40">
          <div className="max-w-5xl mx-auto bg-gradient-to-br from-surface-container-high to-surface-container-lowest rounded-[2rem] md:rounded-[3rem] p-6 sm:p-10 md:p-16 lg:p-24 text-center space-y-8 md:space-y-12 relative overflow-hidden border border-outline-variant/10">
            <div className="absolute inset-0 bg-primary/5 blur-[120px] pointer-events-none" />
            <div className="relative z-10 space-y-4 md:space-y-6">
              <h2 className="text-3xl sm:text-5xl md:text-7xl lg:text-8xl font-headline font-extrabold text-on-surface tracking-tighter leading-tight">
                {t("landing.ctaTitle")} <br /><span className="text-secondary">{t("landing.ctaHighlight")}</span>
              </h2>
              <p className="text-base sm:text-xl md:text-3xl font-light text-on-surface-variant max-w-3xl mx-auto font-body leading-relaxed">
                {t("landing.ctaDescription")}
              </p>
            </div>
            <div className="relative z-10 flex flex-col md:flex-row justify-center items-center gap-6 md:gap-8 mt-4">
              <Link
                to="/hub"
                className="w-full md:w-auto px-8 sm:px-10 md:px-14 py-5 sm:py-6 md:py-8 bg-gradient-to-br from-primary to-primary-container text-on-primary font-headline font-extrabold text-base sm:text-lg md:text-2xl uppercase tracking-widest rounded-3xl shadow-[0_20px_40px_rgba(0,209,255,0.2)] hover:shadow-[0_20px_60px_rgba(0,209,255,0.4)] hover:scale-105 active:scale-95 transition-all"
              >
                {t("common.deployPlatform")}
              </Link>
              <Link
                to="/scanner"
                className="text-on-surface-variant hover:text-on-surface font-headline font-bold uppercase tracking-[0.2em] text-sm md:text-lg underline decoration-primary underline-offset-8 transition-colors"
              >
                {t("common.tryMediaForensics")}
              </Link>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Landing;
