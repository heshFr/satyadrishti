import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, useInView } from "framer-motion";
import { useAuth } from "@/contexts/AuthContext";
import Footer from "@/components/Footer";
import MaterialIcon from "@/components/MaterialIcon";
import { useRef, useState, useEffect } from "react";

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
      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? "bg-surface/80 backdrop-blur-xl shadow-ambient" : "bg-transparent"}`}>
        <div className="flex justify-between items-center max-w-7xl mx-auto px-12 py-6">
          <Link to="/" className="text-2xl font-black tracking-tighter text-on-surface font-headline">
            Satya Drishti
          </Link>
          <div className="hidden md:flex gap-10 items-center">
            <a href="#features" className="text-on-surface-variant hover:text-on-surface transition-colors duration-300 font-headline tracking-tight font-bold text-sm uppercase">
              Platform
            </a>
            <Link to="/scanner" className="text-on-surface-variant hover:text-on-surface transition-colors duration-300 font-headline tracking-tight font-bold text-sm uppercase">
              Scanner
            </Link>
            <Link to="/call-protection" className="text-on-surface-variant hover:text-on-surface transition-colors duration-300 font-headline tracking-tight font-bold text-sm uppercase">
              Protection
            </Link>
            <Link to="/help" className="text-on-surface-variant hover:text-on-surface transition-colors duration-300 font-headline tracking-tight font-bold text-sm uppercase">
              Support
            </Link>
          </div>
          <div className="flex gap-6 items-center">
            {isAuthenticated ? (
              <Link to="/call-protection" className="btn-sentinel px-6 py-2 rounded-lg text-sm">
                Call Protection
              </Link>
            ) : (
              <>
                <Link to="/login" className="text-on-surface-variant hover:text-on-surface font-headline tracking-tight font-bold text-sm uppercase transition-all">
                  {t("common.login")}
                </Link>
                <Link to="/login" className="bg-gradient-to-br from-primary to-primary-container text-on-primary px-6 py-2 rounded-lg font-headline tracking-tight font-bold text-sm uppercase transition-transform active:scale-95">
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      </nav>

      <main className="pt-32 relative">
        {/* Global continuous background */}
        <div className="fixed inset-0 pointer-events-none z-0">
          <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-primary/[0.03] rounded-full blur-[200px]" />
          <div className="absolute bottom-[30%] left-0 w-[600px] h-[600px] bg-secondary/[0.02] rounded-full blur-[150px]" />
        </div>

        {/* ── Hero Section ── */}
        <section className="relative px-12 py-24 overflow-hidden">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center gap-16 relative z-10">
            <motion.div
              className="w-full md:w-1/2 space-y-8"
              variants={stagger.container}
              initial="initial"
              animate="animate"
            >
              <motion.div variants={stagger.item} className="inline-flex items-center gap-2 px-4 py-2 rounded-[0.75rem] bg-surface-container-high border border-outline-variant/15">
                <span className="w-2 h-2 rounded-full bg-secondary shadow-[0_0_10px_rgba(78,222,163,0.5)]" />
                <span className="text-xs font-label tracking-widest uppercase text-on-surface-variant">Real-time threat monitoring active</span>
              </motion.div>

              <motion.h1 variants={stagger.item} className="text-5xl md:text-7xl font-headline font-extrabold tracking-tighter leading-none text-on-surface">
                <span className="text-6xl md:text-8xl">Satya Drishti</span><br /><span className="text-primary-container text-3xl md:text-4xl font-bold">The Truth, Authenticated.</span>
              </motion.h1>

              <motion.p variants={stagger.item} className="text-xl text-on-surface-variant font-body leading-relaxed max-w-lg">
                In an era of synthetic deception, Satya Drishti serves as your digital curator. We detect deepfakes, voice clones, and scams with 99.9% precision before they reach your network.
              </motion.p>

              <motion.div variants={stagger.item} className="flex flex-wrap gap-6 pt-4">
                <Link
                  to={isAuthenticated ? "/call-protection" : "/login"}
                  className="px-10 py-5 bg-gradient-to-br from-primary to-primary-container text-on-primary font-headline font-bold uppercase tracking-tight rounded-xl shadow-2xl hover:brightness-110 transition-all"
                >
                  Start Protecting Your Family
                </Link>
              </motion.div>
            </motion.div>

            {/* Hero visual — logo with orbiting elements */}
            <motion.div
              className="w-full md:w-1/2 relative flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            >
              <div className="relative w-[420px] h-[420px] flex items-center justify-center">
                {/* Glow behind */}
                <div className="absolute inset-0 bg-primary/8 rounded-3xl blur-[80px]" />

                {/* Logo fills the square */}
                <img
                  src="/logo.png"
                  alt="Satya Drishti"
                  className="relative z-10 w-full h-full object-contain rounded-2xl drop-shadow-[0_0_40px_rgba(0,209,255,0.4)]"
                />

                {/* Orbital elements around the image */}
                <div className="absolute inset-[-20px] animate-spin-orbit">
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 w-4 h-4 bg-secondary rounded-full shadow-[0_0_15px_rgba(78,222,163,0.6)]" />
                </div>
                <div className="absolute inset-[-30px] animate-spin-orbit-reverse">
                  <div className="absolute bottom-0 right-1/2 translate-x-1/2 w-3 h-3 bg-primary rounded-full shadow-[0_0_15px_rgba(0,209,255,0.6)]" />
                </div>
              </div>
            </motion.div>
          </div>

          {/* Background gradient */}
          <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-primary/5 to-transparent blur-3xl pointer-events-none" />
        </section>

        {/* ── Gradient transition ── */}
        <div className="h-32 section-transition relative z-10" />

        {/* ── Stats Section ── */}
        <section className="px-12 py-24 relative z-10">
          <div className="absolute inset-0 section-glow-cyan pointer-events-none" />
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-20">
              <div className="space-y-4 border-l border-outline-variant/15 pl-8">
                <div className="text-5xl font-headline font-black text-on-surface tracking-tighter">
                  $<AnimatedCounter target={10} suffix="B+" />
                </div>
                <p className="text-on-surface-variant uppercase tracking-widest text-xs font-label">Lost Annually to Deepfakes</p>
              </div>
              <div className="space-y-4 border-l border-outline-variant/15 pl-8">
                <div className="text-5xl font-headline font-black text-secondary tracking-tighter">
                  <AnimatedCounter target={98} suffix="%" />
                </div>
                <p className="text-on-surface-variant uppercase tracking-widest text-xs font-label">Increase in Voice Cloning Fraud</p>
              </div>
              <div className="space-y-4 border-l border-outline-variant/15 pl-8">
                <div className="text-5xl font-headline font-black text-primary-container tracking-tighter">0.3s</div>
                <p className="text-on-surface-variant uppercase tracking-widest text-xs font-label">Real-time Detection Latency</p>
              </div>
            </div>
          </div>
        </section>

        {/* ── Gradient transition ── */}
        <div className="h-24 section-transition relative z-10" />

        {/* ── Features Bento Grid ── */}
        <section id="features" className="px-12 py-24 relative z-10">
          <div className="absolute inset-0 section-glow-emerald pointer-events-none" />
          <div className="max-w-7xl mx-auto space-y-16">
            <div className="text-center space-y-4">
              <h2 className="text-4xl md:text-5xl font-headline font-extrabold text-on-surface">The Sentinel Ecosystem</h2>
              <p className="text-on-surface-variant max-w-2xl mx-auto">Sophisticated protection layers designed to outpace synthetic identity theft and neural network manipulation.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
              {/* Voice Cloning Detection — Large */}
              <div className="md:col-span-8 bg-surface-container-low/60 backdrop-blur-sm p-12 rounded-3xl group relative overflow-hidden border border-outline-variant/10 hover:border-primary/15 transition-all duration-500">
                <div className="relative z-10 space-y-6">
                  <MaterialIcon icon="record_voice_over" filled size={48} className="text-secondary" />
                  <h3 className="text-3xl font-headline font-bold text-on-surface">Voice Cloning Detection</h3>
                  <p className="text-on-surface-variant max-w-md">Our 7-layer neural acoustic analyzer detects synthetic artifacts in audio streams with laboratory-grade precision, neutralizing impersonation scams instantly.</p>
                </div>
                <div className="absolute -bottom-20 -right-20 w-80 h-80 bg-secondary/10 rounded-full blur-[100px] group-hover:bg-secondary/20 transition-all" />
              </div>

              {/* Visual Verifier */}
              <div className="md:col-span-4 bg-surface-container/60 backdrop-blur-sm p-12 rounded-3xl border border-outline-variant/10 hover:border-primary/15 flex flex-col justify-between transition-all duration-500">
                <MaterialIcon icon="visibility" size={48} className="text-primary-container" />
                <div className="space-y-4 mt-8">
                  <h3 className="text-2xl font-headline font-bold text-on-surface">Visual Verifier</h3>
                  <p className="text-sm text-on-surface-variant">Frame-by-frame forensic analysis of images and videos to identify AI-generated content.</p>
                </div>
              </div>

              {/* Digital Signature */}
              <div className="md:col-span-4 bg-surface-container-high/60 backdrop-blur-sm p-12 rounded-3xl border border-outline-variant/10 hover:border-primary/15 transition-all duration-500">
                <MaterialIcon icon="verified_user" filled size={48} className="text-primary" />
                <div className="mt-8 space-y-4">
                  <h3 className="text-2xl font-headline font-bold text-on-surface">Voice Prints</h3>
                  <p className="text-sm text-on-surface-variant">Biometric voice enrollment for your family — verify caller identity in real-time.</p>
                </div>
              </div>

              {/* Real-time Scanning — Large */}
              <div className="md:col-span-8 bg-surface-container-low/60 backdrop-blur-sm p-12 rounded-3xl relative overflow-hidden group border border-outline-variant/10 hover:border-primary/15 transition-all duration-500">
                <div className="relative z-10 space-y-6">
                  <h3 className="text-3xl font-headline font-bold text-on-surface">Real-time Call Protection</h3>
                  <p className="text-on-surface-variant max-w-sm">Live monitoring that analyzes incoming calls for voice cloning, coercion patterns, and identity mismatch — all in under 3 seconds.</p>
                  <div className="flex gap-4">
                    <span className="px-4 py-2 bg-secondary/10 text-secondary rounded-[0.75rem] text-xs font-bold uppercase tracking-widest">Always On</span>
                    <span className="px-4 py-2 bg-primary/10 text-primary rounded-[0.75rem] text-xs font-bold uppercase tracking-widest">7-Layer Detection</span>
                  </div>
                </div>
                <div className="absolute -bottom-20 -right-20 w-80 h-80 bg-primary/10 rounded-full blur-[100px] group-hover:bg-primary/20 transition-all" />
              </div>
            </div>
          </div>
        </section>

        {/* ── Why Trust Satya Drishti ── */}
        <section className="px-12 py-32 relative">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/3 to-transparent pointer-events-none" />
          <div className="max-w-7xl mx-auto relative z-10">
            <div className="text-center space-y-6 mb-20">
              <h2 className="text-4xl md:text-5xl font-headline font-extrabold text-on-surface">Built for Indian Families</h2>
              <p className="text-on-surface-variant text-lg max-w-2xl mx-auto">Protecting families from voice cloning scams, digital kidnapping threats, and AI-powered coercion. Multilingual support for Hindi, Marathi, and English.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                { icon: "lock", title: "Privacy First", desc: "All analysis is performed locally. Audio is processed in real-time and immediately discarded — we never store your calls." },
                { icon: "speed", title: "Sub-Second Detection", desc: "Our 9-layer neural engine analyzes voice patterns in under 0.3 seconds, detecting synthetic artifacts before they reach you." },
                { icon: "translate", title: "Multilingual Safety", desc: "Full support for English, Hindi, and Marathi — ensuring every family member can use and understand the protection." },
              ].map((item) => (
                <div key={item.title} className="p-10 rounded-3xl bg-surface-container-low/60 backdrop-blur-sm border border-outline-variant/10 space-y-5 group hover:bg-surface-container-high/40 transition-all duration-300">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                    <MaterialIcon icon={item.icon} filled size={28} className="text-primary" />
                  </div>
                  <h3 className="text-2xl font-headline font-bold text-on-surface">{item.title}</h3>
                  <p className="text-on-surface-variant leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Final CTA ── */}
        <section className="px-12 py-40">
          <div className="max-w-5xl mx-auto bg-gradient-to-br from-surface-container-high to-surface-container-lowest rounded-[3rem] p-16 md:p-24 text-center space-y-12 relative overflow-hidden border border-outline-variant/10">
            <div className="absolute inset-0 bg-primary/5 blur-[120px] pointer-events-none" />
            <div className="relative z-10 space-y-6">
              <h2 className="text-5xl md:text-7xl font-headline font-extrabold text-on-surface tracking-tighter">
                Your Shield in the <br /><span className="text-secondary">Synthetic Age.</span>
              </h2>
              <p className="text-xl text-on-surface-variant max-w-2xl mx-auto font-body">
                Deploy Satya Drishti today and reclaim the truth. Protect your family from the rising tide of AI-powered fraud.
              </p>
            </div>
            <div className="relative z-10 flex flex-col md:flex-row justify-center items-center gap-6">
              <Link
                to={isAuthenticated ? "/call-protection" : "/login"}
                className="w-full md:w-auto px-12 py-6 bg-gradient-to-br from-primary to-primary-container text-on-primary font-headline font-extrabold text-lg uppercase tracking-tight rounded-2xl shadow-[0_20px_40px_rgba(0,209,255,0.2)] hover:scale-105 active:scale-95 transition-all"
              >
                Start Protecting Your Family
              </Link>
              <Link
                to="/scanner"
                className="text-on-surface-variant hover:text-on-surface font-headline font-bold uppercase tracking-widest text-sm underline decoration-primary underline-offset-8 transition-all"
              >
                Try the Scanner Free
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
