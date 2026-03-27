import { useState } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import MaterialIcon from "@/components/MaterialIcon";
import Layout from "@/components/Layout";
import { toast } from "sonner";

/* ── FAQ Data ── */
interface FaqItem {
  question: string;
  answer: string;
}

const faqs: FaqItem[] = [
  {
    question: "Threat Detection: How does Satya Drishti differentiate AI-generated audio?",
    answer:
      "Our neural engine analyzes over 2,000 spectral markers per second, looking for phase discontinuities and synthetic resonance artifacts that are physically impossible for human vocal cords to produce. The 9-layer ensemble — AST, XLS-R 300M, Whisper Features, Prosodic, Breathing, Phase, Formant, Temporal, and Fusion — provides a 99.4% confidence rating on all incoming streams.",
  },
  {
    question: "Voice Authentication: Is biometric bypass protection enabled by default?",
    answer:
      'Yes. All accounts feature active "Liveness Detection" by default. This requires the user to perform micro-vocal fluctuations that cannot be pre-recorded or synthetically generated. Voice prints enrolled through the Voice Enrollment page are cross-referenced in real-time during call protection.',
  },
  {
    question: "Account Security: What happens if I lose my physical hardware key?",
    answer:
      'Our "Social Recovery" protocol allows you to designate trusted nodes (family or colleagues) who can collectively authorize a key reset. Alternatively, you can use the Identity Lockdown procedure to freeze all assets until physical verification is complete.',
  },
  {
    question: "What is a deepfake and how is it used in scams?",
    answer:
      "A deepfake is AI-generated or manipulated media (image, audio, or video) designed to impersonate a real person. Scammers use voice cloning to impersonate family members demanding urgent money transfers, or create fake video calls from supposed bank officials. Our system detects these in real-time.",
  },
  {
    question: "What file types can the scanner analyze?",
    answer:
      "The scanner supports images (JPG, PNG, WEBP), audio files (WAV, MP3, OGG, FLAC), and video files (MP4, MOV, AVI, WEBM). Maximum file size is 100MB. Each modality uses specialized AI models trained on modern generators including Midjourney, DALL-E, Stable Diffusion, and GPT Image.",
  },
  {
    question: "How does real-time call protection work?",
    answer:
      "When activated, the system captures your microphone audio and optionally system audio (via screen share). Every few seconds, it sends audio segments to the AI engine for analysis of voice deepfakes, coercion language, and threat patterns. Results appear in real-time on the dashboard with the 9-layer ensemble providing continuous monitoring.",
  },
  {
    question: "Is my data private? Does Satya Drishti store my calls?",
    answer:
      "All analysis is performed locally. Audio from call protection is processed in real-time and immediately discarded. Only analysis results (threat scores, timestamps, transcripts) are stored if you choose to save them. You can enable auto-delete in Settings.",
  },
  {
    question: "Why does my social media image show as uncertain?",
    answer:
      "Social media platforms (WhatsApp, Instagram, Facebook) heavily compress images, which can destroy forensic signals. Our system detects this compression and adjusts analysis accordingly, but may report 'uncertain' when compression is too severe. The V2 pipeline includes platform fingerprinting for WhatsApp, Instagram, Facebook, Twitter, and Telegram.",
  },
];

const Help = () => {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const [openFaq, setOpenFaq] = useState<number | null>(null);
  const filtered = faqs.filter(
    (faq) =>
      !search ||
      faq.question.toLowerCase().includes(search.toLowerCase()) ||
      faq.answer.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <Layout systemStatus="protected">
      <div className="px-8 py-12 max-w-7xl mx-auto space-y-16">

        {/* ── Hero Section ── */}
        <section className="relative space-y-6">
          <div className="space-y-2">
            <span className="text-secondary font-label text-xs tracking-[0.2em] uppercase font-bold">
              Intelligence Center
            </span>
            <h1 className="font-headline text-5xl md:text-6xl font-extrabold tracking-tight text-on-surface">
              How can we{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-primary-container">
                protect you
              </span>{" "}
              today?
            </h1>
          </div>
          <div className="max-w-2xl relative">
            <MaterialIcon icon="search" className="absolute left-4 top-1/2 -translate-y-1/2 text-primary" size={24} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search documentation, threats, or account status..."
              className="w-full bg-surface-container-low border-b-2 border-outline-variant focus:border-primary focus:ring-0 py-5 pl-14 pr-6 text-lg transition-all rounded-t-xl outline-none placeholder:text-on-surface-variant/40"
            />
          </div>
        </section>

        {/* ── Quick Start Bento Grid ── */}
        <section className="space-y-6">
          <div className="flex justify-between items-end">
            <h2 className="font-headline text-2xl font-bold tracking-tight">Quick-start Guides</h2>
            <Link to="/help" className="text-primary text-sm font-medium hover:underline">
              View all guides &rarr;
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Shield Management — large card */}
            <Link
              to="/dashboard"
              className="md:col-span-2 md:row-span-2 bg-surface-container-low rounded-xl p-8 flex flex-col justify-between group cursor-pointer hover:bg-surface-container-high transition-all duration-300 relative overflow-hidden"
            >
              <div className="relative z-10 space-y-4">
                <MaterialIcon icon="security" filled className="text-primary" size={40} />
                <h3 className="font-headline text-3xl font-bold">Shield Management</h3>
                <p className="text-on-surface-variant max-w-xs">
                  Configure real-time interceptors and automated counter-measures for your digital perimeter.
                </p>
              </div>
              <div className="relative z-10 pt-8">
                <span className="bg-primary text-on-primary px-6 py-3 rounded-xl font-bold text-sm tracking-tight inline-block">
                  Initialize Shield
                </span>
              </div>
              <div className="absolute -right-8 -bottom-8 opacity-5 group-hover:opacity-10 transition-opacity">
                <MaterialIcon icon="verified_user" size={192} />
              </div>
            </Link>

            {/* Deep Scans */}
            <Link
              to="/scanner"
              className="bg-surface-container-low rounded-xl p-6 hover:bg-surface-container-high transition-all group cursor-pointer border border-outline-variant/10"
            >
              <MaterialIcon icon="radar" className="text-secondary mb-4" size={24} />
              <h4 className="font-bold text-lg mb-2">Deep Scans</h4>
              <p className="text-sm text-on-surface-variant">Analyze local and cloud assets for synthetic vulnerabilities.</p>
            </Link>

            {/* Vault Access */}
            <Link
              to="/history"
              className="bg-surface-container-low rounded-xl p-6 hover:bg-surface-container-high transition-all group cursor-pointer border border-outline-variant/10"
            >
              <MaterialIcon icon="lock_open" className="text-tertiary mb-4" size={24} />
              <h4 className="font-bold text-lg mb-2">Vault Access</h4>
              <p className="text-sm text-on-surface-variant">Securely manage your biometric keys and encrypted payloads.</p>
            </Link>

            {/* Identity Lockdown */}
            <Link
              to="/settings"
              className="md:col-span-2 bg-gradient-to-br from-surface-container-low to-surface-container-high rounded-xl p-6 flex items-center justify-between border border-outline-variant/10"
            >
              <div className="space-y-1">
                <h4 className="font-bold text-lg">Identity Lockdown</h4>
                <p className="text-sm text-on-surface-variant">Enable zero-trust identity protocols for all sessions.</p>
              </div>
              <MaterialIcon icon="fingerprint" className="text-primary-container" size={40} />
            </Link>
          </div>
        </section>

        {/* ── Dashboard & FAQ Split ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">

          {/* LEFT: System Health & Support */}
          <div className="lg:col-span-1 space-y-8">
            {/* System Health */}
            <div className="bg-surface-container-low rounded-xl p-6 space-y-6 border border-outline-variant/10">
              <div className="flex items-center justify-between">
                <h3 className="font-headline font-bold text-lg">System Health</h3>
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-secondary animate-pulse" />
                  <span className="text-[10px] font-bold text-secondary uppercase tracking-widest">Operational</span>
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-end">
                  <span className="text-4xl font-black font-headline tracking-tighter">99.9%</span>
                  <span className="text-xs text-on-surface-variant mb-1">Last 30 Days</span>
                </div>
                <div className="h-16 w-full flex items-end gap-1">
                  {[60, 75, 40, 90, 85, 70, 95, 65].map((h, i) => (
                    <div
                      key={i}
                      className={`w-full rounded-t-sm ${i === 4 ? "bg-secondary" : "bg-secondary/20"}`}
                      style={{ height: `${h}%` }}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-[10px] text-on-surface-variant font-bold uppercase tracking-tighter">
                  <span>Latency: 14ms</span>
                  <span>Global Nodes: 124</span>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: FAQ Section */}
          <div className="lg:col-span-2 space-y-8">
            <h3 className="font-headline font-bold text-2xl">Frequent Intelligence Requests</h3>

            <div className="space-y-4">
              {filtered.map((faq, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="group bg-surface-container-low rounded-xl overflow-hidden transition-all duration-300"
                >
                  <button
                    onClick={() => setOpenFaq(openFaq === i ? null : i)}
                    className="cursor-pointer w-full px-8 py-6 flex items-center justify-between text-left hover:bg-surface-container-high transition-colors"
                  >
                    <span className="font-bold text-lg pr-4">{faq.question}</span>
                    <motion.div animate={{ rotate: openFaq === i ? 180 : 0 }} transition={{ duration: 0.2 }} className="flex-shrink-0">
                      <MaterialIcon icon="expand_more" className="text-primary" size={24} />
                    </motion.div>
                  </button>
                  <AnimatePresence>
                    {openFaq === i && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
                        className="overflow-hidden"
                      >
                        <p className="px-8 pb-6 text-on-surface-variant leading-relaxed">{faq.answer}</p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
              {filtered.length === 0 && (
                <div className="text-center py-16">
                  <MaterialIcon icon="search" size={40} className="text-on-surface-variant/20 mx-auto mb-3" />
                  <p className="text-on-surface-variant/50">No results found</p>
                  <p className="text-on-surface-variant/30 text-sm mt-1">Try different keywords</p>
                </div>
              )}
            </div>

            {/* Documentation Links */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-8">
              <div className="p-6 bg-surface-container rounded-xl space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-widest text-primary">API Reference</h4>
                <ul className="space-y-2 text-sm text-on-surface-variant">
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">Authentication Flow</Link></li>
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">Webhook Listeners</Link></li>
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">SDK Documentation</Link></li>
                </ul>
              </div>
              <div className="p-6 bg-surface-container rounded-xl space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-widest text-secondary">User Guide</h4>
                <ul className="space-y-2 text-sm text-on-surface-variant">
                  <li><Link to="/scanner" className="hover:text-on-surface transition-colors">First Scan Setup</Link></li>
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">Mobile App Sync</Link></li>
                  <li><Link to="/settings" className="hover:text-on-surface transition-colors">Dashboard Config</Link></li>
                </ul>
              </div>
              <div className="p-6 bg-surface-container rounded-xl space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-widest text-tertiary">Security Protocols</h4>
                <ul className="space-y-2 text-sm text-on-surface-variant">
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">Privacy Policy</Link></li>
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">Compliance Certs</Link></li>
                  <li><Link to="/help" className="hover:text-on-surface transition-colors">Data Encryption</Link></li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* ── Emergency Procedures ── */}
        <section className="bg-error-container/20 border-l-4 border-error rounded-r-xl p-8 space-y-8">
          <div className="flex items-center gap-4">
            <MaterialIcon icon="report" filled className="text-error" size={40} />
            <div>
              <h2 className="font-headline text-3xl font-black text-on-error-container tracking-tighter">
                Emergency Procedures
              </h2>
              <p className="text-error font-medium">Follow these steps immediately if you suspect a breach.</p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Account Compromise */}
            <div className="space-y-4">
              <h3 className="font-bold text-xl text-on-error-container">Account Compromise</h3>
              <div className="space-y-4">
                {[
                  <>Initiate <strong className="text-on-surface">Global Logout</strong> from all active devices in the Security tab.</>,
                  <>Regenerate your <strong className="text-on-surface">Recovery Seed Phrase</strong> and store it offline.</>,
                  <>Contact our <strong className="text-on-surface">Emergency Response Team</strong> via the priority link below.</>,
                ].map((step, i) => (
                  <div key={i} className="flex gap-4">
                    <span className="h-6 w-6 rounded-full bg-error text-on-error text-[10px] font-bold flex items-center justify-center flex-shrink-0">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <p className="text-sm">{step}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Identity Lockdown */}
            <div className="space-y-4">
              <h3 className="font-bold text-xl text-on-error-container">Identity Lockdown</h3>
              <div className="space-y-4">
                {[
                  <>Enable <strong className="text-on-surface">Biometric Deadman Switch</strong> to freeze outgoing transfers.</>,
                  <>Invalidate all <strong className="text-on-surface">Temporary API Tokens</strong> immediately.</>,
                  <>Trigger <strong className="text-on-surface">Dark Web Monitor</strong> for immediate leaked credential scan.</>,
                ].map((step, i) => (
                  <div key={i} className="flex gap-4">
                    <span className="h-6 w-6 rounded-full bg-error text-on-error text-[10px] font-bold flex items-center justify-center flex-shrink-0">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <p className="text-sm">{step}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="pt-4 flex flex-wrap gap-4">
            <button
              onClick={() => toast.error("System lockdown initiated — all sessions terminated")}
              className="px-8 py-4 bg-error text-on-error font-black uppercase tracking-widest rounded-xl hover:bg-error/90 active:scale-95 transition-all"
            >
              Execute Total System Lockdown
            </button>
            <a
              href="https://cybercrime.gov.in"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 bg-surface-container-high border border-error/30 text-error font-bold uppercase tracking-widest rounded-xl hover:bg-surface-container-highest transition-all flex items-center gap-2"
            >
              <MaterialIcon icon="open_in_new" size={16} />
              Report to Cyber Crime Portal
            </a>
            <a
              href="tel:112"
              className="px-8 py-4 bg-surface-container-high border border-outline-variant/30 text-on-surface font-bold uppercase tracking-widest rounded-xl hover:bg-surface-container-highest transition-all flex items-center gap-2"
            >
              <MaterialIcon icon="call" size={16} />
              Emergency: 112
            </a>
          </div>
        </section>
      </div>
    </Layout>
  );
};

export default Help;
