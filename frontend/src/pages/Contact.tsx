import { useState } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { toast } from "sonner";
import Layout from "@/components/Layout";
import { useAuth } from "@/contexts/AuthContext";
import emailjs from "@emailjs/browser";

// ── EmailJS Configuration ──
// Sign up at https://emailjs.com and fill in these values:
const EMAILJS_SERVICE_ID = import.meta.env.VITE_EMAILJS_SERVICE_ID || "service_satyadrishti";
const EMAILJS_TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID || "template_contact";
const EMAILJS_PUBLIC_KEY = import.meta.env.VITE_EMAILJS_PUBLIC_KEY || "";

const subjectKeys = ["reportDeepfake", "technicalSupport", "accountHelp", "partnership", "other"] as const;

const Contact = () => {
  const { t } = useTranslation();
  const { user } = useAuth();

  const [name, setName] = useState(user?.name || "");
  const [email, setEmail] = useState(user?.email || "");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Send email via EmailJS (directly from browser to eksatyadrishti@gmail.com)
      await emailjs.send(
        EMAILJS_SERVICE_ID,
        EMAILJS_TEMPLATE_ID,
        {
          from_name: name,
          from_email: email,
          subject: subject,
          message: message,
          to_email: "eksatyadrishti@gmail.com",
        },
        EMAILJS_PUBLIC_KEY,
      );
      setSubmitted(true);
      toast.success(t("contact.success"));
    } catch (err) {
      console.error("[Contact] EmailJS error:", err);
      toast.error("Failed to send message. Please try again or email us directly at eksatyadrishti@gmail.com");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setMessage("");
    setSubject("");
    if (!user) {
      setName("");
      setEmail("");
    }
    setSubmitted(false);
  };

  return (
    <Layout systemStatus="protected">
      <div className="pt-32 pb-20 px-6 md:px-12 max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>

          {/* Header */}
          <div className="mb-12">
            <h1 className="font-headline text-4xl md:text-5xl font-extrabold tracking-tighter text-on-surface mb-4">{t("contact.title")}</h1>
            <p className="text-on-surface-variant text-lg max-w-2xl">{t("contact.subtitle")}</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
            {/* Form Column */}
            <div className="lg:col-span-2 space-y-8">

              {/* How to reach us guide */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 }}
                className="bg-surface-container-low rounded-2xl border border-primary/10 p-8"
              >
                <div className="flex items-center gap-4 mb-5">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-2.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="help" size={20} className="text-primary" />
                    </div>
                  </div>
                  <h3 className="text-lg font-headline font-bold text-on-surface">{t("contact.stepsTitle")}</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  {[t("contact.step1"), t("contact.step2"), t("contact.step3")].map((step, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-primary to-primary-container flex items-center justify-center text-sm font-headline font-bold text-on-primary-container">
                        {i + 1}
                      </div>
                      <p className="text-base text-on-surface-variant/70">{step}</p>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Form / Success State */}
              <AnimatePresence mode="wait">
                {submitted ? (
                  <motion.div
                    key="success"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-12 text-center"
                  >
                    <div className="w-16 h-16 rounded-full bg-secondary/10 border border-secondary/20 flex items-center justify-center mx-auto mb-5">
                      <MaterialIcon icon="check_circle" size={32} className="text-secondary" />
                    </div>
                    <h3 className="text-2xl font-headline font-bold text-on-surface mb-3">Message Sent</h3>
                    <p className="text-on-surface-variant/60 text-base mb-8 max-w-md mx-auto">
                      {t("contact.success")}
                    </p>
                    <button
                      onClick={resetForm}
                      className="cursor-pointer inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-bold uppercase tracking-widest text-base transition-all"
                    >
                      <MaterialIcon icon="chat" size={16} /> Send another message
                    </button>
                  </motion.div>
                ) : (
                  <motion.form
                    key="form"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ delay: 0.1 }}
                    onSubmit={handleSubmit}
                    className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-8 space-y-6"
                  >
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                      <div>
                        <label className="text-sm font-label uppercase tracking-widest text-on-surface-variant block mb-2.5">
                          {t("contact.formName")}
                        </label>
                        <input
                          type="text"
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          required
                          className="w-full bg-surface-container-high border-none focus:ring-1 focus:ring-primary text-on-surface text-base rounded-lg px-5 py-4 placeholder:text-on-surface-variant/30 transition-all"
                          placeholder="Your full name"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-label uppercase tracking-widest text-on-surface-variant block mb-2.5">
                          {t("contact.formEmail")}
                        </label>
                        <input
                          type="email"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          required
                          className="w-full bg-surface-container-high border-none focus:ring-1 focus:ring-primary text-on-surface text-base rounded-lg px-5 py-4 placeholder:text-on-surface-variant/30 transition-all"
                          placeholder="you@example.com"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-label uppercase tracking-widest text-on-surface-variant block mb-2.5">
                        {t("contact.formSubject")}
                      </label>
                      <select
                        value={subject}
                        onChange={(e) => setSubject(e.target.value)}
                        required
                        className="cursor-pointer w-full bg-surface-container-high border-none focus:ring-1 focus:ring-primary text-on-surface text-base rounded-lg px-5 py-4 transition-all"
                      >
                        <option value="" disabled>
                          {t("contact.selectSubject")}
                        </option>
                        {subjectKeys.map((key) => (
                          <option key={key} value={key} className="bg-surface-container-low text-on-surface">
                            {t(`contact.subjects.${key}`)}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-label uppercase tracking-widest text-on-surface-variant">
                          {t("contact.formMessage")}
                        </label>
                        <span className={`text-sm font-mono ${message.length > 1800 ? "text-primary" : "text-on-surface-variant/30"}`}>
                          {message.length}/2000
                        </span>
                      </div>
                      <textarea
                        value={message}
                        onChange={(e) => setMessage(e.target.value.slice(0, 2000))}
                        required
                        rows={6}
                        className="w-full bg-surface-container-high border-none focus:ring-1 focus:ring-primary text-on-surface text-base rounded-lg px-5 py-4 placeholder:text-on-surface-variant/30 transition-all resize-none"
                        placeholder="Describe your issue or question in detail..."
                      />
                    </div>
                    <motion.button
                      type="submit"
                      disabled={loading}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      className="cursor-pointer inline-flex items-center gap-3 px-10 py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-bold text-base uppercase tracking-widest transition-all disabled:opacity-50"
                    >
                      <MaterialIcon icon="send" size={18} />
                      {loading ? t("common.loading") : t("contact.formSubmit")}
                    </motion.button>
                  </motion.form>
                )}
              </AnimatePresence>
            </div>

            {/* Sidebar */}
            <div className="space-y-5">
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-surface-container-low rounded-2xl p-7 border-error/10 border relative overflow-hidden"
              >
                <div className="absolute -top-10 -right-10 w-24 h-24 bg-error/10 blur-[40px] rounded-full" />
                <div className="flex items-center gap-4 mb-4 relative z-10">
                  <div className="bg-gradient-to-br from-error to-error/60 p-[1px] rounded-lg">
                    <div className="p-2.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="warning" size={22} className="text-error" />
                    </div>
                  </div>
                  <p className="text-on-surface font-headline font-bold text-base">Emergency</p>
                </div>
                <p className="text-5xl font-headline font-extrabold text-error relative z-10">112</p>
                <p className="text-on-surface-variant/60 text-sm mt-2 relative z-10">{t("contact.emergencyNote")}</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-surface-container-low rounded-2xl p-7 border-primary/10 border relative overflow-hidden"
              >
                <div className="absolute -top-10 -right-10 w-24 h-24 bg-primary/10 blur-[40px] rounded-full" />
                <div className="flex items-center gap-4 mb-4 relative z-10">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-2.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="phone" size={22} className="text-primary" />
                    </div>
                  </div>
                  <p className="text-on-surface font-headline font-bold text-base">{t("help.emergency")}</p>
                </div>
                <p className="text-5xl font-headline font-extrabold text-primary relative z-10">1930</p>
                <p className="text-on-surface-variant/60 text-sm mt-2 relative z-10">National Cyber Crime Helpline</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-7"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-2.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="mail" size={22} className="text-primary/60" />
                    </div>
                  </div>
                  <p className="text-on-surface font-headline font-bold text-base">{t("contact.emailLabel")}</p>
                </div>
                <p className="text-primary text-base font-mono font-medium">eksatyadrishti@gmail.com</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 }}
                className="bg-surface-container-low rounded-2xl p-7 border-secondary/10 border relative overflow-hidden"
              >
                <div className="absolute -top-10 -right-10 w-24 h-24 bg-secondary/10 blur-[40px] rounded-full" />
                <div className="flex items-center gap-4 mb-4 relative z-10">
                  <div className="bg-gradient-to-br from-secondary to-primary p-[1px] rounded-lg">
                    <div className="p-2.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="language" size={22} className="text-secondary" />
                    </div>
                  </div>
                  <p className="text-on-surface font-headline font-bold text-base">Online Portal</p>
                </div>
                <a
                  href="https://cybercrime.gov.in"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="cursor-pointer text-secondary hover:text-secondary/80 text-base font-medium transition-colors flex items-center gap-1.5 relative z-10"
                >
                  cybercrime.gov.in <MaterialIcon icon="open_in_new" size={16} />
                </a>
                <p className="text-on-surface-variant/50 text-sm mt-2 relative z-10">File complaints online 24/7</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-surface-container-low rounded-2xl border border-outline-variant/10 p-7"
              >
                <div className="flex items-center gap-4 mb-3">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-2.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="schedule" size={22} className="text-on-surface-variant/40" />
                    </div>
                  </div>
                  <p className="text-on-surface font-headline font-bold text-base">Response Time</p>
                </div>
                <p className="text-on-surface-variant/60 text-base">{t("contact.responseTime")}</p>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>
    </Layout>
  );
};

export default Contact;
