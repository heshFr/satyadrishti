import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { useAuth } from "@/contexts/AuthContext";
import { ApiError } from "@/lib/api";

const Register = () => {
  const { t } = useTranslation();
  const { register } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (password.length < 8) {
      setError(t("auth.passwordTooShort"));
      return;
    }
    if (password !== confirmPassword) {
      setError(t("auth.passwordMismatch"));
      return;
    }

    setLoading(true);
    try {
      await register(email, password, name);
      navigate("/scanner", { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(t("common.error"));
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface flex items-center justify-center px-4 relative overflow-hidden">
      {/* Background blobs */}
      <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
        <div className="absolute top-[15%] right-[10%] w-[500px] h-[500px] rounded-full bg-primary/8 blur-[140px]" />
        <div className="absolute bottom-[10%] left-[5%] w-[450px] h-[450px] rounded-full bg-secondary/8 blur-[120px]" />
        <div className="absolute top-[60%] right-[40%] w-[300px] h-[300px] rounded-full bg-primary-container/6 blur-[100px]" />
      </div>

      {/* Floating glass bubbles */}
      <motion.div
        className="absolute top-[12%] left-[8%] w-28 h-28 rounded-3xl bg-primary/5 backdrop-blur-md border border-primary/10 shadow-[0_8px_32px_rgba(0,209,255,0.08)]"
        animate={{ y: [0, -18, 0], rotate: [0, 6, 0] }}
        transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute top-[25%] right-[12%] w-20 h-20 rounded-2xl bg-secondary/5 backdrop-blur-md border border-secondary/10 shadow-[0_8px_32px_rgba(78,222,163,0.08)]"
        animate={{ y: [0, 14, 0], rotate: [0, -8, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
      />
      <motion.div
        className="absolute bottom-[15%] right-[18%] w-24 h-24 rounded-[1.5rem] bg-primary-container/5 backdrop-blur-md border border-primary-container/10 shadow-[0_8px_32px_rgba(0,148,196,0.08)]"
        animate={{ y: [0, -12, 0], rotate: [0, 4, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 2 }}
      />
      <motion.div
        className="absolute bottom-[30%] left-[12%] w-16 h-16 rounded-2xl bg-secondary/4 backdrop-blur-md border border-secondary/8 shadow-[0_8px_32px_rgba(78,222,163,0.06)]"
        animate={{ y: [0, 10, 0], rotate: [0, -5, 0] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
      />
      <motion.div
        className="absolute top-[55%] left-[25%] w-14 h-14 rounded-xl bg-primary/3 backdrop-blur-md border border-primary/8"
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 3 }}
      />

      <motion.div
        className="w-full max-w-md relative z-10"
        initial={{ opacity: 0, y: 30, filter: "blur(4px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 0.5 }}
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2.5 cursor-pointer">
            <motion.div
              animate={{ y: [0, -4, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            >
              <MaterialIcon icon="security" size={40} filled className="text-primary" />
            </motion.div>
            <span className="font-headline text-2xl font-extrabold tracking-tighter text-on-surface">
              SATYA DRISHTI<span className="text-primary-container">.</span>
            </span>
          </Link>
        </div>

        {/* Glass Card */}
        <div className="bg-surface-container-low/60 backdrop-blur-xl rounded-3xl border border-outline-variant/15 p-8 shadow-[0_8px_60px_rgba(0,0,0,0.25),0_0_0_1px_rgba(255,255,255,0.03)_inset]">
          <h1 className="font-headline text-2xl font-extrabold tracking-tighter text-on-surface mb-1">
            {t("auth.registerTitle")}
          </h1>
          <p className="text-on-surface-variant text-sm mb-8">{t("auth.registerSubtitle")}</p>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm text-on-surface-variant mb-2 font-medium">
                {t("auth.nameLabel")}
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="w-full bg-surface-container-high/50 backdrop-blur-sm border border-outline-variant/10 focus:ring-1 focus:ring-primary focus:border-primary/30 text-on-surface rounded-2xl px-4 py-3.5 placeholder:text-on-surface-variant/30 transition-all focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-on-surface-variant mb-2 font-medium">
                {t("auth.emailLabel")}
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full bg-surface-container-high/50 backdrop-blur-sm border border-outline-variant/10 focus:ring-1 focus:ring-primary focus:border-primary/30 text-on-surface rounded-2xl px-4 py-3.5 placeholder:text-on-surface-variant/30 transition-all font-mono text-sm focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-on-surface-variant mb-2 font-medium">
                {t("auth.passwordLabel")}
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full bg-surface-container-high/50 backdrop-blur-sm border border-outline-variant/10 focus:ring-1 focus:ring-primary focus:border-primary/30 text-on-surface rounded-2xl px-4 py-3.5 placeholder:text-on-surface-variant/30 transition-all pr-12 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant/60 hover:text-primary transition-colors cursor-pointer"
                >
                  <MaterialIcon
                    icon={showPassword ? "visibility_off" : "visibility"}
                    size={20}
                  />
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm text-on-surface-variant mb-2 font-medium">
                {t("auth.confirmPasswordLabel")}
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  className="w-full bg-surface-container-high/50 backdrop-blur-sm border border-outline-variant/10 focus:ring-1 focus:ring-primary focus:border-primary/30 text-on-surface rounded-2xl px-4 py-3.5 placeholder:text-on-surface-variant/30 transition-all pr-12 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant/60 hover:text-primary transition-colors cursor-pointer"
                >
                  <MaterialIcon
                    icon={showConfirmPassword ? "visibility_off" : "visibility"}
                    size={20}
                  />
                </button>
              </div>
            </div>

            {error && (
              <motion.p
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-error text-sm bg-error/10 border border-error/20 rounded-xl px-3 py-2"
              >
                {error}
              </motion.p>
            )}

            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className="w-full py-4 rounded-2xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-base uppercase tracking-widest overflow-hidden relative transition-all shadow-[0_4px_25px_rgba(0,209,255,0.25)] hover:shadow-[0_4px_35px_rgba(0,209,255,0.4)] disabled:opacity-50 cursor-pointer"
            >
              <span className="relative z-10">
                {loading ? t("common.loading") : t("common.register")}
              </span>
            </motion.button>
          </form>

          <div className="mt-8 text-center space-y-3">
            <p className="text-sm text-on-surface-variant">
              {t("auth.alreadyHaveAccount")}{" "}
              <Link
                to="/login"
                className="text-primary hover:text-primary/80 font-medium transition-colors cursor-pointer"
              >
                {t("common.login")}
              </Link>
            </p>
            <Link
              to="/hub"
              className="block text-sm text-on-surface-variant/50 hover:text-primary transition-colors cursor-pointer"
            >
              {t("common.continueWithout")} →
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Register;
