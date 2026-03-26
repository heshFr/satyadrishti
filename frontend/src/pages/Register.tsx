import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { useAuth } from "@/contexts/AuthContext";
import { ApiError } from "@/lib/api";

/* ── Password Requirements Checker ── */
const PasswordRequirements = ({ password }: { password: string }) => {
  const requirements = [
    { label: "At least 6 characters", met: password.length >= 6 },
    { label: "One uppercase letter", met: /[A-Z]/.test(password) },
    { label: "One number", met: /\d/.test(password) },
  ];

  if (!password) return null;

  return (
    <div className="space-y-1.5 mt-2">
      {requirements.map((req) => (
        <div key={req.label} className="flex items-center gap-2">
          <MaterialIcon
            icon={req.met ? "check_circle" : "cancel"}
            size={14}
            className={req.met ? "text-secondary" : "text-on-surface-variant/30"}
          />
          <span className={`text-[10px] uppercase tracking-wider font-bold ${req.met ? "text-secondary" : "text-on-surface-variant/40"}`}>
            {req.label}
          </span>
        </div>
      ))}
    </div>
  );
};

/* ── Floating Particles ── */
const FloatingParticles = () => {
  const particles = Array.from({ length: 15 }, (_, i) => ({
    id: i, x: Math.random() * 100, y: Math.random() * 100,
    size: Math.random() * 3 + 1, duration: Math.random() * 15 + 10, delay: Math.random() * 5,
  }));
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {particles.map((p) => (
        <motion.div key={p.id} className="absolute rounded-full bg-primary/20"
          style={{ width: p.size, height: p.size, left: `${p.x}%`, top: `${p.y}%` }}
          animate={{ y: [0, -40, 0], x: [0, Math.random() * 20 - 10, 0], opacity: [0, 0.6, 0] }}
          transition={{ duration: p.duration, repeat: Infinity, delay: p.delay, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
};

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

  const [currentTime, setCurrentTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const timeStr = currentTime.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const [sessionId] = useState(() => `SD-${Math.floor(Math.random() * 9000 + 1000)}-REG`);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!name.trim()) { setError("Full name is required"); return; }
    if (password.length < 6) { setError("Password must be at least 6 characters"); return; }
    if (!/[A-Z]/.test(password)) { setError("Password must contain at least one uppercase letter"); return; }
    if (!/\d/.test(password)) { setError("Password must contain at least one number"); return; }
    if (password !== confirmPassword) { setError(t("auth.passwordMismatch")); return; }

    setLoading(true);
    try {
      await register(email, password, name);
      navigate("/call-protection", { replace: true });
    } catch (err) {
      if (err instanceof ApiError) { setError(err.message); }
      else { setError(t("common.error")); }
    } finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col lg:flex-row relative overflow-hidden">

      {/* ═══ LEFT PANEL — Cinematic Hero (mirrors Login.tsx) ═══ */}
      <div className="hidden lg:flex lg:w-[55%] xl:w-[60%] relative flex-col justify-between p-12 xl:p-16 overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <img src="/bg-deepfake-2.png" alt="" className="w-full h-full object-cover opacity-[0.7]" />
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-surface/70 via-surface/50 to-surface/30" />
        <div className="absolute top-0 right-0 w-[600px] h-[600px] rounded-full bg-primary/5 blur-[150px]" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] rounded-full bg-secondary/5 blur-[120px]" />
        <FloatingParticles />
        <div className="absolute inset-0 opacity-[0.02]" style={{
          backgroundImage: `linear-gradient(rgba(0,209,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,209,255,0.3) 1px, transparent 1px)`,
          backgroundSize: "80px 80px",
        }} />

        <div className="relative z-10 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group cursor-pointer">
            <motion.div animate={{ y: [0, -3, 0] }} transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}>
              <MaterialIcon icon="security" size={32} filled className="text-primary drop-shadow-[0_0_10px_rgba(0,209,255,0.3)]" />
            </motion.div>
            <span className="font-headline text-lg font-extrabold tracking-tighter text-on-surface">
              SATYA DRISHTI<span className="text-primary-container">.</span>
            </span>
          </Link>
          <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-high/60 backdrop-blur-sm rounded-full border border-outline-variant/10">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-secondary shadow-[0_0_6px_rgba(78,222,163,0.6)]" />
            </span>
            <span className="text-[10px] font-label uppercase tracking-[0.15em] text-on-surface-variant font-bold">All Systems Operational</span>
          </div>
        </div>

        <div className="relative z-10 flex-1 flex flex-col justify-center max-w-xl">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }} className="space-y-8">
            <div className="flex items-center gap-3">
              <div className="h-[1px] w-12 bg-gradient-to-r from-primary to-transparent" />
              <span className="text-[10px] font-label uppercase tracking-[0.3em] text-primary font-bold">Identity Initialization</span>
            </div>
            <h1 className="font-headline text-5xl xl:text-6xl font-extrabold tracking-tighter text-on-surface leading-[1.05]">
              Join the Digital<br />Shield Network<span className="text-primary-container">.</span>
            </h1>
            <p className="text-on-surface-variant text-base xl:text-lg leading-relaxed max-w-md font-light">
              Create your Satya Drishti identity to unlock full protection — real-time call monitoring, voice print enrollment, and AI-powered threat detection for your family.
            </p>
            <div className="flex flex-wrap gap-3 pt-2">
              {[
                { icon: "mic", label: "Voice Shield" },
                { icon: "image_search", label: "Media Forensics" },
                { icon: "shield", label: "AI Protection" },
              ].map((pill) => (
                <div key={pill.label} className="flex items-center gap-2 px-4 py-2 bg-surface-container-high/40 rounded-full border border-outline-variant/10">
                  <MaterialIcon icon={pill.icon} size={14} className="text-primary" />
                  <span className="text-[10px] font-label uppercase tracking-wider text-on-surface-variant font-bold">{pill.label}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        <div className="relative z-10 flex items-center justify-between text-[9px] font-mono text-outline uppercase tracking-wider">
          <span>Session: {sessionId}</span>
          <span>{timeStr} UTC+5:30</span>
          <span>Encryption: AES-256-GCM</span>
        </div>
      </div>

      {/* ═══ RIGHT PANEL — Sign Up Form ═══ */}
      <div className="flex-1 lg:w-[45%] xl:w-[40%] flex items-center justify-center px-6 py-12 lg:px-12 xl:px-16 relative">
        <div className="absolute inset-0 lg:hidden">
          <div className="absolute top-1/4 right-0 w-[300px] h-[300px] rounded-full bg-primary/5 blur-[100px]" />
          <div className="absolute bottom-0 left-0 w-[250px] h-[250px] rounded-full bg-secondary/5 blur-[80px]" />
        </div>
        <div className="absolute inset-0 bg-surface-container-lowest/50 hidden lg:block" />

        <motion.div className="w-full max-w-md relative z-10" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.1 }}>
          {/* Mobile Logo */}
          <div className="lg:hidden text-center mb-10">
            <Link to="/" className="inline-flex items-center gap-2.5 cursor-pointer">
              <MaterialIcon icon="security" size={36} filled className="text-primary" />
              <span className="font-headline text-xl font-extrabold tracking-tighter text-on-surface">
                SATYA DRISHTI<span className="text-primary-container">.</span>
              </span>
            </Link>
          </div>

          <div className="bg-surface-container-low/80 backdrop-blur-xl rounded-2xl p-8 md:p-10 border border-outline-variant/10 relative overflow-hidden">
            <div className="absolute -top-20 -right-20 w-40 h-40 bg-primary/5 rounded-full blur-[60px]" />

            <div className="flex justify-center mb-6">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
                className="relative w-24 h-24 flex items-center justify-center"
              >
                <motion.div className="absolute inset-0 rounded-full border-2 border-primary/30"
                  animate={{ scale: [1, 1.15, 1], opacity: [0.4, 0, 0.4] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                />
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-primary-container/10 backdrop-blur-sm flex items-center justify-center border border-primary/25 shadow-[0_0_40px_rgba(0,209,255,0.15)]">
                  <MaterialIcon icon="person_add" size={36} filled className="text-primary drop-shadow-[0_0_12px_rgba(0,209,255,0.5)]" />
                </div>
              </motion.div>
            </div>

            <div className="text-center mb-8">
              <h2 className="font-headline text-2xl font-extrabold tracking-tighter text-on-surface">Create Account</h2>
              <p className="text-xs text-on-surface-variant mt-2 font-light">Initialize your secure perimeter credentials.</p>
            </div>

            {/* Tab Switcher */}
            <div className="flex mb-8 bg-surface-container-high/50 rounded-lg p-1">
              <Link to="/login" className="flex-1 py-2.5 text-[10px] font-label font-bold uppercase tracking-[0.2em] rounded-md transition-all duration-300 text-center text-on-surface-variant hover:text-on-surface cursor-pointer">
                Login
              </Link>
              <div className="flex-1 py-2.5 text-[10px] font-label font-bold uppercase tracking-[0.2em] rounded-md bg-surface-container-highest text-on-surface shadow-sm text-center">
                Sign Up
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Name */}
              <div className="space-y-2">
                <label className="block text-[10px] font-label uppercase tracking-[0.2em] text-on-surface-variant font-bold px-1">Full Name</label>
                <div className="relative group">
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10"><MaterialIcon icon="person" size={18} className="text-outline" /></div>
                  <input type="text" value={name} onChange={(e) => setName(e.target.value)} required placeholder="Your full name"
                    className="w-full bg-transparent border-b-2 border-outline-variant/30 focus:border-primary focus:outline-none text-on-surface py-3 pl-7 placeholder:text-outline-variant/40 transition-all duration-300 font-body text-sm" />
                </div>
              </div>

              {/* Email */}
              <div className="space-y-2">
                <label className="block text-[10px] font-label uppercase tracking-[0.2em] text-on-surface-variant font-bold px-1">Email Address</label>
                <div className="relative group">
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10"><MaterialIcon icon="mail" size={18} className="text-outline" /></div>
                  <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="you@example.com"
                    className="w-full bg-transparent border-b-2 border-outline-variant/30 focus:border-primary focus:outline-none text-on-surface py-3 pl-7 placeholder:text-outline-variant/40 transition-all duration-300 font-body text-sm" />
                </div>
              </div>

              {/* Password */}
              <div className="space-y-2">
                <label className="block text-[10px] font-label uppercase tracking-[0.2em] text-on-surface-variant font-bold px-1">Password</label>
                <div className="relative group">
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10"><MaterialIcon icon="lock" size={18} className="text-outline" /></div>
                  <input type={showPassword ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)} required placeholder="••••••••••"
                    className="w-full bg-transparent border-b-2 border-outline-variant/30 focus:border-primary focus:outline-none text-on-surface py-3 pl-7 pr-10 placeholder:text-outline-variant/40 transition-all duration-300 font-body text-sm" />
                  <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-0 top-1/2 -translate-y-1/2 text-outline hover:text-primary transition-colors cursor-pointer p-1">
                    <MaterialIcon icon={showPassword ? "visibility_off" : "visibility"} size={18} />
                  </button>
                </div>
                <PasswordRequirements password={password} />
              </div>

              {/* Confirm Password */}
              <div className="space-y-2">
                <label className="block text-[10px] font-label uppercase tracking-[0.2em] text-on-surface-variant font-bold px-1">Confirm Password</label>
                <div className="relative group">
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10"><MaterialIcon icon="lock" size={18} className="text-outline" /></div>
                  <input type={showConfirmPassword ? "text" : "password"} value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required placeholder="••••••••••"
                    className="w-full bg-transparent border-b-2 border-outline-variant/30 focus:border-primary focus:outline-none text-on-surface py-3 pl-7 pr-10 placeholder:text-outline-variant/40 transition-all duration-300 font-body text-sm" />
                  <button type="button" onClick={() => setShowConfirmPassword(!showConfirmPassword)} className="absolute right-0 top-1/2 -translate-y-1/2 text-outline hover:text-primary transition-colors cursor-pointer p-1">
                    <MaterialIcon icon={showConfirmPassword ? "visibility_off" : "visibility"} size={18} />
                  </button>
                </div>
              </div>

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div initial={{ opacity: 0, y: -5, height: 0 }} animate={{ opacity: 1, y: 0, height: "auto" }} exit={{ opacity: 0, y: -5, height: 0 }}
                    className="text-error text-xs bg-error-container/10 border border-error/20 rounded-lg px-4 py-3 flex items-center gap-2">
                    <MaterialIcon icon="error" size={14} className="text-error shrink-0" />{error}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit */}
              <motion.button type="submit" disabled={loading || !name || !email || !password || !confirmPassword}
                whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}
                className="w-full py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm uppercase tracking-[0.2em] transition-all duration-300 disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,209,255,0.15)] hover:shadow-[0_0_30px_rgba(0,209,255,0.25)]">
                {loading ? (
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
                    Creating Account...
                  </div>
                ) : "Sign Up"}
              </motion.button>
            </form>

            {/* Footer Links */}
            <div className="mt-6 text-center space-y-3">
              <p className="text-xs text-on-surface-variant">
                Already have an account?{" "}
                <Link to="/login" className="text-primary hover:text-primary-container transition-colors cursor-pointer font-bold">Login</Link>
              </p>
              <Link to="/" className="block text-[10px] text-on-surface-variant/40 hover:text-primary transition-colors cursor-pointer uppercase tracking-wider font-bold">
                ← Back to Home
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Register;
