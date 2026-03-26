import { useState, useEffect, useRef, useCallback } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { useAuth } from "@/contexts/AuthContext";
import { api, ApiError } from "@/lib/api";
import { toast } from "sonner";

// ═════════════════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═════════════════════════════════════════════════════════════════════════

// ─── Animated Shield Logo ──────────────────────────────────────────────
const AnimatedShield = ({ mode }: { mode: AuthMode }) => {
  const icon = mode === "twofa" ? "verified_user" : mode === "register" ? "person_add" : mode === "forgot" ? "lock_reset" : mode === "reset" ? "password" : "security";
  return (
    <motion.div
      key={mode}
      initial={{ scale: 0.8, opacity: 0, rotateY: 90 }}
      animate={{ scale: 1, opacity: 1, rotateY: 0 }}
      exit={{ scale: 0.8, opacity: 0, rotateY: -90 }}
      transition={{ type: "spring", stiffness: 200, damping: 20 }}
      className="relative w-24 h-24 md:w-28 md:h-28 flex items-center justify-center"
    >
      {/* Outer pulsing ring */}
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-primary/30"
        animate={{ scale: [1, 1.15, 1], opacity: [0.4, 0, 0.4] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
      />
      {/* Second ring */}
      <motion.div
        className="absolute inset-2 rounded-full border border-primary/20"
        animate={{ scale: [1, 1.1, 1], opacity: [0.3, 0, 0.3] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
      />
      {/* Core icon container */}
      <div className="w-16 h-16 md:w-20 md:h-20 rounded-full bg-gradient-to-br from-primary/20 to-primary-container/10 backdrop-blur-sm flex items-center justify-center border border-primary/25 shadow-[0_0_40px_rgba(0,209,255,0.15)]">
        <MaterialIcon icon={icon} size={36} filled className="text-primary drop-shadow-[0_0_12px_rgba(0,209,255,0.5)]" />
      </div>
    </motion.div>
  );
};

// ─── Floating Particles Background ─────────────────────────────────────
const FloatingParticles = () => {
  const particles = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 3 + 1,
    duration: Math.random() * 15 + 10,
    delay: Math.random() * 5,
  }));

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full bg-primary/20"
          style={{ width: p.size, height: p.size, left: `${p.x}%`, top: `${p.y}%` }}
          animate={{ y: [0, -40, 0], x: [0, Math.random() * 20 - 10, 0], opacity: [0, 0.6, 0] }}
          transition={{ duration: p.duration, repeat: Infinity, delay: p.delay, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
};

// ─── Underline Input Component ─────────────────────────────────────────
const UnderlineInput = ({
  label, type = "text", value, onChange, placeholder, icon,
  required = false, disabled = false, autoComplete, rightElement,
}: {
  label: string; type?: string; value: string; onChange: (v: string) => void;
  placeholder?: string; icon?: string; required?: boolean; disabled?: boolean;
  autoComplete?: string; rightElement?: React.ReactNode;
}) => {
  const [focused, setFocused] = useState(false);
  return (
    <div className="space-y-2">
      <label className="block text-[10px] font-label uppercase tracking-[0.2em] text-on-surface-variant font-bold px-1">
        {label}
      </label>
      <div className="relative group">
        {icon && (
          <div className="absolute left-0 top-1/2 -translate-y-1/2 z-10">
            <MaterialIcon icon={icon} size={18} className={`transition-colors duration-300 ${focused ? "text-primary" : "text-outline"}`} />
          </div>
        )}
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          required={required}
          disabled={disabled}
          autoComplete={autoComplete}
          className={`w-full bg-transparent border-b-2 ${focused ? "border-primary" : "border-outline-variant/30"} focus:outline-none text-on-surface py-3 placeholder:text-outline-variant/40 transition-all duration-300 font-body text-sm ${icon ? "pl-7" : ""} ${rightElement ? "pr-10" : ""} ${disabled ? "opacity-40" : ""}`}
        />
        {rightElement && (
          <div className="absolute right-0 top-1/2 -translate-y-1/2">{rightElement}</div>
        )}
        {/* Focus glow line */}
        <div className={`absolute bottom-0 left-1/2 -translate-x-1/2 h-[2px] bg-primary transition-all duration-500 ${focused ? "w-full shadow-[0_0_8px_rgba(0,209,255,0.4)]" : "w-0"}`} />
      </div>
    </div>
  );
};

// ─── OTP Input Component (for 2FA verification) ───────────────────────
const OtpInput = ({ value, onChange, autoFocus }: { value: string; onChange: (v: string) => void; autoFocus?: boolean }) => {
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
  const digits = value.padEnd(6, " ").split("").slice(0, 6);

  useEffect(() => {
    if (autoFocus) inputRefs.current[0]?.focus();
  }, [autoFocus]);

  const handleDigit = (index: number, digit: string) => {
    if (!/^\d?$/.test(digit)) return;
    const newVal = digits.map((d, i) => (i === index ? (digit || " ") : d)).join("");
    onChange(newVal.replace(/ /g, ""));
    if (digit && index < 5) inputRefs.current[index + 1]?.focus();
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && digits[index] === " " && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (pasted) {
      onChange(pasted);
      const focusIdx = Math.min(pasted.length, 5);
      inputRefs.current[focusIdx]?.focus();
    }
  };

  return (
    <div className="flex gap-3 justify-center" onPaste={handlePaste}>
      {digits.map((d, i) => (
        <input
          key={i}
          ref={(el) => { inputRefs.current[i] = el; }}
          type="text"
          inputMode="numeric"
          maxLength={1}
          value={d === " " ? "" : d}
          onChange={(e) => handleDigit(i, e.target.value)}
          onKeyDown={(e) => handleKeyDown(i, e)}
          className="w-12 h-14 md:w-14 md:h-16 text-center text-xl font-headline font-bold text-on-surface bg-surface-container-high rounded-lg border border-outline-variant/20 focus:border-primary focus:ring-1 focus:ring-primary/30 focus:outline-none transition-all"
        />
      ))}
    </div>
  );
};

// ─── System Status Indicator ───────────────────────────────────────────
const SystemStatusBadge = () => (
  <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-high/60 backdrop-blur-sm rounded-full border border-outline-variant/10">
    <span className="relative flex h-2 w-2">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75" />
      <span className="relative inline-flex rounded-full h-2 w-2 bg-secondary shadow-[0_0_6px_rgba(78,222,163,0.6)]" />
    </span>
    <span className="text-[10px] font-label uppercase tracking-[0.15em] text-on-surface-variant font-bold">
      All Systems Operational
    </span>
  </div>
);

// ─── OAuth Social Login Button ─────────────────────────────────────────
const SocialButton = ({
  provider, label, loading, onClick,
}: {
  provider: "google" | "github"; label: string; loading: boolean; onClick: () => void;
}) => {
  const icon = provider === "google" ? "g_mobiledata" : "code";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-lg bg-surface-container-high/40 border border-outline-variant/15 hover:border-primary/30 hover:bg-surface-container-highest/50 transition-all duration-300 cursor-pointer group disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {loading ? (
        <div className="w-4 h-4 border-2 border-on-surface-variant/30 border-t-primary rounded-full animate-spin" />
      ) : (
        <>
          <MaterialIcon icon={icon} size={18} className="text-on-surface-variant group-hover:text-primary transition-colors" />
          <span className="text-xs font-label font-bold text-on-surface-variant group-hover:text-on-surface transition-colors uppercase tracking-wider">
            {label}
          </span>
        </>
      )}
    </button>
  );
};

// ─── Password Strength Meter ───────────────────────────────────────────
const PasswordStrengthMeter = ({ password }: { password: string }) => {
  if (!password) return null;

  // Calculate strength score
  let score = 0;
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score++;
  if (/\d/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;

  const level = score <= 1 ? "weak" : score <= 3 ? "moderate" : "strong";
  const colors = { weak: "bg-error", moderate: "bg-primary", strong: "bg-secondary" };
  const labels = { weak: "Weak", moderate: "Moderate", strong: "Strong" };

  return (
    <div className="space-y-1.5">
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className={`flex-1 h-1 rounded-full transition-colors duration-500 ${
              i <= score ? colors[level] : "bg-surface-container-highest"
            }`}
          />
        ))}
      </div>
      <div className="flex justify-between items-center">
        <p className="text-[9px] font-label text-on-surface-variant uppercase tracking-wider">
          {labels[level]} passphrase
        </p>
        <p className="text-[9px] font-mono text-outline">
          {password.length} chars
        </p>
      </div>
    </div>
  );
};


// ═════════════════════════════════════════════════════════════════════════
// MAIN LOGIN COMPONENT
// ═════════════════════════════════════════════════════════════════════════
type AuthMode = "login" | "register" | "twofa" | "forgot" | "reset";

const Login = () => {
  const { t } = useTranslation();
  const { login, register, startOAuth, verify2FA } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const redirect = searchParams.get("redirect") || "/call-protection";

  // ── Auth State ──
  const [mode, setMode] = useState<AuthMode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [rememberDevice, setRememberDevice] = useState(() => {
    try { return JSON.parse(localStorage.getItem("satya-remember-device") || "false"); }
    catch { return false; }
  });

  // ── 2FA State ──
  const [twoFARequired, setTwoFARequired] = useState(false);
  const [tempToken, setTempToken] = useState("");
  const [otpValue, setOtpValue] = useState("");
  const [twoFALoading, setTwoFALoading] = useState(false);

  // ── OAuth State ──
  const [googleLoading, setGoogleLoading] = useState(false);
  const [githubLoading, setGithubLoading] = useState(false);

  // ── Forgot Password State ──
  const [resetEmail, setResetEmail] = useState("");
  const [resetCode, setResetCode] = useState("");
  const [newResetPassword, setNewResetPassword] = useState("");
  const [confirmResetPassword, setConfirmResetPassword] = useState("");
  const [resetLoading, setResetLoading] = useState(false);
  const [devResetCode, setDevResetCode] = useState("");

  // ── Clock for hero section ──
  const [currentTime, setCurrentTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const timeStr = currentTime.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });

  // ── Session ID ──
  const [sessionId] = useState(() => `SD-${Math.floor(Math.random() * 9000 + 1000)}-AUTH`);

  // ── Remember device persistence ──
  useEffect(() => {
    localStorage.setItem("satya-remember-device", JSON.stringify(rememberDevice));
  }, [rememberDevice]);

  // ══════════════════════════════════════
  // AUTH HANDLERS
  // ══════════════════════════════════════

  // ── Login ──
  const handleLogin = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const result = await login(email, password);
      // Check if 2FA is required
      if (result.requires2FA && result.tempToken) {
        setTempToken(result.tempToken);
        setTwoFARequired(true);
        setMode("twofa");
        toast.info("Two-factor authentication required");
        setLoading(false);
        return;
      }
      // Store remember preference
      if (rememberDevice) {
        localStorage.setItem("satya-trusted-device", "true");
      }
      toast.success("Authentication successful");
      navigate(redirect, { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(t("common.error"));
      }
    } finally {
      setLoading(false);
    }
  }, [email, password, login, navigate, redirect, rememberDevice, t]);

  // ── Register ──
  const handleRegister = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Validation
    if (!fullName.trim()) {
      setError("Full name is required");
      return;
    }
    if (password.length < 8) {
      setError("Passphrase must be at least 8 characters");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passphrases do not match");
      return;
    }

    setLoading(true);
    try {
      await register(email, password, fullName.trim());
      toast.success("Identity initialized successfully");
      navigate(redirect, { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Registration failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }, [email, password, confirmPassword, fullName, register, navigate, redirect]);

  // ── 2FA Verify ──
  const handle2FAVerify = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (otpValue.length < 6) {
      setError("Please enter the full 6-digit code");
      return;
    }
    setError("");
    setTwoFALoading(true);
    try {
      await verify2FA(tempToken, otpValue);
      if (rememberDevice) {
        localStorage.setItem("satya-trusted-device", "true");
      }
      toast.success("Two-factor verification successful");
      navigate(redirect, { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Invalid verification code. Please try again.");
      }
      setOtpValue("");
    } finally {
      setTwoFALoading(false);
    }
  }, [otpValue, tempToken, verify2FA, navigate, redirect, rememberDevice]);

  // ── Google OAuth ──
  const handleGoogleLogin = useCallback(async () => {
    setGoogleLoading(true);
    setError("");
    try {
      await startOAuth("google");
      toast.success("Google authentication successful");
      navigate(redirect, { replace: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Google sign-in failed";
      if (msg.includes("not configured") || msg.includes("501")) {
        toast.info("Google OAuth requires server configuration. Use email sign-in instead.");
      } else {
        setError(msg);
        toast.error(msg);
      }
    } finally {
      setGoogleLoading(false);
    }
  }, [startOAuth, navigate, redirect]);

  // ── GitHub OAuth ──
  const handleGithubLogin = useCallback(async () => {
    setGithubLoading(true);
    setError("");
    try {
      await startOAuth("github");
      toast.success("GitHub authentication successful");
      navigate(redirect, { replace: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "GitHub sign-in failed";
      if (msg.includes("not configured") || msg.includes("501")) {
        toast.info("GitHub OAuth requires server configuration. Use email sign-in instead.");
      } else {
        setError(msg);
        toast.error(msg);
      }
    } finally {
      setGithubLoading(false);
    }
  }, [startOAuth, navigate, redirect]);

  // ── Forgot Password — Request Code ──
  const handleForgotPassword = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!resetEmail.trim()) { setError("Enter your email address"); return; }
    setResetLoading(true);
    try {
      const res = await api.auth.requestPasswordReset(resetEmail.trim());
      // Dev mode: server returns the code for testing
      if (res.code) {
        setDevResetCode(res.code);
      }
      toast.success("Recovery code generated");
      setMode("reset");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Failed to request password reset");
      }
    } finally {
      setResetLoading(false);
    }
  }, [resetEmail]);

  // ── Forgot Password — Confirm Reset ──
  const handleResetPassword = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (resetCode.length !== 6) { setError("Enter the full 6-digit recovery code"); return; }
    if (newResetPassword.length < 8) { setError("New password must be at least 8 characters"); return; }
    if (newResetPassword !== confirmResetPassword) { setError("Passwords do not match"); return; }
    setResetLoading(true);
    try {
      await api.auth.resetPassword(resetEmail.trim(), resetCode, newResetPassword);
      toast.success("Password reset successfully! You can now sign in.");
      setResetEmail(""); setResetCode(""); setNewResetPassword(""); setConfirmResetPassword(""); setDevResetCode("");
      setMode("login");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Failed to reset password");
      }
    } finally {
      setResetLoading(false);
    }
  }, [resetEmail, resetCode, newResetPassword, confirmResetPassword]);

  // ── Mode Switcher ──
  const switchMode = useCallback((newMode: AuthMode) => {
    setError("");
    setMode(newMode);
    setOtpValue("");
    setTwoFARequired(false);
    setTempToken("");
    if (newMode === "forgot") {
      setResetEmail(email); // Pre-fill from login form
      setResetCode(""); setNewResetPassword(""); setConfirmResetPassword(""); setDevResetCode("");
    }
  }, [email]);


  // ═════════════════════════════════════════════════════════════════════
  // RENDER — Split-Screen Stitch Layout
  // ═════════════════════════════════════════════════════════════════════
  return (
    <div className="min-h-screen bg-surface flex flex-col lg:flex-row relative overflow-hidden">

      {/* ═══ LEFT PANEL — Cinematic Hero ═══ */}
      <div className="hidden lg:flex lg:w-[55%] xl:w-[60%] relative flex-col justify-between p-12 xl:p-16 overflow-hidden">
        {/* Background — laptop dashboard covering full area */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <img src="/bg-deepfake-2.png" alt="" className="w-full h-full object-cover opacity-[0.7]" />
        </div>
        {/* Gradient overlay — text readability */}
        <div className="absolute inset-0 bg-gradient-to-r from-surface/70 via-surface/50 to-surface/30" />
        <div className="absolute top-0 right-0 w-[600px] h-[600px] rounded-full bg-primary/5 blur-[150px]" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] rounded-full bg-secondary/5 blur-[120px]" />
        <div className="absolute top-1/3 left-1/3 w-[400px] h-[400px] rounded-full bg-primary-container/3 blur-[100px]" />

        {/* Floating particles */}
        <FloatingParticles />

        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: `linear-gradient(rgba(0,209,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,209,255,0.3) 1px, transparent 1px)`,
            backgroundSize: "80px 80px",
          }}
        />

        {/* Top — Logo & System Status */}
        <div className="relative z-10 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group cursor-pointer">
            <motion.div
              animate={{ y: [0, -3, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            >
              <MaterialIcon icon="security" size={32} filled className="text-primary drop-shadow-[0_0_10px_rgba(0,209,255,0.3)]" />
            </motion.div>
            <span className="font-headline text-lg font-extrabold tracking-tighter text-on-surface">
              SATYA DRISHTI<span className="text-primary-container">.</span>
            </span>
          </Link>
          <SystemStatusBadge />
        </div>

        {/* Center — Hero Content */}
        <div className="relative z-10 flex-1 flex flex-col justify-center max-w-xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="space-y-8"
          >
            {/* Eyebrow */}
            <div className="flex items-center gap-3">
              <div className="h-[1px] w-12 bg-gradient-to-r from-primary to-transparent" />
              <span className="text-[10px] font-label uppercase tracking-[0.3em] text-primary font-bold">Sentinel Authentication</span>
            </div>

            {/* Main Headline */}
            <h1 className="font-headline text-5xl xl:text-6xl font-extrabold tracking-tighter text-on-surface leading-[1.05]">
              Protecting the<br />
              Digital Truth<span className="text-primary-container">.</span>
            </h1>

            {/* Subtext */}
            <p className="text-on-surface-variant text-base xl:text-lg leading-relaxed max-w-md font-light">
              Satya Drishti uses advanced neural scanning to verify authenticity in an age of synthetic deception. Enter the vault to access your data curation suite.
            </p>

            {/* Feature Pills */}
            <div className="flex flex-wrap gap-3 pt-2">
              {[
                { icon: "mic", label: "Voice Shield" },
                { icon: "image_search", label: "Media Forensics" },
                { icon: "shield", label: "AI Protection" },
              ].map((pill) => (
                <div
                  key={pill.label}
                  className="flex items-center gap-2 px-4 py-2 bg-surface-container-high/40 rounded-full border border-outline-variant/10"
                >
                  <MaterialIcon icon={pill.icon} size={14} className="text-primary" />
                  <span className="text-[10px] font-label uppercase tracking-wider text-on-surface-variant font-bold">{pill.label}</span>
                </div>
              ))}
            </div>

            {/* Security Trust Indicators */}
            <div className="flex items-center gap-6 pt-4">
              {[
                { value: "256-bit", label: "AES Encryption" },
                { value: "99.9%", label: "Uptime" },
                { value: "SOC 2", label: "Compliant" },
              ].map((stat) => (
                <div key={stat.label} className="space-y-0.5">
                  <p className="text-sm font-headline font-extrabold text-primary tracking-tight">{stat.value}</p>
                  <p className="text-[9px] font-label text-outline uppercase tracking-wider">{stat.label}</p>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Bottom — Technical Metadata */}
        <div className="relative z-10 flex items-center justify-between text-[9px] font-mono text-outline uppercase tracking-wider">
          <span>Session: {sessionId}</span>
          <span>{timeStr} UTC+5:30</span>
          <span>Encryption: AES-256-GCM</span>
        </div>
      </div>

      {/* ═══ RIGHT PANEL — Authentication Form ═══ */}
      <div className="flex-1 lg:w-[45%] xl:w-[40%] flex items-center justify-center px-6 py-12 lg:px-12 xl:px-16 relative">
        {/* Subtle mobile background */}
        <div className="absolute inset-0 lg:hidden">
          <div className="absolute top-1/4 right-0 w-[300px] h-[300px] rounded-full bg-primary/5 blur-[100px]" />
          <div className="absolute bottom-0 left-0 w-[250px] h-[250px] rounded-full bg-secondary/5 blur-[80px]" />
        </div>

        {/* Right panel tint */}
        <div className="absolute inset-0 bg-surface-container-lowest/50 hidden lg:block" />

        <motion.div
          className="w-full max-w-md relative z-10"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          {/* Mobile Logo */}
          <div className="lg:hidden text-center mb-10">
            <Link to="/" className="inline-flex items-center gap-2.5 cursor-pointer">
              <MaterialIcon icon="security" size={36} filled className="text-primary" />
              <span className="font-headline text-xl font-extrabold tracking-tighter text-on-surface">
                SATYA DRISHTI<span className="text-primary-container">.</span>
              </span>
            </Link>
          </div>

          {/* Glassmorphic Form Card */}
          <div className="bg-surface-container-low/80 backdrop-blur-xl rounded-2xl p-8 md:p-10 border border-outline-variant/10 relative overflow-hidden">
            {/* Card accent glow */}
            <div className="absolute -top-20 -right-20 w-40 h-40 bg-primary/5 rounded-full blur-[60px]" />

            {/* ═══ Animated Shield ═══ */}
            <div className="flex justify-center mb-6">
              <AnimatePresence mode="wait">
                <AnimatedShield mode={mode} />
              </AnimatePresence>
            </div>

            {/* ═══ Section Title ═══ */}
            <AnimatePresence mode="wait">
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3 }}
                className="text-center mb-8"
              >
                <h2 className="font-headline text-2xl font-extrabold tracking-tighter text-on-surface">
                  {mode === "login" && "Vault Entrance"}
                  {mode === "register" && "Initialize Identity"}
                  {mode === "twofa" && "Two-Factor Verification"}
                  {mode === "forgot" && "Account Recovery"}
                  {mode === "reset" && "Set New Password"}
                </h2>
                <p className="text-xs text-on-surface-variant mt-2 font-light">
                  {mode === "login" && "Identity verification required for access."}
                  {mode === "register" && "Create your secure perimeter credentials."}
                  {mode === "twofa" && "Enter the 6-digit code from your authenticator app."}
                  {mode === "forgot" && "Enter your email to receive a recovery code."}
                  {mode === "reset" && "Enter the 6-digit code and your new password."}
                </p>
              </motion.div>
            </AnimatePresence>

            {/* ═══ LOGIN / REGISTER TAB SWITCHER ═══ */}
            {(mode === "login" || mode === "register") && (
              <div className="flex mb-8 bg-surface-container-high/50 rounded-lg p-1">
                {(["login", "register"] as const).map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => switchMode(tab)}
                    className={`flex-1 py-2.5 text-[10px] font-label font-bold uppercase tracking-[0.2em] rounded-md transition-all duration-300 cursor-pointer ${
                      mode === tab
                        ? "bg-surface-container-highest text-on-surface shadow-sm"
                        : "text-on-surface-variant hover:text-on-surface"
                    }`}
                  >
                    {tab === "login" ? "Login" : "Sign Up"}
                  </button>
                ))}
              </div>
            )}

            {/* ═══ FORMS ═══ */}
            <AnimatePresence mode="wait">

              {/* ──────── LOGIN FORM ──────── */}
              {mode === "login" && (
                <motion.form
                  key="login"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                  onSubmit={handleLogin}
                  className="space-y-6"
                >
                  <UnderlineInput
                    label="Email Address"
                    type="email"
                    value={email}
                    onChange={setEmail}
                    placeholder="sentinel@satyaDrishti.in"
                    icon="mail"
                    required
                    autoComplete="email"
                  />
                  <UnderlineInput
                    label="Passphrase"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={setPassword}
                    placeholder="••••••••••"
                    icon="lock"
                    required
                    autoComplete="current-password"
                    rightElement={
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="text-outline hover:text-primary transition-colors cursor-pointer p-1"
                      >
                        <MaterialIcon icon={showPassword ? "visibility_off" : "visibility"} size={18} />
                      </button>
                    }
                  />

                  {/* Remember Device + Forgot Password */}
                  <div className="flex items-center justify-between pt-1">
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={rememberDevice}
                        onChange={(e) => setRememberDevice(e.target.checked)}
                        className="w-4 h-4 rounded border-outline-variant/30 bg-surface-container-high text-primary focus:ring-primary/30 cursor-pointer"
                      />
                      <span className="text-[10px] font-label text-on-surface-variant group-hover:text-on-surface transition-colors uppercase tracking-wider">
                        Trust this device
                      </span>
                    </label>
                    <button
                      type="button"
                      onClick={() => switchMode("forgot")}
                      className="text-[10px] font-label text-primary hover:text-primary-container transition-colors cursor-pointer uppercase tracking-wider font-bold"
                    >
                      Forgot Password?
                    </button>
                  </div>

                  {/* Error */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -5, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: "auto" }}
                        exit={{ opacity: 0, y: -5, height: 0 }}
                        className="text-error text-xs bg-error-container/10 border border-error/20 rounded-lg px-4 py-3 flex items-center gap-2"
                      >
                        <MaterialIcon icon="error" size={14} className="text-error shrink-0" />
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Submit Button */}
                  <motion.button
                    type="submit"
                    disabled={loading || !email || !password}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm uppercase tracking-[0.2em] transition-all duration-300 disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,209,255,0.15)] hover:shadow-[0_0_30px_rgba(0,209,255,0.25)]"
                  >
                    {loading ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
                        Authenticating...
                      </span>
                    ) : (
                      "Secure Sign In"
                    )}
                  </motion.button>

                  {/* Divider */}
                  <div className="flex items-center gap-4 py-2">
                    <div className="flex-1 h-[1px] bg-outline-variant/15" />
                    <span className="text-[9px] font-label uppercase tracking-[0.2em] text-outline">Or continue with</span>
                    <div className="flex-1 h-[1px] bg-outline-variant/15" />
                  </div>

                  {/* Social Login — Google & GitHub */}
                  <div className="flex gap-3">
                    <SocialButton
                      provider="google"
                      label="Google"
                      loading={googleLoading}
                      onClick={handleGoogleLogin}
                    />
                    <SocialButton
                      provider="github"
                      label="GitHub"
                      loading={githubLoading}
                      onClick={handleGithubLogin}
                    />
                  </div>
                </motion.form>
              )}

              {/* ──────── REGISTER FORM ──────── */}
              {mode === "register" && (
                <motion.form
                  key="register"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                  onSubmit={handleRegister}
                  className="space-y-5"
                >
                  <UnderlineInput
                    label="Full Name"
                    value={fullName}
                    onChange={setFullName}
                    placeholder="Elena Richards"
                    icon="person"
                    required
                    autoComplete="name"
                  />
                  <UnderlineInput
                    label="Email Address"
                    type="email"
                    value={email}
                    onChange={setEmail}
                    placeholder="sentinel@satyaDrishti.in"
                    icon="mail"
                    required
                    autoComplete="email"
                  />
                  <UnderlineInput
                    label="Create Passphrase"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={setPassword}
                    placeholder="Min. 8 characters"
                    icon="lock"
                    required
                    autoComplete="new-password"
                    rightElement={
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="text-outline hover:text-primary transition-colors cursor-pointer p-1"
                      >
                        <MaterialIcon icon={showPassword ? "visibility_off" : "visibility"} size={18} />
                      </button>
                    }
                  />
                  <UnderlineInput
                    label="Confirm Passphrase"
                    type="password"
                    value={confirmPassword}
                    onChange={setConfirmPassword}
                    placeholder="Re-enter passphrase"
                    icon="lock"
                    required
                    autoComplete="new-password"
                  />

                  {/* Password strength meter */}
                  <PasswordStrengthMeter password={password} />

                  {/* Error */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -5, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: "auto" }}
                        exit={{ opacity: 0, y: -5, height: 0 }}
                        className="text-error text-xs bg-error-container/10 border border-error/20 rounded-lg px-4 py-3 flex items-center gap-2"
                      >
                        <MaterialIcon icon="error" size={14} className="text-error shrink-0" />
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <motion.button
                    type="submit"
                    disabled={loading || !email || !password || !fullName}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm uppercase tracking-[0.2em] transition-all duration-300 disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,209,255,0.15)] hover:shadow-[0_0_30px_rgba(0,209,255,0.25)]"
                  >
                    {loading ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
                        Creating Identity...
                      </span>
                    ) : (
                      "Initialize Secure Account"
                    )}
                  </motion.button>

                  {/* Divider */}
                  <div className="flex items-center gap-4 py-1">
                    <div className="flex-1 h-[1px] bg-outline-variant/15" />
                    <span className="text-[9px] font-label uppercase tracking-[0.2em] text-outline">Or sign up with</span>
                    <div className="flex-1 h-[1px] bg-outline-variant/15" />
                  </div>

                  {/* Social Register */}
                  <div className="flex gap-3">
                    <SocialButton
                      provider="google"
                      label="Google"
                      loading={googleLoading}
                      onClick={handleGoogleLogin}
                    />
                    <SocialButton
                      provider="github"
                      label="GitHub"
                      loading={githubLoading}
                      onClick={handleGithubLogin}
                    />
                  </div>
                </motion.form>
              )}

              {/* ──────── 2FA VERIFICATION ──────── */}
              {mode === "twofa" && (
                <motion.form
                  key="twofa"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                  onSubmit={handle2FAVerify}
                  className="space-y-8"
                >
                  {/* Visual indicator */}
                  <div className="text-center space-y-3">
                    <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                      <MaterialIcon icon="phonelink_lock" size={28} className="text-primary" />
                    </div>
                    <p className="text-xs text-on-surface-variant">
                      Open your authenticator app and enter the<br />
                      <span className="text-on-surface font-bold">6-digit verification code</span>
                    </p>
                  </div>

                  {/* OTP Input */}
                  <OtpInput value={otpValue} onChange={setOtpValue} autoFocus />

                  {/* Timer / Info */}
                  <div className="flex items-center justify-center gap-2 text-[10px] text-on-surface-variant font-label uppercase tracking-wider">
                    <MaterialIcon icon="schedule" size={12} className="text-outline" />
                    <span>Code refreshes every 30 seconds</span>
                  </div>

                  {/* Error */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -5, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: "auto" }}
                        exit={{ opacity: 0, y: -5, height: 0 }}
                        className="text-error text-xs bg-error-container/10 border border-error/20 rounded-lg px-4 py-3 flex items-center gap-2"
                      >
                        <MaterialIcon icon="error" size={14} className="text-error shrink-0" />
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Verify Button */}
                  <motion.button
                    type="submit"
                    disabled={twoFALoading || otpValue.length < 6}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm uppercase tracking-[0.2em] transition-all duration-300 disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,209,255,0.15)]"
                  >
                    {twoFALoading ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
                        Verifying...
                      </span>
                    ) : (
                      "Verify & Authenticate"
                    )}
                  </motion.button>

                  {/* Back to login */}
                  <button
                    type="button"
                    onClick={() => switchMode("login")}
                    className="w-full flex items-center justify-center gap-2 py-3 text-on-surface-variant hover:text-primary transition-colors cursor-pointer"
                  >
                    <MaterialIcon icon="arrow_back" size={16} />
                    <span className="text-xs font-label font-bold uppercase tracking-wider">Back to Login</span>
                  </button>

                  {/* Backup code option */}
                  <div className="text-center">
                    <button
                      type="button"
                      className="text-[10px] text-outline hover:text-on-surface-variant transition-colors cursor-pointer font-label uppercase tracking-wider"
                    >
                      Use a backup code instead
                    </button>
                  </div>
                </motion.form>
              )}

              {/* ──────── FORGOT PASSWORD — Enter Email ──────── */}
              {mode === "forgot" && (
                <motion.form
                  key="forgot"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                  onSubmit={handleForgotPassword}
                  className="space-y-6"
                >
                  <div className="text-center space-y-3">
                    <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                      <MaterialIcon icon="mail" size={28} className="text-primary" />
                    </div>
                    <p className="text-xs text-on-surface-variant">
                      Enter the email address associated with your account.<br />
                      <span className="text-on-surface font-bold">We'll generate a 6-digit recovery code.</span>
                    </p>
                  </div>

                  <UnderlineInput
                    label="Email Address"
                    type="email"
                    value={resetEmail}
                    onChange={setResetEmail}
                    placeholder="sentinel@satyaDrishti.in"
                    icon="mail"
                    required
                    autoComplete="email"
                  />

                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -5, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: "auto" }}
                        exit={{ opacity: 0, y: -5, height: 0 }}
                        className="text-error text-xs bg-error-container/10 border border-error/20 rounded-lg px-4 py-3 flex items-center gap-2"
                      >
                        <MaterialIcon icon="error" size={14} className="text-error shrink-0" />
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <motion.button
                    type="submit"
                    disabled={resetLoading || !resetEmail}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm uppercase tracking-[0.2em] transition-all duration-300 disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,209,255,0.15)]"
                  >
                    {resetLoading ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
                        Generating Code...
                      </span>
                    ) : (
                      "Send Recovery Code"
                    )}
                  </motion.button>

                  <button
                    type="button"
                    onClick={() => switchMode("login")}
                    className="w-full flex items-center justify-center gap-2 py-3 text-on-surface-variant hover:text-primary transition-colors cursor-pointer"
                  >
                    <MaterialIcon icon="arrow_back" size={16} />
                    <span className="text-xs font-label font-bold uppercase tracking-wider">Back to Sign In</span>
                  </button>
                </motion.form>
              )}

              {/* ──────── RESET PASSWORD — Enter Code + New Password ──────── */}
              {mode === "reset" && (
                <motion.form
                  key="reset"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.3 }}
                  onSubmit={handleResetPassword}
                  className="space-y-6"
                >
                  {/* Dev mode: show the code */}
                  {devResetCode && (
                    <div className="p-4 rounded-xl bg-secondary/10 border border-secondary/20 text-center space-y-2">
                      <div className="flex items-center justify-center gap-2">
                        <MaterialIcon icon="science" size={16} className="text-secondary" />
                        <span className="text-[10px] font-label uppercase tracking-wider text-secondary font-bold">Dev Mode — Recovery Code</span>
                      </div>
                      <p className="text-2xl font-headline font-extrabold tracking-[0.3em] text-secondary">
                        {devResetCode}
                      </p>
                      <p className="text-[9px] text-on-surface-variant">
                        In production, this would be sent to <span className="font-bold">{resetEmail}</span> via email.
                      </p>
                    </div>
                  )}

                  <div className="text-center space-y-1">
                    <p className="text-xs text-on-surface-variant">
                      Enter the <span className="text-on-surface font-bold">6-digit code</span> and your new password
                    </p>
                  </div>

                  {/* OTP input reuse for the 6-digit code */}
                  <OtpInput value={resetCode} onChange={setResetCode} autoFocus />

                  <div className="flex items-center justify-center gap-2 text-[10px] text-on-surface-variant font-label uppercase tracking-wider">
                    <MaterialIcon icon="schedule" size={12} className="text-outline" />
                    <span>Code expires in 15 minutes</span>
                  </div>

                  <UnderlineInput
                    label="New Password"
                    type={showPassword ? "text" : "password"}
                    value={newResetPassword}
                    onChange={setNewResetPassword}
                    placeholder="Min. 8 characters"
                    icon="lock"
                    required
                    autoComplete="new-password"
                    rightElement={
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="text-outline hover:text-primary transition-colors cursor-pointer p-1"
                      >
                        <MaterialIcon icon={showPassword ? "visibility_off" : "visibility"} size={18} />
                      </button>
                    }
                  />

                  <UnderlineInput
                    label="Confirm New Password"
                    type="password"
                    value={confirmResetPassword}
                    onChange={setConfirmResetPassword}
                    placeholder="Re-enter password"
                    icon="lock"
                    required
                    autoComplete="new-password"
                  />

                  <PasswordStrengthMeter password={newResetPassword} />

                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -5, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: "auto" }}
                        exit={{ opacity: 0, y: -5, height: 0 }}
                        className="text-error text-xs bg-error-container/10 border border-error/20 rounded-lg px-4 py-3 flex items-center gap-2"
                      >
                        <MaterialIcon icon="error" size={14} className="text-error shrink-0" />
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <motion.button
                    type="submit"
                    disabled={resetLoading || resetCode.length < 6 || !newResetPassword}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm uppercase tracking-[0.2em] transition-all duration-300 disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,209,255,0.15)]"
                  >
                    {resetLoading ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-on-primary-container/30 border-t-on-primary-container rounded-full animate-spin" />
                        Resetting Password...
                      </span>
                    ) : (
                      "Reset Password"
                    )}
                  </motion.button>

                  <div className="flex items-center justify-center gap-4">
                    <button
                      type="button"
                      onClick={() => setMode("forgot")}
                      className="text-[10px] text-outline hover:text-on-surface-variant transition-colors cursor-pointer font-label uppercase tracking-wider"
                    >
                      Resend code
                    </button>
                    <span className="text-outline">|</span>
                    <button
                      type="button"
                      onClick={() => switchMode("login")}
                      className="text-[10px] text-outline hover:text-on-surface-variant transition-colors cursor-pointer font-label uppercase tracking-wider"
                    >
                      Back to Sign In
                    </button>
                  </div>
                </motion.form>
              )}
            </AnimatePresence>

            {/* ═══ "Continue Without Account" ═══ */}
            {(mode === "login" || mode === "register") && (
              <div className="mt-6 text-center">
                <Link
                  to="/scanner"
                  className="text-[10px] text-outline hover:text-primary transition-colors cursor-pointer font-label uppercase tracking-[0.2em]"
                >
                  Continue without account →
                </Link>
              </div>
            )}
          </div>

          {/* ═══ Footer Links ═══ */}
          <div className="mt-8 flex items-center justify-center gap-6 flex-wrap">
            {["Privacy Policy", "Security Protocol", "Terms of Service"].map((link) => (
              <button
                key={link}
                type="button"
                className="text-[9px] font-label uppercase tracking-[0.15em] text-outline hover:text-on-surface-variant transition-colors cursor-pointer"
              >
                {link}
              </button>
            ))}
          </div>

          {/* Mobile system status */}
          <div className="mt-6 flex justify-center lg:hidden">
            <SystemStatusBadge />
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Login;
