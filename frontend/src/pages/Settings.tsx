import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import TopBar from "@/components/TopBar";
import MaterialIcon from "@/components/MaterialIcon";
import { useAuth } from "@/contexts/AuthContext";
import { api, ApiError } from "@/lib/api";

/* ─────────────────────────────────────────────────────────────────────
 * Local state hook (for purely client-side preferences)
 * ──────────────────────────────────────────────────────────────────── */
function useLocalState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : initial;
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (v: T) => {
      setValue(v);
      localStorage.setItem(key, JSON.stringify(v));
    },
    [key],
  );
  return [value, set];
}

/* ─────────────────────────────────────────────────────────────────────
 * Big chunky toggle that matches the project's typography
 * ──────────────────────────────────────────────────────────────────── */
const Toggle = ({
  enabled,
  onChange,
  disabled,
}: {
  enabled: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) => (
  <button
    type="button"
    role="switch"
    aria-checked={enabled}
    disabled={disabled}
    onClick={() => !disabled && onChange(!enabled)}
    className={`relative w-14 h-8 rounded-full transition-colors duration-300 shrink-0 ${
      disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"
    } ${enabled ? "bg-primary" : "bg-surface-container-highest"}`}
  >
    <motion.span
      layout
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
      className={`absolute top-1 w-6 h-6 rounded-full shadow-lg ${
        enabled ? "right-1 bg-on-primary" : "left-1 bg-on-surface"
      }`}
    />
  </button>
);

/* ─────────────────────────────────────────────────────────────────────
 * Section card primitive — matches the rest of the project
 * ──────────────────────────────────────────────────────────────────── */
const SectionCard = ({
  id,
  icon,
  title,
  subtitle,
  children,
}: {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) => (
  <motion.section
    id={id}
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true, margin: "-80px" }}
    transition={{ duration: 0.6 }}
    className="scroll-mt-32"
  >
    <div className="flex items-start gap-5 mb-8">
      <div className="w-14 h-14 shrink-0 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
        <MaterialIcon icon={icon} size={28} className="text-primary" />
      </div>
      <div className="space-y-1.5 min-w-0">
        <h2 className="text-3xl md:text-4xl font-headline font-black uppercase tracking-tighter leading-none">
          {title}
        </h2>
        <p className="text-base text-on-surface-variant font-light">{subtitle}</p>
      </div>
    </div>
    <div className="rounded-[2rem] bg-surface-container-low/30 border border-white/5 backdrop-blur-xl p-8 md:p-10 space-y-8">
      {children}
    </div>
  </motion.section>
);

/* Row primitive used for toggle/pill-style settings */
const Row = ({
  title,
  desc,
  children,
}: {
  title: string;
  desc?: string;
  children: React.ReactNode;
}) => (
  <div className="flex items-center justify-between gap-6 py-2">
    <div className="min-w-0">
      <p className="font-headline font-bold text-lg text-on-surface">{title}</p>
      {desc && <p className="text-sm text-on-surface-variant font-light mt-1">{desc}</p>}
    </div>
    {children}
  </div>
);

const inputCls =
  "w-full bg-surface-container-high/40 border border-outline-variant/30 focus:border-primary focus:ring-2 focus:ring-primary/20 text-on-surface text-base px-5 py-3.5 rounded-2xl outline-none transition-all placeholder:text-on-surface-variant/50";

const labelCls =
  "block text-[13px] font-mono uppercase tracking-[0.3em] text-on-surface-variant/80 mb-2";

const SECTIONS = [
  { id: "profile", icon: "person", label: "Profile" },
  { id: "security", icon: "shield", label: "Security" },
  { id: "notifications", icon: "notifications_active", label: "Alerts" },
  { id: "appearance", icon: "palette", label: "Appearance" },
  { id: "family", icon: "family_restroom", label: "Family" },
  { id: "privacy", icon: "policy", label: "Privacy" },
  { id: "danger", icon: "warning", label: "Danger Zone" },
] as const;

interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  relationship: string;
}

const SettingsPage = () => {
  const { i18n } = useTranslation();
  const { user, isAuthenticated, refreshUser, setUser, setup2FA, confirm2FA, disable2FA, logout } = useAuth();
  const navigate = useNavigate();

  /* ── Local-only preferences (real, working) ── */
  const [theme, setTheme] = useLocalState("satya-theme", "dark");
  const [fontSize, setFontSize] = useLocalState("satya-font-size", 16);
  const [language, setLanguage] = useState(i18n.language || "en");

  /* ── Sync App-level theme/font when these change (same-tab) ── */
  useEffect(() => {
    window.dispatchEvent(new StorageEvent("storage", { key: "satya-theme" }));
  }, [theme]);
  useEffect(() => {
    window.dispatchEvent(new StorageEvent("storage", { key: "satya-font-size" }));
  }, [fontSize]);

  /* ── Profile (server-backed) — live edit buffers ── */
  const [profileName, setProfileName] = useState("");
  const [profileDirty, setProfileDirty] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);

  /* ── Notification prefs (server-backed) ── */
  const [savingNotif, setSavingNotif] = useState(false);

  /* ── Password ── */
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);

  /* ── 2FA setup flow ── */
  const [twoFactorOpen, setTwoFactorOpen] = useState(false);
  const [twoFactorMode, setTwoFactorMode] = useState<"setup" | "disable">("setup");
  const [twoFactorSecret, setTwoFactorSecret] = useState("");
  const [twoFactorQrUrl, setTwoFactorQrUrl] = useState("");
  const [twoFactorBackupCodes, setTwoFactorBackupCodes] = useState<string[]>([]);
  const [twoFactorCode, setTwoFactorCode] = useState("");
  const [twoFactorSubmitting, setTwoFactorSubmitting] = useState(false);

  /* ── Emergency contacts (multi, localStorage; primary syncs to server) ── */
  const [contacts, setContacts] = useLocalState<EmergencyContact[]>("satya-emergency-contacts", []);
  const [showContactForm, setShowContactForm] = useState(false);
  const [newContactName, setNewContactName] = useState("");
  const [newContactPhone, setNewContactPhone] = useState("");
  const [newContactRelation, setNewContactRelation] = useState("family");

  /* ── Privacy: scan deletion ── */
  const [clearingScans, setClearingScans] = useState(false);
  const [confirmClearScans, setConfirmClearScans] = useState(false);

  /* ── Danger: account deletion ── */
  const [deleteAccountOpen, setDeleteAccountOpen] = useState(false);
  const [deleteAccountPassword, setDeleteAccountPassword] = useState("");
  const [deletingAccount, setDeletingAccount] = useState(false);

  /* ── Hydrate buffers from user ── */
  useEffect(() => {
    if (user) {
      setProfileName(user.name || "");
      setProfileDirty(false);
    }
  }, [user]);

  /* ── Anchor nav active highlighting ── */
  const [activeAnchor, setActiveAnchor] = useState<string>("profile");
  useEffect(() => {
    const handler = () => {
      const offset = 200;
      let current: string = "profile";
      for (const s of SECTIONS) {
        const el = document.getElementById(s.id);
        if (el && el.getBoundingClientRect().top <= offset) {
          current = s.id;
        }
      }
      setActiveAnchor(current);
    };
    handler();
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const handleLanguageChange = async (lang: string) => {
    setLanguage(lang);
    i18n.changeLanguage(lang);
    localStorage.setItem("satya-lang", lang);
    if (isAuthenticated) {
      try {
        const updated = await api.auth.updateProfile({ language_pref: lang });
        setUser(updated);
      } catch {
        // non-fatal — local language change still works
      }
    }
  };

  /* ─── Profile save ─── */
  const saveProfile = async () => {
    if (!isAuthenticated) {
      toast.error("Sign in to save profile changes");
      return;
    }
    if (!profileName.trim()) {
      toast.error("Name cannot be empty");
      return;
    }
    setSavingProfile(true);
    try {
      const updated = await api.auth.updateProfile({ name: profileName.trim() });
      setUser(updated);
      setProfileDirty(false);
      toast.success("Profile saved");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to save profile");
    } finally {
      setSavingProfile(false);
    }
  };

  /* ─── Notification prefs (single-toggle save with optimistic UI) ─── */
  const updateNotifPref = async (
    field: "notify_email_threats" | "notify_email_reports" | "notify_push_enabled",
    value: boolean,
  ) => {
    if (!isAuthenticated || !user) {
      toast.error("Sign in to manage notification preferences");
      return;
    }
    // For push, request browser permission first
    if (field === "notify_push_enabled" && value) {
      if (!("Notification" in window)) {
        toast.error("This browser does not support push notifications");
        return;
      }
      const permission = await Notification.requestPermission();
      if (permission !== "granted") {
        toast.error("Permission denied — enable notifications in your browser settings");
        return;
      }
    }
    setSavingNotif(true);
    // Optimistic: update local user state immediately
    setUser({ ...user, [field]: value });
    try {
      const updated = await api.auth.updateProfile({ [field]: value });
      setUser(updated);
    } catch (err) {
      // Roll back
      setUser({ ...user, [field]: !value });
      toast.error(err instanceof ApiError ? err.message : "Failed to update preference");
    } finally {
      setSavingNotif(false);
    }
  };

  /* ─── Password ─── */
  const handlePasswordChange = async () => {
    if (!currentPassword.trim()) { toast.error("Enter your current password"); return; }
    if (newPassword.length < 8) { toast.error("New password must be at least 8 characters"); return; }
    if (newPassword !== confirmPassword) { toast.error("Passwords do not match"); return; }
    if (currentPassword === newPassword) { toast.error("New password must differ from current"); return; }
    if (!isAuthenticated) { toast.error("Sign in to change your password"); return; }

    setChangingPassword(true);
    try {
      await api.auth.changePassword(currentPassword, newPassword);
      toast.success("Password updated");
      setShowPasswordForm(false);
      setCurrentPassword(""); setNewPassword(""); setConfirmPassword("");
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        toast.error("Current password is incorrect");
      } else {
        toast.error(err instanceof ApiError ? err.message : "Failed to change password");
      }
    } finally {
      setChangingPassword(false);
    }
  };

  /* ─── 2FA flow ─── */
  const open2FASetup = async () => {
    if (!isAuthenticated) { toast.error("Sign in to enable 2FA"); return; }
    setTwoFactorMode("setup");
    setTwoFactorOpen(true);
    setTwoFactorCode("");
    try {
      const data = await setup2FA();
      setTwoFactorSecret(data.secret);
      setTwoFactorQrUrl(data.qr_url);
      setTwoFactorBackupCodes(data.backup_codes || []);
    } catch (err) {
      setTwoFactorOpen(false);
      toast.error(err instanceof ApiError ? err.message : "Failed to start 2FA setup");
    }
  };

  const open2FADisable = () => {
    if (!isAuthenticated) { toast.error("Sign in to manage 2FA"); return; }
    setTwoFactorMode("disable");
    setTwoFactorOpen(true);
    setTwoFactorCode("");
  };

  const submit2FA = async () => {
    if (twoFactorCode.length < 6) { toast.error("Enter your 6-digit code"); return; }
    setTwoFactorSubmitting(true);
    try {
      if (twoFactorMode === "setup") {
        await confirm2FA(twoFactorCode);
        toast.success("Two-factor authentication enabled");
      } else {
        await disable2FA(twoFactorCode);
        toast.success("Two-factor authentication disabled");
      }
      await refreshUser();
      setTwoFactorOpen(false);
      setTwoFactorCode("");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Verification failed");
    } finally {
      setTwoFactorSubmitting(false);
    }
  };

  /* ─── Emergency contacts ─── */
  const addContact = async () => {
    if (!newContactName.trim() || !newContactPhone.trim()) {
      toast.error("Fill in name and phone");
      return;
    }
    const next: EmergencyContact = {
      id: Date.now().toString(),
      name: newContactName.trim(),
      phone: newContactPhone.trim(),
      relationship: newContactRelation,
    };
    const list = [...contacts, next];
    setContacts(list);
    // First contact becomes the primary — sync to backend
    if (list.length === 1 && isAuthenticated) {
      try {
        const updated = await api.auth.updateProfile({
          emergency_contact_name: next.name,
          emergency_contact_phone: next.phone,
        });
        setUser(updated);
      } catch {
        /* non-fatal */
      }
    }
    setNewContactName(""); setNewContactPhone(""); setNewContactRelation("family");
    setShowContactForm(false);
    toast.success("Contact added");
  };

  const removeContact = async (id: string) => {
    const next = contacts.filter((c) => c.id !== id);
    setContacts(next);
    // If the primary was removed, sync the new primary (or clear) on the backend
    if (isAuthenticated) {
      const newPrimary = next[0];
      try {
        const updated = await api.auth.updateProfile({
          emergency_contact_name: newPrimary?.name || "",
          emergency_contact_phone: newPrimary?.phone || "",
        });
        setUser(updated);
      } catch {
        /* non-fatal */
      }
    }
    toast.success("Contact removed");
  };

  /* ─── Clear all scans ─── */
  const handleClearScans = async () => {
    if (!isAuthenticated) { toast.error("Sign in to manage scans"); return; }
    setClearingScans(true);
    try {
      let total = 0;
      let removed = 0;
      let failed = 0;
      // Hard cap on rounds so a buggy DELETE endpoint can't infinite-loop
      for (let round = 0; round < 50; round++) {
        const result = await api.scans.list(1, 100);
        if (result.items.length === 0) break;
        total = total || result.total;
        const beforeRemoved = removed;
        for (const scan of result.items) {
          try {
            await api.scans.delete(scan.id);
            removed++;
          } catch {
            failed++;
          }
        }
        // If we made no progress this round, stop
        if (removed === beforeRemoved) break;
      }
      if (failed > 0) {
        toast.success(`Cleared ${removed} of ${total} scans (${failed} couldn't be deleted)`);
      } else {
        toast.success(`Cleared ${removed} scan${removed === 1 ? "" : "s"}`);
      }
      setConfirmClearScans(false);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to clear scan history");
    } finally {
      setClearingScans(false);
    }
  };

  /* ─── Account deletion ─── */
  const handleDeleteAccount = async () => {
    if (!deleteAccountPassword.trim()) {
      toast.error("Enter your password to confirm");
      return;
    }
    setDeletingAccount(true);
    try {
      await api.auth.deleteAccount(deleteAccountPassword);
      toast.success("Account deleted");
      logout();
      navigate("/");
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        toast.error("Password is incorrect");
      } else {
        toast.error(err instanceof ApiError ? err.message : "Failed to delete account");
      }
    } finally {
      setDeletingAccount(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-surface overflow-x-hidden">
      <TopBar systemStatus="protected" />

      {/* Hero */}
      <section className="pt-40 pb-16 px-6 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-6"
        >
          <span className="inline-block px-4 py-1.5 bg-primary/10 border border-primary/20 rounded-full text-primary font-mono text-[12px] font-black uppercase tracking-[0.3em]">
            System Configuration
          </span>
          <h1 className="text-6xl md:text-7xl font-black font-headline tracking-tighter leading-[0.85] uppercase">
            Settings
          </h1>
          <p className="max-w-2xl text-lg md:text-xl text-on-surface-variant font-light leading-relaxed">
            Manage your identity, security, and how Satya Drishti behaves on your devices. Changes save automatically, except where you see an explicit save button.
          </p>
        </motion.div>
      </section>

      {/* Sticky anchor nav */}
      <nav
        className="sticky top-24 z-30 bg-background/80 backdrop-blur-xl border-y border-white/5"
      >
        <div className="max-w-7xl mx-auto px-6 overflow-x-auto">
          <ul className="flex items-center gap-1 py-3 min-w-max">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-full font-headline font-bold text-[14px] uppercase tracking-[0.15em] whitespace-nowrap transition-all ${
                    activeAnchor === s.id
                      ? "bg-primary text-on-primary"
                      : "text-on-surface-variant hover:bg-surface-container-high/40 hover:text-on-surface"
                  }`}
                >
                  <MaterialIcon icon={s.icon} size={16} />
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-16 space-y-24">
        {/* ─── Sign-in nudge for unauthenticated users ─── */}
        {!isAuthenticated && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-[2rem] bg-error/5 border border-error/20 p-8 flex items-start gap-5"
          >
            <MaterialIcon icon="info" size={28} className="text-error shrink-0 mt-1" />
            <div className="space-y-3 flex-1">
              <h3 className="text-xl font-headline font-black uppercase tracking-tight">
                Sign in for full settings
              </h3>
              <p className="text-base text-on-surface-variant font-light">
                Profile, security, notifications, and account deletion all require an authenticated session. Your local preferences (theme, font size, language) work without sign-in.
              </p>
              <div className="flex gap-3">
                <Link to="/login" className="px-6 py-2.5 bg-primary text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-sm hover:bg-primary/90 transition-colors">
                  Sign In
                </Link>
                <Link to="/register" className="px-6 py-2.5 bg-surface-container-high/60 border border-outline-variant/20 hover:border-primary/40 rounded-full font-headline font-bold uppercase tracking-wider text-sm transition-colors">
                  Create Account
                </Link>
              </div>
            </div>
          </motion.div>
        )}

        {/* ─────────────────────────────── PROFILE ─────────────────────────────── */}
        <SectionCard
          id="profile"
          icon="person"
          title="Profile"
          subtitle="Your identity within the platform."
        >
          <div className="flex flex-col md:flex-row gap-8 items-start">
            <div className="w-32 h-32 shrink-0 rounded-3xl bg-gradient-to-br from-primary/20 to-secondary/10 border border-primary/20 flex items-center justify-center">
              <span className="text-5xl font-headline font-black uppercase text-primary">
                {(user?.name || "U").charAt(0)}
              </span>
            </div>
            <div className="flex-1 w-full grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className={labelCls}>Full Name</label>
                <input
                  value={profileName}
                  onChange={(e) => { setProfileName(e.target.value); setProfileDirty(true); }}
                  placeholder="Your name"
                  className={inputCls}
                  disabled={!isAuthenticated}
                />
              </div>
              <div>
                <label className={labelCls}>Email Address</label>
                <div className="relative">
                  <input
                    value={user?.email || ""}
                    disabled
                    className={`${inputCls} cursor-not-allowed opacity-70`}
                  />
                  {user?.email_verified && (
                    <span className="absolute right-4 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 px-2.5 py-1 bg-secondary/15 text-secondary border border-secondary/30 rounded-full text-[12px] font-mono font-bold uppercase tracking-wider">
                      <MaterialIcon icon="verified" size={12} /> Verified
                    </span>
                  )}
                </div>
              </div>
              <div className="md:col-span-2">
                <label className={labelCls}>Account Type</label>
                <div className={`${inputCls} flex items-center gap-3 cursor-default`}>
                  <MaterialIcon
                    icon={user?.oauth_provider ? "key" : "lock"}
                    size={18}
                    className="text-on-surface-variant"
                  />
                  <span className="capitalize">
                    {user?.oauth_provider
                      ? `${user.oauth_provider} OAuth`
                      : isAuthenticated
                      ? "Email & Password"
                      : "Not signed in"}
                  </span>
                </div>
              </div>
              {isAuthenticated && (
                <div className="md:col-span-2 flex items-center gap-3 pt-2">
                  <button
                    onClick={saveProfile}
                    disabled={!profileDirty || savingProfile}
                    className="px-7 py-3 bg-primary text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-sm hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {savingProfile ? "Saving..." : "Save Profile"}
                  </button>
                  {profileDirty && (
                    <button
                      onClick={() => { setProfileName(user?.name || ""); setProfileDirty(false); }}
                      className="text-sm text-on-surface-variant hover:text-on-surface transition-colors"
                    >
                      Discard
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </SectionCard>

        {/* ─────────────────────────────── SECURITY ─────────────────────────────── */}
        <SectionCard
          id="security"
          icon="shield"
          title="Security"
          subtitle="Protect your account from unauthorized access."
        >
          {/* 2FA */}
          <Row
            title="Two-Factor Authentication"
            desc={
              user?.totp_enabled
                ? "Active — TOTP codes required at every sign-in."
                : "Add an extra layer with an authenticator app like Authy, Google Authenticator, or 1Password."
            }
          >
            {user?.totp_enabled ? (
              <button
                onClick={open2FADisable}
                className="px-5 py-2.5 bg-error/10 border border-error/30 text-error rounded-full font-headline font-bold uppercase tracking-wider text-xs hover:bg-error/20 transition-colors"
              >
                Disable
              </button>
            ) : (
              <button
                onClick={open2FASetup}
                disabled={!isAuthenticated}
                className="px-5 py-2.5 bg-primary text-on-primary rounded-full font-headline font-bold uppercase tracking-wider text-xs hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Enable
              </button>
            )}
          </Row>

          <div className="border-t border-outline-variant/15" />

          {/* Password */}
          <div>
            <button
              onClick={() => setShowPasswordForm((v) => !v)}
              disabled={!isAuthenticated || !!user?.oauth_provider}
              className="w-full flex items-center justify-between gap-4 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="text-left">
                <p className="font-headline font-bold text-lg text-on-surface">Change Password</p>
                <p className="text-sm text-on-surface-variant font-light mt-1">
                  {user?.oauth_provider
                    ? `Sign-in is managed by ${user.oauth_provider} — password change is not available.`
                    : "Update the password used to sign in to your account."}
                </p>
              </div>
              <motion.div animate={{ rotate: showPasswordForm ? 180 : 0 }}>
                <MaterialIcon icon="expand_more" size={24} className="text-on-surface-variant" />
              </motion.div>
            </button>
            <AnimatePresence>
              {showPasswordForm && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden"
                >
                  <div className="pt-6 space-y-4">
                    <div>
                      <label className={labelCls}>Current Password</label>
                      <input
                        type="password"
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)}
                        placeholder="Enter your current password"
                        className={inputCls}
                        autoComplete="current-password"
                      />
                    </div>
                    <div>
                      <label className={labelCls}>New Password</label>
                      <input
                        type="password"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        placeholder="At least 8 characters"
                        className={inputCls}
                        autoComplete="new-password"
                      />
                    </div>
                    <div>
                      <label className={labelCls}>Confirm New Password</label>
                      <input
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Repeat the new password"
                        className={inputCls}
                        autoComplete="new-password"
                      />
                    </div>
                    <button
                      onClick={handlePasswordChange}
                      disabled={!currentPassword || !newPassword || changingPassword}
                      className="px-7 py-3 bg-primary text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-sm hover:bg-primary/90 transition-colors disabled:opacity-40"
                    >
                      {changingPassword ? "Updating..." : "Update Password"}
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </SectionCard>

        {/* ─────────────────────────────── NOTIFICATIONS ─────────────────────────────── */}
        <SectionCard
          id="notifications"
          icon="notifications_active"
          title="Alerts"
          subtitle="Choose how you hear from Satya Drishti."
        >
          <Row
            title="Email — Threat Alerts"
            desc="Receive an email immediately when a high-confidence threat is detected on your account."
          >
            <Toggle
              enabled={!!user?.notify_email_threats}
              onChange={(v) => updateNotifPref("notify_email_threats", v)}
              disabled={!isAuthenticated || savingNotif}
            />
          </Row>
          <div className="border-t border-outline-variant/15" />
          <Row
            title="Email — Weekly Reports"
            desc="A weekly digest summarising scans, detections, and protection status."
          >
            <Toggle
              enabled={!!user?.notify_email_reports}
              onChange={(v) => updateNotifPref("notify_email_reports", v)}
              disabled={!isAuthenticated || savingNotif}
            />
          </Row>
          <div className="border-t border-outline-variant/15" />
          <Row
            title="Browser Push Notifications"
            desc="Real-time desktop alerts when this tab is open. Requires browser permission."
          >
            <Toggle
              enabled={!!user?.notify_push_enabled}
              onChange={(v) => updateNotifPref("notify_push_enabled", v)}
              disabled={!isAuthenticated || savingNotif}
            />
          </Row>
        </SectionCard>

        {/* ─────────────────────────────── APPEARANCE ─────────────────────────────── */}
        <SectionCard
          id="appearance"
          icon="palette"
          title="Appearance"
          subtitle="Tune how Satya Drishti looks on your device."
        >
          <div>
            <label className={labelCls}>Theme</label>
            <div className="grid grid-cols-3 gap-4">
              {(["dark", "light", "system"] as const).map((id) => (
                <button
                  key={id}
                  onClick={() => setTheme(id)}
                  className={`p-5 rounded-2xl space-y-3 transition-all border-2 ${
                    theme === id
                      ? "border-primary bg-primary/10"
                      : "border-outline-variant/20 hover:border-outline-variant/40 bg-surface-container-high/30"
                  }`}
                >
                  <div
                    className={`h-16 w-full rounded-xl flex flex-col p-2 gap-1.5 ${
                      id === "dark"
                        ? "bg-surface-container-lowest border border-white/5"
                        : id === "light"
                        ? "bg-white"
                        : "bg-gradient-to-br from-surface-container-lowest to-white"
                    }`}
                  >
                    {id !== "system" && (
                      <>
                        <div className={`h-1.5 w-2/3 rounded-full ${id === "dark" ? "bg-primary/50" : "bg-black/30"}`} />
                        <div className={`h-1.5 w-full rounded-full ${id === "dark" ? "bg-primary/20" : "bg-black/15"}`} />
                        <div className={`h-1.5 w-1/2 rounded-full ${id === "dark" ? "bg-primary/15" : "bg-black/10"}`} />
                      </>
                    )}
                  </div>
                  <p className="text-sm font-headline font-black uppercase tracking-wider text-center capitalize">
                    {id}
                  </p>
                </button>
              ))}
            </div>
          </div>

          <div className="border-t border-outline-variant/15" />

          <div>
            <div className="flex items-center justify-between mb-4">
              <label className={`${labelCls} mb-0`}>Interface Font Size</label>
              <span className="font-mono font-bold text-primary text-base">{fontSize}px</span>
            </div>
            <input
              type="range"
              min={14}
              max={22}
              value={fontSize}
              onChange={(e) => setFontSize(Number(e.target.value))}
              className="w-full h-2 bg-surface-container-high rounded-full appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-on-surface-variant/60 font-mono mt-2">
              <span>14px (Compact)</span>
              <span>18px</span>
              <span>22px (Large)</span>
            </div>
          </div>

          <div className="border-t border-outline-variant/15" />

          <div>
            <label className={labelCls}>Language</label>
            <div className="grid grid-cols-3 gap-3">
              {[
                { code: "en", label: "English", flag: "🇺🇸" },
                { code: "hi", label: "हिन्दी", flag: "🇮🇳" },
                { code: "mr", label: "मराठी", flag: "🇮🇳" },
              ].map((l) => (
                <button
                  key={l.code}
                  onClick={() => handleLanguageChange(l.code)}
                  className={`flex items-center justify-center gap-3 px-5 py-4 rounded-2xl font-headline font-bold transition-all border-2 ${
                    language === l.code
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-outline-variant/20 bg-surface-container-high/30 hover:border-outline-variant/40"
                  }`}
                >
                  <span className="text-2xl leading-none">{l.flag}</span>
                  <span className="text-base">{l.label}</span>
                </button>
              ))}
            </div>
          </div>
        </SectionCard>

        {/* ─────────────────────────────── FAMILY ─────────────────────────────── */}
        <SectionCard
          id="family"
          icon="family_restroom"
          title="Family & Voice Prints"
          subtitle="Trusted contacts and biometric voice enrolment."
        >
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-headline font-bold text-lg text-on-surface">Emergency Contacts</p>
                <p className="text-sm text-on-surface-variant font-light mt-1">
                  Quick-call directory. The first contact also syncs to your account as your primary.
                </p>
              </div>
              <button
                onClick={() => setShowContactForm((v) => !v)}
                className="px-5 py-2.5 bg-primary text-on-primary rounded-full font-headline font-bold uppercase tracking-wider text-xs hover:bg-primary/90 transition-colors flex items-center gap-2"
              >
                <MaterialIcon icon={showContactForm ? "close" : "person_add"} size={16} />
                {showContactForm ? "Cancel" : "Add"}
              </button>
            </div>

            <AnimatePresence>
              {showContactForm && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden"
                >
                  <div className="bg-surface-container-high/30 rounded-2xl p-6 space-y-4">
                    <div>
                      <label className={labelCls}>Name</label>
                      <input value={newContactName} onChange={(e) => setNewContactName(e.target.value)} placeholder="e.g. Mom" className={inputCls} />
                    </div>
                    <div>
                      <label className={labelCls}>Phone</label>
                      <input
                        value={newContactPhone}
                        onChange={(e) => setNewContactPhone(e.target.value)}
                        placeholder="+91 98765 43210"
                        type="tel"
                        className={`${inputCls} font-mono`}
                      />
                    </div>
                    <div>
                      <label className={labelCls}>Relationship</label>
                      <select
                        value={newContactRelation}
                        onChange={(e) => setNewContactRelation(e.target.value)}
                        className={inputCls}
                      >
                        <option value="family">Family</option>
                        <option value="spouse">Spouse</option>
                        <option value="parent">Parent</option>
                        <option value="child">Child</option>
                        <option value="friend">Friend</option>
                      </select>
                    </div>
                    <button
                      onClick={addContact}
                      className="px-7 py-3 bg-primary text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-sm hover:bg-primary/90 transition-colors"
                    >
                      Save Contact
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {contacts.length > 0 ? (
              <div className="space-y-3">
                {contacts.map((c, i) => (
                  <div
                    key={c.id}
                    className="flex items-center justify-between gap-4 p-5 rounded-2xl bg-surface-container-high/30 border border-outline-variant/10"
                  >
                    <div className="flex items-center gap-4 min-w-0">
                      <div className="h-12 w-12 shrink-0 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-on-primary font-headline font-black text-lg">
                        {c.name.charAt(0).toUpperCase()}
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-headline font-bold text-base text-on-surface truncate">{c.name}</p>
                          {i === 0 && (
                            <span className="px-2 py-0.5 bg-primary/15 text-primary border border-primary/30 rounded-full text-[11px] font-mono font-black uppercase tracking-wider">
                              Primary
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-on-surface-variant font-mono truncate">
                          {c.phone} · <span className="capitalize">{c.relationship}</span>
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <a
                        href={`tel:${c.phone.replace(/\s+/g, "")}`}
                        className="p-3 rounded-xl hover:bg-primary/10 text-on-surface-variant hover:text-primary transition-colors"
                        title="Call"
                      >
                        <MaterialIcon icon="call" size={20} />
                      </a>
                      <button
                        onClick={() => removeContact(c.id)}
                        className="p-3 rounded-xl hover:bg-error/10 text-on-surface-variant hover:text-error transition-colors"
                        title="Remove"
                      >
                        <MaterialIcon icon="delete" size={20} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 rounded-2xl bg-surface-container-high/20 border border-dashed border-outline-variant/30">
                <MaterialIcon icon="contacts" size={36} className="text-on-surface-variant/40 mx-auto mb-3" />
                <p className="text-base text-on-surface-variant font-light">No emergency contacts yet.</p>
              </div>
            )}
          </div>

          <div className="border-t border-outline-variant/15" />

          {/* Voice prints */}
          <div className="flex items-start justify-between gap-6">
            <div className="min-w-0">
              <p className="font-headline font-bold text-lg text-on-surface flex items-center gap-2">
                <MaterialIcon icon="record_voice_over" size={20} className="text-secondary" />
                Voice Prints
              </p>
              <p className="text-sm text-on-surface-variant font-light mt-1 max-w-lg">
                Enrol biometric voice prints for family members so the call protection engine can verify caller identity in real time.
              </p>
            </div>
            <Link
              to="/voice-prints"
              className="shrink-0 px-5 py-2.5 bg-secondary/10 border border-secondary/30 text-secondary rounded-full font-headline font-bold uppercase tracking-wider text-xs hover:bg-secondary/20 transition-colors flex items-center gap-2"
            >
              <MaterialIcon icon="settings_voice" size={16} />
              Manage
            </Link>
          </div>
        </SectionCard>

        {/* ─────────────────────────────── PRIVACY ─────────────────────────────── */}
        <SectionCard
          id="privacy"
          icon="policy"
          title="Privacy"
          subtitle="Control your data on this server."
        >
          <Row
            title="Clear Scan History"
            desc="Permanently delete every scan and analysis report stored under your account. This cannot be undone."
          >
            {confirmClearScans ? (
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={handleClearScans}
                  disabled={clearingScans}
                  className="px-5 py-2.5 bg-error text-on-primary rounded-full font-headline font-bold uppercase tracking-wider text-xs hover:bg-error/90 transition-colors disabled:opacity-50"
                >
                  {clearingScans ? "Clearing..." : "Yes, delete all"}
                </button>
                <button
                  onClick={() => setConfirmClearScans(false)}
                  disabled={clearingScans}
                  className="px-4 py-2.5 text-sm text-on-surface-variant hover:text-on-surface transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirmClearScans(true)}
                disabled={!isAuthenticated}
                className="px-5 py-2.5 bg-error/10 border border-error/30 text-error rounded-full font-headline font-bold uppercase tracking-wider text-xs hover:bg-error/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Clear All
              </button>
            )}
          </Row>

          <div className="border-t border-outline-variant/15" />

          <Row title="Sign Out" desc="End the current session on this device.">
            <button
              onClick={() => { logout(); toast.success("Signed out"); navigate("/"); }}
              disabled={!isAuthenticated}
              className="px-5 py-2.5 bg-surface-container-high border border-outline-variant/30 hover:border-primary/40 rounded-full font-headline font-bold uppercase tracking-wider text-xs transition-colors disabled:opacity-40"
            >
              Sign Out
            </button>
          </Row>
        </SectionCard>

        {/* ─────────────────────────────── DANGER ZONE ─────────────────────────────── */}
        <SectionCard
          id="danger"
          icon="warning"
          title="Danger Zone"
          subtitle="Irreversible actions on your account."
        >
          <div className="rounded-2xl bg-error/5 border border-error/20 p-6">
            <div className="flex items-start justify-between gap-6">
              <div className="min-w-0">
                <p className="font-headline font-bold text-lg text-error">Delete Account</p>
                <p className="text-sm text-on-surface-variant font-light mt-1 max-w-lg">
                  Permanently remove your account, all scans, voice prints, cases, and personal data. There is no recovery.
                </p>
              </div>
              <button
                onClick={() => setDeleteAccountOpen(true)}
                disabled={!isAuthenticated}
                className="shrink-0 px-5 py-2.5 bg-error text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-xs hover:bg-error/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Delete...
              </button>
            </div>
          </div>
        </SectionCard>

        <p className="text-center text-xs font-mono uppercase tracking-[0.4em] text-on-surface-variant/50 pt-12">
          Local preferences live on this device · Account changes sync across all sessions
        </p>
      </main>

      {/* ──────────────────────────── 2FA Modal ──────────────────────────── */}
      <AnimatePresence>
        {twoFactorOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
            onClick={() => !twoFactorSubmitting && setTwoFactorOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ duration: 0.25 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md bg-surface-container-low border border-outline-variant/30 rounded-[2rem] p-8 space-y-6"
            >
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                  <MaterialIcon icon="shield" size={24} className="text-primary" />
                </div>
                <h3 className="text-2xl font-headline font-black uppercase tracking-tight">
                  {twoFactorMode === "setup" ? "Enable 2FA" : "Disable 2FA"}
                </h3>
              </div>

              {twoFactorMode === "setup" ? (
                <>
                  <p className="text-base text-on-surface-variant font-light">
                    Scan this QR with an authenticator app, or enter the secret manually.
                  </p>
                  {twoFactorQrUrl ? (
                    <div className="flex justify-center">
                      <img
                        src={`https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(twoFactorQrUrl)}`}
                        alt="2FA QR"
                        className="w-56 h-56 rounded-2xl bg-white p-3"
                      />
                    </div>
                  ) : (
                    <div className="h-56 flex items-center justify-center">
                      <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                  {twoFactorSecret && (
                    <div>
                      <label className={labelCls}>Secret (manual entry)</label>
                      <div className={`${inputCls} font-mono select-all break-all text-sm`}>
                        {twoFactorSecret}
                      </div>
                    </div>
                  )}
                  {twoFactorBackupCodes.length > 0 && (
                    <details className="group">
                      <summary className="cursor-pointer text-sm font-headline font-bold uppercase tracking-wider text-primary">
                        Backup codes ({twoFactorBackupCodes.length})
                      </summary>
                      <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-xs">
                        {twoFactorBackupCodes.map((c) => (
                          <div key={c} className="px-3 py-2 bg-surface-container-high/40 rounded-lg select-all">
                            {c}
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-on-surface-variant/70 mt-2 italic">
                        Save these — each can be used once if you lose your authenticator.
                      </p>
                    </details>
                  )}
                </>
              ) : (
                <p className="text-base text-on-surface-variant font-light">
                  Enter your current 6-digit authenticator code to disable two-factor protection.
                </p>
              )}

              <div>
                <label className={labelCls}>Authenticator code</label>
                <input
                  value={twoFactorCode}
                  onChange={(e) => setTwoFactorCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="000000"
                  inputMode="numeric"
                  autoFocus
                  className={`${inputCls} text-center font-mono text-2xl tracking-[0.5em]`}
                />
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={submit2FA}
                  disabled={twoFactorCode.length < 6 || twoFactorSubmitting}
                  className="flex-1 px-7 py-3 bg-primary text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-sm hover:bg-primary/90 transition-colors disabled:opacity-40"
                >
                  {twoFactorSubmitting ? "Verifying..." : twoFactorMode === "setup" ? "Confirm" : "Disable"}
                </button>
                <button
                  onClick={() => setTwoFactorOpen(false)}
                  disabled={twoFactorSubmitting}
                  className="px-5 py-3 text-sm text-on-surface-variant hover:text-on-surface transition-colors"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ──────────────────────────── Delete Account Modal ──────────────────────────── */}
      <AnimatePresence>
        {deleteAccountOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
            onClick={() => !deletingAccount && setDeleteAccountOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ duration: 0.25 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md bg-surface-container-low border border-error/30 rounded-[2rem] p-8 space-y-6"
            >
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-2xl bg-error/10 border border-error/20 flex items-center justify-center">
                  <MaterialIcon icon="warning" size={24} className="text-error" />
                </div>
                <h3 className="text-2xl font-headline font-black uppercase tracking-tight text-error">
                  Delete Account
                </h3>
              </div>
              <p className="text-base text-on-surface-variant font-light">
                This permanently deletes your account, scans, voice prints, and cases. <span className="font-bold text-on-surface">There is no recovery.</span> Enter your password to confirm.
              </p>
              <div>
                <label className={labelCls}>Password</label>
                <input
                  type="password"
                  value={deleteAccountPassword}
                  onChange={(e) => setDeleteAccountPassword(e.target.value)}
                  placeholder="Your account password"
                  className={inputCls}
                  autoComplete="current-password"
                />
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={handleDeleteAccount}
                  disabled={!deleteAccountPassword || deletingAccount}
                  className="flex-1 px-7 py-3 bg-error text-on-primary rounded-full font-headline font-black uppercase tracking-wider text-sm hover:bg-error/90 transition-colors disabled:opacity-40"
                >
                  {deletingAccount ? "Deleting..." : "Delete Forever"}
                </button>
                <button
                  onClick={() => { setDeleteAccountOpen(false); setDeleteAccountPassword(""); }}
                  disabled={deletingAccount}
                  className="px-5 py-3 text-sm text-on-surface-variant hover:text-on-surface transition-colors"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="py-20 text-center text-outline text-[12px] font-mono uppercase tracking-[0.5em] opacity-30">
        Satya Drishti • Defending Humanity in the Age of AI
      </footer>
    </div>
  );
};

export default SettingsPage;
