import { useState, useCallback, useEffect } from "react";
import Layout from "@/components/Layout";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { api, ApiError } from "@/lib/api";

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

const Toggle = ({ enabled, onChange }: { enabled: boolean; onChange: (v: boolean) => void }) => (
  <div
    onClick={() => onChange(!enabled)}
    className={`w-12 h-6 rounded-full relative p-1 cursor-pointer transition-colors ${
      enabled ? "bg-secondary-container" : "bg-surface-container-highest"
    }`}
  >
    <div
      className="w-4 h-4 bg-on-surface rounded-full absolute transition-all"
      style={{ right: enabled ? "4px" : "auto", left: enabled ? "auto" : "4px" }}
    />
  </div>
);

interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  relationship: string;
}

type Section = "profile" | "privacy-security" | "notifications" | "ai" | "family" | "appearance" | "history";

const SIDEBAR_ITEMS: { id: Section; icon: string; label: string }[] = [
  { id: "profile", icon: "person", label: "Profile" },
  { id: "privacy-security", icon: "shield", label: "Privacy & Security" },
  { id: "notifications", icon: "notifications_active", label: "Notifications" },
  { id: "ai", icon: "psychology", label: "AI Config" },
  { id: "family", icon: "family_restroom", label: "Family" },
  { id: "appearance", icon: "palette", label: "Appearance" },
  { id: "history", icon: "history", label: "History" },
];

const SettingsPage = () => {
  const { i18n } = useTranslation();
  const { isAuthenticated } = useAuth();
  const [activeSection, setActiveSection] = useState<Section>("profile");
  const [language, setLanguage] = useState(i18n.language || "en");

  // Protection & AI
  const [voiceProtection, setVoiceProtection] = useLocalState("satya-voice-protection", true);
  const [videoProtection, setVideoProtection] = useLocalState("satya-video-protection", true);
  const [conversationMonitoring, setConversationMonitoring] = useLocalState("satya-conversation-monitoring", true);
  const [sensitivity, setSensitivity] = useLocalState("satya-sensitivity", 75);
  const [autoBlock, setAutoBlock] = useLocalState("satya-auto-block", true);
  const [neuralEngine, setNeuralEngine] = useLocalState("satya-neural-engine", "ensemble");

  // Notifications
  const [emailAlerts, setEmailAlerts] = useLocalState("satya-email-alerts", true);
  const [pushNotifs, setPushNotifs] = useLocalState("satya-push-notifs", false);

  // Appearance
  const [theme, setTheme] = useLocalState("satya-theme", "dark");
  const [fontSize, setFontSize] = useLocalState("satya-font-size", 14);

  // Privacy
  const [autoDelete, setAutoDelete] = useLocalState("satya-auto-delete", "never");
  const [anonymousMode, setAnonymousMode] = useLocalState("satya-anonymous", false);

  // Security
  const [twoFactor, setTwoFactor] = useLocalState("satya-2fa", true);
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  // Emergency Contacts
  const [contacts, setContacts] = useLocalState<EmergencyContact[]>("satya-emergency-contacts", []);
  const [showContactForm, setShowContactForm] = useState(false);
  const [newContactName, setNewContactName] = useState("");
  const [newContactPhone, setNewContactPhone] = useState("");
  const [newContactRelation, setNewContactRelation] = useState("family");

  // Profile (persisted to localStorage)
  const [profileName, setProfileName] = useLocalState("satya-profile-name", "User");
  const [profileEmail, setProfileEmail] = useLocalState("satya-profile-email", "");
  const [profilePhone, setProfilePhone] = useLocalState("satya-profile-phone", "");
  const [profileTimezone, setProfileTimezone] = useLocalState("satya-profile-timezone", "Asia/Kolkata");

  const handleLanguageChange = (lang: string) => {
    setLanguage(lang);
    i18n.changeLanguage(lang);
  };

  const [changingPassword, setChangingPassword] = useState(false);

  const handlePasswordChange = async () => {
    if (!currentPassword.trim()) { toast.error("Enter your current password"); return; }
    if (newPassword.length < 8) { toast.error("New password must be at least 8 characters"); return; }
    if (newPassword !== confirmPassword) { toast.error("Passwords do not match"); return; }
    if (currentPassword === newPassword) { toast.error("New password must be different from current password"); return; }
    if (!isAuthenticated) { toast.error("You must be logged in to change your password"); return; }

    setChangingPassword(true);
    try {
      await api.auth.changePassword(currentPassword, newPassword);
      toast.success("Password updated successfully");
      setShowPasswordForm(false);
      setCurrentPassword(""); setNewPassword(""); setConfirmPassword("");
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 403) {
          toast.error("Current password is incorrect");
        } else {
          toast.error(err.message);
        }
      } else {
        toast.error("Failed to change password");
      }
    } finally {
      setChangingPassword(false);
    }
  };

  const addContact = () => {
    if (!newContactName.trim() || !newContactPhone.trim()) { toast.error("Fill in all fields"); return; }
    setContacts([...contacts, { id: Date.now().toString(), name: newContactName.trim(), phone: newContactPhone.trim(), relationship: newContactRelation }]);
    setNewContactName(""); setNewContactPhone(""); setNewContactRelation("family"); setShowContactForm(false);
    toast.success("Contact added");
  };

  // Notify App-level theme sync when settings change (same-tab)
  useEffect(() => {
    window.dispatchEvent(new StorageEvent("storage", { key: "satya-theme" }));
  }, [theme]);

  useEffect(() => {
    window.dispatchEvent(new StorageEvent("storage", { key: "satya-font-size" }));
  }, [fontSize]);

  const show = (s: Section) => activeSection === "profile" || activeSection === s;
  const inputCls = "w-full bg-surface-container-low border-b border-outline-variant focus:border-primary-fixed focus:ring-0 text-on-surface py-2 outline-none transition-all";

  return (
    <Layout systemStatus="protected">
      <div className="flex min-h-screen">

        {/* SideNavBar */}
        <aside className="hidden md:flex flex-col w-64 fixed left-0 top-20 h-[calc(100vh-5rem)] bg-surface border-r border-outline-variant/15 z-40">
          <div className="px-6 py-4">
            <h2 className="text-lg font-bold text-on-surface">Settings</h2>
            <p className="text-xs text-on-surface-variant opacity-70">System Configuration</p>
          </div>
          <nav className="flex-1 overflow-y-auto mt-4">
            {SIDEBAR_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={`w-full flex items-center gap-3 px-6 py-4 text-sm font-medium cursor-pointer transition-all duration-300 ${
                  activeSection === item.id
                    ? "text-primary-container bg-surface-container-low border-r-4 border-primary-container font-semibold"
                    : "text-on-surface-variant opacity-70 hover:bg-surface-container-high hover:opacity-100"
                }`}
              >
                <MaterialIcon icon={item.icon} size={20} />
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="p-6 mt-auto space-y-4">
            <div className="flex flex-col gap-2">
              <Link to="/contact" className="text-on-surface-variant opacity-70 hover:bg-surface-container-high hover:opacity-100 transition-all duration-300 flex items-center gap-3 px-2 py-2 cursor-pointer text-xs">
                <MaterialIcon icon="contact_support" size={16} />
                <span>Support</span>
              </Link>
              <Link to="/help" className="text-on-surface-variant opacity-70 hover:bg-surface-container-high hover:opacity-100 transition-all duration-300 flex items-center gap-3 px-2 py-2 cursor-pointer text-xs">
                <MaterialIcon icon="menu_book" size={16} />
                <span>Docs</span>
              </Link>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 ml-0 md:ml-64 p-8 lg:p-12">
          <div className="max-w-6xl mx-auto space-y-12">

            {/* Page Header */}
            <section>
              <h1 className="text-4xl font-headline font-extrabold text-on-surface tracking-tight mb-2">Hub Settings</h1>
              <p className="text-on-surface-variant">Manage your identity, security protocols, and autonomous AI modules.</p>
            </section>

            {/* Bento Grid */}
            <div className="grid grid-cols-12 gap-6">

              {/* Profile Section (8 cols) */}
              {show("profile") && (
                <div className="col-span-12 lg:col-span-8 glass-panel border border-outline-variant/15 rounded-xl p-8 flex flex-col md:flex-row gap-8 items-start">
                  <div className="relative group">
                    <div className="w-32 h-32 rounded-full p-1 bg-gradient-to-tr from-primary to-secondary shadow-[0_0_20px_rgba(0,209,255,0.4)]">
                      <div className="w-full h-full rounded-full bg-surface-container-low overflow-hidden border-4 border-surface flex items-center justify-center">
                        <MaterialIcon icon="person" className="text-primary" size={56} />
                      </div>
                    </div>
                  </div>
                  <div className="flex-1 w-full space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-1">
                        <label className="text-xs font-label uppercase tracking-widest text-outline">Full Name</label>
                        <input value={profileName} onChange={(e) => setProfileName(e.target.value)} className={inputCls} />
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs font-label uppercase tracking-widest text-outline">Email Address</label>
                        <input type="email" value={profileEmail} onChange={(e) => setProfileEmail(e.target.value)} placeholder="you@example.com" className={inputCls} />
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs font-label uppercase tracking-widest text-outline">Phone Sequence</label>
                        <input value={profilePhone} onChange={(e) => setProfilePhone(e.target.value)} placeholder="+91 xxxx xxx xxx" className={inputCls} />
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs font-label uppercase tracking-widest text-outline">Timezone</label>
                        <select value={profileTimezone} onChange={(e) => setProfileTimezone(e.target.value)} className={`${inputCls} appearance-none`}>
                          <option value="Asia/Kolkata">UTC+05:30 India (IST)</option>
                          <option value="UTC">UTC+00:00 London (GMT)</option>
                          <option value="America/New_York">UTC-05:00 New York (EST)</option>
                          <option value="Asia/Tokyo">UTC+09:00 Tokyo (JST)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* AI Config Section (4 cols) */}
              {show("ai") && (
                <div className="col-span-12 lg:col-span-4 glass-panel border border-outline-variant/15 rounded-xl p-8 space-y-6">
                  <div className="flex items-center gap-3">
                    <MaterialIcon icon="psychology" className="text-primary-fixed-dim" size={24} />
                    <h3 className="font-headline font-bold text-lg">AI Configuration</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs font-label text-outline uppercase tracking-wider">
                        <span>Sensitivity</span>
                        <span className="text-primary">{sensitivity < 40 ? "Low" : sensitivity < 70 ? "Medium" : "High"}</span>
                      </div>
                      <input
                        type="range" min={0} max={100} value={sensitivity}
                        onChange={(e) => setSensitivity(Number(e.target.value))}
                        className="w-full h-1 bg-surface-container-highest rounded-full appearance-none accent-primary-container cursor-pointer"
                      />
                    </div>
                    <div className="flex items-center justify-between p-4 bg-surface-container-low rounded-lg">
                      <div className="text-sm">
                        <p className="font-medium">Auto-Block</p>
                        <p className="text-xs text-on-surface-variant">Prevent deepfake ingress</p>
                      </div>
                      <Toggle enabled={autoBlock} onChange={setAutoBlock} />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs font-label uppercase text-outline">Neural Engine</label>
                      <select
                        value={neuralEngine}
                        onChange={(e) => setNeuralEngine(e.target.value)}
                        className="w-full bg-surface-container-low border border-outline-variant/30 rounded-lg p-2 text-sm text-on-surface outline-none"
                      >
                        <option value="ensemble">Satya-X Core (v2.4)</option>
                        <option value="fast">Fast Mode (AST Only)</option>
                        <option value="thorough">Deep Scan (All Layers)</option>
                      </select>
                    </div>
                  </div>

                  {/* Protection toggles */}
                  <div className="space-y-4 pt-4 border-t border-outline-variant/10">
                    {[
                      { icon: "mic", label: "Voice Protection", enabled: voiceProtection, onChange: setVoiceProtection },
                      { icon: "videocam", label: "Video Protection", enabled: videoProtection, onChange: setVideoProtection },
                      { icon: "chat", label: "Conversation Monitor", enabled: conversationMonitoring, onChange: setConversationMonitoring },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <MaterialIcon icon={item.icon} className="text-primary" size={18} />
                          <span className="text-sm text-on-surface">{item.label}</span>
                        </div>
                        <Toggle enabled={item.enabled} onChange={item.onChange} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Privacy & Security (combined — 7 cols) */}
              {activeSection === "privacy-security" && (
                <div className="col-span-12 lg:col-span-7 glass-panel border border-outline-variant/15 rounded-xl p-8">
                  <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-3">
                      <MaterialIcon icon="security" filled className="text-secondary" size={24} />
                      <h3 className="font-headline font-bold text-lg">Privacy & Security</h3>
                    </div>
                    <span className="px-3 py-1 bg-secondary/10 text-secondary text-xs rounded-full border border-secondary/20">All Clear</span>
                  </div>
                  <div className="space-y-6">
                    {/* 2FA */}
                    <div className="flex items-center justify-between border-b border-outline-variant/10 pb-4">
                      <div>
                        <p className="font-medium">Two-Factor Authentication</p>
                        <p className="text-xs text-on-surface-variant">
                          {twoFactor ? "Hardware keys and mobile auth active" : "Not enabled \u2014 enable for maximum security"}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        {twoFactor && (
                          <>
                            <span className="text-[10px] text-secondary uppercase font-bold tracking-tighter">Active</span>
                            <div className="w-4 h-4 rounded-full bg-secondary shadow-[0_0_8px_#4edea3]" />
                          </>
                        )}
                        <Toggle enabled={twoFactor} onChange={(v) => { setTwoFactor(v); toast.success(v ? "2FA enabled" : "2FA disabled"); }} />
                      </div>
                    </div>

                    {/* Change Password accordion */}
                    <div>
                      <button
                        onClick={() => setShowPasswordForm(!showPasswordForm)}
                        className="cursor-pointer w-full flex items-center justify-between p-3 hover:bg-surface-container-high rounded-xl transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <MaterialIcon icon="password" className="text-on-surface-variant" size={20} />
                          <span className="font-medium text-sm">Change Password</span>
                        </div>
                        <motion.div animate={{ rotate: showPasswordForm ? 180 : 0 }}>
                          <MaterialIcon icon="expand_more" size={16} className="text-on-surface-variant" />
                        </motion.div>
                      </button>
                      <AnimatePresence>
                        {showPasswordForm && (
                          <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                            <div className="px-4 pb-4 space-y-3 pt-2">
                              <input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} placeholder="Current password" className={inputCls} />
                              <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="New password (min 8 chars)" className={inputCls} />
                              <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Confirm new password" className={inputCls} />
                              <button onClick={handlePasswordChange} disabled={!currentPassword || !newPassword || changingPassword}
                                className="btn-sentinel px-4 py-2 rounded-lg text-xs disabled:opacity-40">
                                {changingPassword ? "Updating..." : "Update Password"}
                              </button>
                              {!isAuthenticated && (
                                <p className="text-xs text-error">Sign in to change your password</p>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Privacy settings */}
                    <div className="border-t border-outline-variant/10 pt-6 space-y-5">
                      <h4 className="text-xs font-label uppercase tracking-widest text-outline">Data & Privacy</h4>
                      <div className="flex items-center justify-between">
                        <div><p className="font-medium">Auto-Delete Scans</p><p className="text-xs text-on-surface-variant">Automatically remove scan history</p></div>
                        <select value={autoDelete} onChange={(e) => setAutoDelete(e.target.value)}
                          className="bg-surface-container-low border-b border-outline-variant text-on-surface py-2 text-sm outline-none">
                          <option value="never">Never</option><option value="7">After 7 days</option><option value="30">After 30 days</option><option value="90">After 90 days</option>
                        </select>
                      </div>
                      <div className="flex items-center justify-between">
                        <div><p className="font-medium">Anonymous Mode</p><p className="text-xs text-on-surface-variant">Don't store any identifying data</p></div>
                        <Toggle enabled={anonymousMode} onChange={setAnonymousMode} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Appearance & Theme (5 cols) */}
              {show("appearance") && (
                <div className="col-span-12 lg:col-span-5 glass-panel border border-outline-variant/15 rounded-xl p-8 space-y-6">
                  <div className="flex items-center gap-3">
                    <MaterialIcon icon="palette" className="text-primary" size={24} />
                    <h3 className="font-headline font-bold text-lg">Appearance</h3>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    {([
                      { id: "dark", label: "Dark" },
                      { id: "light", label: "Light" },
                      { id: "system", label: "System" },
                    ] as const).map((t) => (
                      <button
                        key={t.id}
                        onClick={() => setTheme(t.id)}
                        className={`cursor-pointer p-3 rounded-xl space-y-2 transition-all ${
                          theme === t.id
                            ? "border-2 border-primary bg-surface-container-highest"
                            : "border border-outline-variant/30 opacity-50 hover:opacity-100"
                        }`}
                      >
                        <div
                          className={`h-12 w-full rounded-md flex flex-col p-1 gap-1 ${
                            t.id === "dark"
                              ? "bg-surface-container-lowest"
                              : t.id === "light"
                              ? "bg-white"
                              : "bg-gradient-to-br from-surface to-white"
                          }`}
                        >
                          {t.id !== "system" && (
                            <>
                              <div className={`h-1 w-2/3 rounded-full ${t.id === "dark" ? "bg-primary/20" : "bg-black/20"}`} />
                              <div className={`h-1 w-full rounded-full ${t.id === "dark" ? "bg-primary/10" : "bg-black/10"}`} />
                            </>
                          )}
                        </div>
                        <p className="text-[10px] font-bold text-center">{t.label}</p>
                      </button>
                    ))}
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs font-label text-outline uppercase tracking-wider">
                        <span>Interface Font Size</span>
                        <span className="text-primary">{fontSize}px</span>
                      </div>
                      <input
                        type="range" min={12} max={20} value={fontSize}
                        onChange={(e) => setFontSize(Number(e.target.value))}
                        className="w-full h-1 bg-surface-container-highest rounded-full appearance-none accent-primary-container"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs font-label uppercase text-outline">Language</label>
                      <select
                        value={language}
                        onChange={(e) => handleLanguageChange(e.target.value)}
                        className="w-full bg-surface-container-low border border-outline-variant/30 rounded-lg p-2 text-sm text-on-surface outline-none"
                      >
                        <option value="en">English (Indian Standard)</option>
                        <option value="hi">Hindi</option>
                        <option value="mr">Marathi</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {/* Notifications */}
              {activeSection === "notifications" && (
                <div className="col-span-12 glass-panel border border-outline-variant/15 rounded-xl p-8 space-y-6">
                  <h3 className="font-headline font-bold text-lg tracking-tight">Notifications</h3>
                  <div className="space-y-5">
                    <div className="flex items-center justify-between">
                      <div><p className="font-medium">Email Alerts</p><p className="text-xs text-on-surface-variant">Get notified about threats via email</p></div>
                      <Toggle enabled={emailAlerts} onChange={setEmailAlerts} />
                    </div>
                    <div className="flex items-center justify-between">
                      <div><p className="font-medium">Push Notifications</p><p className="text-xs text-on-surface-variant">Browser push for real-time alerts</p></div>
                      <Toggle enabled={pushNotifs} onChange={setPushNotifs} />
                    </div>
                  </div>
                </div>
              )}

              {/* History Section */}
              {activeSection === "history" && (
                <div className="col-span-12 glass-panel border border-outline-variant/15 rounded-xl p-8 space-y-6">
                  <div className="flex items-center gap-3">
                    <MaterialIcon icon="history" className="text-primary" size={24} />
                    <h3 className="font-headline font-bold text-lg tracking-tight">Scan & Analysis History</h3>
                  </div>
                  <div className="text-center py-12 space-y-4">
                    <MaterialIcon icon="folder_open" className="text-on-surface-variant/30" size={48} />
                    <p className="text-on-surface-variant">Your scan history will appear here.</p>
                    <p className="text-xs text-on-surface-variant/50">All past scanner results and call protection logs will be shown in this section.</p>
                    <Link to="/scanner" className="inline-flex items-center gap-2 px-6 py-3 bg-primary/10 text-primary rounded-xl text-sm font-bold hover:bg-primary/20 transition-colors">
                      <MaterialIcon icon="image_search" size={18} />
                      Start a New Scan
                    </Link>
                  </div>
                </div>
              )}

              {/* Family / Emergency Contacts */}
              {activeSection === "family" && (
                <div className="col-span-12 glass-panel border border-outline-variant/15 rounded-xl p-8 space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="font-headline font-bold text-lg tracking-tight">Emergency Contacts</h3>
                    <button onClick={() => setShowContactForm(true)} className="btn-sentinel px-4 py-2 rounded-lg text-xs flex items-center gap-2">
                      <MaterialIcon icon="person_add" size={16} /> Add Contact
                    </button>
                  </div>
                  {contacts.length > 0 ? (
                    <div className="space-y-3">
                      {contacts.map((c) => (
                        <div key={c.id} className="flex items-center justify-between p-4 rounded-xl bg-surface-container-low">
                          <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-primary-container flex items-center justify-center text-on-primary-container font-bold text-sm">
                              {c.name.charAt(0).toUpperCase()}
                            </div>
                            <div>
                              <p className="font-bold text-sm">{c.name}</p>
                              <p className="text-xs text-on-surface-variant font-mono">{c.phone} &bull; {c.relationship}</p>
                            </div>
                          </div>
                          <button onClick={() => { setContacts(contacts.filter((x) => x.id !== c.id)); toast.success("Removed"); }}
                            className="cursor-pointer p-2 rounded-lg hover:bg-error/10 text-on-surface-variant/40 hover:text-error transition-colors">
                            <MaterialIcon icon="delete" size={16} />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-on-surface-variant text-center py-8">No emergency contacts added.</p>
                  )}
                  <AnimatePresence>
                    {showContactForm && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="p-4 rounded-xl bg-surface-container-low space-y-3">
                          <input value={newContactName} onChange={(e) => setNewContactName(e.target.value)} placeholder="Name" className={inputCls} />
                          <input value={newContactPhone} onChange={(e) => setNewContactPhone(e.target.value)} placeholder="Phone" type="tel" className={`${inputCls} font-mono`} />
                          <select value={newContactRelation} onChange={(e) => setNewContactRelation(e.target.value)}
                            className="w-full bg-surface-container-low border-b border-outline-variant text-on-surface py-2.5 text-sm outline-none">
                            <option value="family">Family</option><option value="spouse">Spouse</option><option value="parent">Parent</option><option value="child">Child</option><option value="friend">Friend</option>
                          </select>
                          <div className="flex gap-3">
                            <button onClick={addContact} className="btn-sentinel px-4 py-2 rounded-lg text-xs">Save</button>
                            <button onClick={() => setShowContactForm(false)} className="cursor-pointer text-xs text-on-surface-variant hover:text-on-surface">Cancel</button>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>

            {/* Action Footer */}
            <div className="flex flex-col md:flex-row items-center justify-between pt-8 gap-4">
              <div className="flex items-center gap-2">
                <MaterialIcon icon="info" className="text-primary" size={16} />
                <p className="text-xs text-on-surface-variant">All settings are saved automatically to your browser.</p>
              </div>
              <div className="flex items-center gap-4 w-full md:w-auto">
                <button
                  onClick={() => window.location.reload()}
                  className="cursor-pointer flex-1 md:flex-none px-8 py-3 bg-surface-container-high rounded-xl text-sm font-bold border border-outline-variant/30 hover:bg-surface-container-highest transition-all"
                >
                  Reset
                </button>
                <button
                  onClick={() => toast.success("All settings saved")}
                  className="cursor-pointer flex-1 md:flex-none px-8 py-3 bg-gradient-to-r from-primary to-primary-container text-on-primary rounded-xl text-sm font-extrabold shadow-[0_4px_20px_rgba(0,209,255,0.3)] hover:scale-105 transition-all"
                >
                  Confirm Settings
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </Layout>
  );
};

export default SettingsPage;
