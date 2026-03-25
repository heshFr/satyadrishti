import { useState } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import MaterialIcon from "@/components/MaterialIcon";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";
import { toast } from "sonner";

const verdictConfig: Record<string, { color: string; bg: string; border: string }> = {
  "ai-generated": { color: "text-error", bg: "bg-error/10", border: "border-error/20" },
  authentic: { color: "text-secondary", bg: "bg-secondary/10", border: "border-secondary/20" },
  uncertain: { color: "text-primary", bg: "bg-primary/10", border: "border-primary/20" },
};

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.06 } } },
  item: {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as const } },
  },
};

const Profile = () => {
  const { t } = useTranslation();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(user?.name || "");

  const { data: scansData, isLoading } = useQuery({
    queryKey: ["scans", 1],
    queryFn: () => api.scans.list(1, 10),
    enabled: !!user,
  });

  const updateMutation = useMutation({
    mutationFn: (name: string) => api.auth.updateProfile({ name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["auth"] });
      toast.success("Profile updated");
      setIsEditing(false);
    },
    onError: () => toast.error("Failed to update profile"),
  });

  if (!user) return null;

  const totalScans = scansData?.total ?? 0;
  const threats = scansData?.items.filter((s) => s.verdict === "ai-generated").length ?? 0;
  const safe = scansData?.items.filter((s) => s.verdict === "authentic").length ?? 0;
  const avgConfidence =
    scansData?.items.length
      ? Math.round(scansData.items.reduce((a, s) => a + s.confidence, 0) / scansData.items.length)
      : 0;

  const memberDays = Math.floor(
    (Date.now() - new Date(user.created_at).getTime()) / (1000 * 60 * 60 * 24)
  );

  const stats = [
    { label: "Total Scans", value: totalScans, icon: "search", color: "text-primary", bg: "bg-primary/8" },
    { label: "Threats Found", value: threats, icon: "warning", color: "text-error", bg: "bg-error/8" },
    { label: "Verified Safe", value: safe, icon: "verified_user", color: "text-secondary", bg: "bg-secondary/8" },
    { label: "Avg Confidence", value: `${avgConfidence}%`, icon: "trending_up", color: "text-primary", bg: "bg-primary/8" },
  ];

  return (
    <Layout systemStatus="protected">
      <div className="max-w-5xl mx-auto pt-32 pb-20 px-6 md:px-8">
        <motion.div variants={stagger.container} initial="initial" animate="animate" className="space-y-6">

          {/* HERO HEADER */}
          <motion.div
            variants={stagger.item}
            className="relative bg-surface-container-low rounded-xl border border-outline-variant/10 overflow-hidden"
          >
            {/* Background accents */}
            <div className="absolute top-0 right-0 w-64 h-64 rounded-full bg-primary/5 blur-[100px]" />
            <div className="absolute bottom-0 left-0 w-48 h-48 rounded-full bg-secondary/5 blur-[80px]" />

            <div className="relative z-10 p-8 md:p-10">
              <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                {/* Avatar */}
                <div className="relative group">
                  <div className="absolute -inset-1 rounded-2xl bg-gradient-to-br from-primary to-primary-container opacity-60 blur-sm group-hover:opacity-80 transition-opacity" />
                  <div className="relative w-20 h-20 md:w-24 md:h-24 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 border border-outline-variant/10 flex items-center justify-center">
                    <span className="text-3xl md:text-4xl font-headline font-bold text-primary">
                      {user.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-secondary border-2 border-surface flex items-center justify-center">
                    <MaterialIcon icon="check" size={14} className="text-on-surface" />
                  </div>
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <AnimatePresence mode="wait">
                    {isEditing ? (
                      <motion.div
                        key="editing"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="flex items-center gap-3"
                      >
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="text-2xl font-headline font-bold bg-surface-container-high border-none focus:ring-1 focus:ring-primary text-on-surface rounded-lg px-4 py-2 w-full max-w-xs focus:outline-none"
                          autoFocus
                        />
                        <button
                          onClick={() => updateMutation.mutate(editName)}
                          disabled={updateMutation.isPending || !editName.trim()}
                          className="p-2 rounded-lg bg-secondary/10 border border-secondary/20 text-secondary hover:bg-secondary/20 transition-colors cursor-pointer disabled:opacity-50"
                        >
                          <MaterialIcon icon="check" size={16} />
                        </button>
                        <button
                          onClick={() => { setIsEditing(false); setEditName(user.name); }}
                          className="p-2 rounded-lg bg-surface-container-high border border-outline-variant/10 text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer"
                        >
                          <MaterialIcon icon="close" size={16} />
                        </button>
                      </motion.div>
                    ) : (
                      <motion.div key="display" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        <div className="flex items-center gap-3">
                          <h1 className="font-headline text-2xl md:text-3xl font-extrabold tracking-tighter text-on-surface truncate">
                            {user.name}
                          </h1>
                          <button
                            onClick={() => { setEditName(user.name); setIsEditing(true); }}
                            className="p-1.5 rounded-lg hover:bg-surface-container-high text-on-surface-variant/50 hover:text-primary transition-all cursor-pointer"
                          >
                            <MaterialIcon icon="edit" size={16} />
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <div className="flex flex-wrap items-center gap-4 mt-2">
                    <span className="flex items-center gap-1.5 text-on-surface-variant/60 text-sm">
                      <MaterialIcon icon="mail" size={14} /> {user.email}
                    </span>
                    <span className="flex items-center gap-1.5 text-on-surface-variant/60 text-sm">
                      <MaterialIcon icon="calendar_today" size={14} />
                      Joined {new Date(user.created_at).toLocaleDateString("en-US", { month: "long", year: "numeric" })}
                    </span>
                    <span className="flex items-center gap-1.5 text-on-surface-variant/60 text-sm">
                      <MaterialIcon icon="language" size={14} />
                      {user.language_pref === "hi" ? "Hindi" : user.language_pref === "mr" ? "Marathi" : "English"}
                    </span>
                  </div>

                  {/* Member badge */}
                  <div className="flex items-center gap-3 mt-4">
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-headline font-semibold bg-primary/10 text-primary border border-primary/20">
                      <MaterialIcon icon="shield" size={12} />
                      {memberDays < 30 ? "New Member" : memberDays < 365 ? "Active Member" : "Veteran Member"}
                    </span>
                    <span className="text-on-surface-variant/40 text-xs font-mono">
                      {memberDays}d active
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* STATS GRID */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {stats.map((stat) => (
              <motion.div
                key={stat.label}
                variants={stagger.item}
                className="relative bg-surface-container-low rounded-xl border border-outline-variant/10 p-5 overflow-hidden"
              >
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-3">
                    <div className={`p-2 rounded-lg ${stat.bg} ${stat.color}`}>
                      <MaterialIcon icon={stat.icon} size={16} />
                    </div>
                    {!isLoading && typeof stat.value === "number" && stat.value > 0 && (
                      <span className="text-[10px] font-mono text-on-surface-variant/40 uppercase">Active</span>
                    )}
                  </div>
                  <p className={`text-3xl font-headline font-bold ${stat.color} tabular-nums`}>
                    {isLoading ? (
                      <span className="inline-block w-12 h-8 rounded bg-surface-container-high animate-pulse" />
                    ) : (
                      stat.value
                    )}
                  </p>
                  <p className="text-on-surface-variant/50 text-xs mt-1 font-medium">{stat.label}</p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* MAIN CONTENT GRID */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Recent Scans (2/3 width) */}
            <motion.div
              variants={stagger.item}
              className="lg:col-span-2 bg-surface-container-low rounded-xl border border-outline-variant/10 p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-headline font-semibold text-on-surface flex items-center gap-2.5">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-1.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="activity_zone" size={16} className="text-primary" />
                    </div>
                  </div>
                  Recent Activity
                </h2>
                <Link
                  to="/history"
                  className="text-primary text-sm hover:text-primary/80 flex items-center gap-1 transition-colors group cursor-pointer"
                >
                  {t("common.viewAll")}
                  <MaterialIcon icon="arrow_forward" size={14} className="group-hover:translate-x-0.5 transition-transform" />
                </Link>
              </div>

              {isLoading ? (
                <div className="space-y-3">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="h-16 rounded-xl bg-surface-container-high animate-pulse" />
                  ))}
                </div>
              ) : scansData?.items.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/10 flex items-center justify-center mx-auto mb-4">
                    <MaterialIcon icon="search" size={32} className="text-primary/40" />
                  </div>
                  <p className="text-on-surface-variant/50 mb-4">No scans yet</p>
                  <Link
                    to="/scanner"
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-primary to-primary-container text-on-primary-container text-sm font-headline font-bold uppercase tracking-widest transition-all cursor-pointer"
                  >
                    <MaterialIcon icon="search" size={16} /> Try the Scanner
                  </Link>
                </div>
              ) : (
                <div className="space-y-2">
                  {scansData?.items.slice(0, 8).map((scan, i) => {
                    const cfg = verdictConfig[scan.verdict] || verdictConfig.uncertain;
                    return (
                      <motion.div
                        key={scan.id}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        className="group flex items-center justify-between py-3 px-4 rounded-xl hover:bg-surface-container-high/50 transition-all cursor-default"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <div className={`w-2 h-2 rounded-full ${cfg.color === "text-error" ? "bg-error" : cfg.color === "text-secondary" ? "bg-secondary" : "bg-primary"}`} />
                          <div className="min-w-0">
                            <p className="text-on-surface text-sm font-medium truncate">{scan.file_name}</p>
                            <p className="text-on-surface-variant/40 text-xs flex items-center gap-1">
                              <MaterialIcon icon="schedule" size={12} />
                              {new Date(scan.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${cfg.bg} ${cfg.color} ${cfg.border}`}>
                            {scan.verdict}
                          </span>
                          <span className="text-on-surface-variant/30 text-xs font-mono w-10 text-right">
                            {scan.confidence}%
                          </span>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              )}
            </motion.div>

            {/* Sidebar (1/3 width) */}
            <div className="space-y-4">
              {/* Security Overview */}
              <motion.div
                variants={stagger.item}
                className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-6"
              >
                <h3 className="text-sm font-headline font-semibold text-on-surface flex items-center gap-2 mb-5">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-1.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="fingerprint" size={16} className="text-secondary" />
                    </div>
                  </div>
                  Security Overview
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-on-surface-variant/60 text-sm">Account Status</span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-secondary/10 text-secondary border border-secondary/20">Active</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-on-surface-variant/60 text-sm">Password</span>
                    <span className="text-on-surface-variant/40 text-xs">Set</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-on-surface-variant/60 text-sm">Two-Factor</span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">Off</span>
                  </div>
                  <div className="h-px bg-outline-variant/10" />
                  <Link
                    to="/settings"
                    className="flex items-center justify-between text-primary text-sm hover:text-primary/80 transition-colors group cursor-pointer"
                  >
                    Manage Security
                    <MaterialIcon icon="chevron_right" size={16} className="group-hover:translate-x-0.5 transition-transform" />
                  </Link>
                </div>
              </motion.div>

              {/* Quick Actions */}
              <motion.div
                variants={stagger.item}
                className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-6"
              >
                <h3 className="text-sm font-headline font-semibold text-on-surface flex items-center gap-2 mb-5">
                  <div className="bg-gradient-to-br from-primary to-primary-container p-[1px] rounded-lg">
                    <div className="p-1.5 rounded-lg bg-surface-container-low">
                      <MaterialIcon icon="visibility" size={16} className="text-primary" />
                    </div>
                  </div>
                  Quick Actions
                </h3>
                <div className="space-y-2">
                  {[
                    { icon: "search", label: "New Scan", to: "/scanner", color: "text-primary" },
                    { icon: "shield", label: "Call Protection", to: "/call-protection", color: "text-primary" },
                    { icon: "download", label: "Export History", to: "/history", color: "text-secondary" },
                  ].map((action) => (
                    <Link
                      key={action.label}
                      to={action.to}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-surface-container-high/50 transition-colors group cursor-pointer"
                    >
                      <MaterialIcon icon={action.icon} size={16} className={action.color} />
                      <span className="text-sm text-on-surface-variant group-hover:text-on-surface transition-colors">{action.label}</span>
                      <MaterialIcon icon="chevron_right" size={14} className="text-on-surface-variant/30 ml-auto group-hover:translate-x-0.5 transition-transform" />
                    </Link>
                  ))}
                </div>
              </motion.div>

              {/* Scan Distribution */}
              {scansData && scansData.items.length > 0 && (
                <motion.div
                  variants={stagger.item}
                  className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-6"
                >
                  <h3 className="text-sm font-headline font-semibold text-on-surface mb-4">Verdict Distribution</h3>
                  <div className="space-y-3">
                    {[
                      { label: "AI Generated", count: threats, total: scansData.items.length, color: "bg-error" },
                      { label: "Authentic", count: safe, total: scansData.items.length, color: "bg-secondary" },
                      { label: "Uncertain", count: scansData.items.length - threats - safe, total: scansData.items.length, color: "bg-primary" },
                    ].map((item) => (
                      <div key={item.label}>
                        <div className="flex items-center justify-between text-xs mb-1.5">
                          <span className="text-on-surface-variant/60">{item.label}</span>
                          <span className="text-on-surface font-mono">{item.count}</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-surface-container-high overflow-hidden">
                          <motion.div
                            className={`h-full rounded-full ${item.color}`}
                            initial={{ width: 0 }}
                            animate={{ width: item.total > 0 ? `${(item.count / item.total) * 100}%` : "0%" }}
                            transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1], delay: 0.3 }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </Layout>
  );
};

export default Profile;
