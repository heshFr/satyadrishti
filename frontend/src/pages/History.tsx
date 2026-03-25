import { useState, useMemo, useCallback } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import Layout from "@/components/Layout";
import MaterialIcon from "@/components/MaterialIcon";
import { api } from "@/lib/api";
import type { Scan } from "@/lib/types";

type VerdictFilter = "all" | "ai-generated" | "authentic" | "uncertain";

const verdictConfig = {
  "ai-generated": { label: "AI Generated", icon: "gpp_bad", bg: "bg-error-container/20", text: "text-error", border: "border-error/20", badgeBg: "bg-error-container", badgeText: "text-on-error-container" },
  authentic: { label: "Authentic", icon: "verified_user", bg: "bg-secondary-container/20", text: "text-secondary", border: "border-secondary/20", badgeBg: "bg-secondary-container", badgeText: "text-on-secondary-container" },
  uncertain: { label: "Inconclusive", icon: "help", bg: "bg-surface-container-high", text: "text-on-surface-variant", border: "border-outline-variant/30", badgeBg: "bg-surface-container-high", badgeText: "text-on-surface-variant" },
};

const fileIcon = (name: string) => {
  const ext = name.split(".").pop()?.toLowerCase() || "";
  if (["mp4", "mov", "avi", "webm"].includes(ext)) return "movie";
  if (["wav", "mp3", "ogg", "flac"].includes(ext)) return "audio_file";
  return "image";
};

const CallHistory = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [selectedScan, setSelectedScan] = useState<Scan | null>(null);
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("all");
  const [showFilterMenu, setShowFilterMenu] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["scans", page],
    queryFn: () => api.scans.list(page),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.scans.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["scans"] });
      setSelectedScan(null);
      toast.success(t("history.deleted"));
    },
  });

  const totalPages = data ? Math.ceil(data.total / data.per_page) : 0;

  const verdictCounts = useMemo(() => {
    if (!data || data.items.length === 0) return null;
    const counts = { "ai-generated": 0, authentic: 0, uncertain: 0 };
    for (const scan of data.items) {
      const key = scan.verdict as keyof typeof counts;
      if (key in counts) counts[key]++;
      else counts.uncertain++;
    }
    return counts;
  }, [data]);

  const total = verdictCounts
    ? verdictCounts["ai-generated"] + verdictCounts.authentic + verdictCounts.uncertain
    : 0;

  const filteredItems = useMemo(() => {
    if (!data) return [];
    if (verdictFilter === "all") return data.items;
    return data.items.filter((s) => s.verdict === verdictFilter);
  }, [data, verdictFilter]);

  const handleExport = useCallback(() => {
    if (!data || data.items.length === 0) { toast.error("No scans to export"); return; }
    const rows = [
      ["File Name", "Verdict", "Confidence", "Type", "Date"].join(","),
      ...data.items.map((s) =>
        [`"${s.file_name}"`, s.verdict, `${s.confidence}%`, s.file_type, new Date(s.created_at).toLocaleString()].join(",")
      ),
    ];
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `satyadrishti_history_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click(); URL.revokeObjectURL(url);
    toast.success("History exported as CSV");
  }, [data]);

  return (
    <Layout systemStatus="protected">
      <div className="pt-32 pb-20 px-6 md:px-10 max-w-5xl mx-auto space-y-10">
        {/* Header */}
        <div className="flex justify-between items-end">
          <div>
            <h1 className="font-headline text-4xl font-extrabold tracking-tighter text-on-surface">
              Scan History<span className="text-primary-container">.</span>
            </h1>
            <p className="text-on-surface-variant font-label text-xs uppercase tracking-[0.2em] mt-2">
              {t("history.subtitle")}
            </p>
          </div>
          <div className="flex gap-4 relative">
            <div className="relative">
              <button
                onClick={() => setShowFilterMenu((p) => !p)}
                className={`px-4 py-2 rounded flex items-center gap-2 text-xs font-bold tracking-widest uppercase border transition-all cursor-pointer ${
                  verdictFilter !== "all"
                    ? "bg-primary/10 border-primary/40 text-primary"
                    : "bg-surface-container-high border-outline-variant/20 hover:border-primary/40"
                }`}
              >
                <MaterialIcon icon="filter_list" size={16} /> Filter{verdictFilter !== "all" ? ": " + verdictFilter : ""}
              </button>
              {showFilterMenu && (
                <div className="absolute right-0 top-full mt-2 w-48 bg-surface-container-low border border-outline-variant/20 rounded-xl shadow-xl z-50 overflow-hidden">
                  {(["all", "ai-generated", "authentic", "uncertain"] as VerdictFilter[]).map((f) => (
                    <button
                      key={f}
                      onClick={() => { setVerdictFilter(f); setShowFilterMenu(false); }}
                      className={`w-full text-left px-4 py-3 text-xs font-bold uppercase tracking-widest transition-colors cursor-pointer ${
                        verdictFilter === f ? "bg-primary/10 text-primary" : "text-on-surface-variant hover:bg-surface-container-high"
                      }`}
                    >
                      {f === "all" ? "All Verdicts" : verdictConfig[f as keyof typeof verdictConfig]?.label || f}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={handleExport}
              className="px-4 py-2 bg-surface-container-high rounded flex items-center gap-2 text-xs font-bold tracking-widest uppercase border border-outline-variant/20 hover:border-primary/40 transition-all cursor-pointer"
            >
              <MaterialIcon icon="download" size={16} /> Export
            </button>
          </div>
        </div>

        {/* Verdict distribution */}
        {verdictCounts && total > 0 && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              {([
                { key: "ai-generated" as const, icon: "gpp_bad", color: "text-error", borderColor: "border-error/20" },
                { key: "authentic" as const, icon: "verified_user", color: "text-secondary", borderColor: "border-secondary/20" },
                { key: "uncertain" as const, icon: "help", color: "text-on-surface-variant", borderColor: "border-outline-variant/20" },
              ] as const).map(({ key, icon, color, borderColor }) => (
                <div key={key} className={`bg-surface-container-low p-4 rounded-xl border-b-2 ${borderColor}`}>
                  <MaterialIcon icon={icon} size={20} className={color} />
                  <p className={`text-2xl font-headline font-black mt-2 ${color}`}>{verdictCounts[key]}</p>
                  <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mt-1">{verdictConfig[key].label}</p>
                </div>
              ))}
            </div>
            <div className="h-1.5 rounded-full overflow-hidden flex bg-surface-container-highest">
              {verdictCounts["ai-generated"] > 0 && (
                <div className="h-full bg-error transition-all duration-500" style={{ width: `${(verdictCounts["ai-generated"] / total) * 100}%` }} />
              )}
              {verdictCounts.authentic > 0 && (
                <div className="h-full bg-secondary transition-all duration-500" style={{ width: `${(verdictCounts.authentic / total) * 100}%` }} />
              )}
              {verdictCounts.uncertain > 0 && (
                <div className="h-full bg-outline transition-all duration-500" style={{ width: `${(verdictCounts.uncertain / total) * 100}%` }} />
              )}
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {isLoading && (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-surface-container-low p-6 rounded-xl border border-outline-variant/10">
                <div className="flex items-center gap-6">
                  <div className="w-12 h-12 rounded-lg bg-surface-container-high animate-pulse" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-surface-container-high animate-pulse rounded w-1/3" />
                    <div className="h-3 bg-surface-container-high animate-pulse rounded w-1/4" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {!isLoading && (!data || data.items.length === 0) && (
          <div className="text-center py-20">
            <div className="w-20 h-20 rounded-xl bg-surface-container-high flex items-center justify-center mx-auto mb-6">
              <MaterialIcon icon="search" size={40} className="text-on-surface-variant/30" />
            </div>
            <p className="text-on-surface-variant mb-6">{t("history.empty")}</p>
            <Link to="/scanner"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-lg bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-headline font-extrabold text-sm tracking-widest uppercase transition-all hover:scale-[1.02] cursor-pointer"
            >
              {t("history.goToScanner")}
            </Link>
          </div>
        )}

        {/* Scan list */}
        {data && data.items.length > 0 && (
          <div className="flex flex-col gap-4">
            {filteredItems.length === 0 && verdictFilter !== "all" && (
              <p className="text-center text-on-surface-variant py-10">No scans match filter &ldquo;{verdictFilter}&rdquo;</p>
            )}
            {filteredItems.map((scan, i) => {
              const config = verdictConfig[scan.verdict as keyof typeof verdictConfig] || verdictConfig.uncertain;
              return (
                <motion.div
                  key={scan.id}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => setSelectedScan(scan)}
                  className="glass-panel p-6 rounded-xl border border-outline-variant/10 hover:border-primary/30 transition-all cursor-pointer group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                      <div className="w-12 h-12 rounded-lg bg-surface-container-highest flex items-center justify-center">
                        <MaterialIcon icon={fileIcon(scan.file_name)} size={24} className={config.text} />
                      </div>
                      <div>
                        <h3 className="font-headline font-bold text-on-surface group-hover:text-primary transition-colors">{scan.file_name}</h3>
                        <div className="flex gap-4 mt-1">
                          <span className="text-[10px] text-on-surface-variant uppercase tracking-widest">
                            {new Date(scan.created_at).toLocaleString()}
                          </span>
                          <span className="text-[10px] text-on-surface-variant uppercase tracking-widest">
                            {scan.file_type || "Media"}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-8 text-right">
                      <div>
                        <div className={`px-3 py-1 rounded-full ${config.badgeBg} ${config.badgeText} text-[10px] font-black uppercase tracking-widest`}>
                          Verdict: {config.label}
                        </div>
                      </div>
                      <div className="hidden sm:block">
                        <p className="text-xl font-headline font-black text-on-surface">{scan.confidence}%</p>
                        <p className="text-[10px] text-on-surface-variant uppercase tracking-widest">Confidence</p>
                      </div>
                      <MaterialIcon icon="chevron_right" size={20} className="text-on-surface-variant group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </motion.div>
              );
            })}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center gap-2 pt-6">
                {Array.from({ length: totalPages }, (_, i) => (
                  <button
                    key={i}
                    onClick={() => setPage(i + 1)}
                    className={`w-9 h-9 rounded-lg text-sm font-medium cursor-pointer transition-all duration-300 ${
                      page === i + 1
                        ? "bg-primary text-on-primary"
                        : "bg-surface-container-high text-on-surface-variant hover:text-on-surface"
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Detail modal */}
      <AnimatePresence>
        {selectedScan && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedScan(null)}
          >
            <div className="absolute inset-0 bg-surface/80 backdrop-blur-md" />

            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              transition={{ type: "spring", stiffness: 200, damping: 25 }}
              className="relative w-full max-w-lg bg-surface-container-low rounded-xl shadow-2xl overflow-hidden border border-outline-variant/10"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => setSelectedScan(null)}
                className="absolute top-5 right-5 p-2 rounded-lg hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface transition-all z-10 cursor-pointer"
              >
                <MaterialIcon icon="close" size={20} />
              </button>

              <div className="p-8">
                <div className="flex items-center gap-4 mb-8">
                  <div className="w-14 h-14 rounded-xl bg-surface-container-high flex items-center justify-center">
                    <MaterialIcon icon={fileIcon(selectedScan.file_name)} size={28} className="text-primary" />
                  </div>
                  <div>
                    <h2 className="text-xl font-headline font-bold text-on-surface">{selectedScan.file_name}</h2>
                    <p className="text-on-surface-variant text-sm">
                      {new Date(selectedScan.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="p-5 rounded-xl bg-surface-container-high text-center">
                    <p className="text-xs text-on-surface-variant uppercase tracking-widest mb-2 font-label">Verdict</p>
                    <p className={`text-lg font-headline font-bold ${
                      (verdictConfig[selectedScan.verdict as keyof typeof verdictConfig] || verdictConfig.uncertain).text
                    }`}>
                      {selectedScan.verdict}
                    </p>
                  </div>
                  <div className="p-5 rounded-xl bg-surface-container-high text-center">
                    <p className="text-xs text-on-surface-variant uppercase tracking-widest mb-2 font-label">
                      {t("scanner.confidenceLabel")}
                    </p>
                    <p className="text-lg font-headline font-bold text-on-surface">{selectedScan.confidence}%</p>
                  </div>
                </div>

                {selectedScan.forensic_data && selectedScan.forensic_data.length > 0 && (
                  <div className="space-y-2 mb-8">
                    <h4 className="text-sm font-headline font-semibold text-on-surface mb-3">{t("scanner.forensicEvidence")}</h4>
                    {selectedScan.forensic_data.map((item, i) => (
                      <div key={i} className="flex items-center justify-between py-2.5 px-4 rounded-lg bg-surface-container-high text-sm">
                        <span className="text-on-surface-variant">{item.name || item.label}</span>
                        <span className={`font-semibold ${
                          item.status === "pass" ? "text-secondary" : item.status === "fail" ? "text-error" : "text-primary"
                        }`}>
                          {item.status.toUpperCase()}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div className="flex items-center gap-3">
                  <a
                    href={api.scans.reportUrl(selectedScan.id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-primary to-primary-container text-on-primary-container text-sm font-headline font-bold transition-all cursor-pointer"
                  >
                    <MaterialIcon icon="download" size={16} /> {t("common.downloadReport")}
                  </a>
                  <button
                    onClick={() => deleteMutation.mutate(selectedScan.id)}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-error/20 text-error text-sm font-medium hover:bg-error-container/10 transition-all cursor-pointer"
                  >
                    <MaterialIcon icon="delete" size={16} />
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </Layout>
  );
};

export default CallHistory;
