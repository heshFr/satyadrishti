import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/contexts/AuthContext";
import LanguageToggle from "./LanguageToggle";
import UserMenu from "./UserMenu";
import MaterialIcon from "./MaterialIcon";

interface TopBarProps {
  systemStatus: "protected" | "monitoring" | "alert";
}

const TopBar = ({ systemStatus }: TopBarProps) => {
  const { t } = useTranslation();
  const location = useLocation();
  const { isAuthenticated } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const navItems: { path: string; label: string; icon?: string; highlight?: boolean }[] = [
    { path: "/call-protection", label: "Call Protection", icon: "shield", highlight: true },
    { path: "/scanner", label: t("common.scanner"), icon: "search" },
    { path: "/history", label: t("common.history"), icon: "history" },
    { path: "/help", label: "Help", icon: "help_outline" },
    { path: "/contact", label: "Contact", icon: "mail" },
  ];

  const isLanding = location.pathname === "/";

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled || !isLanding
          ? "bg-surface/85 backdrop-blur-2xl shadow-[0_4px_30px_rgba(0,0,0,0.3)] border-b border-outline-variant/10"
          : "bg-transparent"
      }`}
    >
      <div className="flex justify-between items-center max-w-[1600px] mx-auto px-6 lg:px-10 h-24">
        {/* Logo */}
        <div className="flex items-center gap-10 lg:gap-14">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-[0_0_20px_rgba(0,209,255,0.3)] group-hover:shadow-[0_0_30px_rgba(0,209,255,0.5)] transition-shadow">
              <MaterialIcon icon="shield" size={22} filled className="text-on-primary" />
            </div>
            <span className="text-2xl font-black tracking-tighter text-on-surface font-headline">
              Satya Drishti
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden lg:flex gap-2 items-center">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`font-headline text-[15px] tracking-tight transition-all duration-300 flex items-center gap-2 px-5 py-2.5 rounded-xl ${
                    isActive
                      ? item.highlight
                        ? "bg-primary/15 text-primary font-extrabold shadow-[0_0_15px_rgba(0,209,255,0.15)]"
                        : "bg-surface-container-high/60 text-on-surface font-bold"
                      : item.highlight
                        ? "text-primary font-bold hover:bg-primary/10 hover:shadow-[0_0_15px_rgba(0,209,255,0.1)]"
                        : "text-on-surface-variant font-semibold hover:text-on-surface hover:bg-surface-container-high/40"
                  }`}
                >
                  {item.icon && <MaterialIcon icon={item.icon} size={20} filled={isActive || !!item.highlight} />}
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-4 lg:gap-5">
          {/* System status */}
          <div className="hidden md:flex items-center gap-2.5 px-4 py-2 rounded-xl bg-surface-container-high/50 border border-outline-variant/10">
            <div
              className={`h-2.5 w-2.5 rounded-full ${
                systemStatus === "protected"
                  ? "bg-secondary shadow-[0_0_12px_rgba(78,222,163,0.6)]"
                  : systemStatus === "alert"
                  ? "bg-error animate-pulse shadow-[0_0_12px_rgba(255,180,171,0.6)]"
                  : "bg-primary animate-pulse shadow-[0_0_12px_rgba(0,209,255,0.6)]"
              }`}
            />
            <span className="text-sm font-headline font-bold uppercase tracking-widest text-on-surface-variant">
              {systemStatus === "protected" ? "Protected" : systemStatus === "alert" ? "Alert" : "Monitoring"}
            </span>
          </div>

          <LanguageToggle />

          <button className="p-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high/50 transition-all duration-300 rounded-xl">
            <MaterialIcon icon="notifications" size={24} />
          </button>

          {isAuthenticated ? (
            <UserMenu />
          ) : (
            <Link
              to="/login"
              className="hidden sm:flex items-center gap-2 bg-gradient-to-br from-primary to-primary-container text-on-primary px-7 py-3 rounded-xl font-headline tracking-tight font-extrabold text-[15px] uppercase shadow-[0_4px_20px_rgba(0,209,255,0.25)] hover:shadow-[0_4px_30px_rgba(0,209,255,0.4)] transition-all active:scale-95"
            >
              <MaterialIcon icon="login" size={20} />
              {t("common.login")}
            </Link>
          )}

          {/* Mobile hamburger */}
          <button
            className="lg:hidden p-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high/50 transition-all rounded-xl"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            <MaterialIcon icon={mobileMenuOpen ? "close" : "menu"} size={28} />
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="lg:hidden overflow-hidden bg-surface-container-low/95 backdrop-blur-2xl border-t border-outline-variant/10"
          >
            <div className="px-6 py-5 space-y-2">
              {navItems.map((item, i) => (
                <motion.div
                  key={item.path}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <Link
                    to={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center gap-3 px-5 py-4 rounded-xl font-headline text-base tracking-wide uppercase transition-all ${
                      location.pathname === item.path
                        ? "text-primary font-extrabold bg-primary/10 border border-primary/20"
                        : item.highlight
                          ? "text-primary font-bold hover:bg-primary/10"
                          : "text-on-surface-variant font-semibold hover:text-on-surface hover:bg-surface-container-high/30"
                    }`}
                  >
                    {item.icon && <MaterialIcon icon={item.icon} size={22} filled={!!item.highlight} />}
                    {item.label}
                  </Link>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

export default TopBar;
