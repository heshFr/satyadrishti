import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/contexts/AuthContext";

import UserMenu from "./UserMenu";
import MaterialIcon from "./MaterialIcon";
import LanguageDropdown from "./LanguageDropdown";

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

  const isLanding = location.pathname === "/";

  const navItems = isLanding
    ? [
        { path: "/about", label: t("common.about"), icon: "info" },
        { path: "/donate", label: t("common.donate"), icon: "favorite" },
      ]
    : [
        { path: "/call-protection", label: t("common.callProtection"), icon: "shield", highlight: true },
        { path: "/scanner", label: t("common.scanner"), icon: "search" },
        { path: "/history", label: t("common.history"), icon: "history" },
        { path: "/help", label: t("common.help"), icon: "help_outline" },
        { path: "/contact", label: t("common.contact"), icon: "mail" },
        { path: "/donate", label: t("common.donate"), icon: "favorite" },
      ];

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-all duration-500 ${
        scrolled || !isLanding
          ? "bg-black/40 backdrop-blur-xl border-b border-white/5 shadow-[0_4px_30px_rgba(0,0,0,0.3)]"
          : "bg-transparent border-b border-transparent"
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
          <div className="hidden lg:flex gap-8 items-center ml-4">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`relative group font-headline text-[15px] tracking-[0.15em] uppercase transition-all duration-300 flex items-center gap-2 py-2 ${
                    isActive
                      ? "text-primary font-extrabold"
                      : "text-on-surface-variant font-bold hover:text-primary-container"
                  }`}
                >
                  {item.icon && <MaterialIcon icon={item.icon} size={18} filled={isActive} />}
                  {item.label}
                  {/* Underline expansion animation */}
                  <span 
                    className={`absolute bottom-0 left-0 h-[2px] bg-gradient-to-r from-primary-container to-secondary transition-all duration-300 ease-out ${
                      isActive ? "w-full" : "w-0 group-hover:w-full"
                    }`}
                  />
                </Link>
              );
            })}
          </div>
        </div>

        {/* Right side actions */}
        <div className="flex items-center gap-4 lg:gap-6">
          <div className="hidden lg:flex items-center gap-6">
            <LanguageDropdown />
            
            <div className="h-6 w-px bg-outline-variant/30 mx-2"></div>

            {/* Settings — only on app pages */}
            {!isLanding && (
              <Link
                to="/settings"
                className="p-2.5 text-on-surface-variant hover:text-primary transition-all duration-300 rounded-full hover:bg-white/5 active:scale-95"
                title="Settings"
              >
                <MaterialIcon icon="settings" size={22} />
              </Link>
            )}

            {isAuthenticated ? (
              <UserMenu />
            ) : isLanding ? (
              /* Landing: Login + Sign Up (cyan glow) */
              <div className="flex items-center gap-6">
                <Link
                  to="/login"
                  className="text-on-surface-variant hover:text-primary font-headline tracking-widest font-bold text-[14px] uppercase transition-colors active:scale-95"
                >
                  {t("common.login")}
                </Link>
                <Link
                  to="/register"
                  className="px-7 py-2.5 bg-gradient-to-r from-[#00d1ff] to-[#00d1ff]/80 text-[#070d1f] font-headline font-black text-[14px] uppercase tracking-widest rounded-full shadow-[0_0_25px_rgba(0,209,255,0.4)] hover:shadow-[0_0_40px_rgba(0,209,255,0.6)] transition-all active:scale-95 hover:scale-[1.05]"
                >
                  {t("common.signUp")}
                </Link>
              </div>
            ) : (
              /* App pages: Login + Get Protected */
              <div className="flex items-center gap-8">
                <Link
                  to="/login"
                  className="text-on-surface-variant hover:text-primary font-headline tracking-widest font-bold text-[14px] uppercase transition-colors active:scale-95"
                >
                  {t("common.login")}
                </Link>
                <Link
                  to="/call-protection"
                  className="px-6 py-2.5 bg-gradient-to-r from-primary to-secondary text-surface font-headline font-black text-[14px] uppercase tracking-widest rounded-full shadow-[0_0_20px_rgba(0,209,255,0.2)] hover:shadow-[0_0_30px_rgba(0,209,255,0.4)] transition-all active:scale-95 hover:scale-[1.05]"
                >
                  {t("common.getProtected")} →
                </Link>
              </div>
            )}
          </div>

          {/* Mobile controls */}
          <div className="lg:hidden flex items-center gap-4">
            <LanguageDropdown />
            <button
              className="p-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high/50 transition-all rounded-xl active:scale-95"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <MaterialIcon icon={mobileMenuOpen ? "close" : "menu"} size={28} />
            </button>
          </div>
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
