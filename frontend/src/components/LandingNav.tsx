import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Shield, Menu, X, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/contexts/AuthContext";


const navLinks = [
  { to: "#features", label: "Features" },
  { to: "/call-protection", label: "Call Protection" },
  { to: "/scanner", label: "Scanner" },
];

const LandingNav = () => {
  const { t } = useTranslation();
  const { isAuthenticated } = useAuth();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrolled ? "py-3" : "py-5"
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 md:px-12">
          <div className={`flex items-center justify-between transition-all duration-500 ${
            scrolled
              ? "floating-nav rounded-2xl px-6 h-14"
              : "bg-transparent px-2 h-16"
          }`}>
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2.5 flex-shrink-0 cursor-pointer">
              <div className="relative">
                <div className="absolute inset-0 bg-primary/20 blur-lg rounded-full" />
                <Shield className="w-6 h-6 text-primary relative drop-shadow-[0_0_10px_rgba(6,182,212,0.5)]" fill="currentColor" fillOpacity={0.15} />
              </div>
              <span className="text-lg font-display font-bold text-white">{t("common.appName")}</span>
            </Link>

            {/* Desktop nav links */}
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  className="px-4 py-2 rounded-lg text-sm font-display font-medium text-muted-foreground hover:text-white transition-colors duration-200 cursor-pointer"
                >
                  {link.label}
                </Link>
              ))}
            </div>

            {/* Right side */}
            <div className="flex items-center gap-3">


              {isAuthenticated ? (
                <Link
                  to="/hub"
                  className="hidden md:flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-primary to-accent text-white text-sm font-display font-semibold hover:shadow-glow-sm transition-all cursor-pointer"
                >
                  {t("common.dashboard")}
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              ) : (
                <div className="hidden md:flex items-center gap-2">
                  <Link
                    to="/login"
                    className="px-4 py-2 rounded-lg text-sm font-display font-medium text-muted-foreground hover:text-white transition-colors cursor-pointer"
                  >
                    {t("common.login")}
                  </Link>
                  <Link
                    to="/register"
                    className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-primary to-accent text-white text-sm font-display font-semibold hover:shadow-glow-sm transition-all cursor-pointer"
                  >
                    Sign Up
                  </Link>
                </div>
              )}

              <button
                onClick={() => setMobileOpen(!mobileOpen)}
                className="md:hidden p-2 rounded-lg text-muted-foreground hover:text-white transition-colors cursor-pointer"
              >
                {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="fixed top-20 left-4 right-4 z-50 floating-nav rounded-2xl p-4 md:hidden"
          >
            <div className="space-y-1">
              {navLinks.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileOpen(false)}
                  className="block px-4 py-3 rounded-xl text-sm font-display font-medium text-muted-foreground hover:text-white hover:bg-white/[0.04] transition-all cursor-pointer"
                >
                  {link.label}
                </Link>
              ))}
              <div className="border-t border-white/[0.04] my-2" />

              {!isAuthenticated && (
                <div className="flex items-center gap-2 pt-2">
                  <Link
                    to="/login"
                    onClick={() => setMobileOpen(false)}
                    className="flex-1 text-center px-4 py-2.5 rounded-xl text-sm font-display font-medium text-white border border-white/10 cursor-pointer"
                  >
                    {t("common.login")}
                  </Link>
                  <Link
                    to="/register"
                    onClick={() => setMobileOpen(false)}
                    className="flex-1 text-center px-4 py-2.5 rounded-xl bg-gradient-to-r from-primary to-accent text-white text-sm font-display font-semibold cursor-pointer"
                  >
                    Sign Up
                  </Link>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default LandingNav;
