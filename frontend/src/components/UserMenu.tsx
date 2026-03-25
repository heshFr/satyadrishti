import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { User, Settings, LogOut } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const UserMenu = () => {
  const { t } = useTranslation();
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!user) return null;

  const initial = user.name.charAt(0).toUpperCase();

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 text-primary text-sm font-display font-bold flex items-center justify-center hover:shadow-glow-sm transition-all cursor-pointer"
      >
        {initial}
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-48 glass-card border border-white/[0.08] rounded-xl shadow-lg py-1 z-50">
          <div className="px-4 py-2 border-b border-white/[0.06]">
            <p className="text-white text-sm font-display font-medium truncate">{user.name}</p>
            <p className="text-muted-foreground text-xs font-mono truncate">{user.email}</p>
          </div>
          <Link
            to="/profile"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-4 py-2.5 text-sm text-muted-foreground hover:text-white hover:bg-white/[0.04] transition-colors cursor-pointer"
          >
            <User className="w-4 h-4" /> {t("common.profile")}
          </Link>
          <Link
            to="/settings"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-4 py-2.5 text-sm text-muted-foreground hover:text-white hover:bg-white/[0.04] transition-colors cursor-pointer"
          >
            <Settings className="w-4 h-4" /> {t("common.settings")}
          </Link>
          <button
            onClick={() => {
              logout();
              setOpen(false);
            }}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-danger hover:bg-danger/10 transition-colors cursor-pointer"
          >
            <LogOut className="w-4 h-4" /> {t("common.logout")}
          </button>
        </div>
      )}
    </div>
  );
};

export default UserMenu;
