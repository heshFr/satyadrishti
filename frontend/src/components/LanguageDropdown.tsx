import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "./MaterialIcon";

const languages = [
  { code: "en", label: "English", flag: "🇺🇸" },
  { code: "hi", label: "Hindi", flag: "🇮🇳" },
  { code: "mr", label: "Marathi", flag: "🇮🇳" },
] as const;

const LanguageDropdown = () => {
  const { i18n } = useTranslation();
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentLang = languages.find(l => l.code === (i18n.language || "en")) || languages[0];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const changeLanguage = (code: string) => {
    i18n.changeLanguage(code);
    localStorage.setItem("satya-lang", code);
    setOpen(false);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button 
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-surface-container-high transition-colors border border-transparent hover:border-outline-variant/20 cursor-pointer"
      >
        <span className="text-[1.35rem] leading-none drop-shadow-sm">{currentLang.flag}</span>
        <span className="text-sm font-headline font-semibold text-on-surface hidden sm:block">{currentLang.label}</span>
        <MaterialIcon icon={open ? "expand_less" : "expand_more"} className="text-on-surface-variant text-sm ml-1" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div 
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute right-0 mt-3 w-44 bg-surface/95 backdrop-blur-2xl border border-outline-variant/20 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.2)] overflow-hidden z-50 p-1.5"
          >
            {languages.map((lang) => (
              <button
                key={lang.code}
                onClick={() => changeLanguage(lang.code)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all cursor-pointer ${
                  currentLang.code === lang.code 
                    ? "bg-primary/10 text-primary" 
                    : "hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface"
                }`}
              >
                <span className="text-[1.35rem] leading-none drop-shadow-sm">{lang.flag}</span>
                <span className="font-headline font-semibold text-sm">{lang.label}</span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LanguageDropdown;
