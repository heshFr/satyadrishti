import { useTranslation } from "react-i18next";

const languages = [
  { code: "en", label: "EN" },
  { code: "hi", label: "HI" },
  { code: "mr", label: "MR" },
] as const;

const LanguageToggle = () => {
  const { i18n } = useTranslation();
  const current = i18n.language || "en";

  const toggle = (lang: string) => {
    i18n.changeLanguage(lang);
    localStorage.setItem("satya-lang", lang);
  };

  return (
    <div className="flex items-center rounded-lg border border-border overflow-hidden text-xs">
      {languages.map(({ code, label }) => (
        <button
          key={code}
          onClick={() => toggle(code)}
          className={`px-2.5 py-1.5 font-mono font-medium transition-colors cursor-pointer ${
            current === code || (code === "en" && !languages.some((l) => l.code === current))
              ? "bg-gradient-to-r from-primary to-accent text-white"
              : "text-muted-foreground hover:text-white"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
};

export default LanguageToggle;
