import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Sentinel Design System — "The Digital Curator"
        surface: {
          DEFAULT: "#0c1324",
          dim: "#0c1324",
          bright: "#33394c",
          container: {
            DEFAULT: "#191f31",
            low: "#151b2d",
            high: "#23293c",
            highest: "#2e3447",
            lowest: "#070d1f",
          },
          variant: "#2e3447",
          tint: "#4cd6ff",
        },
        "on-surface": {
          DEFAULT: "#dce1fb",
          variant: "#bbc9cf",
        },
        primary: {
          DEFAULT: "#a4e6ff",
          container: "#00d1ff",
          fixed: "#b7eaff",
          "fixed-dim": "#4cd6ff",
        },
        "on-primary": {
          DEFAULT: "#003543",
          container: "#00566a",
          fixed: "#001f28",
          "fixed-variant": "#004e60",
        },
        secondary: {
          DEFAULT: "#4edea3",
          container: "#00a572",
          fixed: "#6ffbbe",
          "fixed-dim": "#4edea3",
        },
        "on-secondary": {
          DEFAULT: "#003824",
          container: "#00311f",
          fixed: "#002113",
          "fixed-variant": "#005236",
        },
        tertiary: {
          DEFAULT: "#cfdcff",
          container: "#a4c0ff",
          fixed: "#d8e2ff",
          "fixed-dim": "#adc6ff",
        },
        "on-tertiary": {
          DEFAULT: "#002e6a",
          container: "#004ba3",
          fixed: "#001a42",
          "fixed-variant": "#004395",
        },
        error: {
          DEFAULT: "#ffb4ab",
          container: "#93000a",
        },
        "on-error": {
          DEFAULT: "#690005",
          container: "#ffdad6",
        },
        outline: {
          DEFAULT: "#859399",
          variant: "#3c494e",
        },
        inverse: {
          surface: "#dce1fb",
          "on-surface": "#2a3043",
          primary: "#00677f",
        },
        background: "#0c1324",
        "on-background": "#dce1fb",

        // Legacy aliases for compatibility
        foreground: "#dce1fb",
        safe: "#4edea3",
        warning: "#F59E0B",
        danger: "#ffb4ab",
      },
      fontFamily: {
        headline: ['"Manrope"', "system-ui", "sans-serif"],
        body: ['"Inter"', "system-ui", "sans-serif"],
        label: ['"Inter"', "system-ui", "sans-serif"],
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
      fontSize: {
        "display-lg": ["3.5rem", { lineHeight: "1.05", fontWeight: "800", letterSpacing: "-0.02em" }],
        "display-md": ["2.5rem", { lineHeight: "1.1", fontWeight: "800", letterSpacing: "-0.02em" }],
        "title-lg": ["2rem", { lineHeight: "1.2", fontWeight: "700" }],
        "title-sm": ["1.25rem", { lineHeight: "1.3", fontWeight: "600" }],
        "label-lg": ["0.875rem", { lineHeight: "1.4", fontWeight: "500" }],
        "label-md": ["0.75rem", { lineHeight: "1.4", fontWeight: "600", letterSpacing: "0.05em" }],
        "label-sm": ["0.625rem", { lineHeight: "1.4", fontWeight: "700", letterSpacing: "0.2em" }],
        // Legacy
        "hero-xl": ["5.5rem", { lineHeight: "1.05", fontWeight: "800", letterSpacing: "-0.03em" }],
        "hero-lg": ["4rem", { lineHeight: "1.08", fontWeight: "800", letterSpacing: "-0.025em" }],
        "hero-md": ["2.5rem", { lineHeight: "1.1", fontWeight: "700", letterSpacing: "-0.02em" }],
      },
      borderRadius: {
        DEFAULT: "0.125rem",
        lg: "0.25rem",
        xl: "0.5rem",
        "2xl": "1rem",
        "3xl": "1.5rem",
        "4xl": "2rem",
        full: "0.75rem",
      },
      animation: {
        "float": "float 6s ease-in-out infinite",
        "float-slow": "float-slow 10s ease-in-out infinite",
        "glow-pulse": "glow-pulse 3s ease-in-out infinite",
        "gradient-x": "gradient-x 8s ease infinite",
        "gradient-xy": "gradient-xy 12s ease infinite",
        "scan-line": "scan-line 2s ease-in-out infinite",
        "pulse-ring": "pulse-ring 2s ease-out infinite",
        "slide-up": "slide-up-fade 0.6s cubic-bezier(0.16,1,0.3,1) both",
        "shimmer": "shimmer 2.5s linear infinite",
        "spin-slow": "spin 8s linear infinite",
        "spin-medium": "spin 6s linear infinite",
        "spin-orbit": "spin 20s linear infinite",
        "spin-orbit-reverse": "spin 15s linear infinite reverse",
        "aurora": "aurora 8s ease-in-out infinite alternate",
        "morph": "morph 15s ease-in-out infinite",
        "grid-fade": "grid-fade 4s ease-in-out infinite",
        "border-spin": "border-spin 3s linear infinite",
        "danger-pulse": "danger-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "waveform": "pulse-height 1.5s ease-in-out infinite",
        "protection-glow": "protection-glow 3s ease-in-out infinite",
      },
      keyframes: {
        "float": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-15px)" },
        },
        "float-slow": {
          "0%, 100%": { transform: "translateY(0px) rotate(0deg)" },
          "33%": { transform: "translateY(-20px) rotate(2deg)" },
          "66%": { transform: "translateY(-10px) rotate(-1deg)" },
        },
        "glow-pulse": {
          "0%, 100%": { opacity: "0.4", transform: "scale(1)" },
          "50%": { opacity: "0.8", transform: "scale(1.05)" },
        },
        "gradient-x": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        "gradient-xy": {
          "0%, 100%": { backgroundPosition: "0% 0%" },
          "25%": { backgroundPosition: "100% 0%" },
          "50%": { backgroundPosition: "100% 100%" },
          "75%": { backgroundPosition: "0% 100%" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "aurora": {
          "0%": { transform: "translate(0, 0) scale(1)" },
          "50%": { transform: "translate(30px, -20px) scale(1.05)" },
          "100%": { transform: "translate(-20px, 10px) scale(0.95)" },
        },
        "morph": {
          "0%, 100%": { borderRadius: "60% 40% 30% 70% / 60% 30% 70% 40%" },
          "25%": { borderRadius: "30% 60% 70% 40% / 50% 60% 30% 60%" },
          "50%": { borderRadius: "50% 60% 30% 60% / 30% 60% 70% 40%" },
          "75%": { borderRadius: "60% 40% 50% 40% / 40% 50% 60% 50%" },
        },
        "grid-fade": {
          "0%, 100%": { opacity: "0.3" },
          "50%": { opacity: "0.6" },
        },
        "border-spin": {
          "0%": { "--border-angle": "0deg" },
          "100%": { "--border-angle": "360deg" },
        },
        "scan-line": {
          "0%": { top: "0%", opacity: "1" },
          "50%": { opacity: "0.5" },
          "100%": { top: "100%", opacity: "0" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(1)", opacity: "0.6" },
          "100%": { transform: "scale(2.5)", opacity: "0" },
        },
        "slide-up-fade": {
          from: { opacity: "0", transform: "translateY(24px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "danger-pulse": {
          "0%, 100%": { borderColor: "rgba(255, 180, 171, 0.6)" },
          "50%": { borderColor: "rgba(255, 180, 171, 1)" },
        },
        "pulse-height": {
          "0%, 100%": { height: "10px" },
          "50%": { height: "40px" },
        },
        "protection-glow": {
          "0%, 100%": { boxShadow: "0 0 60px 10px rgba(78, 222, 163, 0.2)" },
          "50%": { boxShadow: "0 0 80px 20px rgba(78, 222, 163, 0.3)" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      boxShadow: {
        "glow-sm": "0 0 15px rgba(0, 209, 255, 0.15)",
        "glow": "0 0 30px rgba(0, 209, 255, 0.2)",
        "glow-lg": "0 0 60px rgba(0, 209, 255, 0.25)",
        "glow-cyan": "0 0 40px rgba(0, 209, 255, 0.3)",
        "glow-emerald": "0 0 40px rgba(78, 222, 163, 0.3)",
        "glow-danger": "0 0 30px rgba(255, 180, 171, 0.3)",
        "glow-safe": "0 0 30px rgba(78, 222, 163, 0.3)",
        "ambient": "0 20px 80px rgba(220, 225, 251, 0.04)",
        "card": "0 4px 24px rgba(0, 0, 0, 0.6)",
        "card-hover": "0 8px 40px rgba(0, 209, 255, 0.08), 0 20px 60px rgba(0, 0, 0, 0.5)",
        "elevation-1": "0 1px 3px rgba(0,0,0,0.4)",
        "elevation-2": "0 4px 6px rgba(0,0,0,0.4)",
        "elevation-3": "0 10px 20px rgba(0,0,0,0.4)",
        "elevation-4": "0 20px 40px rgba(0,0,0,0.5)",
        "orb": "0 0 80px rgba(78, 222, 163, 0.3)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
