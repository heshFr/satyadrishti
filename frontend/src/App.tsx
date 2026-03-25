import { lazy, Suspense, useEffect, useSyncExternalStore } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { AuthProvider } from "./contexts/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";

// Eager-loaded core pages
import Landing from "./pages/Landing";
import Scanner from "./pages/Scanner";

// Lazy-loaded pages
const CallProtection = lazy(() => import("./pages/CallProtection"));
const CallHistory = lazy(() => import("./pages/History"));
const SettingsPage = lazy(() => import("./pages/Settings"));
const Advanced = lazy(() => import("./pages/Advanced"));
const Login = lazy(() => import("./pages/Login"));
const Register = lazy(() => import("./pages/Register"));
const Help = lazy(() => import("./pages/Help"));
const Contact = lazy(() => import("./pages/Contact"));
const Profile = lazy(() => import("./pages/Profile"));
const VoiceEnroll = lazy(() => import("./pages/VoiceEnroll"));
const Privacy = lazy(() => import("./pages/Privacy"));
const NotFound = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

// Global theme + font size watcher — syncs localStorage → DOM on any page
function useThemeSync() {
  // Subscribe to storage events (cross-tab) and re-read on changes
  const theme = useSyncExternalStore(
    (cb) => { window.addEventListener("storage", cb); return () => window.removeEventListener("storage", cb); },
    () => { try { return JSON.parse(localStorage.getItem("satya-theme") || '"dark"'); } catch { return "dark"; } },
  );
  const fontSize = useSyncExternalStore(
    (cb) => { window.addEventListener("storage", cb); return () => window.removeEventListener("storage", cb); },
    () => { try { return JSON.parse(localStorage.getItem("satya-font-size") || "14"); } catch { return 14; } },
  );

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "light") {
      root.classList.remove("dark");
      root.classList.add("light");
    } else if (theme === "system") {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.classList.toggle("dark", prefersDark);
      root.classList.toggle("light", !prefersDark);
    } else {
      root.classList.add("dark");
      root.classList.remove("light");
    }
  }, [theme]);

  useEffect(() => {
    if (fontSize >= 12 && fontSize <= 20) {
      document.documentElement.style.fontSize = `${fontSize}px`;
    }
    return () => { document.documentElement.style.fontSize = ""; };
  }, [fontSize]);

  return theme === "light" ? "light" as const : "dark" as const;
}

function LoadingSpinner() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

function App() {
  const toasterTheme = useThemeSync();
  return (
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Toaster position="top-right" theme={toasterTheme} />
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              {/* Public routes */}
              <Route path="/" element={<Landing />} />
              <Route path="/scanner" element={<Scanner />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/help" element={<Help />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/privacy" element={<Privacy />} />

              {/* Auth-optional routes */}
              <Route path="/call-protection" element={<CallProtection />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/advanced" element={<Advanced />} />
              <Route path="/voice-prints" element={<VoiceEnroll />} />

              {/* Auth-required routes */}
              <Route
                path="/history"
                element={
                  <ProtectedRoute requireAuth>
                    <CallHistory />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/profile"
                element={
                  <ProtectedRoute requireAuth>
                    <Profile />
                  </ProtectedRoute>
                }
              />

              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </QueryClientProvider>
    </AuthProvider>
  );
}

export default App;
