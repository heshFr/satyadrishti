import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from "react";
import { api, ApiError } from "@/lib/api";
import type { User } from "@/lib/types";

// ─── Types ─────────────────────────────────────────────────────────────
interface TwoFactorSetupData {
  secret: string;
  qr_url: string;
  backup_codes: string[];
}

interface LoginResult {
  /** If 2FA is required, this contains the temp token. Normal login returns nothing. */
  requires2FA?: boolean;
  tempToken?: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<LoginResult>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<User | null>;
  setUser: (u: User | null) => void;
  // OAuth
  startOAuth: (provider: "google" | "github") => Promise<void>;
  handleOAuthCallback: (provider: "google" | "github", code: string, state?: string) => Promise<void>;
  // 2FA
  setup2FA: () => Promise<TwoFactorSetupData>;
  confirm2FA: (code: string) => Promise<boolean>;
  verify2FA: (tempToken: string, code: string) => Promise<void>;
  disable2FA: (code: string) => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ─── OAuth Popup Utility ───────────────────────────────────────────────
function openOAuthPopup(url: string, name: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const width = 500;
    const height = 600;
    const left = window.screenX + (window.innerWidth - width) / 2;
    const top = window.screenY + (window.innerHeight - height) / 2;
    const features = `width=${width},height=${height},left=${left},top=${top},toolbar=no,menubar=no,scrollbars=yes,resizable=yes`;

    const popup = window.open(url, name, features);
    if (!popup) {
      reject(new Error("Popup blocked. Please allow popups for this site."));
      return;
    }

    // Poll for the redirect back with the auth code
    const pollInterval = setInterval(() => {
      try {
        if (popup.closed) {
          clearInterval(pollInterval);
          reject(new Error("Authentication cancelled"));
          return;
        }

        const popupUrl = popup.location.href;
        if (popupUrl.includes("/auth/callback") || popupUrl.includes("code=")) {
          clearInterval(pollInterval);
          const urlParams = new URL(popupUrl).searchParams;
          const code = urlParams.get("code");
          popup.close();
          if (code) {
            resolve(code);
          } else {
            reject(new Error("No authorization code received"));
          }
        }
      } catch {
        // Cross-origin error — popup is still on external domain, keep polling
      }
    }, 300);

    // Timeout after 5 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      if (!popup.closed) popup.close();
      reject(new Error("Authentication timed out"));
    }, 5 * 60 * 1000);
  });
}

// ─── Provider ──────────────────────────────────────────────────────────
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Hydrate user from token on mount
  useEffect(() => {
    const token = localStorage.getItem("satya-token");
    if (!token) {
      setIsLoading(false);
      return;
    }

    api.auth
      .me()
      .then(setUser)
      .catch(() => {
        localStorage.removeItem("satya-token");
      })
      .finally(() => setIsLoading(false));
  }, []);

  // ── Standard Login ──
  const login = useCallback(async (email: string, password: string): Promise<LoginResult> => {
    try {
      const response = await api.auth.login(email, password);
      // Check if server requires 2FA verification
      if ((response as any).requires_2fa && (response as any).temp_token) {
        return {
          requires2FA: true,
          tempToken: (response as any).temp_token,
        };
      }
      localStorage.setItem("satya-token", response.token);
      setUser(response.user);
      return {};
    } catch (err) {
      // If the server responds with a 2FA-required status
      if (err instanceof ApiError && err.status === 202) {
        const data = JSON.parse(err.message);
        return { requires2FA: true, tempToken: data.temp_token };
      }
      throw err;
    }
  }, []);

  // ── Registration ──
  const register = useCallback(async (email: string, password: string, name: string) => {
    const response = await api.auth.register(email, password, name);
    localStorage.setItem("satya-token", response.token);
    setUser(response.user);
  }, []);

  // ── Logout ──
  const logout = useCallback(() => {
    localStorage.removeItem("satya-token");
    setUser(null);
  }, []);

  // ── Refresh user from /me ──
  const refreshUser = useCallback(async (): Promise<User | null> => {
    const token = localStorage.getItem("satya-token");
    if (!token) return null;
    try {
      const fresh = await api.auth.me();
      setUser(fresh);
      return fresh;
    } catch {
      return null;
    }
  }, []);

  // ── OAuth: Start Flow ──
  const startOAuth = useCallback(async (provider: "google" | "github") => {
    try {
      // Try to get the OAuth URL from the backend
      const { url } = await api.auth.oauthUrl(provider);
      const code = await openOAuthPopup(url, `${provider}_oauth`);
      const response = await api.auth.oauthCallback(provider, code);
      localStorage.setItem("satya-token", response.token);
      setUser(response.user);
    } catch (err) {
      // If the backend returns 501 (not configured), throw a clear message
      if (err instanceof ApiError && err.status === 501) {
        throw new Error(`${provider} OAuth is not configured on the server.`);
      }
      // If the backend doesn't have OAuth configured yet, use a fallback URL
      if (err instanceof ApiError && (err.status === 404 || err.status === 0)) {
        // Construct OAuth URLs directly (fallback for when backend is offline)
        const clientIds: Record<string, string> = {
          google: import.meta.env.VITE_GOOGLE_CLIENT_ID || "",
          github: import.meta.env.VITE_GITHUB_CLIENT_ID || "",
        };
        const redirectUri = `${window.location.origin}/auth/callback`;

        if (!clientIds[provider]) {
          throw new Error(`${provider} OAuth is not configured.`);
        }

        let authUrl: string;
        if (provider === "google") {
          const params = new URLSearchParams({
            client_id: clientIds.google,
            redirect_uri: redirectUri,
            response_type: "code",
            scope: "openid email profile",
            access_type: "offline",
            prompt: "consent",
          });
          authUrl = `https://accounts.google.com/o/oauth2/v2/auth?${params}`;
        } else {
          const params = new URLSearchParams({
            client_id: clientIds.github,
            redirect_uri: redirectUri,
            scope: "user:email read:user",
          });
          authUrl = `https://github.com/login/oauth/authorize?${params}`;
        }

        const code = await openOAuthPopup(authUrl, `${provider}_oauth`);
        const response = await api.auth.oauthCallback(provider, code);
        localStorage.setItem("satya-token", response.token);
        setUser(response.user);
      } else {
        throw err;
      }
    }
  }, []);

  // ── OAuth: Handle Callback ──
  const handleOAuthCallback = useCallback(async (provider: "google" | "github", code: string, state?: string) => {
    const response = await api.auth.oauthCallback(provider, code, state);
    localStorage.setItem("satya-token", response.token);
    setUser(response.user);
  }, []);

  // ── 2FA: Setup ──
  const setup2FA = useCallback(async (): Promise<TwoFactorSetupData> => {
    return await api.auth.twoFactorSetup();
  }, []);

  // ── 2FA: Confirm Setup ──
  const confirm2FA = useCallback(async (code: string): Promise<boolean> => {
    const result = await api.auth.twoFactorConfirm(code);
    return result.success;
  }, []);

  // ── 2FA: Verify During Login ──
  const verify2FA = useCallback(async (tempToken: string, code: string) => {
    const response = await api.auth.twoFactorVerify(tempToken, code);
    localStorage.setItem("satya-token", response.token);
    setUser(response.user);
  }, []);

  // ── 2FA: Disable ──
  const disable2FA = useCallback(async (code: string): Promise<boolean> => {
    const result = await api.auth.twoFactorDisable(code);
    return result.success;
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        refreshUser,
        setUser,
        startOAuth,
        handleOAuthCallback,
        setup2FA,
        confirm2FA,
        verify2FA,
        disable2FA,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
