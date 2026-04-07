import type { User, PaginatedScans, Scan, AnalysisResult, TokenResponse, ContactForm, Case, PaginatedCases, ApiKey, CreateApiKeyResponse } from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
export const WS_BASE = import.meta.env.VITE_WS_BASE ||
  (import.meta.env.VITE_API_BASE
    ? import.meta.env.VITE_API_BASE.replace(/^http/, "ws")
    : "ws://localhost:8000");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem("satya-token");
  const headers: Record<string, string> = {};

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (options?.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...headers, ...(options?.headers as Record<string, string>) },
  });

  if (response.status === 401) {
    localStorage.removeItem("satya-token");
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, error.detail || "Unknown error");
  }

  // Handle empty responses (204, etc.)
  const text = await response.text();
  return text ? JSON.parse(text) : ({} as T);
}

export const api = {
  auth: {
    login: (email: string, password: string) =>
      apiFetch<TokenResponse>("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      }),
    register: (email: string, password: string, name: string) =>
      apiFetch<TokenResponse>("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password, name }),
      }),
    me: () => apiFetch<User>("/api/auth/me"),
    updateProfile: (data: { name?: string; language_pref?: string }) =>
      apiFetch<User>("/api/auth/me", {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    // OAuth — get the redirect URL to start the OAuth flow
    oauthUrl: (provider: "google" | "github") =>
      apiFetch<{ url: string }>(`/api/auth/oauth/${provider}/url`),
    // OAuth — exchange the authorization code for a token
    oauthCallback: (provider: "google" | "github", code: string, state?: string) =>
      apiFetch<TokenResponse>(`/api/auth/oauth/${provider}/callback`, {
        method: "POST",
        body: JSON.stringify({ code, state }),
      }),
    // 2FA — enable (returns QR / secret)
    twoFactorSetup: () =>
      apiFetch<{ secret: string; qr_url: string; backup_codes: string[] }>("/api/auth/2fa/setup", {
        method: "POST",
      }),
    // 2FA — confirm setup with a TOTP code
    twoFactorConfirm: (code: string) =>
      apiFetch<{ success: boolean }>("/api/auth/2fa/confirm", {
        method: "POST",
        body: JSON.stringify({ code }),
      }),
    // 2FA — verify during login
    twoFactorVerify: (tempToken: string, code: string) =>
      apiFetch<TokenResponse>("/api/auth/2fa/verify", {
        method: "POST",
        body: JSON.stringify({ temp_token: tempToken, code }),
      }),
    // 2FA — disable
    twoFactorDisable: (code: string) =>
      apiFetch<{ success: boolean }>("/api/auth/2fa/disable", {
        method: "POST",
        body: JSON.stringify({ code }),
      }),
    // Change password (authenticated)
    changePassword: (currentPassword: string, newPassword: string) =>
      apiFetch<{ success: boolean; message: string }>("/api/auth/password", {
        method: "PUT",
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      }),
    // Password reset (forgot password)
    requestPasswordReset: (email: string) =>
      apiFetch<{ success: boolean; message: string; code?: string }>("/api/auth/password-reset", {
        method: "POST",
        body: JSON.stringify({ email }),
      }),
    resetPassword: (email: string, code: string, newPassword: string) =>
      apiFetch<{ success: boolean; message: string }>("/api/auth/password-reset/confirm", {
        method: "POST",
        body: JSON.stringify({ email, code, new_password: newPassword }),
      }),
    // Email verification
    verifyEmail: (code: string) =>
      apiFetch<{ success: boolean; message: string }>("/api/auth/verify-email", {
        method: "POST",
        body: JSON.stringify({ code }),
      }),
    resendVerification: () =>
      apiFetch<{ success: boolean; message: string; code?: string }>("/api/auth/resend-verification", {
        method: "POST",
      }),
    deleteAccount: (password: string) =>
      apiFetch<{ success: boolean; message: string }>("/api/auth/account", {
        method: "DELETE",
        body: JSON.stringify({ password, confirm: "DELETE" }),
      }),
  },
  apiKeys: {
    list: () => apiFetch<ApiKey[]>("/api/keys"),
    create: (name: string) =>
      apiFetch<CreateApiKeyResponse>("/api/keys", {
        method: "POST",
        body: JSON.stringify({ name }),
      }),
    revoke: (keyId: string) =>
      apiFetch<{ status: string }>(`/api/keys/${keyId}`, { method: "DELETE" }),
  },
  scans: {
    list: (page = 1, perPage = 20) => apiFetch<PaginatedScans>(`/api/scans?page=${page}&per_page=${perPage}`),
    get: (id: string) => apiFetch<Scan>(`/api/scans/${id}`),
    delete: (id: string) => apiFetch<{ status: string }>(`/api/scans/${id}`, { method: "DELETE" }),
    reportUrl: (id: string) => `${API_BASE}/api/scans/${id}/report`,
  },
  analyze: {
    media: (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return apiFetch<AnalysisResult>("/api/analyze/media", { method: "POST", body: formData });
    },
  },
  cases: {
    list: (page = 1, perPage = 20, status?: string) => {
      const params = new URLSearchParams({ page: String(page), per_page: String(perPage) });
      if (status) params.set("status", status);
      return apiFetch<PaginatedCases>(`/api/cases?${params}`);
    },
    get: (id: string) => apiFetch<Case>(`/api/cases/${id}`),
    create: (title: string, description = "", scanIds: string[] = []) =>
      apiFetch<Case>("/api/cases", {
        method: "POST",
        body: JSON.stringify({ title, description, scan_ids: scanIds }),
      }),
    update: (id: string, data: { title?: string; description?: string; status?: string }) =>
      apiFetch<Case>(`/api/cases/${id}`, {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    delete: (id: string) => apiFetch<{ status: string }>(`/api/cases/${id}`, { method: "DELETE" }),
    addScan: (caseId: string, scanId: string) =>
      apiFetch<Case>(`/api/cases/${caseId}/scans`, {
        method: "POST",
        body: JSON.stringify({ scan_id: scanId }),
      }),
    removeScan: (caseId: string, scanId: string) =>
      apiFetch<Case>(`/api/cases/${caseId}/scans/${scanId}`, { method: "DELETE" }),
  },
  contact: {
    submit: (data: ContactForm) =>
      apiFetch<{ success: boolean; message: string }>("/api/contact", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },
  monitoring: {
    feedback: (scanId: string, feedback: "correct" | "incorrect" | "unsure", groundTruth?: string) =>
      apiFetch<{ status: string }>("/api/monitoring/feedback", {
        method: "POST",
        body: JSON.stringify({ scan_id: scanId, feedback, ground_truth: groundTruth }),
      }),
    accuracy: () => apiFetch<any>("/api/monitoring/accuracy"),
    stats: () => apiFetch<any>("/api/monitoring/stats"),
    calibration: () => apiFetch<any>("/api/monitoring/calibration"),
    drift: () => apiFetch<any>("/api/monitoring/drift"),
    checks: () => apiFetch<any>("/api/monitoring/checks"),
    deepHealth: () => apiFetch<any>("/api/monitoring/health/deep"),
  },
};
