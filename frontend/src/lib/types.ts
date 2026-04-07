export interface User {
  id: string;
  email: string;
  name: string;
  language_pref: string;
  email_verified: boolean;
  totp_enabled: boolean;
  oauth_provider: string | null;
  notify_email_threats: boolean;
  notify_email_reports: boolean;
  notify_push_enabled: boolean;
  emergency_contact_name: string | null;
  emergency_contact_phone: string | null;
  created_at: string;
}

export interface Scan {
  id: string;
  file_name: string;
  file_type: string;
  verdict: "ai-generated" | "authentic" | "uncertain";
  confidence: number;
  forensic_data: ForensicItem[];
  raw_scores: Record<string, number>;
  created_at: string;
}

export interface ForensicItem {
  id?: string;
  label?: string;
  name?: string;
  status: "pass" | "fail" | "warn" | "info";
  detail?: string;
  description?: string;
}

export interface PaginatedScans {
  items: Scan[];
  total: number;
  page: number;
  per_page: number;
}

export interface ContactForm {
  name: string;
  email: string;
  subject: string;
  message: string;
}

export interface AnalysisResult {
  verdict: string;
  confidence: number;
  forensic_checks: ForensicItem[];
  forensic_data?: ForensicItem[];
  report_url?: string | null;
  raw_scores?: Record<string, number>;
  scan_id?: string | null;
}

export interface TokenResponse {
  token: string;
  user: User;
  requires_2fa?: boolean;
  temp_token?: string;
}

export interface ApiKey {
  id: string;
  name: string;
  key_prefix: string;
  is_active: boolean;
  last_used: string | null;
  request_count: number;
  created_at: string;
}

export interface CreateApiKeyResponse extends ApiKey {
  full_key: string;
}

export interface Case {
  id: string;
  title: string;
  description: string;
  scan_ids: string[];
  status: "open" | "investigating" | "resolved";
  created_at: string;
  updated_at: string;
}

export interface PaginatedCases {
  items: Case[];
  total: number;
  page: number;
  per_page: number;
}
