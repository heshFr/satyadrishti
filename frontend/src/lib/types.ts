export interface User {
  id: string;
  email: string;
  name: string;
  language_pref: string;
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
