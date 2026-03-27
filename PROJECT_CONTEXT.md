# Satya Drishti — Full Project Context

> **One-liner pitch:** Satya Drishti is a **domain-specialized AI Forensic Compliance Agent** for the financial sector that detects AI voice clones, deepfakes, and synthetic fraud in real-time using a 9-layer neural ensemble with a **Biological Veto System**.
>
> **Hackathon target:** ET AI Hackathon 2026, Problem Statement #5 — "Domain-Specialized AI Agents with Compliance Guardrails"

---

## 1. Project Overview

**Satya Drishti** ("True Vision" in Sanskrit) is a full-stack AI forensic platform that:
- Detects AI-generated voice clones (ElevenLabs, RVC, XTTS, etc.) in **real-time** during calls
- Scans uploaded images, audio, and video for deepfake manipulation
- Provides speaker identity verification via biometric voiceprints
- Enforces **compliance guardrails** through a Biological Veto System that no neural network can override
- Maintains a **full audit trail** with per-layer analysis and veto reasons

### Why it matters
- ₹1,947 Cr in financial fraud losses in India (RBI 2025)
- 500% rise in voice clone fraud since 2023
- Traditional AI detectors fail on modern TTS (ElevenLabs, RVC) because they rely solely on spectral analysis

### Unique differentiator
The **Biological Veto System** checks for physiological impossibilities (breathing patterns, prosodic micro-jitter, formant transitions) that AI voice clones cannot replicate. If any layer detects physiological impossibility, it triggers an **irreversible veto** that overrides all neural network scores.

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + TypeScript, Vite, Tailwind CSS, Framer Motion, i18next |
| **Backend** | Python 3.11, FastAPI, Uvicorn, WebSocket |
| **AI/ML** | PyTorch, Whisper, Wav2Vec2, ECAPA-TDNN, librosa, soundfile |
| **Database** | SQLite (local dev), JWT auth |
| **Audio Pipeline** | AudioWorklet (browser) → WebSocket → 9-layer ensemble |
| **Deployment** | Vercel (frontend), Render (backend), or local |

---

## 3. Project Structure

```
satyadrishti/
├── frontend/                     # React + Vite frontend
│   ├── public/
│   │   ├── logo.png              # Project logo
│   │   └── pcm-processor.js      # AudioWorklet for real-time PCM capture
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx        # Landing page (hero, stats, features)
│   │   │   ├── Hub.tsx            # Agent module selector (3 cards)
│   │   │   ├── CallProtection.tsx # Real-time voice forensics dashboard
│   │   │   ├── Scanner.tsx        # Image/audio/video upload forensics
│   │   │   ├── VoiceEnroll.tsx    # Biometric voiceprint enrollment
│   │   │   ├── Login.tsx          # Auth (Google, GitHub OAuth, 2FA)
│   │   │   ├── Register.tsx       # Registration
│   │   │   ├── Profile.tsx        # User profile
│   │   │   ├── Settings.tsx       # App settings
│   │   │   ├── History.tsx        # Scan history (requires login)
│   │   │   ├── Help.tsx           # Help/FAQ
│   │   │   ├── Contact.tsx        # Contact page
│   │   │   ├── Privacy.tsx        # Privacy policy
│   │   │   ├── Terms.tsx          # Terms of service
│   │   │   ├── Advanced.tsx       # Advanced settings
│   │   │   └── SecurityProtocol.tsx # Security protocol info
│   │   ├── hooks/
│   │   │   └── useCallProtection.ts # Core hook: audio capture, WebSocket, state
│   │   ├── components/
│   │   │   ├── Layout.tsx         # App layout wrapper
│   │   │   ├── TopBar.tsx         # Navigation bar
│   │   │   ├── LandingNav.tsx     # Landing page nav
│   │   │   ├── Footer.tsx         # Footer
│   │   │   ├── MaterialIcon.tsx   # Google Material Icons wrapper
│   │   │   ├── LanguageDropdown.tsx # i18n language switcher
│   │   │   └── ... (more UI components)
│   │   └── contexts/
│   │       └── AuthContext.tsx     # Auth state management
│
├── server/                        # FastAPI backend
│   ├── app.py                     # Main app: REST routes + WebSocket handler
│   ├── inference_engine.py        # 9-layer forensic ensemble orchestrator
│   ├── auth.py                    # JWT authentication
│   ├── database.py                # SQLite database
│   ├── config.py                  # Server configuration
│   ├── models.py                  # Pydantic models
│   ├── alert_system.py            # Alert/notification system
│   ├── rate_limiter.py            # API rate limiting
│   ├── report_generator.py        # Forensic report PDF generation
│   └── routes/                    # API route modules
│
├── engine/                        # Core AI/ML forensic engine
│   ├── audio/                     # Audio forensic analyzers (THE CORE)
│   │   ├── ssl_detector.py        # Layer 1: Self-Supervised Learning (Wav2Vec2)
│   │   ├── whisper_features.py    # Layer 2: Whisper-based feature extraction
│   │   ├── rawnet3.py             # Layer 3: RawNet3 raw waveform analysis
│   │   ├── prosodic_analyzer.py   # Layer 4: Prosodic micro-jitter analysis
│   │   ├── breathing_detector.py  # Layer 5: Breathing pattern detection (BIO VETO)
│   │   ├── formant_analyzer.py    # Layer 6: Formant transition analysis (BIO VETO)
│   │   ├── phase_analyzer.py      # Layer 7: Phase coherence analysis
│   │   ├── temporal_tracker.py    # Layer 8: Cross-chunk temporal tracking
│   │   ├── ensemble_fusion.py     # Layer 9: Weighted ensemble fusion + BIO VETO
│   │   ├── ast_spoof.py           # AST (Audio Spectrogram Transformer) model
│   │   ├── features.py            # Low-level audio feature extraction
│   │   ├── speaker_verify.py      # ECAPA-TDNN speaker verification
│   │   └── transcriber.py         # Whisper-based speech transcription
│   ├── image_forensics/           # Image deepfake detection
│   ├── video/                     # Video forensic analysis
│   ├── text/                      # Text coercion/scam detection
│   ├── fusion/                    # Multi-modal fusion
│   └── optimization/              # Model optimization utilities
│
├── models/                        # Pre-trained model weights
├── configs/                       # Configuration files
├── tests/                         # Test suite
└── scripts/                       # Utility scripts
```

---

## 4. The 9-Layer Audio Forensic Engine (Core IP)

This is the heart of Satya Drishti. Each audio chunk passes through **all 9 layers** in parallel, and results are fused in `ensemble_fusion.py`:

| # | Layer | File | What It Does | Veto Power? |
|---|-------|------|-------------|-------------|
| 1 | **SSL Detector** | `ssl_detector.py` | Wav2Vec2 self-supervised features — detects artifacts invisible to human ear | No |
| 2 | **Whisper Features** | `whisper_features.py` | Whisper encoder features — detects TTS-specific patterns | No |
| 3 | **RawNet3** | `rawnet3.py` | Raw waveform CNN — no preprocessing, catches low-level artifacts | No |
| 4 | **Prosodic Analyzer** | `prosodic_analyzer.py` | Micro-jitter, pitch variation, rhythm — AI voices are "too perfect" | **YES (BIO VETO)** |
| 5 | **Breathing Detector** | `breathing_detector.py` | Detects breathing patterns, pauses, respiratory cycles | **YES (BIO VETO)** |
| 6 | **Formant Analyzer** | `formant_analyzer.py` | Formant transitions, vocal tract resonance patterns | **YES (BIO VETO)** |
| 7 | **Phase Analyzer** | `phase_analyzer.py` | Phase coherence between harmonics — synthetic voices have unnatural phase | No |
| 8 | **Temporal Tracker** | `temporal_tracker.py` | Cross-chunk consistency — checks if voice characteristics drift over time | No |
| 9 | **Ensemble Fusion** | `ensemble_fusion.py` | Weighted fusion of all layers + **Biological Veto logic** | FINAL DECISION |

### Biological Veto System
- Layers 4, 5, 6 (prosodic, breathing, formant) check for **physiological impossibility**
- If ANY of these layers flag a "physiological impossibility" → **VETO IS TRIGGERED**
- The veto **overrides** all other layer scores, even if neural networks say "human"
- This is the compliance guardrail that ensures zero false negatives on synthetic voices
- Every veto includes an auditable `veto_reason` string explaining exactly what was impossible

---

## 5. Real-Time Audio Pipeline

### Frontend → Backend Flow

```
Browser Tab/Call
    ↓
getDisplayMedia() / getUserMedia()
    ↓
AudioWorklet (pcm-processor.js)
    ↓ (PCM Float32 samples)
5-second chunking + silence gating (RMS > 0.005)
    ↓
Base64 WAV encoding
    ↓
WebSocket → ws://localhost:8000/ws/call/{session_id}
    ↓
Backend: Rolling buffer (last 3 chunks ≈ 15s)
    ↓
Concatenated WAV → inference_engine.analyze()
    ↓
9-layer parallel analysis
    ↓
Ensemble fusion + Biological Veto
    ↓
JSON response → WebSocket → Frontend state update
```

### Key Frontend State (useCallProtection.ts)
```typescript
{
  isActive: boolean,           // Is monitoring active?
  isScreenShare: boolean,      // Is screen being shared?
  callState: "safe" | "warning" | "danger" | "critical",
  threatEscalation: number,    // 0-1 threat score
  biologicalVeto: boolean,     // Was biological veto triggered?
  vetoReason: string | null,   // Why the veto was triggered
  audio: { status, confidence, details },
  text: { status, confidence, details },
  speakerVerified: boolean,    // Speaker identity match
  speakerMatch: string | null, // Matched speaker name
  speakerSimilarity: number,   // 0-1 similarity score
  frequencyData: Uint8Array,   // Live frequency data for visualization
  audioLevel: number,          // RMS audio level 0-1
  transcript: Array<{ text, time, flagged }>,
}
```

---

## 6. Backend API Endpoints

### REST API (FastAPI)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze/audio` | Upload audio file for forensic analysis |
| POST | `/api/analyze/image` | Upload image for deepfake detection |
| POST | `/api/analyze/video` | Upload video for forensic analysis |
| POST | `/api/analyze/text` | Analyze text for coercion patterns |
| POST | `/api/enroll-voice` | Enroll a biometric voiceprint |
| POST | `/api/verify-speaker` | Verify speaker against enrolled prints |
| POST | `/api/auth/login` | JWT login |
| POST | `/api/auth/register` | Create account |
| GET  | `/api/history` | Get scan history (auth required) |
| GET  | `/api/report/{id}` | Generate forensic PDF report |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `ws://host:8000/ws/call/{session_id}` | Real-time call protection audio stream |

WebSocket message types:
- **Client → Server:** `{ type: "audio", data: "<base64 WAV>" }` or `{ type: "text", data: "..." }`
- **Server → Client:** `{ type: "analysis_result", modality: "audio", is_synthetic: bool, confidence: float, biological_veto: bool, veto_reason: str, threat_escalation: float, ... }`

---

## 7. Frontend Pages

| Page | Route | Purpose |
|------|-------|---------|
| **Landing** | `/` | Marketing/intro page with stats, features bento grid, CTA |
| **Hub** | `/hub` | Agent module selector (3 cards: Dashboard, Voice Forensics, Media Forensics) |
| **Call Protection** | `/call-protection` | Real-time voice forensics dashboard with idle CTA / active monitoring |
| **Scanner** | `/scanner` | Upload media files for forensic analysis |
| **Voice Enroll** | `/voice-enroll` | Record voiceprint for speaker verification |
| **Login** | `/login` | Google/GitHub OAuth + 2FA login |
| **Register** | `/register` | Account registration |
| **Settings** | `/settings` | App configuration |
| **History** | `/history` | Past scan results (requires auth) |
| **Profile** | `/profile` | User profile management |
| **Help** | `/help` | FAQ and documentation |
| **Contact** | `/contact` | Contact form |

---

## 8. Design System

- **Color scheme:** Dark theme with cyan (`#00D1FF`) primary, emerald (`#4EDEA3`) secondary
- **Typography:** Google Fonts — headline font for titles, body font for content
- **UI Library:** Custom components with Tailwind CSS utility classes
- **Animation:** Framer Motion for all transitions, micro-animations, stagger effects
- **Icons:** Google Material Symbols (via MaterialIcon component)
- **Aesthetic:** Glassmorphism, blur effects, glow shadows, gradient backgrounds
- **Responsive:** Mobile-first with md/lg breakpoints

---

## 9. How to Run Locally

### Backend
```bash
cd satyadrishti
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements-server.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd satyadrishti/frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

### Environment Variables
- `SATYA_JWT_SECRET` — JWT signing secret (defaults to insecure value in dev)
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — Google OAuth
- `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` — GitHub OAuth

---

## 10. Hackathon Framing (PS #5 — Domain-Specialized Agents)

### Problem
Financial institutions are adopting AI voice agents for customer service. The biggest risk: **voice-clone fraud bypasses traditional security.** Existing detectors cannot catch modern TTS clones (ElevenLabs, RVC).

### Solution
Satya Drishti is the **Biological Guardrail** — a domain-specialized AI agent that enforces compliance through physiological analysis that no AI clone can fool.

### Evaluation Criteria Alignment
| Criteria | How We Address It |
|----------|-------------------|
| **Domain expertise depth** | 9 specialized audio forensic layers built for voice fraud detection |
| **Compliance guardrail enforcement** | Biological Veto System — hard physiological limits |
| **Edge-case handling** | Handles ElevenLabs, RVC, XTTS, and future TTS through biology, not patterns |
| **Full task completion** | End-to-end: capture → analyze → detect → alert → audit log |
| **Auditability** | Every decision includes veto reasons, confidence scores, per-layer breakdown |

### Key Talking Points
1. "We don't just detect AI voices — we prove they're not human through biology"
2. "Our Biological Veto cannot be bypassed, even as AI voice cloning improves"
3. "Every detection is explainable and auditable — compliance-grade"
4. "Real-time, sub-5-second detection while the call is happening"
5. "9 independent layers — no single point of failure"
