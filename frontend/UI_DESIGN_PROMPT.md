# Satya Drishti — UI/UX Design Prompt

## What is this app?
A real-time deepfake and scam detection platform built to protect Indian families from voice cloning, deepfake video calls, and coercion scams. Think of it as a "digital bodyguard" for your parents and grandparents.

## Target audience
Indian families — especially non-tech-savvy parents who receive scam calls. The app is used by their tech-savvy children to set up protection. Emotional angle: protecting your mother from a scammer pretending to be you.

## Tech stack
React + TypeScript + Tailwind CSS + Framer Motion. Dark mode only.

## Pages needed

1. **Landing page** — Cinematic, emotional, high-conversion. Hero section should evoke urgency and protection. Include stats about cyber crime in India, feature highlights, trust signals, and a strong CTA.

2. **Dashboard** — The main hub after login. Shows real-time protection status (protected/monitoring/alert), recent scan history, threat stats, and a live call protection panel with audio waveform visualization.

3. **Scanner** — Drag-and-drop file upload zone for images, audio, and video. Shows analysis progress with steps, then displays a verdict (safe/suspicious/fake) with forensic breakdown details.

4. **History** — Table/list of past scans with filters, search, verdict badges, and confidence scores. Expandable rows for details.

5. **Voice Prints** — Enroll family members' voices via microphone recording with live waveform visualization. List enrolled voices with delete option.

6. **Settings** — Protection toggles, account security (password change, 2FA), emergency contacts management, notification preferences.

7. **Help/FAQ** — Categorized FAQ accordion, quick start guides, emergency resources (Indian police/cybercrime helplines).

8. **Contact** — Contact form with subject selection, success state animation.

9. **Login & Register** — Clean auth forms with validation.

10. **Profile** — User stats, activity timeline, security overview.

11. **Advanced** — Live WebSocket call protection interface with real-time audio analysis, transcript panel, threat indicators.

## Navigation
- Landing page has its own floating/transparent navbar
- All other pages share a common top navigation bar with logo, nav links, status indicator pill, language toggle (EN/HI), and user menu

## Design direction
- Premium, modern, world-class — think Linear, Vercel, Raycast, Arc Browser level polish
- Dark theme exclusively
- Generous whitespace, smooth micro-interactions, subtle glassmorphism
- The app deals with security and protection — the design should feel safe, trustworthy, and calm but powerful
- Avoid looking like a generic SaaS dashboard — this should feel unique and intentional
- Mobile responsive

## Key UI elements
- Status indicator (green dot = protected, blue = monitoring, red pulsing = alert)
- Audio waveform visualizer (canvas-based, real-time)
- File drop zone with drag states
- Verdict cards with confidence meters
- Animated progress steps during analysis
- Glass-effect cards with subtle borders

## Mood & feeling
Protective. Calm confidence. Like having a guardian watching over your family. Not aggressive or fear-based — empowering.

## Don't
- Don't use light mode
- Don't make it look like a generic admin template
- Don't use stock illustrations or clipart
- Don't overcrowd — let the design breathe
