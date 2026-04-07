# Satya Drishti — Deployment Guide

End-to-end deployment for the zero-cost production stack:

| Tier              | Service              | Hosts                             | Cost |
| ----------------- | -------------------- | --------------------------------- | ---- |
| Frontend          | Vercel               | React SPA                         | $0   |
| API Gateway       | Render (web)         | FastAPI + auth + DB               | $0   |
| ML Inference      | HuggingFace Spaces   | All ML models (CPU)               | $0   |
| Database          | Neon.tech (Postgres) | Users, scans, cases               | $0   |
| Object Storage    | (none — disk temp)   | Uploaded files deleted after scan | $0   |
| Tunnel (optional) | Cloudflare Tunnel    | Local dev → public URL            | $0   |

The gateway runs **without** torch/transformers installed (`requirements-gateway.txt`); all heavy ML lives behind `INFERENCE_URL`.

---

## 1. Prerequisites

- GitHub repo: `https://github.com/<you>/satyadrishti`
- Accounts: Render, HuggingFace, Neon.tech, Vercel, (optional) Cloudflare
- Local Python 3.10, Node 18+

---

## 2. Provision the Database (Neon.tech)

1. Sign in at https://neon.tech and create a new project: `satyadrishti`.
2. In the project dashboard, copy the **connection string** (the pooled `-pooler` host is recommended for serverless).
3. Convert the URL scheme: Neon gives `postgresql://...`; the gateway auto-rewrites this to `postgresql+asyncpg://` at startup (`server/config.py:18`).
4. Save it as `SATYA_DATABASE_URL` — you will paste this into Render in step 4.

---

## 3. Deploy the ML Inference Worker (HuggingFace Spaces)

The Spaces worker hosts the heavy models. The gateway calls it over HTTPS with a shared secret.

1. Create a new **Space** at https://huggingface.co/new-space.
2. SDK: **Docker** (or Gradio if `hf_inference/` ships a `app.py`). Hardware: **CPU basic (free)**.
3. Upload the `hf_inference/` directory contents (or push as a Git remote).
4. In the Space **Settings → Variables and secrets**, add:
   - `INFERENCE_SECRET` — a long random string (run `python -c "import secrets; print(secrets.token_urlsafe(32))"`). This **must match** the value you set on Render in step 4.
5. Wait for the Space to build, then copy the public URL: `https://<user>-satyadrishti.hf.space`. This becomes `INFERENCE_URL` on the gateway.
6. Verify: `curl https://<user>-satyadrishti.hf.space/health` should return `{"status":"ok"}`.

---

## 4. Deploy the API Gateway (Render)

Render reads `render.yaml` from the repo root automatically.

1. Push the repo to GitHub.
2. At https://dashboard.render.com → **New → Blueprint** → connect the repo. Render detects `render.yaml`.
3. When prompted, fill the unset env vars:
   - `SATYA_DATABASE_URL` — Neon connection string from step 2.
   - `INFERENCE_URL` — Spaces URL from step 3 (e.g., `https://user-satyadrishti.hf.space`).
   - `INFERENCE_SECRET` — same value as on the Spaces worker. (If you let Render `generateValue` for this, copy the generated value back to the Space.)
4. The remaining vars are pre-set in `render.yaml`:
   - `SATYA_JWT_SECRET` — auto-generated.
   - `SATYA_CORS_ORIGINS` — `https://satyadrishti.vercel.app,http://localhost:3000` (extend if you use a custom domain).
   - `SATYA_MAX_CONCURRENT_ANALYSES=2` — caps simultaneous heavy inferences (semaphore in `server/app.py`).
   - `SATYA_LOG_FORMAT=json` — structured logs for ingestion by Render's log drain.
   - `PYTHON_VERSION=3.10.13`.
5. Click **Apply**. First build runs `pip install -r requirements-gateway.txt` then `uvicorn server.app:app --host 0.0.0.0 --port $PORT`.
6. The healthcheck (`/api/health`) determines deploy success.

### Run database migrations on Render

The gateway calls `init_db()` on startup which creates tables via `Base.metadata.create_all` for first-time setup. For schema changes after launch, switch to Alembic:

**One-time baseline** (mark the existing schema as migrated):
```bash
# Locally, with SATYA_DATABASE_URL pointing at Neon:
alembic stamp 0001_initial
```

**Apply pending migrations** (locally before deploying, or via a Render shell):
```bash
alembic upgrade head
```

**Generate a new migration** after editing `server/models.py`:
```bash
alembic revision --autogenerate -m "add new column foo"
# Review the generated file in alembic/versions/, then:
alembic upgrade head
```

> Render's free web service does not support `preDeployCommand`. Run migrations from your local machine against the Neon URL, then redeploy.

---

## 5. Deploy the Frontend (Vercel)

1. Import the repo at https://vercel.com/new.
2. **Root directory:** `frontend/`
3. **Framework preset:** Vite (auto-detected).
4. **Build command:** `npm run build` &nbsp; **Output dir:** `dist`
5. Environment variable:
   - `VITE_API_URL` — your Render URL, e.g., `https://satyadrishti-api.onrender.com`
6. Deploy. Vercel serves at `https://satyadrishti.vercel.app` (or your chosen subdomain).
7. Add the Vercel URL to `SATYA_CORS_ORIGINS` on Render if it differs from the default.

---

## 6. Cloudflare Tunnel (optional, for local dev)

Use this when you want a public URL pointing at your laptop (e.g., demo to investors without deploying).

1. Install `cloudflared` and authenticate: `cloudflared tunnel login`.
2. Create the tunnel: `cloudflared tunnel create satyadrishti-dev`.
3. Run the gateway locally: `python -m server.app`.
4. Start the tunnel: `cloudflared tunnel --url http://localhost:8000` — Cloudflare prints a `*.trycloudflare.com` URL.
5. The provided `start-backend.sh` script wraps this for Linux/WSL.

---

## 7. Post-Deploy Verification

Run these against the production URLs:

```bash
# Gateway healthcheck
curl https://satyadrishti-api.onrender.com/api/health

# Inference worker reachable from the gateway
curl https://satyadrishti-api.onrender.com/api/monitoring/health/deep

# Prometheus metrics
curl https://satyadrishti-api.onrender.com/metrics

# OpenAPI docs
open https://satyadrishti-api.onrender.com/docs

# Frontend
open https://satyadrishti.vercel.app
```

Expected: `/api/health` returns `{"status":"ok"}`, `/api/monitoring/health/deep` reports the inference worker as `up`, `/metrics` returns Prometheus exposition format.

---

## 8. Operations Cheatsheet

| Task                          | Command / URL                                                       |
| ----------------------------- | ------------------------------------------------------------------- |
| View live logs                | Render dashboard → Logs (or `render logs --tail` via CLI)          |
| Roll back a deploy            | Render → Deploys → click previous deploy → **Redeploy**             |
| Reset Postgres                | Neon dashboard → Branch → **Reset** (or create a new branch)        |
| Rotate `INFERENCE_SECRET`     | Update on Render env vars **and** Spaces secrets, redeploy both     |
| Scale concurrency             | Render env: bump `SATYA_MAX_CONCURRENT_ANALYSES` (free tier ≤ 2)    |
| Inspect Prometheus metrics    | `curl …/metrics` or scrape from your Grafana instance               |
| Check rate-limit state        | `/api/monitoring/stats` (requires admin auth)                        |
| Run E2E tests against prod    | `pytest tests/test_e2e.py --base-url https://…onrender.com`         |

---

## 9. Free-Tier Gotchas

- **Render free web spins down** after 15 min idle. First request after idle takes ~30 s. Use a cron pinger or upgrade to paid for always-on.
- **HF Spaces sleeps** after 48 h inactivity. Ping `/health` periodically.
- **Neon free tier** suspends compute after 5 min idle; first connection adds latency. Use a pooler URL.
- **Render free has 512 MB RAM** — that's why the gateway has zero ML deps. Do **not** add `torch` or `transformers` to `requirements-gateway.txt`.
- **Vercel free** has function execution limits — irrelevant here since we serve a static SPA.

---

## 10. Local Development

```bash
# Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-gateway.txt -e ".[dev]"
python -m server.app             # http://localhost:8000

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000

# Tests
pytest tests/test_e2e.py -v
```

For a full local stack (with actual ML models loaded in-process), use `requirements-server.txt` instead of `requirements-gateway.txt` and unset `INFERENCE_URL`.
