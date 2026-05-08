# Satya Drishti — Deployment

The current setup is intentionally simple:

| Tier     | Where                                | Notes                              |
| -------- | ------------------------------------ | ---------------------------------- |
| Frontend | Vercel                               | React SPA, built with Vite         |
| Backend  | Local laptop (or Cloudflare Tunnel)  | FastAPI + all ML engines in-process |
| Database | Local SQLite (`satyadrishti.db`)     | Switch to Postgres when needed      |

There is no Render / HuggingFace Spaces / Neon split deployment. All ML
inference runs in the same process as the API server.

---

## 1. Local Development

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
python -m server.app             # http://localhost:8000
```

The first run downloads HuggingFace models into `.hf_cache/` (a few GB).
Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` after that to avoid
re-fetching them.

### Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
```

Vite reads `frontend/.env` for `VITE_API_BASE` (defaults to
`http://localhost:8000`).

---

## 2. Frontend on Vercel

The frontend is the only piece deployed publicly today.

1. Import the repo at https://vercel.com/new.
2. **Root directory:** `frontend/`
3. **Framework preset:** Vite (auto-detected).
4. **Build command:** `npm run build` &nbsp; **Output dir:** `dist`
5. Set `VITE_API_BASE` in the Vercel dashboard to your backend's public URL.

If your backend isn't publicly reachable, the deployed Vercel app can
load but its API calls will fail (browser tries `localhost:8000` from
the visitor's machine). To make the deployed frontend work end-to-end,
you need a public backend URL — see the Cloudflare Tunnel option below.

---

## 3. Public Backend via Cloudflare Tunnel (optional)

Use this when you want the deployed Vercel frontend to talk to your
laptop, e.g. for a demo, without paying for a server.

```bash
bash start-backend.sh
```

`start-backend.sh` boots `python -m server.app` and a Cloudflare quick
tunnel. Cloudflare prints a `*.trycloudflare.com` URL — paste that as
`VITE_API_BASE` in Vercel and redeploy. The URL changes every run.

For a stable URL, use a named tunnel:
https://developers.cloudflare.com/cloudflare-one/connections/connect-apps

---

## 4. Database

For local dev the backend creates `satyadrishti.db` (SQLite) on first
run via `Base.metadata.create_all`. To move to Postgres later:

1. Provision a Postgres database (Neon, Supabase, RDS, etc.).
2. Set `SATYA_DATABASE_URL=postgresql+asyncpg://...` in `.env`.
3. Run migrations:
   ```bash
   alembic upgrade head
   ```

`alembic/` and `alembic.ini` are already configured.

---

## 5. Environment Variables

Read by `server/config.py`. Set in `.env` for local, in the Vercel
dashboard for the frontend.

| Var                              | Default                     | Purpose                          |
| -------------------------------- | --------------------------- | -------------------------------- |
| `SATYA_DATABASE_URL`             | `sqlite+aiosqlite:///./satyadrishti.db` | DB connection                    |
| `SATYA_JWT_SECRET`               | (warns if unset)            | JWT signing key                  |
| `SATYA_CORS_ORIGINS`             | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed origins  |
| `SATYA_MAX_CONCURRENT_ANALYSES`  | `2`                         | Cap on simultaneous heavy inferences |
| `SATYA_LOG_FORMAT`               | text                        | Set to `json` for structured logs |
| `VITE_API_BASE` (frontend)       | `http://localhost:8000`     | Backend URL the SPA calls        |
