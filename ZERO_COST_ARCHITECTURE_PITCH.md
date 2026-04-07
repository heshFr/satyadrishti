# Satya Drishti: Zero-Cost Production Architecture Pitch

**Objective:** Transition Satya Drishti from a local/hackathon project to a **Live, 24/7 Production Product** capable of running a massive 9-engine forensic AI pipeline—with an ongoing budget of **$0.00**.

To achieve this without triggering Out-Of-Memory (OOM) errors or incurring massive AWS/GCP bills, we must adopt a highly decoupled, free-tier "Serverless Microservices" architecture.

Below is the detailed architectural pitch outling each layer, the free-tier service provider, and why it works.

---

## The Architectural Split

We cannot host the AI inference on the same machine as the web server if we want zero cost. Free tiers typically cap at 512MB RAM, while our models (Wav2Vec2, ViT, DeBERTa, Whisper) require gigabytes. The solution is splitting the monolith into exactly four layers:

### 1. The Presentation Layer (Frontend)

- **Provider:** **Vercel** (already live)
- **What it does:** Hosts the React 19 / Vite frontend, handling static assets, routing, and i18n completely at the edge.
- **Why it's $0:** Vercel provides 100GB of bandwidth per month for free, capable of serving tens of thousands of users without a single cent charged.

### 2. The Persistence Layer (Database)

- **Provider:** **Neon.tech** or **Supabase** (Free Tier)
- **What it does:** Replaces the local `sqlite` database. Handles our users, scan history logs, and metadata.
- **Why it's $0:** Both Neon and Supabase offer **500MB of managed, serverless PostgreSQL storage forever for free**. They automatically pause when inactive and wake up instantly, which fits our zero-cost requirement perfectly.

### 3. The API / Gateway Layer (Auth & Orchestration)

- **Provider:** **Render.com** (Free Web Service) or **Vercel Serverless Functions**
- **What it does:** We strip all heavy `torch` / `transformers` logic out of the FastAPI server. This layer is now just a lightweight "traffic cop". It handles JWT Authentication, connection pooling to Postgres, CORS, and Rate Limiting. When it receives a media file, it forwards it to the inference layer.
- **Why it's $0:** Because it no longer loads machine learning weights into RAM, it comfortably fits inside Render's 512MB memory limit or Vercel's 50MB function limit.

---

## 4. The AI Inference Layer (The Deepfake Engines)

This is the heavy lifting. To run our GPU-intensive code for free, we have two excellent options. We will implement one based on our priority (Speed vs. Permanent Uncapped Availability).

### Inference Option A: Modal.com (Optimal & Professional) 🚀

- **How it works:** We convert our `engine/` logic into Modal Python functions. Modal spins up high-end GPUs (A10G, T4, A100s) on demand to process the request, runs the inference, returns the result to our API API Gateway, and immediately spins down.
- **Why it's $0:** Modal gives every developer **$30.00 of free compute credits per month**. Because they charge by the exact millisecond of compute _only when a user uploads a file_, $30 funds thousands of scans per month.
- **Why choose this:** We get GPU speeds. The user experience remains premium (results in seconds, not minutes). If usage exceeds $30/month, the product has proven market traction justifying monetization.

### Inference Option B: Hugging Face Spaces (The Uncapped Fallback) 🤗

- **How it works:** We deploy a standalone FastAPI instance inside a Hugging Face Space specifically meant for inference via an API call returning JSON.
- **Why it's $0:** Hugging Face Spaces offers a permanent, uncapped free tier running on 2 vCPUs and **16GB of RAM**.
- **Why choose this:** It's absolutely, permanently free regardless of how much scale or traffic we get. The 16GB RAM is enough to load all our models simultaneously or sequentially.
- **The Catch:** It runs on CPU. Inference for heavy tasks (like processing video or running Wav2Vec2) will be significantly slower, resulting in a UI loading bar that might take 20-30 seconds.

---
