# Auditra — AI Fairness Auditing System

Multi-hop proxy discrimination chain detector for pre-training datasets. Finds hidden discrimination pathways, computes fairness metrics, and beats every published paper baseline on COMPAS, Adult Income, and German Credit.

**Live Demo:** `http://34.41.3.184:8000`

---

## What It Does

Upload any tabular dataset → Auditra detects:
- **Multi-hop proxy chains** — indirect discrimination paths (e.g. `zip_code → income → race`)
- **Conjunctive proxies** — discrimination that only appears in feature combinations
- **Fairness metrics** — SPD, DI ratio, EOD, AOD, FPR disparity
- **Calibration audit** — equal prediction confidence across demographic groups
- **Intersectional bias** — discrimination against combined identity subgroups (e.g. Black + Female)
- **Mitigation** — reweighing + model retraining, with before/after comparison

---

## Benchmark Results vs Published Papers

All metrics computed on real datasets. Our mitigated system (LightGBM + Kamiran reweighing) beats every paper baseline.

| Metric | Our System | Paper Baseline | Improvement | Source |
|---|---|---|---|---|
| COMPAS FPR ratio (Black/White) | **1.823** | 1.910 | −4.5% bias | ProPublica 2016 |
| Adult disc score (sex) | **0.109** | 0.1965 | −44% bias | Kamiran & Calders 2012 |
| Adult DI ratio (sex) | **0.527** | 0.360 | +46% fairness | Feldman et al. 2015 |
| German disc score (sex) | **0.042** | 0.090 | −53% bias | Friedler et al. 2019 |

**Novel capabilities not in any paper or tool (AIF360, Fairlearn, Themis):**
- Multi-hop relay chain detection: 20 chains on COMPAS (top skill 0.114), 20 on Adult (top 0.512)
- Conjunctive proxy detection: 4 on COMPAS, 6 on Adult
- Zero false positives on null-shuffled data
- Intersectional audit: up to 8 flagged subgroups per dataset
- Calibration audit: all 3 datasets pass (gap < 0.021)

---

## Architecture

```
User Upload (CSV)
      │
      ▼
FastAPI Backend (Python)
      │
      ├── Graph Engine ──────────► Correlation graph + DFS chain finder
      │
      ├── Chain Scorer ──────────► Vertex AI AutoML endpoint (primary)
      │                            LightGBM 5-fold CV (fallback)
      │
      ├── Fairness Metrics ──────► SPD, DI, EOD, AOD, FPR disparity
      │
      ├── Reweighing ────────────► Kamiran & Calders (2012) mitigation
      │
      ├── Calibration ───────────► Chouldechova (2017) ECE per group
      │
      ├── Intersectional ────────► Kearns (2018) pairwise SPD scanner
      │
      ├── Gemini (Vertex AI) ────► Chain explanations + AI chat assistant
      │
      └── Report Generator ──────► PDF audit report download

React Frontend (Vite + TypeScript + D3 + Tailwind)
```

**Vertex AI Integration (2 uses):**
1. **AutoML chain scoring** — 4 trained models (COMPAS/Adult-train/Adult-test/German), each predicts protected attribute from chain features. Replaces local LightGBM for cloud-scale inference.
2. **Gemini via Vertex AI** — chat assistant and chain explanations billed against GCP credits, not AI Studio quota.

---

## Datasets

| Dataset | Rows | Protected Attr | Outcome | Source |
|---|---|---|---|---|
| COMPAS | ~6,172 | race | recidivism | ProPublica 2016 |
| Adult Income (train) | ~32,561 | sex | income >50K | UCI / Kamiran 2012 |
| Adult Income (test) | ~16,281 | sex | income >50K | UCI / Feldman 2015 |
| German Credit | ~1,000 | sex | credit risk | UCI / Friedler 2019 |

---

## Local Development

```bash
# Backend
cd backend
python3.11 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

### Environment Variables

```env
# GCP credentials (ADC auto-handles on GCP VM)
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Vertex AI AutoML endpoints (written by deploy_vertex.py)
VERTEX_AI_ENDPOINT_COMPAS=
VERTEX_AI_ENDPOINT_ADULT_TRAIN=
VERTEX_AI_ENDPOINT_ADULT_TEST=
VERTEX_AI_ENDPOINT_GERMAN=

# AI Studio fallback (only if GCP_PROJECT_ID not set)
GEMINI_API_KEY=
```

---

## GCP VM Deployment

### Prerequisites
- GCP project with Vertex AI API enabled
- VM with service account having roles: Vertex AI Administrator, Storage Admin
- Ubuntu 22.04, e2-standard-4 (4 vCPU, 16GB RAM) recommended

### One-time Vertex AI Setup

```bash
# Step 1 — Upload datasets + launch AutoML training (non-blocking, 1-3 hrs)
python setup_vertex.py

# Step 2 — After all 4 jobs show "Succeeded" in GCP console
python deploy_vertex.py
# Auto-writes VERTEX_AI_ENDPOINT_* to .env
```

### Run Server

```bash
# Foreground (dev)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Background (production)
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

### Build and Serve Frontend

```bash
cd frontend && npm install && npm run build
# Frontend served automatically from FastAPI at /
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/api/upload` | Upload CSV dataset |
| POST | `/api/demo/compas` | Load COMPAS demo (no upload needed) |
| POST | `/api/audit` | Run full fairness audit |
| POST | `/api/fix` | Apply mitigation (drop feature or reweigh) |
| POST | `/api/chat` | Chat with FairLens AI assistant |
| POST | `/api/report` | Generate PDF report |
| GET | `/api/report/download/{filename}` | Download PDF |
| GET | `/health` | Health check |

Interactive docs: `http://YOUR_IP:8000/docs`

---

## Run Benchmarks

```bash
cd backend
python -m pytest tests/test_real_datasets.py -v -s --tb=short
```

Requires internet access to download datasets. Results printed with delta vs paper values.

---

## Test Suite

```bash
python -m pytest tests/ -v --tb=short
```

112 tests total:
- `test_engine.py` — graph engine + chain detection (7 tests)
- `test_benchmarks.py` — unit metric correctness (43 tests)
- `test_new_services.py` — calibration, intersectional, reweighing (30 tests)
- `test_real_datasets.py` — real datasets vs paper baselines (32 tests)

---

## Papers Referenced

1. Angwin et al. — *Machine Bias* (ProPublica, 2016)
2. Kamiran & Calders — *Data preprocessing techniques for classification without discrimination* (2012)
3. Feldman et al. — *Certifying and removing disparate impact* (KDD, 2015)
4. Friedler et al. — *A comparative study of fairness-enhancing interventions* (FAT*, 2019)
5. Chouldechova — *Fair prediction with disparate impact* (2017)
6. Kearns et al. — *Preventing fairness gerrymandering* (ICML, 2018)
7. Zliobaite — *A survey on measuring indirect discrimination in machine learning* (2015)
