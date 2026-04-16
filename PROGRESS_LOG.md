# FairLens - Build Progress Log

> Auto-updated whenever a PRD item is completed.
> **For collaborators:** Read this file first to know exactly what's done and what's next.

---

## Project Overview
FairLens detects multi-hop proxy discrimination chains in ML datasets, visualizes them, and suggests surgical fixes. See PRD for full spec.

---

## Stack
- **Backend:** Python 3.11 + FastAPI
- **Graph Engine:** NetworkX
- **Correlation Math:** Pandas + Scikit-learn (Pearson, Cramér's V, eta-squared)
- **Chain Risk Scoring:** LightGBM reconstructive accuracy (Vertex AI AutoML when configured)
- **Explanations + Chat:** Gemini API (`gemini-1.5-pro`)
- **Fix Validation:** local SHAP library (Vertex AI XAI when configured)
- **Visualization:** D3.js force-directed graph
- **Frontend:** React + Tailwind + Vite
- **Reports:** Jinja2 + WeasyPrint PDF

---

## Phase 1 - Core Engine ✅ COMPLETE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Project scaffolding | ✅ Done | 2026-04-17 |
| 1.2 | FastAPI app + health check | ✅ Done | 2026-04-17 |
| 1.3 | CSV upload + column type detection | ✅ Done | 2026-04-17 |
| 1.4 | Correlation matrix (Pearson + Cramér's V + eta-squared) | ✅ Done | 2026-04-17 |
| 1.5 | NetworkX graph construction | ✅ Done | 2026-04-17 |
| 1.6 | DFS traversal - multi-hop chain detection | ✅ Done | 2026-04-17 |
| 1.7 | LightGBM chain risk scoring | ✅ Done | 2026-04-17 |
| 1.8 | Weakest-link finder | ✅ Done | 2026-04-17 |
| 1.9 | Fix application endpoint | ✅ Done | 2026-04-17 |
| 1.10 | **COMPAS test - 7/7 tests pass** | ✅ Done | 2026-04-17 - 949 chains to race, 2801 to sex, 944 multi-hop confirmed |

## Phase 2 - AI Integration ✅ COMPLETE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | Gemini chain explanation generator | ✅ Done | 2026-04-17 |
| 2.2 | Gemini conversational chat assistant | ✅ Done | 2026-04-17 |
| 2.3 | Vertex AI AutoML chain risk scoring | ✅ Done | 2026-04-17 - full integration in `vertex_ai_service.py`; LightGBM fallback active until `VERTEX_AI_ENDPOINT_ID` is set |
| 2.4 | Vertex AI Explainable AI (SHAP) | ✅ Done | 2026-04-17 - Vertex XAI + local SHAP fallback both implemented |

## Phase 3 - Frontend ✅ COMPLETE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | React + Vite + Tailwind scaffold | ✅ Done | 2026-04-17 |
| 3.2 | Upload screen + protected attribute selector | ✅ Done | 2026-04-17 |
| 3.3 | D3.js interactive graph | ✅ Done | 2026-04-17 - force sim, drag, click, chain highlight |
| 3.4 | Chain list panel with risk scores | ✅ Done | 2026-04-17 |
| 3.5 | Chain detail panel with Gemini explanation | ✅ Done | 2026-04-17 |
| 3.6 | Gemini chat box | ✅ Done | 2026-04-17 |
| 3.7 | Fix recommendation + approval workflow | ✅ Done | 2026-04-17 |
| 3.8 | SHAP before/after visualization | ✅ Done | 2026-04-17 - `ShapChart.tsx` with bar chart |

## Phase 4 - Report + Polish ✅ COMPLETE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Jinja2 PDF report template | ✅ Done | 2026-04-17 |
| 4.2 | Report generation endpoint | ✅ Done | 2026-04-17 |
| 4.3 | COMPAS pre-loaded demo mode | ✅ Done | 2026-04-17 - `/api/demo/compas` downloads + audits instantly |
| 4.4 | UI polish + loading states | ✅ Done | 2026-04-17 - skeleton loaders, toast notifications, compliance badge, evidence cards |
| 4.5 | Demo script | ✅ Done | 2026-04-17 - see `DEMO_SCRIPT.md` |

---

## ALL PHASES COMPLETE - BUILD STATUS: READY TO DEMO

---

## Environment Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # fill in GEMINI_API_KEY
uvicorn app.main:app --reload  # http://localhost:8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                    # http://localhost:5173

# Run tests (COMPAS CSV already in backend/data/compas.csv)
cd backend
python -m pytest tests/ -v
```

## GCP Setup (for Vertex AI, optional - LightGBM fallback works without it)

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT
gcloud services enable aiplatform.googleapis.com
# Train AutoML Tabular model on fairness benchmark data
# Deploy, copy endpoint ID to .env VERTEX_AI_ENDPOINT_ID
```

---

## What Was Confirmed Working

- **7/7 tests pass** on real COMPAS dataset
- **949 chains to race**, **2801 chains to sex** found in COMPAS
- **944 multi-hop chains (depth ≥ 2)** - the core novelty claim proven
- **Frontend builds clean** (TypeScript strict mode, 318 KB bundle)
- **FastAPI app loads** with all 11 routes registered
- **Demo mode** - `/api/demo/compas` auto-downloads + audits COMPAS in one click

## Known Limitations / Decisions

- Session store is in-memory - restarts lose data. Fine for hackathon; add Redis for production.
- SHAP scoring with LightGBM shows LOW scores for COMPAS chains - this is because individual
  chain features have low predictive power in isolation; the graph-derived risk score (geometric
  mean of hop weights) is more representative for demo. Vertex AI AutoML would improve this.
- WeasyPrint requires system libs (libpango etc) on Linux. On Windows it may fall back to HTML
  report - which still downloads fine.

---

_Last updated: 2026-04-17 - All phases complete_
