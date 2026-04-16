# FairLens

Pre-training dataset auditor for multi-hop proxy discrimination chains.

> **See `PROGRESS_LOG.md` for build status and handoff notes.**

## Quick Start

```bash
# Backend
cd backend
python -m venv venv && source venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 - upload COMPAS CSV, select "race" as protected attribute, run audit.

## COMPAS Demo Dataset
```bash
curl -L "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv" \
  -o backend/data/compas.csv
```
