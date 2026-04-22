# Startup Guide

This file contains the practical startup commands for running FairLens in a
GitHub Codespace or similar remote Linux development environment.

## First-Time Setup

Run these commands the first time you start the project in a new Codespace.

### Backend terminal

```bash
cd auditra/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

After that, open `auditra/backend/.env` and add your real key:

```env
GEMINI_API_KEY=your_actual_key_here
```

Then start the backend:

```bash
cd auditra/backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend terminal

```bash
cd auditra/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Reopening The Same Codespace Later

If you reopen the same Codespace, you usually do not need to reinstall Python
or npm dependencies again.

### Backend terminal

```bash
cd auditra/backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend terminal

```bash
cd auditra/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## Notes

- Use two terminals: one for backend and one for frontend.
- In Codespaces, the explicit `--host 0.0.0.0` makes port forwarding work more
  reliably than relying on localhost defaults.
- If you create a brand-new Codespace instead of reopening the existing one,
  run the first-time setup again.
- If `.venv` or `node_modules` was deleted, reinstall dependencies.
- The backend health endpoint is available on port `8000` at `/health`.
