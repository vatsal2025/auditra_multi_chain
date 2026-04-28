# Auditra — Architectural Flow

## Overview

Auditra is a fairness auditing system for tabular ML datasets. A user uploads a CSV, selects protected attributes, and the system detects multi-hop proxy discrimination chains, computes standard fairness metrics, audits calibration, and returns a PDF report — all powered by Vertex AI.

The system is a monolith deployed as a single FastAPI process that serves both the REST API and the compiled React frontend as static files.

---

## Request Lifecycle

### 1. Upload Phase — `POST /api/upload`

```
Client → FastAPI → session_store (in-memory dict)
```

- File received as `UploadFile` (multipart form)
- Parsed into a `pandas.DataFrame`
- `detect_column_types(df)` classifies each column as `numeric` or `categorical`
  - Numeric dtype + high cardinality + not ID-like → `numeric`
  - Everything else → `categorical`
  - ID-like columns (uuid, ssn, zip, etc.) auto-excluded from correlation
- `get_excluded_columns(df)` flags near-unique columns (>95% unique values) to skip
- DataFrame stored in `session_store` keyed by a UUID `session_id`
- Returns: column list with types, row count, session_id

### 2. Audit Phase — `POST /api/audit`

This is the core computation pipeline. Runs synchronously (FastAPI async thread pool).

```
AuditRequest
  │
  ├─ [1] Graph Engine ──── build correlation graph
  │
  ├─ [2] Chain Finder ──── DFS over graph to find paths
  │
  ├─ [3] Chain Scorer ──── Vertex AI AutoML (chain-scorer endpoints)
  │                         LightGBM fallback
  │
  ├─ [4] Gemini Explanations ── aicredits.in proxy → gemini-2.0-flash
  │                              (top 5 chains; rest use fallback text)
  │
  ├─ [5] Fairness Metrics ─── Vertex AI AutoML (outcome-scorer endpoints)
  │                             LightGBM 5-fold CV fallback
  │                             → SPD, DI ratio, EOD, AOD, PPD
  │
  ├─ [6] Reweighing ────────── Kamiran & Calders weights → mitigated metrics
  │
  ├─ [7] Conjunctive Proxies ─ pairwise feature interaction scan
  │
  ├─ [8] Calibration Audit ─── ECE per group (Chouldechova 2017)
  │
  └─ [9] Intersectional Audit ─ pairwise protected attr SPD scan (Kearns 2018)
```

**Step 1 — Graph Engine** (`graph_engine.py`)

Builds a directed feature correlation graph:
- Pairwise strengths computed using the appropriate measure per column pair type:
  - numeric-numeric: Pearson |r| with two-tailed p-value
  - categorical-categorical: bias-corrected Cramér's V with chi-squared p-value
  - mixed: eta-squared with ANOVA F-test p-value
- All pairs filtered by Bonferroni-corrected significance (α = 0.05 / n_tests)
- Edges added only when strength ≥ `threshold` (default 0.15) and p-value significant
- Protected attributes are **sink nodes**: no outgoing edges ever leave them

**Step 2 — Chain Finder** (`graph_engine.py`)

Depth-first search for paths terminating at each protected attribute:
- DFS starts from every non-protected feature node
- Paths are collected only when they reach the protected attribute as terminal node
- Protected attributes blocked as intermediates (can only be chain endpoints)
- Cycle prevention: no node visited twice in same path
- Max depth: configurable (default 4 hops)
- Initial risk score = geometric mean of hop weights along the path
- Identical paths deduplicated; highest-scoring copy kept

**Step 3 — Chain Scorer** (`chain_scorer.py` + `vertex_ai_service.py`)

Replaces geometric mean risk with a model-derived skill score:
- Sends chain-path features to the Vertex AI AutoML chain-scorer endpoint
- Model predicts the protected attribute from only those features
- Skill score = (accuracy − majority_baseline) / (1 − majority_baseline)
- Skill = 0 → chain adds nothing over guessing the most common group
- Skill = 1 → chain perfectly reconstructs the protected attribute
- Falls back to LightGBM 5-fold CV on the same data if endpoint unavailable

**Step 4 — Gemini Explanations** (`gemini_service.py`)

- Top 5 chains sent to Vertex AI Gemini 1.5 Flash 8B
- Prompt instructs Gemini to name the historical/social reason, cite regulations
- Cached in `_explanation_cache` (keyed on path + protected attr) — repeat calls free
- Chains 6–20 get a deterministic fallback explanation (no API call)
- Chain 21+ not explained

**Step 5 — Fairness Metrics** (`fairness_metrics.py` + `vertex_ai_service.py`)

When `outcome_column` provided in request:
- Sends stratified sample (500 rows) to Vertex AI AutoML outcome-scorer endpoint
- Outcome model predicts the outcome label from all non-protected features
- Group-level metrics computed from returned predictions:
  - Prediction rate per group: P(Ŷ=1 | group)
  - TPR per group, FPR per group, precision per group
- Aggregate metrics: SPD, DI ratio, EOD, AOD, PPD (all unprivileged vs privileged)
- Falls back to LightGBM 5-fold CV on full dataset if Vertex AI endpoint not configured

**Step 6 — Reweighing** (`reweighing.py`)

- Kamiran & Calders (2012) formula: W_i = P(S) × P(Y) / P_obs(S, Y)
- Produces sample weights making outcome rates identical across demographic groups
- LightGBM retrained with these weights → mitigated fairness metrics computed
- Discrimination score after reweighing converges to 0 by construction

**Step 7 — Conjunctive Proxies** (`interaction_scanner.py`)

Detects Type 2 proxy discrimination (Zliobaite 2015):
- Computes individual skill score for every non-protected feature
- Candidate pairs: features with individual skill ≥ 0.02 (or half that for cross-pairing)
- Joint skill computed for each candidate pair together
- Interaction gain = joint_skill − max(skill_A, skill_B)
- Pairs with gain ≥ 0.05 flagged as conjunctive proxies

**Step 8 — Calibration Audit** (`calibration.py`)

- LightGBM trained with 5-fold CV to get probability outputs
- Expected Calibration Error (ECE) computed per demographic group using 10-bin equal-width bins
- Calibration gap = max(ECE) − min(ECE) across groups
- Gap < 0.05 → pass (Chouldechova threshold)
- High gap means the model is reliably confident for some groups but not others

**Step 9 — Intersectional Audit** (`intersectional.py`)

- Enumerates all combinations of protected attribute values (e.g. race=Black × sex=Female)
- Base rate P(Y=1) computed per intersection subgroup
- SPD vs privileged combo reported for each subgroup
- Subgroups with |SPD| > 0.10 flagged (Kearns 2018 threshold)
- Selects worst-case attribute pair when >2 protected attributes provided

**Result assembly**

All outputs written to an `AuditResponse` and stored in `session_store` alongside the raw graph objects, for use by subsequent fix/report calls.

### 3. Fix Phase — `POST /api/fix`

```
FixRequest (session_id, chain_id, fix_strategy)
  │
  ├─ "drop" strategy ── removes weakest-link feature from df copy
  │                      recomputes fairness metrics before/after
  │                      Vertex AI XAI SHAP attributions (optional)
  │
  └─ "reweigh" strategy ── applies Kamiran & Calders weights
                            recomputes fairness metrics with new weights
```

- Fix operates on a copy of the session DataFrame — original preserved
- Returns `MetricDelta` list (before/after per metric) and SHAP feature importances

### 4. Chat Phase — `POST /api/chat`

```
ChatRequest (session_id, message)
  │
  └─ aicredits.in proxy → gemini-2.0-flash
       System prompt: audit context (chain list + dataset name)
       Multi-turn history reconstructed from request
       Returns plain-text reply
```

### 5. Report Phase — `POST /api/report`

```
ReportRequest (session_id)
  │
  └─ report_generator.py ── pulls audit result from session_store
                              renders to PDF using reportlab/FPDF
                              writes to /tmp/reports/
                              returns download URL

GET /api/report/download/{filename}
  └─ FileResponse
```

---

## Session Storage

```python
# app/core/session_store.py
_store: dict[str, dict] = {}
```

In-memory dict. Keys: `session_id` → dict of `{df, col_types, audit, G, strengths, audit_config}`. Not persisted across server restarts. Sufficient for demo/hackathon scale; would need Redis or DB for production multi-instance.

---

## Frontend → Backend Communication

```
React (Vite, port 5173 dev / bundled for prod)
  │
  ├─ /api/upload     → multipart CSV upload
  ├─ /api/audit      → JSON body, waits for full result
  ├─ /api/fix        → JSON body
  ├─ /api/chat       → JSON body + JSON reply
  ├─ /api/report     → generates PDF
  └─ /api/report/download/{file} → binary stream
```

In production, FastAPI serves the compiled React bundle directly:
- `/assets/*` → `StaticFiles` serving `frontend/dist/assets/`
- All other paths → `frontend/dist/index.html` (React SPA catch-all)

### Audit Request — Optional Fairness Fields

The `/api/audit` request accepts an optional `outcome_column` field. When provided, the full fairness metrics pipeline (Steps 5–9) runs. When omitted, only chain detection + scoring runs (Steps 1–4).

```json
{
  "session_id": "...",
  "protected_attributes": ["sex", "race"],
  "max_depth": 4,
  "outcome_column": "income",      // optional — enables SPD/DI/EOD/AOD
  "privileged_groups": null,        // auto-inferred if null
  "positive_outcome": null          // auto-inferred (minority class)
}
```

The frontend exposes an "Outcome column" dropdown after CSV upload. Selecting a column enables the Fairness Metrics panel in results.

### Frontend Key Features

- **Graph:** D3 force-directed, draggable nodes, scroll-to-zoom, pan, neon-white glowing edges, color-coded nodes by risk level. Native browser Fullscreen API on the fullscreen button.
- **Chat:** aicredits.in → gemini-2.0-flash with full audit context injected as system prompt. 4096 token limit, 60s timeout.
- **Suggestion chip:** Neon green clickable prompt pre-fills chat input.

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `AICREDITS_API_KEY` | Yes | gemini-2.0-flash via aicredits.in proxy |
| `GCP_PROJECT_ID` | Yes | Vertex AI chain/outcome scorer endpoints |
| `GCP_REGION` | Yes | Vertex AI region (us-central1) |
| `VERTEX_AI_ENDPOINT_*` | No | AutoML endpoints — LightGBM fallback if unset |

---

## Vertex AI Communication

All Vertex AI calls use Application Default Credentials (ADC). On the GCP VM the service account attached to the instance handles auth automatically — no JSON key file, no `GOOGLE_APPLICATION_CREDENTIALS` env var needed.

```
VM (Service Account)
  │
  ├─ aiplatform.Endpoint.predict()   chain-scorer endpoints  (4 endpoints)
  ├─ aiplatform.Endpoint.predict()   outcome-scorer endpoints (4 endpoints)
  ├─ aicredits.in (OpenAI-compat proxy) → gemini-2.0-flash  (explanations)
  └─ aicredits.in (OpenAI-compat proxy) → gemini-2.0-flash  (chat)
```

---

## Deployment Topology

```
GCP VM: auditra-vm (us-central1-a, e2-standard-4)
  ├─ Port 8000 (TCP) — FastAPI server via systemd (auditra.service)
  ├─ GCP Firewall rule: allow-8000 (0.0.0.0/0)
  │
  └─ Vertex AI (managed, us-central1)
       ├─ 4 × AutoML TabularDataset
       ├─ 8 × AutoML TrainingJob (4 chain-scorers + 4 outcome-scorers)
       └─ 8 × Endpoint (n1-standard-4, 1 replica each)

GCS Bucket: auditra-ml-6bf0badc
  └─ datasets/ (compas.csv, adult_train.csv, adult_test.csv, german.csv)
```
