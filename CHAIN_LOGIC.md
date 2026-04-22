# Chain Finding Logic

This document explains the current FairLens backend implementation for finding
multi-hop proxy discrimination chains in a dataset.

It reflects the code as it exists today, not an aspirational architecture.

## Relevant Files

- `backend/app/api/routes/upload.py`
- `backend/app/api/routes/demo.py`
- `backend/app/api/routes/audit.py`
- `backend/app/api/routes/fix.py`
- `backend/app/services/graph_engine.py`
- `backend/app/services/chain_scorer.py`
- `backend/app/services/fix_engine.py`
- `backend/app/core/session_store.py`
- `backend/app/models/schemas.py`
- `backend/app/core/config.py`

## High-Level Goal

The system tries to answer this question:

Can a protected attribute such as `race` or `sex` be indirectly reconstructed
from other columns through one or more intermediate features?

The implementation treats a chain like:

`zipcode -> neighborhood -> income_band -> race`

as evidence that a model could infer `race` indirectly even if `race` is not
used as a direct feature.

## End-to-End Flow

There are two ways to start an audit:

1. `POST /api/upload`
2. `POST /api/demo/compas`

Both paths end up storing a Pandas DataFrame in the in-memory session store and
then calling the same audit pipeline.

### 1. Dataset registration

`upload.py` reads a CSV into a DataFrame, validates that it has at least two
columns, generates a `session_id`, and stores the following per-session state:

- `df`
- `col_types`
- `filename`
- `chat_history`
- `fixes_applied`

`demo.py` does the same thing for the built-in COMPAS dataset, then immediately
calls the audit route with:

- protected attributes: `["race", "sex"]`
- `max_depth=4`
- `threshold=0.15`

### 2. Column type detection

`detect_column_types()` in `graph_engine.py` classifies each column as either:

- `numeric`
- `categorical`

Current rule:

- if Pandas thinks the column is numeric and it has more than 10 unique values,
  it is marked `numeric`
- otherwise it is marked `categorical`

This is intentionally simple and fast, but it means low-cardinality numeric
columns are treated as categorical.

### 3. Pairwise predictive strength calculation

`build_graph()` starts by calling `_pairwise_strength()`, which computes a
strength score for every column pair.

The metric depends on the two column types:

- numeric + numeric: absolute Pearson correlation
- categorical + categorical: Cramer's V
- numeric + categorical: eta-squared

Important implementation detail:

- strengths are stored in both directions, so `(a, b)` and `(b, a)` receive the
  same value
- the score is therefore symmetric even though the graph is represented as a
  directed graph later

This means the current graph is direction-aware in data structure only, not in
statistical meaning.

### 4. Graph construction

`build_graph()` creates a `networkx.DiGraph` and adds every column as a node.

Then for each pairwise strength:

- if `a != b`
- and `weight >= threshold`

it adds an edge `a -> b` with that weight.

Because strengths are inserted in both directions, qualifying pairs produce two
directed edges:

- `a -> b`
- `b -> a`

The threshold is the main control for graph density:

- lower threshold -> denser graph -> more possible chains
- higher threshold -> sparser graph -> fewer possible chains

### 5. Multi-hop chain search

`find_chains()` is the core enumerator.

Inputs:

- graph `G`
- pairwise `strengths`
- list of `protected_attributes`
- `max_depth`
- `col_types`

The search works as follows:

1. Build a list of non-protected starting nodes.
2. For each protected attribute:
3. For each non-protected start node:
4. Run DFS until the path reaches the protected node or exceeds the depth limit.

The DFS implementation is `_dfs_chains()`.

Current DFS behavior:

- it grows paths one neighbor at a time
- it stops when `len(current_path) > max_depth + 1`
- it records a chain as soon as the current last node equals the protected node
- it does not revisit nodes already in the current path

That last rule makes every discovered chain a simple path, not a cyclic walk.

### 6. What `max_depth` means

`max_depth` is effectively the maximum number of hops, not the maximum number of
nodes.

Example:

- `max_depth=4`
- maximum path length in nodes is `5`
- example valid chain: `A -> B -> C -> D -> protected`

### 7. Chain object creation

Every raw DFS path is turned into a `Chain` object.

Each chain stores:

- `id`
- `path`
- `hops`
- `risk_score`
- `risk_label`
- `protected_attribute`
- `weakest_link`
- optional `explanation`

Each hop is represented as `ChainHop`:

- `source`
- `target`
- `weight`

### 8. Initial chain risk before model scoring

Inside `find_chains()`, the initial chain score is computed as the geometric
mean of hop weights:

`risk_score = product(weights) ** (1 / number_of_hops)`

Why this matters:

- a single weak edge drags the chain down
- longer chains are penalized naturally
- the score stays in the same 0-1 range as the edge strengths

This is only the first-pass risk estimate.

### 9. Weakest-link selection

The weakest link is chosen as the hop with the smallest weight.

Important current behavior:

- the stored `weakest_link` is the `source` column of that weakest hop
- not the full edge
- not the `target`

So if the weakest hop is:

`income_band -> race`

the stored removable feature is:

`income_band`

This directly affects the fix engine, which drops that feature column later.

### 10. Deduplication and sorting

After all candidate chains are built:

- chains are sorted by descending `risk_score`
- chains are deduplicated by exact `tuple(path)`

So two chains with the same ordered path are treated as duplicates, even if they
were reached from separate traversal branches.

## Post-Discovery Risk Scoring

The graph engine's geometric-mean score is not the final chain score.

`audit.py` calls `score_all_chains()` from `chain_scorer.py`, which rescoring
every chain based on reconstructive accuracy.

### 11. Reconstructive accuracy idea

For a chain ending in a protected attribute:

- all non-protected nodes in the path become features
- the protected attribute becomes the target

The question becomes:

How accurately can these chain features predict the protected attribute?

That prediction accuracy is treated as the chain's final risk score.

### 12. Vertex AI first, LightGBM fallback

`score_chain()` uses this order:

1. Try `score_chain_vertex()`
2. If unavailable or not configured, use local LightGBM

Vertex AI is only used when `VERTEX_AI_ENDPOINT_ID` is configured.

### 13. Local LightGBM scoring

The local path in `_score_via_lgbm()` does the following:

1. Select chain feature columns plus the protected target column.
2. Drop rows with nulls in any of those columns.
3. If fewer than 50 rows remain, return `0.0`.
4. Label-encode categorical features.
5. Label-encode the target when needed.
6. Train `LGBMClassifier`.
7. Run 3-fold cross-validation with accuracy scoring.
8. Use mean accuracy as final score.

Then `score_all_chains()` replaces:

- `risk_score`
- `risk_label`

and re-sorts the chains.

So in practice, the final ranking used by the UI is based on model accuracy, not
on the graph-engine geometric mean.

## Audit Response Construction

After rescoring:

1. `audit.py` generates Gemini explanations for the top 20 chains.
2. `build_graph_schema()` converts the graph into frontend-friendly nodes and
   edges.
3. A text summary is created.
4. The finished `AuditResponse` is stored in the session store.

### 14. Node risk assignment for the UI

`build_graph_schema()` assigns each node the worst risk level of any chain that
includes that node.

This means node coloring in the UI reflects chain participation severity, not an
independent node-level metric.

## Fix Workflow

The chain-finding logic feeds directly into the fix flow.

`POST /api/fix`:

1. looks up the chain by `chain_id`
2. passes the current DataFrame and chain to `apply_fix()`
3. drops the chain's `weakest_link` column
4. stores the modified DataFrame back into the session
5. records the fix in `fixes_applied`
6. removes the chain from the current stored audit result

### 15. SHAP and explanation of the fix

`fix_engine.py` tries to produce before/after feature influence data:

1. try Vertex AI Explainable AI
2. else try local SHAP with LightGBM
3. else fall back to a simple correlation-based proxy

The fixed dataset is currently produced by removing the feature column entirely.

## Session Model

The implementation is session-based and in-memory.

`session_store.py` keeps a dictionary keyed by `session_id`.

Current implications:

- server restarts erase all uploaded data and audit results
- the DataFrame is kept in RAM
- the graph and strengths are also kept in RAM after an audit

This is acceptable for a hackathon demo but would need Redis or a database for a
production deployment.

## Current Assumptions and Limitations

### Statistical simplifications

- Pairwise strengths are symmetric, so direction in the graph is structural, not
  causal.
- Type detection is heuristic and coarse.
- The graph only considers pairwise relationships when building edges.
- DFS explores all qualifying simple paths up to depth, which can grow quickly in
  dense graphs.

### Operational simplifications

- Sessions are in-memory only.
- The fix strategy removes a whole column instead of transforming or masking it.
- Top-20 Gemini explanations are rate-limited by count, not by total token cost.

### Why this still works for the demo

Despite those simplifications, the current implementation is coherent:

- pairwise statistics create a feature relationship graph
- DFS exposes indirect paths to protected attributes
- LightGBM rescoring answers whether the chain can actually reconstruct the
  protected attribute
- weakest-link removal gives a simple, explainable mitigation action

## Short Mental Model

If you need to explain the current system quickly, this is the simplest accurate
summary:

1. Turn the dataset into a graph where columns are nodes and strong statistical
   relationships are edges.
2. Search every simple path from non-protected columns to protected columns.
3. Treat each path as a possible proxy-discrimination chain.
4. Re-score each chain by training a model to see how well the chain predicts
   the protected attribute.
5. Rank the chains, explain the highest-risk ones, and suggest breaking a chain
   by dropping the weakest feature in it.
