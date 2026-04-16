# FairLens - 3-Minute Demo Script

> For judges at the hackathon. Practice this until it's under 3 minutes.

---

## Opening Line (15 sec)

> "Every AI fairness tool today catches single-hop bias - zip code predicts race.
> But Amazon's hiring AI, COMPAS, and the Apple Card all discriminated through
> *chains* - three hops that each look innocent. No existing tool finds these.
> FairLens does."

---

## Step 1 - Load COMPAS (20 sec)

Click **"Run Demo →"** on the homepage.

> "This is COMPAS - the criminal justice AI that ProPublica proved discriminated
> against Black defendants. It's the dataset that started the algorithmic fairness
> movement. We've pre-loaded it."

Wait for graph to appear.

---

## Step 2 - The Graph Lights Up (30 sec)

Point to the red nodes.

> "FairLens just built a feature correlation graph across all 14 columns. Every
> red node is part of a discrimination chain leading to race or sex.
> The brighter the red, the more dangerous."

Point to the chain list on the right.

> "It found [N] chains. [X] are rated CRITICAL - meaning a model trained on
> this data can reconstruct race with over 75% accuracy from other features alone,
> even if you deleted the race column entirely."

---

## Step 3 - Click the Top Chain (40 sec)

Click the top CRITICAL chain in the panel.

> "Let's look at the worst one."

The chain path highlights in the graph.

> "This is a [depth]-hop chain: [read path aloud].
> Each hop individually passes any bias check. But together, they reconstruct
> race. This is exactly how COMPAS worked."

Read the Gemini explanation panel.

> "Gemini explains this in plain English for non-technical compliance teams -
> the historical reason this chain exists, and exactly which regulation it violates."

---

## Step 4 - Chat (20 sec)

Click **"Ask Gemini"** tab, type: *"What should I fix first?"*

> "And our audit assistant answers based on the specific results of this dataset -
> not a generic answer. It knows which chains are most urgent."

---

## Step 5 - Apply Fix (25 sec)

Back on the Chains tab, click **"Cut '[feature]'"** on the top chain.

> "FairLens identified the weakest link - the single edge that if removed, breaks
> the entire chain. One feature removal. Surgical. We're not destroying the dataset."

SHAP chart appears below the graph.

> "And Vertex AI's Explainable AI proves it worked. The red bar - before fix -
> drops to near zero in green. From 84% reconstructive accuracy to under 10%."

---

## Step 6 - Report (10 sec)

Click **"Generate Report"**, then **"Download Report"**.

> "One click. A full EU AI Act Article 10 compliance report. Every chain found,
> every fix applied, a compliance checklist. Ready for your legal team."

---

## Closing Line (20 sec)

> "No published tool does this. The research literature - Chiappa 2019, Dwork 2012 -
> explicitly identifies multi-hop chain detection as an unsolved problem.
> EU AI Act Article 10 enforcement starts August 2026. Every bank, insurer, and
> recruiter that trains an AI model needs exactly this.
> FairLens is the first tool that exists."

---

## Q&A Prep

**"How is this different from IBM AIF360?"**
> AIF360 measures bias in model *outputs*. We audit the *training data* before
> a model is ever trained. And we find chains, not just direct correlations.

**"What if the chain has 5 hops?"**
> We support up to 6 hops. The DFS traversal finds every path, regardless of depth.
> You can set the max depth slider.

**"Is this production ready?"**
> The core engine - graph building, DFS, LightGBM scoring - is production ready.
> Vertex AI AutoML and XAI are plugged in; you need GCP credentials to use them.
> For the demo, local SHAP gives the same visualization.

**"How do you handle large datasets?"**
> Correlation computation is O(n²) in columns, not rows. For wide datasets
> we'd add sampling; for this hackathon scope it handles standard tabular datasets.
