# Claude Session Context — MM26 NCAAM
> **Paste this at the start of every new Claude.ai conversation.**
> Update "Current State" before committing at end of each session.

---

## What We're Building
Full bracket simulation model for NCAAM March Madness.
Repo: https://github.com/billykeim/mm26_ncaam

## My Stack
- **Execution agent:** Cursor (reads repo directly, governed by .cursorrules + PIPELINE.md)
- **Thought partner:** Claude.ai (stateless — needs this file for context each session)
- **Source of truth:** GitHub (DECISIONS.md + PIPELINE.md are canonical)

## Pipeline Protocol
1. Think + decide with Claude.ai → update DECISIONS.md → commit
2. Cursor reads DECISIONS.md + PIPELINE.md + .cursorrules → executes → logs results → commits
3. Next Claude session starts by pasting this file

---

## Current Project State
**Phase:** Data Ingestion — executing full pipeline

**All architecture decisions locked:**
- Prediction target: full bracket simulation (game-level XGBoost + Monte Carlo 10,000 iterations)
- Primary model: XGBoost | Baseline: Logistic Regression (seed-only) | Calibration: Platt scaling
- Train/test: leave-one-tournament-out CV
- Core principle: ALL features are rolling (game-level windows, no end-of-season leakage)
- Data: Torvik (pybart) + Coach history (SR/pandas.read_html) + Game logs (CBBpy)
- Date range: 2008–present (excl. 2020 COVID); 2021 flagged is_bubble_year=1
- 2008–2010: use team_ratings() instead of time_machine() (time_machine only back to 2011)
- Recency decay: λ=0.02 exponential weighting (Decision B)
- Name normalization: team_name_map.json + coach_name_map.json (Decision C)

**What's been built:**
- ✅ Repo scaffolded with full directory structure
- ✅ DECISIONS.md — strategic source of truth
- ✅ PIPELINE.md — full technical reference (schema, crosswalks, feature specs, validation)
- ✅ .cursorrules — Cursor agent behavior rules
- ✅ CLAUDE_CONTEXT.md — this file

**What's in progress / next:**
1. Cursor executing full ingestion pipeline (ingest_torvik → ingest_coaches → ingest_gamelogs)
2. Build processed layer (game_log, coach_store, player_aggregates)
3. Build rolling features + static features
4. Build matchup feature matrix
5. Run validate.py
6. Build training splits

**Open questions:**
- CBBpy coverage for 2008–2010 game logs — validate during ingestion
- Mid-season coach change handling — flag + use start-of-season coach
- 2021 bubble year — include with flag, test sensitivity in v2

---

## Key Files to Reference
- `DECISIONS.md` — all strategic decisions, locked choices, open questions
- `PIPELINE.md` — feature schemas, file layout, coach ingestion design, extensibility contract
- `.cursorrules` — how Cursor should behave

---

## Recent Code / Output to Review
[Paste Cursor output, error messages, or model results here at session start]
