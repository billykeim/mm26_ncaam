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
**Phase:** Modeling v1 complete — Monte Carlo next

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
- ✅ Raw + processed: Torvik, coaches, CBBpy game logs, coach_store, player_aggregates, static_features
- ✅ `build_game_log.py` → `data/processed/game_log.parquet` (196,736 team×game rows)
- ✅ `build_rolling.py` → `data/features/rolling_features.parquet` (shift-safe rolling + recency)
- ✅ `build_matchups.py` → `data/features/matchup_features.parquet` (~1,130 SR bracket games × ~458 model features)
- ✅ `validate.py` → `data/features/validation_report.txt` (0 FAILED on pre-model checks)
- ✅ `build_training.py` → `data/training/train_{Y}.parquet`, `test_{Y}.parquet`, `feature_list_v1.txt`
- ✅ `baseline_lr.py` → `data/outputs/predictions/baseline_lr_predictions.parquet` (seed-only baseline)
- ✅ `xgboost_model.py` → `xgb_v1_predictions.parquet`, `data/training/feature_importance_log/{Y}_importance.csv`, `data/outputs/models/xgb_v1_full.json` (+ meta JSON)
- ✅ `calibration.py` → `data/outputs/calibration_plot.png`, `xgb_v1_calibrated_predictions.parquet`
- ✅ `notebooks/03_model_results.ipynb` — LOO metrics, importance, year/round charts, calibration, upsets
- ✅ `schema_registry.json` extended (game_log, rolling, matchup entries)

**What's next:**
1. Bracket Monte Carlo harness (10,000 sims) using calibrated win probabilities
2. Optional leakage audit given strong LOO headline metrics
3. Address known gaps: 2008/2021 rolling snapshot edge cases, name-map coverage warnings from validate

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
- Modeling v1 scripts under `src/models/` (`baseline_lr`, `xgboost_model`, `calibration`) and outputs under `data/outputs/`
- Results narrative + charts: `notebooks/03_model_results.ipynb`