# MM26 NCAAM — Project Brain
> **Source of truth** for all strategic decisions, model choices, and architecture.
> Claude.ai uses this for context. Cursor uses this as execution guidance. GitHub owns it.
>
> **Workflow:** Update this file after every meaningful decision. Commit after every session.
> **Technical reference:** See `PIPELINE.md` for full feature store schema, crosswalks, and pipeline architecture.

---

## 🔄 How to Use This File

| Role | Action |
|------|--------|
| **Claude.ai** | Paste `CLAUDE_CONTEXT.md` at the start of each new chat session |
| **Cursor** | Read this + `PIPELINE.md` before every task. Reference with `@DECISIONS.md` and `@PIPELINE.md` |
| **You** | Update after decisions are made. Commit to GitHub after each session. |

---

## 📌 Project Overview

**Goal:** Build an ML model to predict NCAAM March Madness tournament outcomes via full bracket simulation
**Repo:** https://github.com/billykeim/mm26_ncaam
**Started:** 2026-03-29
**Current Phase:** `✅ Setup` `🔄 Data` `⬜ Features` `⬜ Modeling` `⬜ Evaluation` `⬜ Deploy`

---

## ✅ Locked Decisions

### Prediction Target
**Full bracket simulation end-to-end**
- Train a game-level win probability model per matchup
- Chain predictions round-by-round via Monte Carlo simulation (10,000 iterations)
- Output: probability distribution over all possible bracket outcomes
- Pick highest-probability outcome per slot as the "official" bracket prediction

---

### Model Architecture (v1)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Primary model** | XGBoost (game-level win probability) | Best-in-class for tabular data at ~1,200 tournament game training rows; captures complex feature interactions natively |
| **Baseline model** | Logistic Regression (seed-only) | Honest measurement of feature engineering contribution |
| **Bracket simulation** | Monte Carlo (10,000 iterations) | Propagates uncertainty across rounds; outputs probability distributions |
| **Calibration** | Platt scaling | Ensures outputs are true probabilities |
| **v2 consideration** | Ensemble (XGBoost + LR + RF) | After v1 baseline is solid |

> **Why not neural nets for v1?** ~1,200 training samples is too small. XGBoost consistently outperforms on structured tabular data at this scale. Neural nets revisited only if we expand to full regular-season game-level training data.

---

### Train / Test Strategy
**Leave-one-tournament-out cross-validation**
- For each test year: train on all other years, predict that year's tournament
- Roll forward chronologically — never let future data inform past predictions
- Features for each game use only data available BEFORE that game's tip-off (rolling window)
- **This is the #1 leakage risk. Cursor must enforce this strictly.**

---

### Core Architecture Decision: Game-Level Rolling Features
**All features are computed as rolling windows — NOT end-of-season aggregates.**

This means:
- Every row in the training set represents a specific game
- Features for that game reflect only data from games played *before* that game's date
- This eliminates leakage AND enables temporal modeling: streaks, trajectory, early vs late season
- Tournament prediction uses features rolled through Selection Sunday (the last pre-tournament snapshot)

See `PIPELINE.md` for full rolling feature specifications.

---

### Data Sources

| Source | Type | Priority | Tool | Status |
|--------|------|----------|------|--------|
| Bart Torvik / T-Rank | Team efficiency, advanced metrics | ⭐ Primary | pybart | 🟢 Ingested |
| Sports-Reference CBB (coaches) | Coach tournament history | ⭐ Primary | pandas.read_html + BeautifulSoup | 🟢 Ingested |
| CBBpy (ESPN) | Game-by-game box scores, game logs | ⭐ Primary | CBBpy pip package | 🟡 In progress (2025 cached; full years running) |
| Historical tournament results | Bracket outcomes | ⭐ Required | pybart.madness | 🟢 Ingested (conf=3 NCAA filter) |
| Betting lines (pre-tournament) | Market-implied probabilities | 🟡 Nice to have | TBD | 🔴 Not ingested |
| Injury / roster news | Player availability proxy | 🟢 Simple proxy only | TBD | 🔴 Not ingested |

**Historical range:** 2008–present (~17 tournaments, excluding 2020 COVID cancellation)
**Pre-2008 gap:** Torvik time_machine only goes to 2011. For 2008–2010, use end-of-regular-season `team_ratings()` before conference tournaments. Document this in experiment log. *(Decision A — accepted)*

---

### Selection Sunday Dates + Time Machine Snapshot Dates

| Year | Selection Sunday | time_machine() date | Notes |
|------|-----------------|--------------------|----|
| 2008 | March 16, 2008 | `20080317` | Last 64-team field. time_machine unavailable — use team_ratings(). |
| 2009 | March 15, 2009 | `20090316` | time_machine unavailable — use team_ratings(). |
| 2010 | March 14, 2010 | `20100315` | time_machine unavailable — use team_ratings(). |
| 2011 | March 13, 2011 | `20110314` | First year of 68-team field + First Four. time_machine available. |
| 2012 | March 11, 2012 | `20120312` | |
| 2013 | March 17, 2013 | `20130318` | |
| 2014 | March 16, 2014 | `20140317` | |
| 2015 | March 15, 2015 | `20150316` | |
| 2016 | March 13, 2016 | `20160314` | |
| 2017 | March 12, 2017 | `20170313` | |
| 2018 | March 11, 2018 | `20180312` | |
| 2019 | March 17, 2019 | `20190318` | |
| 2020 | — | — | **CANCELLED — COVID-19. Exclude entirely from training set.** |
| 2021 | March 14, 2021 | `20210315` | Bubble format — all games Indianapolis. Flag `is_bubble_year=1`. |
| 2022 | March 13, 2022 | `20220314` | |
| 2023 | March 12, 2023 | `20230313` | |
| 2024 | March 17, 2024 | `20240318` | |
| 2025 | March 16, 2025 | `20250317` | |
| 2026 | March 15, 2026 | `20260316` | **Current year — predict only, do not include in training.** |

---

### Ingestion Tool Decisions

| Tool | Used For | Decision | Rationale |
|------|----------|----------|-----------|
| **pybart** | All Torvik data | ✅ Locked | Built for this purpose; caching support; madness module |
| **CBBpy** | Game-by-game box scores, game logs | ✅ Locked *(Decision A)* | ESPN-backed, well-maintained, covers 2008+, no auth required |
| **pandas.read_html + BeautifulSoup** | Sports-Reference coach pages | ✅ Locked *(Decision A)* | CBBpy doesn't cover coaches; SR coach index is clean HTML |
| **sportsipy** | Not used | ❌ Skipped | Redundant with CBBpy + pandas.read_html; older methodology |

---

### Recency Decay
- **Lambda:** `λ = 0.02` for exponential recency weighting *(Decision B)*
- Formula: weight_i = exp(-0.02 × days_ago_i)
- A game 35 days ago carries ~50% weight vs yesterday's game
- Treat λ as a tunable hyperparameter in v2 — log it in experiment log when tested

---

### Name Normalization
- **Team name crosswalk:** `src/utils/team_name_map.json` — maps Torvik names ↔ Sports-Ref names ↔ CBBpy names *(Decision C)*
- **Coach name crosswalk:** `src/utils/coach_name_map.json`
- Must cover all 68+ tournament-caliber teams across all years
- Build this FIRST before any join logic — silent join failures are the #1 risk
- See `PIPELINE.md` for format specification

---

### Feature Engineering Philosophy
See `PIPELINE.md` for full feature specifications. Summary of categories:

1. **Static season-level features** — Torvik efficiency ratings, four factors, SOS (from time_machine snapshot)
2. **Rolling game-level features** — computed from game_log with window ending before each game
3. **Streak + momentum features** — win/loss streaks, margin trends, early vs late season splits
4. **Player aggregate features** — team BPM, height, experience index, star concentration
5. **Coach features** — tournament appearances, win rate, Final Four count
6. **Matchup differential features** — all `delta_*` columns (t1 value − t2 value)
7. **Variance / upset risk flags** — 3pt reliance, low-tempo coin flip, major/mid-major flag

**Provenance:** Every feature column is documented in `data/features/schema_registry.json` with source, is_derived flag, derivation formula, rolling window, and version. See `PIPELINE.md`.

---

### Betting Lines (Nice to Have)
- Not blocking v1 *(Decision from session 1)*
- When added: create `src/data/ingest_betting.py`, register all columns in `schema_registry.json`, join in `build_static.py`
- The extensibility contract in `PIPELINE.md` defines exactly how to do this

---

## 🗂️ Repo Structure

```
mm26_ncaam/
├── DECISIONS.md              ← You are here (strategic decisions)
├── PIPELINE.md               ← Technical reference (schema, crosswalks, architecture)
├── CLAUDE_CONTEXT.md         ← Paste into Claude at session start
├── .cursorrules              ← Cursor agent behavior rules
├── data/
│   ├── raw/                  ← READ ONLY after ingestion
│   │   ├── torvik/
│   │   │   ├── timemachine/          ← {year}_pretournament.parquet
│   │   │   ├── game_results/         ← {year}_game_results.parquet
│   │   │   ├── four_factors/         ← {year}_four_factors.parquet
│   │   │   └── player_stats/         ← {year}_player_stats.parquet
│   │   └── sports_ref/
│   │       ├── coaches_cache/        ← raw HTML (never re-fetch)
│   │       └── game_logs/            ← CBBpy game-by-game box scores
│   ├── processed/
│   │   ├── game_log.parquet          ← master game log, all teams, all years
│   │   ├── coach_store.parquet       ← coach features by team × year
│   │   ├── player_aggregates.parquet ← team player features by team × year
│   │   └── team_name_map.json        ← name normalization crosswalk
│   ├── features/
│   │   ├── rolling_features.parquet  ← team × game_date, all rolling/streak features
│   │   ├── static_features.parquet   ← team × year, season-level features
│   │   ├── matchup_features.parquet  ← game × (t1, t2), all delta_* + labels
│   │   └── schema_registry.json      ← field catalogue (source, derivation, version)
│   └── training/
│       ├── train_{year}.parquet      ← leave-one-year-out splits
│       └── feature_importance_log/   ← XGBoost feature importance per fold
├── notebooks/                ← EDA only, not production
├── src/
│   ├── data/
│   │   ├── ingest_torvik.py
│   │   ├── ingest_coaches.py
│   │   ├── ingest_gamelogs.py
│   │   └── ingest_betting.py         ← placeholder, v2
│   ├── features/
│   │   ├── build_game_log.py
│   │   ├── build_rolling.py
│   │   ├── build_static.py
│   │   ├── build_matchups.py
│   │   └── validate.py
│   ├── models/
│   │   ├── baseline_lr.py
│   │   ├── xgboost_model.py
│   │   ├── calibration.py
│   │   ├── monte_carlo_sim.py
│   │   └── build_training.py
│   └── utils/
│       ├── constants.py
│       ├── team_name_map.json
│       ├── coach_name_map.json
│       └── validation.py
├── outputs/
│   ├── models/               ← saved artifacts with timestamps
│   └── predictions/          ← bracket simulation outputs
└── tests/                    ← pytest, mirrors src/
```

---

## 📋 Open Questions

| # | Question | Status | Resolution |
|---|----------|--------|------------|
| 1 | Exact CBBpy coverage for 2008–2010 game logs? | 🔴 Open | Validate during ingestion |
| 2 | Betting lines source when we pursue it? | 🔴 Open | — |
| 3 | λ=0.02 recency decay — validate against Torvik's own curve | 🔴 Open | Test in v2 |
| 4 | How to handle mid-season coach changes (rare but exists)? | 🔴 Open | Flag games post-change; use new coach features |
| 5 | 2021 bubble year — include with flag or exclude? | 🔴 Open | Include with `is_bubble_year=1`, test sensitivity |

---

## 🧪 Experiment Log

| Date | Experiment | Result | Next Step |
|------|-----------|--------|-----------|
| 2026-03-29 | Pipeline setup | Repo scaffolded, .cursorrules + DECISIONS.md + CLAUDE_CONTEXT.md committed | Begin data ingestion |
| 2026-03-29 | Architecture decisions | All foundational decisions locked; rolling feature design finalized | Execute ingestion pipeline |
| 2026-03-29 | Full raw ingestion | Torvik 191,304 rows; SR coaches 14,334 + index 4,096; team map 67; CBBpy gamelogs 265,867 rows (2025) + multi-year job started | build_game_log / rolling features |

---

## ✅ Session Log

| Date | Type | Summary |
|------|------|---------|
| 2026-03-29 | Setup | Pipeline established: Claude.ai + Cursor + GitHub |
| 2026-03-29 | Strategy | Locked model arch, data sources, rolling feature design, coach ingestion approach, name normalization strategy |

---

## 🚫 What NOT to Do

- **Never** use end-of-season stats as features for mid-season games — rolling windows only
- **Never** train on data that includes the test tournament year — leave-one-out strictly
- **Never** use in-tournament stats as input features — not known at prediction time
- **Never** write to `data/raw/` after initial ingestion — immutable source of truth
- **Never** join across sources without going through `team_name_map.json`
- **Do not** attempt historical injury NLP for v1 — scope creep
- **Do not** add columns to parquet files without registering in `schema_registry.json`
