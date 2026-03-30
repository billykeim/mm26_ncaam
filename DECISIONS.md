# MM26 NCAAM — Project Brain
> **Source of truth** for all strategic decisions, model choices, and architecture.
> Claude.ai uses this for context. Cursor uses this as execution guidance. GitHub owns it.
>
> **Workflow:** Update this file after every meaningful decision. Commit after every session.

---

## 🔄 How to Use This File

| Role | Action |
|------|--------|
| **Claude.ai** | Paste CLAUDE_CONTEXT.md at the start of each new chat session |
| **Cursor** | Read this before every task. Reference with `@DECISIONS.md` |
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
- *Rationale: Captures compounding uncertainty across rounds; more useful than isolated game prediction*

---

### Model Architecture (v1)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Primary model** | XGBoost (game-level win probability) | Best-in-class for tabular data at this sample size (~1,675 games); captures complex feature interactions natively via tree splits |
| **Baseline model** | Logistic Regression (seed-only) | Forces honest measurement of feature engineering contribution; if XGBoost barely beats this, features need work |
| **Bracket simulation** | Monte Carlo (10,000 iterations) | Propagates uncertainty across rounds; gives probability distributions not just point picks |
| **Calibration** | Platt scaling | Ensures model outputs are true probabilities, not just scores |
| **v2 consideration** | Ensemble (XGBoost + LR + RF) | Stack after v1 is solid; neural nets only if sample size expands significantly |

> **Why not neural nets for v1?** ~1,675 training samples (25 tournaments × ~67 games) is too small. Neural nets overfit badly at this scale on tabular data. XGBoost consistently outperforms on structured sports data at this sample size. Revisit if we expand to game-level regular season data (tens of thousands of samples).

---

### Train / Test Strategy
**Leave-one-tournament-out cross-validation**
- For each test year: train on all other years, predict that year's tournament
- Roll forward chronologically — never let future data inform past predictions
- *Critical: this is the #1 leakage risk. Cursor must enforce this strictly.*

---

### Data Sources

| Source | Type | Priority | Status |
|--------|------|----------|--------|
| Bart Torvik / T-Rank | Team efficiency, advanced metrics | ⭐ Primary | 🔴 Not ingested |
| Sports-Reference (CBB) | Box scores, historical results, player stats | ⭐ Primary | 🔴 Not ingested |
| Historical tournament results | Bracket outcomes 2008–present | ⭐ Required | 🔴 Not ingested |
| Betting lines (pre-tournament) | Market-implied probabilities | 🟡 Nice to have | 🔴 Not ingested |
| Coaching data | Tournament experience, coach records | 🟡 Nice to have | 🔴 Not ingested |
| Injury / roster news | Player availability proxy | 🟢 Simple proxy only | 🔴 Not ingested |

**Historical range:** 2008–present (~17 tournaments)
- *Rationale: Game changed significantly with analytics revolution ~2008. Pre-2008 data likely adds noise. Start clean.*

**Injury data approach:** Do NOT attempt historical injury NLP — too sparse and complex for v1. Use a simple binary proxy instead if data is easily available, otherwise omit from v1.

---

### Feature Engineering (to be built)
*Cursor: update Status column as features are implemented.*

| Feature | Type | Source | Status | Notes |
|---------|------|---------|--------|-------|
| Seed | Numeric | NCAA bracket | 🔴 Pending | Include as both raw and differential |
| Seed differential (A - B) | Numeric | NCAA bracket | 🔴 Pending | Core matchup feature |
| Adjusted offensive efficiency | Numeric | Bart Torvik | 🔴 Pending | Points per 100 possessions, opponent-adjusted |
| Adjusted defensive efficiency | Numeric | Bart Torvik | 🔴 Pending | Points allowed per 100 possessions, adjusted |
| Adjusted efficiency margin (AdjEM) | Numeric | Bart Torvik | 🔴 Pending | Off - Def; strongest single team quality signal |
| AdjEM differential (A - B) | Numeric | Bart Torvik | 🔴 Pending | Core matchup feature |
| Tempo (possessions per 40 min) | Numeric | Bart Torvik | 🔴 Pending | Style matchup signal |
| Strength of schedule | Numeric | Bart Torvik | 🔴 Pending | Context for efficiency ratings |
| Coach tournament experience | Numeric | Sports-Reference | 🔴 Pending | # prior tournament appearances as HC |
| Recent form (last 10 games win%) | Numeric | Sports-Reference | 🔴 Pending | Momentum proxy |
| Conference | Categorical | Sports-Reference | 🔴 Pending | Major vs mid-major encoding |
| Pre-tournament betting line | Numeric | TBD | 🟡 Nice to have | If source available; highly predictive |

---

## 🗂️ Repo Structure

```
mm26_ncaam/
├── DECISIONS.md              ← You are here (shared brain)
├── CLAUDE_CONTEXT.md         ← Paste into Claude at session start
├── .cursorrules              ← Cursor agent behavior rules
├── data/
│   ├── raw/                  ← READ ONLY — never modify
│   │   ├── torvik/
│   │   ├── sports_ref/
│   │   └── tournament/
│   ├── processed/            ← Cleaned, merged datasets
│   └── features/             ← Feature matrices ready for modeling
├── notebooks/                ← EDA only, not production
├── src/
│   ├── data/
│   │   ├── ingest_torvik.py
│   │   ├── ingest_sports_ref.py
│   │   └── ingest_tournament.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── baseline_lr.py
│   │   ├── xgboost_model.py
│   │   ├── calibration.py
│   │   └── monte_carlo_sim.py
│   └── utils/
│       ├── constants.py
│       └── validation.py
├── outputs/
│   ├── models/               ← Saved artifacts with timestamps
│   └── predictions/          ← Bracket simulation outputs
└── tests/                    ← pytest, mirrors src/ structure
```

---

## 📋 Open Questions

| # | Question | Status | Resolution |
|---|----------|--------|------------|
| 1 | Bart Torvik ingestion method — API vs scrape vs manual export? | 🔴 Open | — |
| 2 | Betting lines source if we pursue it? | 🔴 Open | — |
| 3 | How to encode coach experience for first-time tournament coaches? | 🔴 Open | — |
| 4 | How to handle teams in test year that had limited prior tournament appearances? | 🔴 Open | — |

---

## 🧪 Experiment Log
*Cursor: append a row here after every experiment run.*

| Date | Experiment | Result | Next Step |
|------|-----------|--------|-----------|
| 2026-03-29 | Pipeline setup | Repo structure + decision framework established | Begin data ingestion |
| 2026-03-29 | Repo scaffold (Cursor) | `data/`, `src/`, `tests/`, `notebooks/`, `outputs/` tree; empty modules and `__init__.py` only | Implement data ingestion |

---

## ✅ Session Log

| Date | Type | Summary |
|------|------|---------|
| 2026-03-29 | Strategy | Locked all foundational decisions — see above |

---

## 🚫 What NOT to Do

- **Never** train on data that includes the test tournament year — #1 leakage risk
- **Never** use in-tournament stats as features — not known at prediction time
- **Never** write to `data/raw/` — immutable source of truth
- **Do not** attempt historical injury NLP for v1 — scope creep
- **Do not** jump to neural nets without XGBoost baseline first
