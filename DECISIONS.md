# MM26 NCAAM — Project Brain
> **Source of truth** for all strategic decisions, model choices, and architecture.
> Claude.ai uses this for context. Cursor uses this as execution guidance. GitHub owns it.
> 
> **Workflow:** Update this file after every meaningful decision. Commit after every session.

---

## 🔄 How to Use This File

| Role | Action |
|------|--------|
| **Claude.ai** | Paste this file at the start of each new chat session for full context |
| **Cursor** | Reads this automatically as part of the project — reference it in prompts with `@DECISIONS.md` |
| **You** | Update it after decisions are made. Commit to GitHub after each session. |

---

## 📌 Project Overview

**Goal:** Build an ML model to predict NCAAM March Madness tournament outcomes  
**Repo:** https://github.com/billykeim/mm26_ncaam  
**Started:** [DATE]  
**Current Phase:** `[ ] Setup` `[ ] Data` `[ ] Features` `[ ] Modeling` `[ ] Evaluation` `[ ] Deploy`

---

## 🏗️ Architecture Decisions

### Data Sources
- [ ] **Primary:** TBD (KenPom / Bart Torvik / ESPN / Sports-Reference)
- [ ] **Secondary:** TBD
- [ ] **Tournament historical data:** TBD (years range)

*Decision notes:*
> [Record why you chose these sources here]

---

### Prediction Target
- [ ] Game-level win probability (each matchup independently)
- [ ] Full bracket simulation (end-to-end)
- [ ] Round-by-round probability
- [ ] Point spread / margin of victory

*Decision notes:*
> [Record choice and rationale here]

---

### Model Architecture
- [ ] **Baseline:** TBD (Logistic Regression / seed-only heuristic)
- [ ] **Primary model:** TBD (XGBoost / LightGBM / Neural Net / Ensemble)
- [ ] **Calibration:** TBD (Platt scaling / isotonic regression)

*Decision notes:*
> [Record choice and rationale here]

---

### Train/Test Split Strategy
- [ ] Leave-one-tournament-out CV (recommended — avoids leakage)
- [ ] Year cutoff (train on years X–Y, test on Z)
- [ ] Other

*Decision notes:*
> [Record choice and rationale here — data leakage is the #1 risk here]

---

### Feature Engineering
*Populate as features are decided:*

| Feature | Type | Source | Status | Notes |
|---------|------|---------|--------|-------|
| Seed differential | Numeric | NCAA bracket | — | Strong baseline signal |
| Adjusted efficiency margin | Numeric | KenPom/Torvik | — | Core predictive feature |
| SOS (strength of schedule) | Numeric | TBD | — | |
| ... | | | | |

---

## 🗂️ Repo Structure

```
mm26_ncaam/
├── DECISIONS.md          ← You are here (shared brain)
├── CLAUDE_CONTEXT.md     ← Paste into Claude at session start
├── .cursorrules          ← Cursor agent behavior rules
├── data/
│   ├── raw/              ← Never modify — source data only
│   ├── processed/        ← Cleaned, merged datasets
│   └── features/         ← Feature matrices ready for modeling
├── notebooks/            ← Exploration and EDA only (not production)
├── src/
│   ├── data/             ← Data ingestion + cleaning scripts
│   ├── features/         ← Feature engineering pipeline
│   ├── models/           ← Model training + evaluation
│   └── utils/            ← Shared utilities
├── outputs/
│   ├── models/           ← Saved model artifacts
│   └── predictions/      ← Bracket predictions
└── tests/                ← Unit tests
```

---

## 📋 Open Questions / Debate Log

*Use this section to track things Claude and you are actively debating — don't delete items, mark them resolved.*

| # | Question | Status | Resolution |
|---|----------|--------|------------|
| 1 | What's our primary data source? | 🔴 Open | — |
| 2 | Game-level or bracket-level prediction? | 🔴 Open | — |
| 3 | How to handle upsets / class imbalance? | 🔴 Open | — |
| 4 | How many years of historical data? | 🔴 Open | — |

---

## 🧪 Experiment Log

*Track what's been tried and what happened. Cursor should append here after each experiment run.*

| Date | Experiment | Result | Next Step |
|------|-----------|--------|-----------|
| — | — | — | — |

---

## ✅ Session Log

*Brief note after each working session — what was decided or built.*

| Date | Type | Summary |
|------|------|---------|
| [Today] | Setup | Established pipeline: Claude.ai + Cursor + GitHub |

---

## 🚫 What NOT to Do (Lessons Learned)

*Populate as you learn — this is gold for future sessions.*

- [ ] Don't train on tournament data that includes outcomes from the test year (leakage)
- [ ] Don't use in-tournament stats as features (they aren't known at prediction time)
- [ ] [Add as you discover them]
