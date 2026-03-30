# Claude Session Context — MM26 NCAAM
> **Paste this at the start of every new Claude.ai conversation.**
> Update the "Current State" section after each session before committing.

---

## What We're Building
Full bracket simulation model for NCAAM March Madness.
Repo: https://github.com/billykeim/mm26_ncaam

## My Stack
- **Execution agent:** Cursor (reads repo directly, governed by .cursorrules)
- **Thought partner:** Claude.ai (stateless — needs this file for context each session)
- **Source of truth:** GitHub (DECISIONS.md is the canonical record)

## Pipeline Protocol
1. Think + decide with Claude.ai → update DECISIONS.md → commit
2. Cursor reads DECISIONS.md + .cursorrules → executes → logs results → commits
3. Next Claude session starts by pasting this file

---

## Current Project State
**Phase:** Data Ingestion (next immediate step)

**What's been decided (all locked):**
- Prediction target: full bracket simulation end-to-end
- Primary model: XGBoost (game-level win probability)
- Baseline model: Logistic Regression (seed-only)
- Bracket sim: Monte Carlo (10,000 iterations)
- Calibration: Platt scaling
- Train/test: leave-one-tournament-out CV
- Data: Bart Torvik + Sports-Reference + historical tournament results
- Date range: 2008–present (~17 tournaments)
- Injury data: simple proxy only (no NLP), or omit from v1
- Betting lines: nice to have, not blocking

**What's been built:**
- Repo structure + pipeline files (DECISIONS.md, CLAUDE_CONTEXT.md, .cursorrules)
- Directory structure defined in DECISIONS.md (not yet scaffolded in repo)

**What's next:**
1. Scaffold the directory structure in the repo
2. Build data ingestion scripts (Torvik + Sports-Reference + tournament results)
3. First EDA notebook — sanity check the data

**Open questions (unresolved):**
- Bart Torvik ingestion method (API vs scrape vs manual export)?
- Betting lines source?
- Coach experience encoding for first-timers?

---

## Key Decisions Summary

| Decision | Choice |
|----------|--------|
| Prediction target | Full bracket simulation (game-level model + Monte Carlo bracket sim) |
| Primary model | XGBoost |
| Baseline | Logistic Regression (seed-only) |
| Simulation | Monte Carlo 10,000 iterations |
| Calibration | Platt scaling |
| Train/test split | Leave-one-tournament-out CV |
| Primary data | Bart Torvik, Sports-Reference CBB |
| Date range | 2008–present |
| Injury data | Simple proxy or omit v1 |
| Neural nets | v2 only, after XGBoost baseline |

---

## Recent Code / Output to Review
[Paste Cursor output, error messages, or model results here at session start]
