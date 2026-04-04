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
**Phase:** Features complete — modeling next (XGBoost v1)

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
- ✅ `build_matchups.py` → `data/features/matchup_features.parquet` (2,218 tournament rows × 494 cols)
- ✅ `validate.py` → `data/features/validation_report.txt` (leakage checks + quality warnings)
- ✅ `build_training.py` → `data/training/train_{Y}.parquet`, `test_{Y}.parquet`, `feature_list_v1.txt` (456 features)
- ✅ `schema_registry.json` extended (game_log, rolling, matchup entries)

**What's next:**
1. Train XGBoost v1 on LOO folds; seed-only logistic baseline; Platt calibration
2. Bracket Monte Carlo harness (10k sims)
3. Address known gaps: 2008/2021 rolling snapshot empty (calendar vs CBBpy year), improve team static join coverage (~18% null barthag)

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
[Here is what was implemented and how it was committed.

Commits (newest first)
Commit	Message
ae2dbb2	data: gamelog ingestion complete — 265867 rows ingested
fd77ce3	data: team name map complete — 67 rows ingested
161b750	data: coach ingestion complete — 14334 rows ingested
95bac92	data: torvik ingestion complete — 191304 rows ingested
86de188	data: dependencies complete — 0 rows ingested
main is 6 commits ahead of origin/main (including prior docs: commit).

Step 1 — Dependencies
Added requirements.txt with pybart @ git+https://github.com/avewright/pybart plus CBBpy, pandas, bs4, requests, lxml, html5lib, pyarrow, scikit-learn, xgboost.
Installed via pip install -r requirements.txt.
.gitignore: .torvik_cache/, .Rhistory, gamelogs_full.log.
Step 2 — ingest_torvik.py
Torvik(cache_dir=<project>/.torvik_cache) (project root, not cwd).
Tournament training set: pybart.madness.tournament_training_set does not match current super_sked (no "NCAA" in text). Implemented conf == 3 NCAA games + team_features_str_keys() so team merges use strings (avoids int/float vs str merges).
Wrote: tournament_training_set.parquet, game_results/{year}_game_results.parquet, four_factors/, player_stats/, timemachine/{year}_pretournament.parquet (2008–2010: team_ratings + timemachine_available=0; 2011–2025: time_machine + flag 1; 2020 skipped).
Logged row counts to stdout; total written (sum across files): 191,304 (includes tournament set + all per-year + timemachine rows).
Step 3 — ingest_coaches.py
Letter pages a-index–z-index ( x-index → 404, skipped).
pandas.read_html on #NCAAM_coaches; unique column names; string coercion for parquet.
Detail pages cached under data/raw/sports_ref/coaches_cache/; 3s sleep between coach fetches.
data/processed/coach_store.parquet: 14,334 cumulative rows; coaches_index.parquet: 4,096 rows.
src/utils/coach_name_map.json: 100 skeleton entries (top coaches by score).
Step 4 — src/utils/team_name_map.json
67 priority teams from PIPELINE.md with canonical, torvik, sports_ref, cbbpy.
Step 5 — ingest_gamelogs.py
get_games_team is broken in CBBpy 2.1.2 (KeyError: 'game_day' on sort). Ingestion uses get_team_schedule + get_game_boxscore per game with a short delay.
2025 full D1 pull: 265,867 boxscore rows, 364 team files.
python -m src.data.ingest_gamelogs --years 2024,2025 supported; default = 2008–2025 excl. 2020; --workers (default 4) parallelizes by year.
A background nohup python3 -m src.data.ingest_gamelogs --workers 4 was started so remaining seasons can fill the cache; new parquets may still be written under data/raw/sports_ref/game_logs/. When that finishes, run git add data/raw/sports_ref/game_logs/ and commit. There is also a long-running get_games_season test in an old terminal session; you can stop it if it is still running (it is not needed for this pipeline).

Other
DECISIONS.md: data-source table + experiment log updated.
pytest passes.
Note: Step 5 commit message uses 265,867 = 2025 season boxscore rows. Filling all years will take a long time; re-run without --years (or list missing years) and commit again when done if you want a single “full history” row total.]

Here is where things stand after finishing the handoff work.

Wrap-up completed
conf_wins_json in historical_reference
Conference win counts by winner’s conference are computed in build_historical_reference and stored as JSON so the stacked conference chart in 02_tournament_analytics.ipynb has data.

02_tournament_analytics.ipynb
Replaced display(sm) with print(sm) so the summary table runs without IPython.

DECISIONS.md
Added one experiment-log row for the EDA → static features → tournament analytics batch.

Separate commits (new work on main):

Commit	Message
a7e26df	feat: expand team name map from Torvik timemachine + coach_store
525b6d3	feat: team×year player aggregates from Torvik player_stats
c71aaa5	feat: static team×year features and schema registry bootstrap (includes constants.py, __init__.py)
1d1db80	docs: EDA notebook for raw Torvik and coach sanity checks
6c44739	data: tournament analytics suite complete
The final commit includes tournament_analytics.py, data/processed/tournament_analytics/*, notebooks/02_tournament_analytics.ipynb, updated data/features/schema_registry.json, and DECISIONS.md.

pytest
Still 1 passed.
Left uncommitted (intentional)
CLAUDE_CONTEXT.md — local edits only.
data/raw/sports_ref/game_logs/** — overnight CBBpy pull; not part of these tasks.
nohup.out — runtime log.
Reminders from the earlier summary
Task F uses chronological round bins and efficiency-based favorites where needed; it is not full parity with every metric in your original Group 1–5 spec.
Seed-pair priors are in seed_pair_win_rates.parquet and registered; joining them into static_features.parquet at matchup level is still a follow-up if you want that wired in.
