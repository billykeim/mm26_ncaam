# MM26 NCAAM — Pipeline Technical Reference
> **Living document.** Update whenever schema, crosswalks, or pipeline logic changes.
> This is the technical complement to `DECISIONS.md` (strategy) and `.cursorrules` (execution rules).
> Cursor should read this alongside `DECISIONS.md` before any feature engineering or ingestion task.

---

## Table of Contents
1. [Pipeline Execution Order](#1-pipeline-execution-order)
2. [Core Architectural Principle: Rolling Features](#2-core-architectural-principle-rolling-features)
3. [File Layout + Schema](#3-file-layout--schema)
4. [Provenance + Schema Registry](#4-provenance--schema-registry)
5. [Feature Specifications](#5-feature-specifications)
   - 5a. Static Season-Level Features
   - 5b. Rolling Game-Level Features
   - 5c. Streak + Momentum Features
   - 5d. Player Aggregate Features
   - 5e. Coach Features
   - 5f. Matchup Differential Features
   - 5g. Variance / Upset Risk Flags
6. [Coach Ingestion Design](#6-coach-ingestion-design)
7. [Name Normalization Crosswalks](#7-name-normalization-crosswalks)
8. [Extensibility Contract](#8-extensibility-contract)
9. [Validation Checks](#9-validation-checks)
10. [Performance Targets](#10-performance-targets)

---

## 1. Pipeline Execution Order

Run in this exact order. Each stage depends on the previous.

```bash
# Stage 0 — install dependencies
pip install pybart CBBpy pandas beautifulsoup4 requests pyarrow fastparquet scikit-learn xgboost --break-system-packages

# Stage 1 — raw data ingestion (run once; cached after first run)
python -m src.data.ingest_torvik        # Torvik ratings, four factors, player stats, time machine
python -m src.data.ingest_coaches       # Sports-Reference coach tournament history (~30 min, rate-limited)
python -m src.data.ingest_tournament_seeds  # Official NCAA seeds from SR brackets (~90s, rate-limited)
python -m src.data.ingest_tournament_results  # Full bracket game rows from SR HTML (cached; ~90s first run)
python -m src.data.ingest_gamelogs      # CBBpy game-by-game box scores (~60-90 min first run)

# Stage 2 — processed layer
python -m src.features.build_game_log   # Master game log with team name normalization
python -m src.features.build_rolling    # Rolling/streak features (game × team)
python -m src.features.build_static     # Season-level features (team × year)

# Stage 3 — feature matrix
python -m src.features.build_matchups   # Matchup pairs with delta_* features + result labels
python -m src.features.validate         # Schema audit, leakage check, null rate check

# Stage 4 — training splits
python -m src.models.build_training     # Leave-one-tournament-out splits → data/training/

# Re-run after any source data update:
# Only stages 2+ need to re-run if raw data unchanged
# Only stage 3+ need to re-run if rolling/static features unchanged
```

---

## 2. Core Architectural Principle: Rolling Features

**Every feature computed for game G uses only data from games played BEFORE game G's date.**

This is enforced by:
1. Always sorting game_log by `game_date` ascending before computing rolling windows
2. Using `pandas.DataFrame.shift(1)` before any `rolling()` call — this shifts the window so game N never sees its own result
3. For the tournament prediction snapshot: use all regular-season games through Selection Sunday

### Two primary tables

```
game_log (data/processed/game_log.parquet)
─────────────────────────────────────────
One row per team per game.
Key columns: team | year | game_date | game_num | opponent | location
             result | margin | pts_scored | pts_allowed
             off_efg | def_efg | off_to | def_to | off_or | def_or | off_ftr | def_ftr
             opp_barthag | quad_result | conf_game_flag | is_tournament_game
             _source | _ingested_at

rolling_features (data/features/rolling_features.parquet)
──────────────────────────────────────────────────────────
One row per team per game date — all features rolled through PRIOR games.
Key columns: team | year | game_date | game_num | ...all rolling feature columns...
             _source | _is_derived | _pipeline_version
```

---

## 3. File Layout + Schema

### data/raw/ — immutable after ingestion

```
data/raw/
├── torvik/
│   ├── timemachine/
│   │   └── {year}_pretournament.parquet     # 45 cols, snapshot day after Selection Sunday
│   │   # 2008-2010: from team_ratings() — documented limitation
│   ├── game_results/
│   │   └── {year}_game_results.parquet      # 54 cols (super_sked), one row per game
│   ├── four_factors/
│   │   └── {year}_four_factors.parquet      # 36 cols, one row per team
│   ├── player_stats/
│   │   └── {year}_player_stats.parquet      # 67 cols, one row per player
│   └── quadrant_records/
│       └── {year}_quadrant.parquet          # Q1-Q4 records per team
└── sports_ref/
    ├── coaches_cache/
    │   └── {slug}.html                      # raw HTML, never re-fetched after first pull
    ├── coaches_index.parquet                # full SR coaches index table
    ├── tournament_seeds.parquet             # (year, team_norm, official_seed) from SR postseason brackets
    ├── tournament_results.parquet           # one row per NCAA tournament game (labels, seeds, round, scores)
    ├── tournament_results_cache/            # {year}.html — SR postseason pages (optional local cache)
    └── game_logs/
        └── {year}_{team_slug}_gamelog.parquet  # CBBpy box score per team per season
```

### data/processed/ — cleaned, joined, normalized

```
data/processed/
├── game_log.parquet                # master: all teams, all games, all years
│   # Rows: ~350 teams × ~35 games × 18 years ≈ 220,000
│   # Key: (team_norm, year, game_date) — team_norm = canonical name from team_name_map
│
├── coach_store.parquet             # coach features, cumulative through prior years
│   # Rows: ~350 teams × 18 years ≈ 6,300
│   # Key: (team_norm, year)
│
├── player_aggregates.parquet       # team-level player summaries
│   # Rows: ~350 teams × 18 years ≈ 6,300
│   # Key: (team_norm, year)
│
└── team_name_map.json              # name normalization (see Section 7)
```

### data/features/ — ML-ready

```
data/features/
├── rolling_features.parquet        # game-level rolling features
│   # Rows: ~220,000 (same as game_log)
│   # Key: (team_norm, year, game_date)
│
├── static_features.parquet         # season-level features
│   # Rows: ~6,300 (one per team per year)
│   # Key: (team_norm, year)
│
├── matchup_features.parquet        # final training-ready table
│   # Rows: ~1,130 (full SR bracket: 64 games × 2008–10 + 67 × 2011–25, excl. 2020)
│   # Key: (year, game_id, t1_team_norm, t2_team_norm)
│   # Columns: t1_* | t2_* | delta_* | result | round | metadata
│
└── schema_registry.json            # field catalogue (see Section 4)
```

### data/training/ — model inputs

```
data/training/
├── train_{year}.parquet            # train on all years except {year}
│   # Used for leave-one-tournament-out CV
└── feature_importance_log/
    └── {year}_importance.csv       # XGBoost feature importances per fold
```

---

## 4. Provenance + Schema Registry

Every feature column in every parquet file must have an entry in `data/features/schema_registry.json`.

### Registry entry format

```json
{
  "delta_adj_em": {
    "source": "barttorvik",
    "endpoint": "team_ratings() / time_machine()",
    "is_derived": true,
    "derivation": "t1_adj_em - t2_adj_em",
    "rolling_window": null,
    "added_version": "v1.0",
    "added_date": "2026-03-29",
    "description": "Matchup adjusted efficiency margin differential. Positive = t1 is more efficient.",
    "feature_group": "efficiency_matchup",
    "nullable": false,
    "expected_range": [-30, 30]
  }
}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Data origin: `"barttorvik"`, `"sports-reference"`, `"cbbpy"`, `"derived"` |
| `endpoint` | string | Specific URL or function that produced the raw data |
| `is_derived` | bool | True if computed from other columns; false if directly ingested |
| `derivation` | string\|null | Formula or description of computation if is_derived=true |
| `rolling_window` | int\|null | Games in rolling window; null if not a rolling feature |
| `added_version` | string | Pipeline version when field was added |
| `added_date` | string | ISO date when field was added |
| `description` | string | Human-readable definition |
| `feature_group` | string | Category (see groups below) |
| `nullable` | bool | Whether nulls are expected |
| `expected_range` | [min, max]\|null | Sanity check bounds for validation |

### Feature groups

```
efficiency_matchup     — adjoe/adjde/adj_em differentials
four_factor_matchup    — eFG%, TO%, OR%, FTR matchups
tempo_matchup          — pace and tempo features
shooting_matchup       — shot location and 3pt features
rebounding_matchup     — rebounding rate matchups
player_quality         — BPM, experience, height aggregates
coach                  — tournament experience features
resume                 — Q1 record, WAB, SOS
momentum_rolling       — rolling win rates, margin trends
streak                 — win/loss streaks
recency                — recency-weighted features
upset_risk             — variance and upset flags
metadata               — year, game_id, seed, is_bubble_year, etc.
```

---

## 5. Feature Specifications

### 5a. Static Season-Level Features
*Source: Torvik time_machine() snapshot. One row per team per year.*

| Column | Source | Derivation | Description |
|--------|--------|------------|-------------|
| `barthag` | barttorvik | raw | Win prob vs avg D1 team (0–1) |
| `adjoe` | barttorvik | raw | Adj offensive efficiency |
| `adjde` | barttorvik | raw | Adj defensive efficiency |
| `adj_em` | barttorvik | derived: adjoe-adjde | Efficiency margin |
| `adjt` | barttorvik | raw | Adjusted tempo |
| `wab` | barttorvik | raw | Wins above bubble |
| `ov_cur_sos` | barttorvik | raw | Overall current SOS |
| `ov_elite_sos` | barttorvik | raw | Overall elite SOS rank |
| `nc_elite_sos` | barttorvik | raw | Non-conf elite SOS rank |
| `off_efg` | barttorvik | raw | Offensive eFG% |
| `def_efg` | barttorvik | raw | Defensive eFG% allowed |
| `off_to` | barttorvik | raw | Offensive turnover rate |
| `def_to` | barttorvik | raw | Defensive forced turnover rate |
| `off_or` | barttorvik | raw | Offensive rebounding rate |
| `def_or` | barttorvik | raw | Defensive rebounding rate allowed |
| `off_ftr` | barttorvik | raw | Offensive free throw rate |
| `def_ftr` | barttorvik | raw | Defensive foul rate |
| `off_rim_fgpct` | barttorvik | raw | Offensive rim FG% |
| `def_rim_fgpct` | barttorvik | raw | Defensive rim FG% allowed |
| `off_three_fgpct` | barttorvik | raw | 3-point FG% |
| `def_three_fgpct` | barttorvik | raw | 3-point FG% allowed |
| `off_three_rate` | barttorvik | raw | 3PA / FGA (3pt reliance) |
| `q1_win_rate` | barttorvik | derived: Q1 W/(W+L) | Win rate vs Q1 opponents |
| `q1_loss_count` | barttorvik | derived: count | Losses to Q1 opponents |
| `seed` | ncaa bracket | raw | Tournament seed (1–16) |
| `conf` | barttorvik | raw | Conference abbreviation |
| `is_major_conf` | derived | derived: conf in power_6 list | 1 if power conference |
| `is_bubble_year` | metadata | raw | 1 if 2021 bubble tournament |

### 5b. Rolling Game-Level Features
*Source: game_log. Window ends BEFORE each game (shift(1) enforced).*

| Column | Window | Source | Description |
|--------|--------|--------|-------------|
| `roll3_win_rate` | 3 games | cbbpy | Win rate, last 3 games |
| `roll5_win_rate` | 5 games | cbbpy | Win rate, last 5 games |
| `roll10_win_rate` | 10 games | cbbpy | Win rate, last 10 games ★ |
| `roll5_avg_margin` | 5 games | cbbpy | Avg point differential, last 5 |
| `roll10_avg_margin` | 10 games | cbbpy | Avg point differential, last 10 ★ |
| `roll10_margin_trend` | 10 games | derived | Slope of margin over last 10 (linreg coef) |
| `roll10_pts_scored` | 10 games | cbbpy | Rolling avg points scored |
| `roll10_pts_allowed` | 10 games | cbbpy | Rolling avg points allowed |
| `roll10_off_efg` | 10 games | cbbpy | Rolling offensive eFG% |
| `roll10_def_efg` | 10 games | cbbpy | Rolling defensive eFG% allowed |
| `roll10_tov_rate` | 10 games | cbbpy | Rolling turnover rate |
| `roll10_opp_barthag` | 10 games | derived | Avg opponent barthag (schedule strength of recent slate) |
| `roll10_q1_win_rate` | last 10 Q1 games | derived | Win rate in Q1 games only ★ |
| `sos_adj_roll10_win_rate` | 10 games | derived | roll10_win_rate × roll10_opp_barthag |
| `recency_wtd_win_rate` | full season | derived | Exp decay weighted win rate (λ=0.02) ★ |
| `late_season_win_rate` | last 40% of season | derived | Win rate when season_pct_elapsed > 0.60 ★ |
| `early_season_win_rate` | first 8 games | derived | Win rate in first 8 games |
| `early_vs_late_win_delta` | — | derived | late_season_win_rate − early_season_win_rate ★ |
| `season_pct_elapsed` | — | derived | game_num / total_games |
| `is_early_season` | — | derived | 1 if game_num ≤ 8 |
| `days_since_last_game` | — | derived | Calendar days since previous game |
| `conf_tourn_rounds_won` | — | sports-ref | Rounds won in conference tournament (0=R1 exit) |
| `top_scorer_roll5_ppg` | 5 games | cbbpy | Rolling PPG of team's highest-usage player ★ |
| `top_scorer_ppg_trend` | 10 games | derived | Slope of star player PPG over last 10 |

### 5c. Streak Features
*Computed from game_log. Reset triggers noted.*

| Column | Reset Trigger | Description |
|--------|--------------|-------------|
| `current_win_streak` | Any loss resets to 0 | Consecutive wins entering this game |
| `current_loss_streak` | Any win resets to 0 | Consecutive losses entering this game |
| `max_win_streak_season` | — | Longest win streak achieved this season so far |
| `games_since_last_loss` | — | Days since last loss (alternative to streak count) |

### 5d. Player Aggregate Features
*Aggregated from Torvik player_stats() to team level. Min-weighted.*

| Column | Weight | Description |
|--------|--------|-------------|
| `team_bpm_wtd` | min_pct | Minutes-weighted avg BPM across rotation ★ |
| `team_obpm_wtd` | min_pct | Minutes-weighted avg offensive BPM |
| `team_dbpm_wtd` | min_pct | Minutes-weighted avg defensive BPM |
| `team_height_wtd` | min_pct | Minutes-weighted avg height (inches) ★ |
| `team_height_max` | — | Height of tallest rotation player (rim protection) |
| `team_experience_idx` | min_pct | Minutes-weighted avg class year (Fr=1…Sr=4) ★ |
| `star_usg_share` | — | Usage rate of top-1 player / team total |
| `top2_usg_share` | — | Combined usage of top-2 players (concentration risk) |
| `depth_score` | — | BPM of 7th-best rotation player (bench depth) |
| `roster_sr_pct` | min_pct | % of minutes from Sr/Jr players |

### 5e. Coach Features
*One row per team per year. Cumulative through prior years only — never includes current year.*

| Column | Description |
|--------|-------------|
| `coach_name` | Head coach full name |
| `coach_years_at_school` | Years as HC at this school entering this season |
| `coach_tourn_appearances` | Career tournament appearances (prior years only) ★ |
| `coach_tourn_games` | Career tournament games played |
| `coach_tourn_win_rate` | Career tournament W/(W+L) ★ |
| `coach_final_four_count` | Career Final Four appearances ★ |
| `coach_deep_run_rate` | % of appearances reaching Sweet 16+ |
| `coach_is_first_tourn` | 1 if this is coach's first tournament |
| `_source` | `"sports-reference.com/cbb/coaches/"` |

### 5f. Matchup Differential Features
*In matchup_features.parquet. Labels come from ``tournament_results.parquet`` (SR bracket). Before the balance swap, t1 is the better (lower) seed; a random 50% of rows swap t1/t2 for label balance.*
*Every t1_*/t2_* static/rolling feature also has a delta_* counterpart = t1 − t2.*

Key delta features (★ = highest expected importance):

```
delta_adj_em              ★  t1_adj_em - t2_adj_em
delta_barthag             ★  t1_barthag - t2_barthag
delta_seed                ★  t1_seed - t2_seed (negative = t1 is better seed)
delta_off_efg_vs_def_efg  ★  t1_off_efg - t2_def_efg  (shooting matchup edge for t1)
delta_roll10_win_rate     ★  t1_roll10_win_rate - t2_roll10_win_rate
delta_off_to_vs_def_to    ★  t1_off_to - t2_def_to  (turnover matchup)
delta_off_or_vs_def_or    ★  t1_off_or - (1 - t2_def_or)  (rebounding matchup)
delta_team_bpm_wtd        ★  t1_team_bpm_wtd - t2_team_bpm_wtd
delta_team_experience_idx ★  t1_team_experience_idx - t2_team_experience_idx
delta_coach_tourn_win_rate   t1_coach_tourn_win_rate - t2_coach_tourn_win_rate
delta_late_season_win_rate   t1_late_season_win_rate - t2_late_season_win_rate
delta_adjt                   t1_adjt - t2_adjt  (tempo preference)
```

Derived matchup-level features (not a simple t1−t2 delta):

```
projected_tempo           (t1_adjt + t2_adjt) / 2
pace_variance_flag        1 if |delta_adjt| > 8 (large tempo mismatch)
three_pt_reliance_flag    1 if t1_off_three_rate > 0.42 OR t2_off_three_rate > 0.42
low_tempo_coin_flip       1 if projected_tempo < 62 AND |delta_seed| <= 4
midmajor_matchup          1 if exactly one team is NOT is_major_conf
```

### 5g. Variance / Upset Risk Flags

| Column | Formula | Description |
|--------|---------|-------------|
| `upset_risk_score` | derived composite | Weighted combination of: low_tempo_coin_flip × 2 + three_pt_reliance_flag + midmajor_matchup + (1 - |delta_barthag| × 5) |
| `torvik_predicted_winner` | t1_wp > 0.50 | Torvik's own pre-game win probability prediction |
| `seed_upset_flag` | t2_seed < t1_seed AND result=t2 | Ground truth label for upsets (for EDA) |

---

## 6. Coach Ingestion Design

### Strategy
- **Primary source:** Sports-Reference CBB coaches index
- **Method:** `pandas.read_html()` for index table + BeautifulSoup for per-coach detail pages
- **Rate limit:** 20 req/min (Sports-Reference ToS) → `time.sleep(3)` between coach page requests
- **Cache:** Raw HTML saved to `data/raw/sports_ref/coaches_cache/{slug}.html` — never re-fetched

### Step-by-step

```python
# Step 1: Pull coaches index
url = "https://www.sports-reference.com/cbb/coaches/"
df_index = pandas.read_html(url)[0]
# Columns: Name, School, Year From, Year To, G, W, L, W-L%, NCAA, NCAA W, NCAA L, NCAA W-L%

# Step 2: For each coach with NCAA tournament games > 0:
coach_url = f"https://www.sports-reference.com/cbb/coaches/{slug}.html"
# Cache HTML → parse season-by-season table
# Extract: year, school, conf_wins, conf_losses, ovr_wins, ovr_losses,
#          ncaa_rounds (0=no bid, 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=champ)

# Step 3: Build cumulative features (strictly prior years)
# For year Y: use sum/rate of all coach data through Y-1

# Step 4: Join to team_name_map using school name
# Flag any unmatched schools for manual review
```

### Edge cases to handle

```
- Coach changes mid-season: use the coach at season start for annual features
  Flag: if SR shows coach change during season, use start-of-season coach,
  add column coach_changed_midseason=1
- First-year coaches: coach_tourn_appearances=0, coach_tourn_win_rate=NaN → impute 0
- Coaches at multiple schools: cumulative stats include all prior schools
- Name variations: "Bill Self" vs "Bill Self Jr." → handled by coach_name_map.json
```

---

## 7. Name Normalization Crosswalks

### team_name_map.json format

```json
{
  "UConn": {
    "torvik": "Connecticut",
    "sports_ref": "Connecticut",
    "cbbpy": "connecticut",
    "canonical": "UConn"
  },
  "UNC": {
    "torvik": "North Carolina",
    "sports_ref": "North Carolina",
    "cbbpy": "north-carolina",
    "canonical": "UNC"
  }
}
```

The `canonical` name is used as the join key in all processed/ and features/ tables.

### Priority teams to cover first (all frequent tournament participants)

```
Alabama, Arizona, Arizona State, Arkansas, Auburn, Baylor, BYU, Cincinnati,
Clemson, Colorado, Connecticut (UConn), Creighton, Davidson, Duke, Florida,
Florida State, Gonzaga, Houston, Illinois, Indiana, Iowa, Iowa State, Kansas,
Kansas State, Kentucky, Louisville, LSU, Marquette, Maryland, Memphis,
Michigan, Michigan State, Minnesota, Mississippi State, Missouri, Murray State,
North Carolina (UNC), Notre Dame, Ohio State, Oklahoma, Oklahoma State, Oregon,
Penn State, Purdue, Saint Mary's, San Diego State, Seton Hall, South Carolina,
St. John's, Tennessee, Texas, Texas A&M, Texas Tech, UCLA, UNLV, USC,
Utah State, Vanderbilt, Villanova, Virginia, Virginia Tech, Wake Forest,
Washington, West Virginia, Wichita State, Wisconsin, Xavier
```

### coach_name_map.json format

```json
{
  "John Calipari": {
    "sports_ref_slug": "calipaj01",
    "canonical": "John Calipari",
    "aliases": ["Cal", "Calipari"]
  }
}
```

---

## 8. Extensibility Contract

Adding a new data source (betting lines, injury reports, etc.) requires exactly three files touched:

### Step 1 — Write ingestion script
```
src/data/ingest_{source}.py
- Saves raw data to data/raw/{source}/
- Logs: source name, date pulled, row count, column count
- Implements caching (never re-download what's already on disk)
- Uses team_name_map.json for any team name normalization
```

### Step 2 — Register all new columns
```
data/features/schema_registry.json
- Add one entry per new column
- Required fields: source, endpoint, is_derived, derivation, rolling_window,
                   added_version, added_date, description, feature_group,
                   nullable, expected_range
```

### Step 3 — Add join logic
```
# Season-level data → src/features/build_static.py
df_static = df_static.merge(df_new_source, on=['team_norm', 'year'], how='left')

# Game-level rolling data → src/features/build_rolling.py
df_rolling = df_rolling.merge(df_new_source, on=['team_norm', 'year', 'game_date'], how='left')
```

`validate.py` automatically picks up new columns and checks them against the registry.

### What you get for free after these 3 steps:
- New features automatically flow into matchup_features.parquet as t1_*/t2_*/delta_* columns
- validate.py checks nulls, ranges, and registry completeness
- schema_registry.json provides full audit trail
- No changes needed to model code — XGBoost sees new columns automatically

---

## 9. Validation Checks

`src/features/validate.py` runs these checks after every pipeline execution:

```python
checks = [
    # Schema completeness
    "all parquet columns exist in schema_registry.json",
    "no parquet columns missing _source attribution",

    # Leakage detection
    "rolling_features: no game uses its own result in any rolling window",
    "matchup_features: no features computed after game_date",
    "tournament training rows: no test-year data in any training fold",

    # Data quality
    "null rate < 5% for all non-nullable columns",
    "all expected_range bounds satisfied (flag violations, don't fail)",
    "team_norm in all tables resolves via team_name_map.json",

    # Join integrity
    "game_log row count matches sum of CBBpy game counts per year",
    "all tournament games in matchup_features have both t1 and t2 features",
    "no duplicate (team_norm, year, game_date) rows in rolling_features",

    # Special cases
    "year 2020 has zero rows in all training tables",
    "year 2021 has is_bubble_year=1 for all rows",
    "years 2008-2010 have timemachine_available=0 flag",
]
```

All checks log to stdout. Leakage checks cause pipeline to abort. Others log warnings.

---

## 10. Performance Targets

| Stage | Input Size | Target Runtime | Bottleneck | Mitigation |
|-------|-----------|----------------|------------|------------|
| Torvik ingestion (cold) | 18 years × bulk files | < 15 min | Network | pybart disk cache |
| Torvik ingestion (warm) | From cache | < 1 min | Disk I/O | Already fast |
| Coach ingestion (cold) | ~500 coach pages | ~30 min | SR rate limit (20 req/min) | 3s sleep + HTML cache |
| Coach ingestion (warm) | From HTML cache | < 2 min | Parsing | Already fast |
| CBBpy game log (cold) | ~90,000 games | 60–90 min | ESPN rate limit | Parallelize by year; cache parquet |
| CBBpy game log (warm) | From cache | < 5 min | Disk I/O | Already fast |
| build_game_log | 220,000 rows | < 1 min | Pandas merge | Vectorized ops only |
| build_rolling | 220,000 rows | < 2 min | GroupBy + rolling | Use pandas rolling() |
| build_static | 6,300 rows | < 30 sec | Multiple merges | All in-memory |
| build_matchups | 1,200 rows | < 10 sec | Tiny | N/A |
| validate | All tables | < 1 min | File reads | Parquet columnar reads |
| **Full pipeline (cold)** | — | **~2 hours** | CBBpy | Run overnight once |
| **Full pipeline (warm)** | — | **< 10 min** | — | Cache everything |

### Parallelization note for CBBpy
```python
# src/data/ingest_gamelogs.py — parallelize by year
from concurrent.futures import ThreadPoolExecutor

years = list(range(2008, 2026))
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(ingest_year, years)
# max_workers=4 stays within ESPN rate limits while cutting runtime ~4x
```

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-29 | v1.0 | Initial pipeline design. All decisions locked. |
