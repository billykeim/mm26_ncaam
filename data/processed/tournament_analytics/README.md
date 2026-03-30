# Tournament analytics outputs

## historical_reference.parquet / .csv
One row per season (excludes 2020). **Upsets** use efficiency favorite
(`t1_net` vs `t2_net`). **Rounds** are six chronological bins within the season.

## historical_reference_summary.parquet
Across-year mean / min / max / std for numeric columns.

## seed_pair_win_rates.parquet
Matchups where **both** teams show seeds 1–16 in the `matchup` string.
`seed_low` is the better (lower-number) seed. `better_seed_wins` counts games
where that seed's team won.

_Source: `data/raw/torvik/tournament_training_set.parquet`._
