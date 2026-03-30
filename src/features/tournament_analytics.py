"""Tournament historical analytics + seed-pair lookup (Torvik training set)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import DATA_PROCESSED, PROJECT_ROOT

TOURNEY_PATH = PROJECT_ROOT / "data" / "raw" / "torvik" / "tournament_training_set.parquet"
OUT_DIR = DATA_PROCESSED / "tournament_analytics"

ROUND_LABELS = ["R64", "R32", "S16", "E8", "F4", "Championship"]
POWER_6 = {"ACC", "B10", "B12", "P12", "SEC", "BE"}


def parse_team_seeds(matchup: str) -> tuple[int | None, int | None]:
    """Return (team1_seed, team2_seed) if both in 1–16; else (None, None)."""
    m = re.match(r"^\s*(\d{1,2})\s+.+?\s+(?:vs\.|at)\s*(\d{1,2})\s+", str(matchup))
    if not m:
        return None, None
    a, b = int(m.group(1)), int(m.group(2))
    if 1 <= a <= 16 and 1 <= b <= 16:
        return a, b
    return None, None


def load_tournament() -> pd.DataFrame:
    df = pd.read_parquet(TOURNEY_PATH)
    df = df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["ord_date"] = pd.to_numeric(df["ord_date"], errors="coerce")
    for c in ("t1pts", "t2pts", "t1adjo", "t1adjd", "t2adjo", "t2adjd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["overtimes"] = pd.to_numeric(df.get("overtimes", 0), errors="coerce").fillna(0)
    df["t1_net"] = df["t1adjo"] - df["t1adjd"]
    df["t2_net"] = df["t2adjo"] - df["t2adjd"]
    df["margin"] = (df["t1pts"] - df["t2pts"]).abs()
    df["total_pts"] = df["t1pts"] + df["t2pts"]
    df["winner_is_t1"] = df["winner"] == df["team1"]
    fav1 = df["t1_net"] >= df["t2_net"]
    df["favorite_won"] = (df["winner_is_t1"] & fav1) | (~df["winner_is_t1"] & ~fav1)
    df["upset"] = ~df["favorite_won"]
    df["is_bubble_year"] = (df["season"] == 2021).astype(int)

    s1 = pd.to_numeric(
        df["matchup"].map(lambda m: parse_team_seeds(m)[0]), errors="coerce"
    )
    s2 = pd.to_numeric(
        df["matchup"].map(lambda m: parse_team_seeds(m)[1]), errors="coerce"
    )
    df["team1_seed"] = s1
    df["team2_seed"] = s2
    win_s = np.where(df["winner_is_t1"], s1, s2)
    df["winner_seed"] = win_s
    min_s = np.minimum(s1.to_numpy(dtype=float), s2.to_numpy(dtype=float))
    df["better_seed_won"] = (win_s == min_s) & s1.notna().to_numpy() & s2.notna().to_numpy()

    return df


def assign_round(df: pd.DataFrame) -> pd.DataFrame:
    """Six chronological buckets per season (proxy for tournament round)."""
    parts: list[pd.DataFrame] = []
    for season, g in df.groupby("season", sort=True):
        g = g.sort_values("ord_date")
        n = len(g)
        if n == 0:
            continue
        edges = [int(x) for x in np.linspace(0, n, 7)]
        rnd = np.zeros(n, dtype=int)
        for i in range(6):
            rnd[edges[i] : edges[i + 1]] = i + 1
        g = g.copy()
        g["round_num"] = rnd
        g["round"] = g["round_num"].map(lambda i: ROUND_LABELS[i - 1])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_seed_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wins by (seed_low, seed_high, round)."""
    sub = df.dropna(subset=["team1_seed", "team2_seed"]).copy()
    sub["seed_low"] = sub[["team1_seed", "team2_seed"]].min(axis=1).astype(int)
    sub["seed_high"] = sub[["team1_seed", "team2_seed"]].max(axis=1).astype(int)
    sub["better_seed_won"] = sub["better_seed_won"].fillna(False).astype(bool)

    rows: list[dict] = []
    for (sl, sh, rnd), g in sub.groupby(["seed_low", "seed_high", "round"]):
        n = len(g)
        wins = int(g["better_seed_won"].sum())
        last5 = g[g["season"].isin([2021, 2022, 2023, 2024, 2025])]
        n5 = len(last5)
        w5 = int(last5["better_seed_won"].sum()) if n5 else 0
        rows.append(
            {
                "seed_low": int(sl),
                "seed_high": int(sh),
                "round": rnd,
                "games_played": n,
                "better_seed_wins": wins,
                "historical_win_rate": wins / n if n else np.nan,
                "last_5yr_win_rate": w5 / n5 if n5 else np.nan,
                "upset_occurred_pct": (n - wins) / n if n else np.nan,
                "_source": "tournament_training_set.parquet",
                "_is_derived": True,
            }
        )
    return pd.DataFrame(rows)


def build_historical_reference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-season summary metrics."""
    years = sorted(int(x) for x in df["season"].dropna().unique() if int(x) != 2020)
    rows: list[dict] = []
    for yr in years:
        g = df[df["season"] == yr]
        by_r = g.groupby("round", dropna=False).agg(
            upsets=("upset", "sum"),
            games=("upset", "count"),
            upset_rate=("upset", "mean"),
            avg_margin=("margin", "mean"),
            avg_total=("total_pts", "mean"),
            ot_rate=("overtimes", lambda x: (x > 0).mean()),
            close_rate=("margin", lambda x: (x <= 5).mean()),
            blowout_rate=("margin", lambda x: (x >= 20).mean()),
            fav_win=("favorite_won", "mean"),
        )
        r64 = g[g["round"] == "R64"]
        seed_parse = g.dropna(subset=["team1_seed"])
        upset_seed = seed_parse.loc[seed_parse["upset"], :]
        max_diff = (
            (upset_seed["team1_seed"] - upset_seed["team2_seed"]).abs().max()
            if len(upset_seed)
            else np.nan
        )
        # first upset (chronological)
        gs = g.sort_values("ord_date")
        fu_rows = gs[gs["upset"]]
        first_upset_round = int(fu_rows.iloc[0]["round_num"]) if len(fu_rows) else np.nan
        # power6 vs mid upset rate
        t1p = g["t1_conf"].isin(POWER_6) if "t1_conf" in g.columns else pd.Series(False, index=g.index)
        t2p = g["t2_conf"].isin(POWER_6) if "t2_conf" in g.columns else pd.Series(False, index=g.index)
        mm = t1p ^ t2p
        p6_up = float(g.loc[mm, "upset"].mean()) if mm.any() else np.nan
        w_team = np.where(g["winner"] == g["team1"], g["t1_conf"], g["t2_conf"])
        conf_wins = pd.Series(w_team).value_counts().to_dict()
        rows.append(
            {
                "season": yr,
                "is_bubble_year": int(yr == 2021),
                "games": len(g),
                "total_upsets": int(g["upset"].sum()),
                "upset_rate": float(g["upset"].mean()),
                "biggest_upset_seed_diff_parseable": float(max_diff) if pd.notna(max_diff) else np.nan,
                "first_upset_round_num": first_upset_round,
                "r64_chalk_rate": float(r64["favorite_won"].mean()) if len(r64) else np.nan,
                "power6_vs_midmajor_upset_rate": p6_up,
                "avg_margin": float(g["margin"].mean()),
                "avg_total": float(g["total_pts"].mean()),
                "ot_rate": float((g["overtimes"] > 0).mean()),
                "close_game_rate": float((g["margin"] <= 5).mean()),
                "blowout_rate": float((g["margin"] >= 20).mean()),
                "conf_wins_json": json.dumps({str(k): int(v) for k, v in conf_wins.items()}),
                "upsets_by_round_json": by_r["upsets"].to_json(),
                "upset_rate_by_round_json": by_r["upset_rate"].to_json(),
                "avg_margin_by_round_json": by_r["avg_margin"].to_json(),
                "entropy_by_round_json": json.dumps(
                    {
                        str(rnd): (
                            -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                            if 0 < (p := float(grp["favorite_won"].mean())) < 1
                            else 0.0
                        )
                        for rnd, grp in g.groupby("round")
                    }
                ),
            }
        )
    per_year = pd.DataFrame(rows)
    num = per_year.select_dtypes(include=[np.number]).drop(columns=["is_bubble_year"], errors="ignore")
    summary = num.agg(["mean", "min", "max", "std"]).reset_index().rename(columns={"index": "stat"})
    return per_year, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_tournament()
    df = assign_round(df)
    per_year, summary = build_historical_reference(df)
    per_year.to_parquet(OUT_DIR / "historical_reference.parquet", index=False)
    per_year.to_csv(OUT_DIR / "historical_reference.csv", index=False)
    summary.to_parquet(OUT_DIR / "historical_reference_summary.parquet", index=False)

    seed_tbl = build_seed_pair_table(df)
    seed_tbl.to_parquet(OUT_DIR / "seed_pair_win_rates.parquet", index=False)

    readme = """# Tournament analytics outputs

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
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"[tournament_analytics] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
