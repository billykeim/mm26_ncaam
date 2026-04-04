"""Aggregate CBBpy player box scores to one row per team per game."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import cbbpy

from src.utils.constants import (
    DATA_PROCESSED,
    GAME_LOGS_RAW,
    SELECTION_SUNDAY_DATES,
    TRAINING_YEARS,
)
from src.utils.name_normalize import load_team_name_map, school_to_canonical

SUM_COLS = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "oreb",
    "dreb",
    "fgm",
    "fga",
    "ftm",
    "fta",
]


def load_espn_team_location_lookup() -> dict[tuple[int, str], str]:
    """Map (season, ESPN boxscore ``team`` string) → short ``location`` label."""
    p = Path(cbbpy.__file__).resolve().parent / "utils" / "mens_team_map.csv"
    tm = pd.read_csv(p)
    out: dict[tuple[int, str], str] = {}
    for _, r in tm.iterrows():
        se = int(r["season"])
        full = str(r["team"]).strip()
        loc = str(r["location"]).strip()
        out[(se, full)] = loc
    return out


def _read_year_raw(year: int, lookup: dict[tuple[int, str], str]) -> pd.DataFrame:
    """Load all team parquets for one season with row order preserved."""
    paths = sorted(GAME_LOGS_RAW.glob(f"{year}_*_gamelog.parquet"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        df["year"] = year
        df["_row_idx"] = np.arange(len(df), dtype=np.int64)
        df["team"] = df["team"].astype(str)
        df["location"] = [lookup.get((year, str(t))) for t in df["team"].values]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _add_team_norm(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Resolve ESPN names to canonical ``team_norm``."""
    if raw.empty:
        return raw
    out = raw.copy()

    def norm_row(loc: object, team: str) -> str:
        if pd.notna(loc) and str(loc).strip():
            return school_to_canonical(str(loc).strip(), mapping)
        return school_to_canonical(str(team), mapping)

    out["team_norm"] = [
        norm_row(loc, t) for loc, t in zip(out["location"].values, out["team"].values)
    ]
    return out


def aggregate_team_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Player rows → one row per (year, game_id, team_norm)."""
    if raw.empty:
        return pd.DataFrame()
    for c in SUM_COLS:
        if c not in raw.columns:
            raw[c] = 0
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
    raw["three_fgm"] = pd.to_numeric(raw.get("3pm", 0), errors="coerce").fillna(0)
    raw["three_fga"] = pd.to_numeric(raw.get("3pa", 0), errors="coerce").fillna(0)
    raw["tov"] = pd.to_numeric(raw.get("to", 0), errors="coerce").fillna(0)

    agg_spec = {c: "sum" for c in SUM_COLS + ["three_fgm", "three_fga", "tov"]}
    agg_spec["_row_idx"] = "min"
    g = (
        raw.groupby(["year", "game_id", "team_norm", "team"], as_index=False)
        .agg(agg_spec)
        .rename(columns={"_row_idx": "_sched_order"})
    )
    g["game_id"] = g["game_id"].astype(str)
    fga = g["fga"].replace(0, np.nan)
    g["efg"] = (g["fgm"] + 0.5 * g["three_fgm"]) / fga
    g["ft_rate"] = g["fta"] / fga
    poss = fga + 0.44 * g["fta"]
    g["tov_rate"] = g["tov"] / poss.replace(0, np.nan)
    return g


def attach_opponent(team_games: pd.DataFrame) -> pd.DataFrame:
    """Self-join on game_id so each row includes opponent totals."""
    if team_games.empty:
        return team_games
    tg = team_games.rename(columns={"pts": "pts_scored"}).copy()
    opp_cols = [
        "year",
        "game_id",
        "team_norm",
        "pts_scored",
        "reb",
        "ast",
        "stl",
        "blk",
        "oreb",
        "dreb",
        "fgm",
        "fga",
        "ftm",
        "fta",
        "three_fgm",
        "three_fga",
        "tov",
        "efg",
        "ft_rate",
        "tov_rate",
    ]
    opp_cols = [c for c in opp_cols if c in tg.columns]
    opp = tg[opp_cols].rename(
        columns={
            "team_norm": "opp_team_norm",
            "pts_scored": "pts_allowed",
            "reb": "opp_reb",
            "ast": "opp_ast",
            "stl": "opp_stl",
            "blk": "opp_blk",
            "oreb": "opp_oreb",
            "dreb": "opp_dreb",
            "fgm": "opp_fgm",
            "fga": "opp_fga",
            "ftm": "opp_ftm",
            "fta": "opp_fta",
            "three_fgm": "opp_three_fgm",
            "three_fga": "opp_three_fga",
            "tov": "opp_tov",
            "efg": "opp_efg",
            "ft_rate": "opp_ft_rate",
            "tov_rate": "opp_tov_rate",
        }
    )
    merged = tg.merge(opp, on=["year", "game_id"], how="inner")
    merged = merged[merged["team_norm"] != merged["opp_team_norm"]].copy()
    merged["margin"] = merged["pts_scored"] - merged["pts_allowed"]
    merged["result"] = (merged["margin"] > 0).astype(np.int8)
    return merged


def add_season_fields(df: pd.DataFrame) -> pd.DataFrame:
    """game_num, synthetic game_date, season_pct_elapsed, flags."""
    if df.empty:
        return df
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out = out.sort_values(["team_norm", "year", "_sched_order"])
    out["game_num"] = out.groupby(["team_norm", "year"]).cumcount() + 1
    max_g = out.groupby(["team_norm", "year"])["game_num"].transform("max")
    out["season_pct_elapsed"] = out["game_num"] / max_g.replace(0, np.nan)
    out["is_early_season"] = (out["game_num"] <= 8).astype(np.int8)
    out["conf_game_flag"] = np.int8(0)
    out["is_tournament_game"] = np.int8(0)
    # CBBpy ``year`` Y is the season ending in March Y. Map each team's game_num linearly from
    # November (Y−1) up to (but not including) that season's Selection Sunday so every row is
    # eligible for pre-tournament snapshots and ordering matches schedule sequence.
    y_int = pd.to_numeric(out["year"], errors="coerce").astype(int)
    nov = pd.to_datetime((y_int - 1).astype(str) + "-11-01", utc=False)
    cut = pd.to_datetime(y_int.map(lambda y: SELECTION_SUNDAY_DATES.get(int(y))), utc=False)
    has_cut = cut.notna()
    denom = max_g.clip(lower=1)
    frac = (out["game_num"] - 1) / denom
    out["game_date"] = pd.NaT
    out.loc[has_cut, "game_date"] = (
        nov[has_cut] + (cut[has_cut] - nov[has_cut]) * frac[has_cut]
    )
    fb = ~has_cut
    if fb.any():
        base = pd.to_datetime((y_int[fb] - 1).astype(str) + "-11-01", utc=False)
        out.loc[fb, "game_date"] = base + pd.to_timedelta(
            (out.loc[fb, "game_num"] - 1) * 4, unit="D"
        )
    return out.drop(columns=["_sched_order"], errors="ignore")


def build_game_log(years: list[int] | None = None) -> pd.DataFrame:
    """Load raw logs, aggregate, join opponent, normalize names."""
    years = years or TRAINING_YEARS
    lookup = load_espn_team_location_lookup()
    mapping = load_team_name_map()
    parts: list[pd.DataFrame] = []
    for yr in years:
        raw = _read_year_raw(yr, lookup)
        if raw.empty:
            print(f"[build_game_log] year {yr}: no raw rows")
            continue
        raw = _add_team_norm(raw, mapping)
        tg = aggregate_team_games(raw)
        merged = attach_opponent(tg)
        merged = add_season_fields(merged)
        parts.append(merged)
        print(f"[build_game_log] year {yr}: team-games={len(merged)}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    df = build_game_log()
    if df.empty:
        print("[build_game_log] empty output")
        return
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        n_teams = sub["team_norm"].nunique()
        print(f"[build_game_log] year {yr}: unique_teams={n_teams}")
        if yr in (2012, 2013) and n_teams < 340:
            print(
                f"[build_game_log] FLAG: year {yr} unique_teams={n_teams} < 340 "
                "(possible coverage gap)"
            )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "game_log.parquet"
    df.to_parquet(out, index=False)
    rpc = df.groupby("year").size()
    print(f"[build_game_log] wrote {out} total_rows={len(df):,} cols={df.shape[1]}")
    print(f"[build_game_log] rows_per_year:\n{rpc.to_string()}")


if __name__ == "__main__":
    main()
