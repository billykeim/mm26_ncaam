"""Ingest Bart Torvik / T-Rank data via pybart (cache_dir under project root)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path for `python src/data/ingest_torvik.py`
_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from pybart import Torvik
from pybart.madness import _find_team_col as _mad_find_team_col

from src.utils.constants import (
    PROJECT_ROOT,
    TIME_MACHINE_DATES,
    TORVIK_CACHE_DIR,
    TORVIK_FOUR_FACTORS,
    TORVIK_GAME_RESULTS,
    TORVIK_PLAYER_STATS,
    TORVIK_RAW,
    TORVIK_TIMEMACHINE,
    TRAINING_YEARS,
)


def _log(step: str, rows: int) -> None:
    """Print ingestion step and row count."""
    print(f"[torvik] {step}: {rows} rows")


def _filter_ncaa_tournament_games(sked: pd.DataFrame) -> pd.DataFrame:
    """NCAA tournament rows in super_sked use ``conf == 3`` (modern schema)."""
    if "conf" not in sked.columns:
        return pd.DataFrame()
    c = pd.to_numeric(sked["conf"], errors="coerce")
    return sked.loc[c == 3].copy()


def tournament_training_set_fixed(torvik: Torvik, years: list[int]) -> pd.DataFrame:
    """
    Labeled NCAA tournament games with regular-season team features.

    Replaces pybart.madness.tournament_training_set() which expects ``NCAA`` in
    string columns; current super_sked marks NCAA games with ``conf == 3``.
    """
    all_games: list[pd.DataFrame] = []
    for yr in years:
        if yr == 2020:
            continue
        try:
            sked = torvik.schedule_stats(yr, fmt="csv")
            features = team_features_str_keys(torvik, yr)
        except Exception as e:
            print(f"[torvik] skip year {yr} tournament set: {e}")
            continue

        team_col = _find_team_col(features)
        if not team_col:
            continue

        tourney = _filter_ncaa_tournament_games(sked)
        if tourney.empty:
            continue

        t1_col = "team1" if "team1" in tourney.columns else None
        t2_col = "team2" if "team2" in tourney.columns else None
        if not t1_col or not t2_col:
            continue

        tourney = tourney.copy()
        tourney[t1_col] = tourney[t1_col].astype(str).str.strip()
        tourney[t2_col] = tourney[t2_col].astype(str).str.strip()
        features = features.copy()
        features[team_col] = features[team_col].astype(str).str.strip()

        t1_feats = features.add_prefix("t1_")
        t2_feats = features.add_prefix("t2_")

        merged = tourney.merge(
            t1_feats, left_on=t1_col, right_on=f"t1_{team_col}", how="left"
        ).merge(
            t2_feats, left_on=t2_col, right_on=f"t2_{team_col}", how="left"
        )
        merged["season"] = yr
        all_games.append(merged)

    if not all_games:
        return pd.DataFrame()
    out = pd.concat(all_games, ignore_index=True)
    return _coerce_for_parquet(out)


def _find_team_col(df: pd.DataFrame) -> str | None:
    return _mad_find_team_col(df)


def team_features_str_keys(torvik: Torvik, year: int) -> pd.DataFrame:
    """
    Same as ``pybart.madness.team_features`` but forces string team keys so merges
    do not mix float team ids with string names (historical seasons).
    """
    df = torvik.team_slice(year, game_type="R")
    team_col = _find_team_col(df)
    if not team_col:
        return df
    df = df.copy()
    df[team_col] = df[team_col].astype(str).str.strip()

    ff = torvik.four_factors(year)
    ff_team = _find_team_col(ff)
    if ff_team:
        ff = ff.copy()
        ff[ff_team] = ff[ff_team].astype(str).str.strip()
        df = df.merge(
            ff,
            left_on=team_col,
            right_on=ff_team,
            how="left",
            suffixes=("", "_ff"),
        )

    sh = torvik.team_shooting_splits(year)
    sh_team = _find_team_col(sh)
    if sh_team:
        sh = sh.copy()
        sh[sh_team] = sh[sh_team].astype(str).str.strip()
        df = df.merge(
            sh,
            left_on=team_col,
            right_on=sh_team,
            how="left",
            suffixes=("", "_shoot"),
        )
    return df


def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid pyarrow failures on object columns with mixed scalars."""
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str)
    return out


def _ensure_dirs() -> None:
    for p in (
        TORVIK_RAW,
        TORVIK_TIMEMACHINE,
        TORVIK_GAME_RESULTS,
        TORVIK_FOUR_FACTORS,
        TORVIK_PLAYER_STATS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def run_ingestion() -> int:
    """
    Pull Torvik bulk files and time machine snapshots.

    Returns
    -------
    int
        Total rows written across all parquet outputs (approximate sum).
    """
    _ensure_dirs()
    cache_path = PROJECT_ROOT / TORVIK_CACHE_DIR
    torvik = Torvik(cache_dir=str(cache_path))

    years_tor = [y for y in range(2008, 2026) if y != 2020]
    total_rows = 0

    # 1) Tournament training set (NCAA games × RS features)
    df_train = tournament_training_set_fixed(torvik, years_tor)
    out_train = TORVIK_RAW / "tournament_training_set.parquet"
    df_train.to_parquet(out_train, index=False)
    total_rows += len(df_train)
    _log("tournament_training_set", len(df_train))

    # 2) Per-year: game_results (super_sked), four_factors, player_stats
    for yr in years_tor:
        try:
            gr = torvik.schedule_stats(yr, fmt="csv")
            path_gr = TORVIK_GAME_RESULTS / f"{yr}_game_results.parquet"
            gr.to_parquet(path_gr, index=False)
            total_rows += len(gr)
            _log(f"game_results {yr}", len(gr))
        except Exception as e:
            print(f"[torvik] game_results {yr} failed: {e}")

        try:
            ff = torvik.four_factors(yr)
            path_ff = TORVIK_FOUR_FACTORS / f"{yr}_four_factors.parquet"
            ff.to_parquet(path_ff, index=False)
            total_rows += len(ff)
            _log(f"four_factors {yr}", len(ff))
        except Exception as e:
            print(f"[torvik] four_factors {yr} failed: {e}")

        try:
            ps = torvik.player_stats(yr)
            path_ps = TORVIK_PLAYER_STATS / f"{yr}_player_stats.parquet"
            ps.to_parquet(path_ps, index=False)
            total_rows += len(ps)
            _log(f"player_stats {yr}", len(ps))
        except Exception as e:
            print(f"[torvik] player_stats {yr} failed: {e}")

    # 3) Pre-tournament snapshots: 2008–2010 team_ratings; 2011–2025 time_machine
    for yr in years_tor:
        if yr in (2008, 2009, 2010):
            df = torvik.team_ratings(yr, fmt="csv")
            df = df.copy()
            df["timemachine_available"] = 0
            path_tm = TORVIK_TIMEMACHINE / f"{yr}_pretournament.parquet"
            df.to_parquet(path_tm, index=False)
            total_rows += len(df)
            _log(f"pretournament (team_ratings) {yr}", len(df))
        elif yr in TIME_MACHINE_DATES:
            d = TIME_MACHINE_DATES[yr]
            df = torvik.time_machine(d)
            df = df.copy()
            df["timemachine_available"] = 1
            df["time_machine_date"] = d
            path_tm = TORVIK_TIMEMACHINE / f"{yr}_pretournament.parquet"
            df.to_parquet(path_tm, index=False)
            total_rows += len(df)
            _log(f"pretournament (time_machine {d}) {yr}", len(df))

    torvik.close()
    return total_rows


def main() -> None:
    """CLI entrypoint."""
    n = run_ingestion()
    print(f"[torvik] TOTAL rows written (sum across files): {n}")


if __name__ == "__main__":
    main()
