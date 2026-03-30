"""Build static (team × year) feature table from Torvik + coach_store (no game_logs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import (
    DATA_FEATURES,
    DATA_PROCESSED,
    TORVIK_FOUR_FACTORS,
    TORVIK_TIMEMACHINE,
    TRAINING_YEARS,
)
from src.utils.name_normalize import load_team_name_map, school_to_canonical, torvik_to_canonical

from src.features import build_player_aggregates as bpa


def _log_nulls(df: pd.DataFrame, label: str) -> None:
    """Print row count and null rate per column (top 15)."""
    n = len(df)
    print(f"[build_static] {label}: rows={n}")
    if n == 0:
        return
    rates = df.isna().mean().sort_values(ascending=False).head(15)
    print(f"[build_static] {label} null rates (worst 15):\n{rates}")


def load_timemachine_stack(years: list[int]) -> pd.DataFrame:
    """Concatenate pretournament snapshots with ``year`` column."""
    frames: list[pd.DataFrame] = []
    for yr in years:
        path = TORVIK_TIMEMACHINE / f"{yr}_pretournament.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["year"] = yr
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_four_factors_stack(years: list[int]) -> pd.DataFrame:
    """Four-factors rows are aligned with timemachine by ``rank`` (same row order per year)."""
    frames: list[pd.DataFrame] = []
    for yr in years:
        path = TORVIK_FOUR_FACTORS / f"{yr}_four_factors.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["year"] = yr
        df["rank"] = np.arange(1, len(df) + 1, dtype=int)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_static(
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Merge timemachine, four factors, player aggregates, coach_store."""
    years = years or TRAINING_YEARS
    mapping = load_team_name_map()

    tm = load_timemachine_stack(years)
    _log_nulls(tm, "timemachine raw")
    tm["team_norm"] = tm["team"].map(lambda x: torvik_to_canonical(str(x), mapping))
    tm["rank"] = pd.to_numeric(tm["rank"], errors="coerce").astype("Int64")

    ff = load_four_factors_stack(years)
    _log_nulls(ff, "four_factors raw")
    ff_cols = [c for c in ff.columns if c not in ("team", "year", "rank")]
    ff_sub = ff[["year", "rank"] + ff_cols]
    merged = tm.merge(ff_sub, on=["year", "rank"], how="left", suffixes=("", "_ff"))
    _log_nulls(merged, "after four_factors merge")

    pa_path = DATA_PROCESSED / "player_aggregates.parquet"
    if pa_path.exists():
        pa = pd.read_parquet(pa_path)
    else:
        pa = bpa.build_all(years)
    if not pa.empty:
        pa["team_norm"] = pa["team"].map(lambda x: torvik_to_canonical(str(x), mapping))
        pa_cols = [c for c in pa.columns if c not in ("team", "year", "team_norm")]
        merged = merged.merge(
            pa[["team_norm", "year"] + pa_cols],
            on=["team_norm", "year"],
            how="left",
        )
    _log_nulls(merged, "after player_aggregates merge")

    coach_path = DATA_PROCESSED / "coach_store.parquet"
    if coach_path.exists():
        cs = pd.read_parquet(coach_path)
        cs["team_norm"] = cs["school"].map(lambda s: school_to_canonical(str(s), mapping))
        cs_agg = (
            cs.groupby(["team_norm", "season_year"], as_index=False)
            .agg(
                {
                    "coach_tourn_appearances": "max",
                    "coach_final_four_count": "max",
                    "coach_champ_count": "max",
                }
            )
        )
        cs_agg = cs_agg.rename(columns={"season_year": "year"})
        cs_agg["year"] = pd.to_numeric(cs_agg["year"], errors="coerce").astype(int)
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype(int)
        merged = merged.merge(cs_agg, on=["team_norm", "year"], how="left")
    _log_nulls(merged, "after coach_store merge")

    merged["is_bubble_year"] = (merged["year"] == 2021).astype(int)
    return merged


def _stringify_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure object columns are parquet-safe (mixed str/float breaks pyarrow)."""
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str)
    return out


def main() -> None:
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    df = merge_static()
    df = _stringify_objects(df)
    out = DATA_FEATURES / "static_features.parquet"
    df.to_parquet(out, index=False)
    print(f"[build_static] wrote {out} shape={df.shape}")


if __name__ == "__main__":
    main()
