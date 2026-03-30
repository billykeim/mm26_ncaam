"""Aggregate Torvik player_stats to team × season player-quality features."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import DATA_PROCESSED, TORVIK_PLAYER_STATS, TRAINING_YEARS

YR_MAP = {"Fr": 1.0, "So": 2.0, "Jr": 3.0, "Sr": 4.0, "--": np.nan}


def parse_height(ht: object) -> float:
    """Convert ``6-10`` style height to total inches."""
    if ht is None or (isinstance(ht, float) and np.isnan(ht)):
        return np.nan
    s = str(ht).strip()
    m = re.match(r"^(\d+)-(\d+)$", s)
    if not m:
        return np.nan
    feet, inches = int(m.group(1)), int(m.group(2))
    return feet * 12.0 + inches


def aggregate_team_season(ps: pd.DataFrame) -> dict:
    """Compute aggregate metrics for one team (rows = players)."""
    if ps.empty:
        return {}
    team = ps["team"].iloc[0]
    year = int(ps["year"].iloc[0])
    w = pd.to_numeric(ps["min_per"], errors="coerce").fillna(0.0)
    total = w.sum()
    if total <= 0:
        total = 1.0
    w_share = w / total

    bpm = pd.to_numeric(ps["bpm"], errors="coerce")
    obpm = pd.to_numeric(ps["obpm"], errors="coerce")
    dbpm = pd.to_numeric(ps["dbpm"], errors="coerce")
    usage = pd.to_numeric(ps["usage"], errors="coerce").fillna(0.0)
    ht_in = ps["ht"].map(parse_height)
    yr_num = ps["yr"].map(lambda x: YR_MAP.get(str(x).strip(), np.nan))

    def wmean(series: pd.Series) -> float:
        return float(np.nansum(series.values * w_share.values) / np.nansum(w_share.values))

    team_bpm_wtd = wmean(bpm)
    team_obpm_wtd = wmean(obpm)
    team_dbpm_wtd = wmean(dbpm)
    team_height_wtd = wmean(ht_in)
    team_experience_idx = wmean(yr_num)

    rot = w_share > 0.10
    if rot.any():
        team_height_max = float(ht_in[rot].max())
    else:
        team_height_max = float(ht_in.max()) if len(ht_in) else np.nan

    u_sum = usage.sum()
    if u_sum > 0:
        star_usg_share = float(usage.max() / u_sum)
        top2 = usage.nlargest(2).sum()
        top2_usg_share = float(top2 / u_sum)
    else:
        star_usg_share = np.nan
        top2_usg_share = np.nan

    order = np.argsort(-w.values)
    idx7 = order[6] if len(order) > 6 else np.nan
    if not (isinstance(idx7, float) and np.isnan(idx7)):
        depth_score = float(bpm.iloc[int(idx7)])
    else:
        depth_score = np.nan

    jr_sr_mask = ps["yr"].isin(["Jr", "Sr"])
    roster_sr_pct = float(w[jr_sr_mask].sum() / total) if total else np.nan

    return {
        "team": team,
        "year": year,
        "team_bpm_wtd": team_bpm_wtd,
        "team_obpm_wtd": team_obpm_wtd,
        "team_dbpm_wtd": team_dbpm_wtd,
        "team_height_wtd": team_height_wtd,
        "team_height_max": team_height_max,
        "team_experience_idx": team_experience_idx,
        "star_usg_share": star_usg_share,
        "top2_usg_share": top2_usg_share,
        "depth_score": depth_score,
        "roster_sr_pct": roster_sr_pct,
    }


def build_all(years: list[int] | None = None) -> pd.DataFrame:
    """Build aggregates for all teams per season."""
    years = years or TRAINING_YEARS
    rows: list[dict] = []
    for yr in years:
        path = TORVIK_PLAYER_STATS / f"{yr}_player_stats.parquet"
        if not path.exists():
            continue
        ps = pd.read_parquet(path)
        for team, g in ps.groupby("team", sort=False):
            rows.append(aggregate_team_season(g))
    return pd.DataFrame(rows)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "player_aggregates.parquet"
    df = build_all()
    print(f"[player_aggregates] rows: {len(df)}")
    df.to_parquet(out, index=False)
    print(f"[player_aggregates] wrote {out}")


if __name__ == "__main__":
    main()
