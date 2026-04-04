"""Generate ``data/features/schema_registry.json`` from raw column sets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import (
    DATA_FEATURES,
    DATA_PROCESSED,
    TORVIK_FOUR_FACTORS,
    TORVIK_PLAYER_STATS,
    TORVIK_TIMEMACHINE,
)

ADDED_DATE = "2026-03-30"
VERSION = "v1.0"


def entry(
    source: str,
    endpoint: str,
    is_derived: bool,
    derivation: str | None,
    description: str,
    feature_group: str,
    nullable: bool = True,
    expected_range: list | None = None,
) -> dict:
    return {
        "source": source,
        "endpoint": endpoint,
        "is_derived": is_derived,
        "derivation": derivation,
        "rolling_window": None,
        "added_version": VERSION,
        "added_date": ADDED_DATE,
        "description": description,
        "feature_group": feature_group,
        "nullable": nullable,
        "expected_range": expected_range,
    }


def main() -> None:
    reg: dict[str, dict] = {}

    tm_path = sorted(TORVIK_TIMEMACHINE.glob("*_pretournament.parquet"))[0]
    tm = pd.read_parquet(tm_path)
    for c in tm.columns:
        derived = c in ("timemachine_available", "time_machine_date")
        reg[f"timemachine.{c}"] = entry(
            "barttorvik",
            "ingest_torvik.py pretournament snapshot",
            derived,
            "Added at ingest; not from Bart flat file." if derived else None,
            f"Pretournament team ratings column `{c}`.",
            "metadata" if derived else "resume",
        )

    ff_path = sorted(TORVIK_FOUR_FACTORS.glob("*_four_factors.parquet"))[0]
    ff = pd.read_parquet(ff_path)
    for c in ff.columns:
        reg[f"four_factors.{c}"] = entry(
            "barttorvik",
            "four_factors() yearly parquet",
            False,
            None,
            f"Four factors column `{c}`.",
            "four_factor_matchup",
        )

    ps_path = sorted(TORVIK_PLAYER_STATS.glob("*_player_stats.parquet"))[0]
    ps = pd.read_parquet(ps_path)
    for c in ps.columns:
        reg[f"player_stats.{c}"] = entry(
            "barttorvik",
            "player_stats() / getadvstats yearly parquet",
            False,
            None,
            f"Torvik player-level stat `{c}`.",
            "player_quality",
        )

    pa = pd.read_parquet(DATA_PROCESSED / "player_aggregates.parquet")
    deriv_cols = {
        "team_bpm_wtd": "Minutes-weighted mean BPM across roster.",
        "team_obpm_wtd": "Minutes-weighted mean OBPM.",
        "team_dbpm_wtd": "Minutes-weighted mean DBPM.",
        "team_height_wtd": "Minutes-weighted mean height (inches).",
        "team_height_max": "Max height among rotation players (>10% min share).",
        "team_experience_idx": "Minutes-weighted class year (Fr=1…Sr=4).",
        "star_usg_share": "Max player usage / sum team usage.",
        "top2_usg_share": "Sum of top-2 usages / sum team usage.",
        "depth_score": "BPM of 7th man by minutes.",
        "roster_sr_pct": "Share of minutes from Jr and Sr players.",
    }
    for c in pa.columns:
        is_d = c in deriv_cols
        reg[f"player_aggregates.{c}"] = entry(
            "barttorvik",
            "build_player_aggregates.py",
            is_d,
            deriv_cols.get(c),
            f"Team-season aggregate `{c}`.",
            "player_quality",
        )

    cs = pd.read_parquet(DATA_PROCESSED / "coach_store.parquet")
    coach_derived = {
        "coach_tourn_appearances",
        "coach_final_four_count",
        "coach_champ_count",
    }
    for c in cs.columns:
        is_d = c in coach_derived
        reg[f"coach_store.{c}"] = entry(
            "sports-reference.com/cbb/coaches/",
            "ingest_coaches.py",
            is_d,
            "Cumulative through prior seasons only." if is_d else None,
            f"Coach history column `{c}`.",
            "coach",
        )

    ta_dir = DATA_PROCESSED / "tournament_analytics"
    seed_path = ta_dir / "seed_pair_win_rates.parquet"
    if seed_path.exists():
        sd = pd.read_parquet(seed_path)
        for c in sd.columns:
            is_agg = c in (
                "historical_win_rate",
                "last_5yr_win_rate",
                "upset_occurred_pct",
            )
            reg[f"seed_pair_win_rates.{c}"] = entry(
                "barttorvik",
                "tournament_analytics.py seed_pair_win_rates.parquet",
                is_agg,
                "Aggregated from tournament_training_set outcomes by seed pair."
                if is_agg
                else None,
                f"Seed-pair lookup column `{c}`.",
                "historical_prior",
            )
    hist_path = ta_dir / "historical_reference.parquet"
    if hist_path.exists():
        hr = pd.read_parquet(hist_path)
        for c in hr.columns:
            reg[f"historical_reference.{c}"] = entry(
                "barttorvik",
                "tournament_analytics.py historical_reference.parquet",
                True,
                "Aggregated from tournament_training_set.parquet",
                f"Season tournament summary `{c}`.",
                "historical_prior",
            )

    static_path = DATA_FEATURES / "static_features.parquet"
    if static_path.exists():
        sf = pd.read_parquet(static_path)
        for c in sf.columns:
            reg[f"static_features.{c}"] = entry(
                "derived",
                "build_static.py",
                True,
                "Merged timemachine + four_factors + player_aggregates + coach_store",
                f"Static team-season feature `{c}`.",
                "metadata" if c in ("is_bubble_year", "year") else "resume",
            )

    gl_path = DATA_PROCESSED / "game_log.parquet"
    if gl_path.exists():
        gl = pd.read_parquet(gl_path)
        raw_gl = {
            "pts_scored",
            "pts_allowed",
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
            "team",
            "game_id",
            "year",
            "team_norm",
            "opp_team_norm",
        }
        for c in gl.columns:
            is_d = c not in raw_gl and c not in ("margin", "result")
            reg[f"game_log.{c}"] = entry(
                "espn/cbbpy",
                "build_game_log.py",
                is_d,
                "Derived from summed box score columns." if is_d else None,
                f"Team-game log column `{c}`.",
                "game_log",
            )

    roll_path = DATA_FEATURES / "rolling_features.parquet"
    if roll_path.exists():
        rf = pd.read_parquet(roll_path)
        for c in rf.columns:
            reg[f"rolling_features.{c}"] = entry(
                "derived",
                "build_rolling.py",
                True,
                "Shift(1) + rolling windows on prior games only.",
                f"Rolling pre-game feature `{c}`.",
                "rolling",
            )

    m_path = DATA_FEATURES / "matchup_features.parquet"
    if m_path.exists():
        mf = pd.read_parquet(m_path)
        for c in mf.columns:
            reg[f"matchup_features.{c}"] = entry(
                "derived",
                "build_matchups.py",
                True,
                "Tournament row joined with static, rolling snapshot, deltas, priors.",
                f"Matchup matrix column `{c}`.",
                "matchup",
            )

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    out = DATA_FEATURES / "schema_registry.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[schema_registry] wrote {len(reg)} entries to {out}")


if __name__ == "__main__":
    main()
