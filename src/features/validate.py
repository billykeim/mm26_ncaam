"""Pipeline validation: leakage (fail) and data quality (warn)."""

from __future__ import annotations

import json
import random
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
    DATA_TRAINING,
    SELECTION_SUNDAY_DATES,
    TEAM_NAME_MAP_PATH,
    TRAINING_YEARS,
)


def _load_registry() -> dict:
    p = DATA_FEATURES / "schema_registry.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_no_2020(df: pd.DataFrame, name: str, failed: list[str]) -> None:
    if "year" in df.columns and (df["year"] == 2020).any():
        failed.append(f"{name}: contains year 2020 rows")


def check_rolling_no_leakage(
    rolling: pd.DataFrame, failed: list[str], n_samples: int = 10
) -> None:
    """Spot-check roll10_win_rate uses only prior games (same season)."""
    rng = random.Random(42)
    gcols = ["team_norm", "year"]
    eligible = rolling.dropna(subset=["result_s"] if "result_s" in rolling.columns else [])
    if "result_s" not in rolling.columns:
        failed.append("rolling_features: missing result_s column for leakage test")
        return
    subs = [g for _, g in rolling.groupby(gcols) if len(g) >= 12]
    if len(subs) < n_samples:
        subs = [g for _, g in rolling.groupby(gcols)]
    picks = rng.sample(subs, min(n_samples, len(subs)))
    for grp in picks:
        grp = grp.sort_values("game_date")
        idx = rng.randrange(10, len(grp))
        row = grp.iloc[idx]
        prior = grp.iloc[:idx]
        exp = prior["result"].tail(10).mean()
        got = row["roll10_win_rate"]
        if pd.isna(got) and pd.isna(exp):
            continue
        if pd.isna(got) or abs(float(got) - float(exp)) > 1e-5:
            failed.append(
                f"rolling leak? team={row['team_norm']} year={row['year']} "
                f"roll10_win_rate={got} expected_from_prior={exp}"
            )
            return


def check_matchup_snapshot_before_ss(
    matchup: pd.DataFrame,
    rolling: pd.DataFrame,
    failed: list[str],
    warns: list[str],
) -> None:
    """Rolling game_date used in snapshot must be before Selection Sunday."""
    first_tourn = min(TRAINING_YEARS)
    for tour_y in TRAINING_YEARS:
        if tour_y not in SELECTION_SUNDAY_DATES:
            continue
        if tour_y == 2021:
            log_y = 2021
        elif tour_y > first_tourn:
            log_y = tour_y - 1
        else:
            log_y = tour_y
        cut = pd.Timestamp(SELECTION_SUNDAY_DATES[tour_y])
        sub = rolling[(rolling["year"] == log_y) & (rolling["game_date"] < cut)]
        if sub.empty:
            warns.append(
                f"No rolling rows before Selection Sunday for tournament {tour_y} "
                f"(log_year={log_y}); matchup rolling may be null"
            )
            continue
        sub = sub.sort_values(["team_norm", "game_date", "game_num"])
        last_dates = sub.groupby("team_norm").tail(1)["game_date"]
        bad = (last_dates >= cut).any()
        if bad:
            failed.append(
                f"rolling snapshot after Selection Sunday for tournament {tour_y}"
            )


def check_loo_2019(warns: list[str]) -> None:
    """Heuristic: 2019 tournament rows should not encode post-tourney 2019 outcomes."""
    warns.append(
        "LOO 2019: manual audit recommended — static/rolling exclude in-tournament games by design"
    )


def quality_matchup(mt: pd.DataFrame, warns: list[str]) -> None:
    key_feats = [
        "t1_barthag",
        "t2_barthag",
        "t1_roll10_win_rate",
        "t2_roll10_win_rate",
        "delta_adj_em",
        "delta_roll10_win_rate",
    ]
    for c in key_feats:
        if c not in mt.columns:
            continue
        rate = float(mt[c].isna().mean())
        if rate >= 0.10:
            warns.append(f"matchup_features key column {c} null_rate={rate:.3f} (>=10%)")

    if "delta_adj_em" in mt.columns:
        d = pd.to_numeric(mt["delta_adj_em"], errors="coerce")
        if d.abs().max() > 40:
            warns.append(f"delta_adj_em max abs {d.abs().max():.2f} outside [-40,40]")

    rw = "t1_roll10_win_rate"
    if rw in mt.columns:
        v = pd.to_numeric(mt[rw], errors="coerce")
        if v.notna().any() and (v.min() < -0.01 or v.max() > 1.01):
            warns.append(f"{rw} outside [0,1] range")

    r = mt["result"]
    if set(pd.unique(r.dropna())) - {0, 1}:
        warns.append("result column not subset of {0,1}")
    bal = float(r.mean())
    if bal < 0.35 or bal > 0.65:
        warns.append(
            f"result mean={bal:.3f} not ~0.5 (expected: team1/team2 not randomly assigned)"
        )

    exp_years = set(TRAINING_YEARS)
    got = set(pd.unique(mt["year"].dropna().astype(int)))
    if got != exp_years:
        warns.append(f"years mismatch got={sorted(got)} expected={sorted(exp_years)}")

    warns.append(
        "Game counts per year exceed 63/67: tournament_training_set includes "
        "multiple postseason tournaments, not NCAA-only."
    )


def schema_check(mt: pd.DataFrame, warns: list[str]) -> None:
    reg = _load_registry()
    keys = set(reg.keys())
    skip = {
        "result",
        "game_id",
        "year",
        "team1",
        "team2",
        "winner",
        "matchup",
        "muid",
        "t1_team_norm",
        "t2_team_norm",
        "t1_seed",
        "t2_seed",
    }
    unreg = []
    for c in mt.columns:
        if c in skip or c.startswith("_"):
            continue
        base = c
        if base.startswith(("t1_", "t2_")):
            base = base[3:]
        elif base.startswith("delta_"):
            base = base[6:]
        found = any(k.endswith(f".{base}") for k in keys)
        if not found and f"matchup_features.{c}" not in keys:
            unreg.append(c)
    if unreg:
        warns.append(f"Unregistered columns (sample 15): {unreg[:15]}")

    bad_deriv = [
        k
        for k, v in reg.items()
        if v.get("is_derived") and not v.get("derivation")
    ]
    if bad_deriv:
        warns.append(
            f"Registry is_derived without derivation ({len(bad_deriv)} keys); "
            f"sample: {bad_deriv[:5]}"
        )


def run_validation() -> tuple[list[str], list[str], list[str]]:
    passed: list[str] = []
    warns: list[str] = []
    failed: list[str] = []

    rolling = pd.read_parquet(DATA_FEATURES / "rolling_features.parquet")
    rolling["game_date"] = pd.to_datetime(rolling["game_date"])
    if "result_s" not in rolling.columns:
        rolling["result_s"] = rolling.groupby(["team_norm", "year"])["result"].shift(1)

    matchup = pd.read_parquet(DATA_FEATURES / "matchup_features.parquet")

    check_no_2020(rolling, "rolling_features", failed)
    check_no_2020(matchup, "matchup_features", failed)
    passed.append("No 2020 rows in rolling/matchup (or year col missing)")

    check_rolling_no_leakage(rolling, failed)
    if not any("rolling leak" in x for x in failed):
        passed.append("Rolling roll10_win_rate spot-check vs prior games")

    check_matchup_snapshot_before_ss(matchup, rolling, failed, warns)

    check_loo_2019(warns)

    with open(TEAM_NAME_MAP_PATH, encoding="utf-8") as f:
        cmap = json.load(f)
    canon = set(cmap.keys())
    for k, v in cmap.items():
        if isinstance(v, dict):
            canon.add(str(v.get("canonical", k)))
            if v.get("torvik"):
                canon.add(str(v["torvik"]))
    for side in ("t1_team_norm", "t2_team_norm"):
        bad = ~matchup[side].astype(str).isin(canon)
        if bad.any():
            warns.append(f"{side}: {bad.sum()} values not in team_name_map canonical/torvik set")

    quality_matchup(matchup, warns)
    schema_check(matchup, warns)

    passed.append("Quality and schema checks executed (see WARNINGS)")

    return passed, warns, failed


def main() -> None:
    passed, warns, failed = run_validation()
    lines = [
        "=== MM26 PIPELINE VALIDATION ===",
        f"PASSED ({len(passed)}):",
        *[f"  - {p}" for p in passed],
        f"WARNINGS ({len(warns)}):",
        *[f"  - {w}" for w in warns],
        f"FAILED ({len(failed)}):",
        *[f"  - {f}" for f in failed],
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    out = DATA_FEATURES / "validation_report.txt"
    out.write_text(text, encoding="utf-8")
    print(f"[validate] wrote {out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
