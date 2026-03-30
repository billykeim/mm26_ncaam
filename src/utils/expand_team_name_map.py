"""Expand ``team_name_map.json`` from Torvik timemachine + coach_store schools."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.constants import DATA_PROCESSED, TEAM_NAME_MAP_PATH, TORVIK_TIMEMACHINE
from src.utils.name_normalize import add_torvik_team, load_team_name_map


def collect_torvik_teams() -> set[str]:
    """Unique team names from all timemachine parquets."""
    names: set[str] = set()
    for path in sorted(TORVIK_TIMEMACHINE.glob("*_pretournament.parquet")):
        df = pd.read_parquet(path, columns=["team"])
        names.update(df["team"].dropna().astype(str).unique())
    return names


def collect_coach_schools() -> set[str]:
    """Unique school names from coach_store."""
    p = DATA_PROCESSED / "coach_store.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=["school"])
    return set(df["school"].dropna().astype(str).unique())


def expand_map() -> dict:
    """Load map, add missing Torvik teams and coach schools; return flags."""
    mapping = load_team_name_map()
    added_torvik: list[str] = []
    torvik_teams = collect_torvik_teams()
    known_torvik = {
        str(v.get("torvik"))
        for v in mapping.values()
        if isinstance(v, dict) and v.get("torvik")
    }
    for t in sorted(torvik_teams):
        if t not in known_torvik:
            mapping, new_e = add_torvik_team(mapping, t)
            if new_e is not None:
                added_torvik.append(t)
                known_torvik.add(t)

    coach_schools = collect_coach_schools()
    in_map_sr = {
        str(v.get("sports_ref"))
        for v in mapping.values()
        if isinstance(v, dict) and v.get("sports_ref")
    }
    coach_not_in_map = sorted(s for s in coach_schools if s and s not in in_map_sr)

    with open(TEAM_NAME_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
        f.write("\n")

    flags = {
        "added_torvik_teams_count": len(added_torvik),
        "coach_schools_not_in_map": coach_not_in_map,
        "torvik_teams_not_in_coach_schools": sorted(
            t for t in torvik_teams if t not in coach_schools
        )[:200],
    }
    return flags


def main() -> None:
    flags = expand_map()
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
