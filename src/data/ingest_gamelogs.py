"""Ingest CBBpy game-by-game box scores (cached per team per season)."""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cbbpy
import pandas as pd
from cbbpy import mens_scraper as ms

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import GAME_LOGS_RAW

TEAM_MAP_CSV = Path(cbbpy.__file__).resolve().parent / "utils" / "mens_team_map.csv"

FINAL_STATUSES = ("Final", "In Progress")

DEFAULT_YEARS: list[int] = [y for y in range(2008, 2026) if y != 2020]


def slugify_team(name: str) -> str:
    """Filename slug from ESPN location label."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s[:120] or "unknown"


def teams_for_season(season: int) -> list[str]:
    """D1 team ``location`` labels for CBBpy (matches schedule lookup)."""
    df = pd.read_csv(TEAM_MAP_CSV)
    sub = df.loc[df["season"] == season]
    return sorted(sub["location"].dropna().unique().tolist())


def fetch_team_gamelog_boxscores(year: int, team: str) -> pd.DataFrame:
    """
    Pull completed-game box scores for one team-season.

    Uses ``get_team_schedule`` + per-game ``get_game_boxscore`` to avoid
    CBBpy ``get_games_team`` sort failures when ``game_day`` is missing.
    """
    sch = ms.get_team_schedule(team, year)
    if sch.empty or "game_id" not in sch.columns:
        return pd.DataFrame()
    stat_col = "game_status" if "game_status" in sch.columns else sch.columns[0]
    gids = sch.loc[sch[stat_col].isin(FINAL_STATUSES), "game_id"].astype(str).tolist()
    frames: list[pd.DataFrame] = []
    for gid in gids:
        try:
            bx = ms.get_game_boxscore(gid)
            if bx is not None and not bx.empty:
                frames.append(bx)
        except Exception:
            continue
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ingest_one_team(year: int, team: str) -> tuple[str, int]:
    """Write ``{year}_{slug}_gamelog.parquet``; return path and row count."""
    GAME_LOGS_RAW.mkdir(parents=True, exist_ok=True)
    slug = slugify_team(team)
    out = GAME_LOGS_RAW / f"{year}_{slug}_gamelog.parquet"
    if out.exists() and out.stat().st_size > 100:
        try:
            cached = pd.read_parquet(out)
            return str(out), len(cached)
        except Exception:
            pass
    try:
        box_df = fetch_team_gamelog_boxscores(year, team)
        if box_df.empty:
            return "", 0
        box_df.to_parquet(out, index=False)
        return str(out), len(box_df)
    except Exception as e:
        print(f"[gamelogs] {year} {team}: {e}")
        return "", 0


def ingest_year(year: int) -> tuple[int, int]:
    """Ingest all D1 teams for one season."""
    teams = teams_for_season(year)
    total_rows = 0
    for team in teams:
        _, n = ingest_one_team(year, team)
        total_rows += n
    print(f"[gamelogs] year {year}: {total_rows} boxscore rows across {len(teams)} teams")
    return total_rows, len(teams)


def run_ingestion(years: list[int], max_workers: int = 4) -> int:
    """
    Parallelize by season (``max_workers`` concurrent years).

    Returns
    -------
    int
        Sum of boxscore rows written across all seasons.
    """
    GAME_LOGS_RAW.mkdir(parents=True, exist_ok=True)
    grand_total = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(ingest_year, y): y for y in years}
        for fut in as_completed(futs):
            y = futs[fut]
            try:
                rows, _n_teams = fut.result()
                grand_total += rows
            except Exception as e:
                print(f"[gamelogs] year {y} failed: {e}")
    return grand_total


def main() -> None:
    parser = argparse.ArgumentParser(description="CBBpy gamelog ingestion.")
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help="Comma-separated ending years (e.g. 2024,2025). Default: 2008–2025 excl. 2020.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel year workers.")
    args = parser.parse_args()
    if args.years.strip():
        years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    else:
        years = list(DEFAULT_YEARS)
    n = run_ingestion(years, max_workers=args.workers)
    print(f"[gamelogs] TOTAL boxscore rows (sum): {n}")


if __name__ == "__main__":
    main()
