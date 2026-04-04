"""Project paths and shared constants."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES: Path = PROJECT_ROOT / "data" / "features"
DATA_TRAINING: Path = PROJECT_ROOT / "data" / "training"
TORVIK_RAW: Path = DATA_RAW / "torvik"
TORVIK_TIMEMACHINE: Path = TORVIK_RAW / "timemachine"
TORVIK_GAME_RESULTS: Path = TORVIK_RAW / "game_results"
TORVIK_FOUR_FACTORS: Path = TORVIK_RAW / "four_factors"
TORVIK_PLAYER_STATS: Path = TORVIK_RAW / "player_stats"
SPORTS_REF_RAW: Path = DATA_RAW / "sports_ref"
TOURNAMENT_SEEDS_PATH: Path = SPORTS_REF_RAW / "tournament_seeds.parquet"
TOURNAMENT_RESULTS_PATH: Path = SPORTS_REF_RAW / "tournament_results.parquet"
TOURNAMENT_RESULTS_CACHE: Path = SPORTS_REF_RAW / "tournament_results_cache"
COACHES_CACHE: Path = SPORTS_REF_RAW / "coaches_cache"
GAME_LOGS_RAW: Path = SPORTS_REF_RAW / "game_logs"
TEAM_NAME_MAP_PATH: Path = PROJECT_ROOT / "src" / "utils" / "team_name_map.json"
COACH_NAME_MAP_PATH: Path = PROJECT_ROOT / "src" / "utils" / "coach_name_map.json"

# Selection Sunday → time_machine() date (YYYYMMDD). From DECISIONS.md; 2008–2010 N/A.
TIME_MACHINE_DATES: dict[int, str] = {
    2011: "20110314",
    2012: "20120312",
    2013: "20130318",
    2014: "20140317",
    2015: "20150316",
    2016: "20160314",
    2017: "20170313",
    2018: "20180312",
    2019: "20190318",
    2021: "20210315",
    2022: "20220314",
    2023: "20230313",
    2024: "20240318",
    2025: "20250317",
}

TORVIK_CACHE_DIR: str = ".torvik_cache"

TRAINING_YEARS: list[int] = [y for y in range(2008, 2026) if y not in (2020,)]

# Selection Sunday calendar dates (YYYY-MM-DD). From DECISIONS.md; used for pre-tournament snapshots.
SELECTION_SUNDAY_DATES: dict[int, str] = {
    2008: "2008-03-16",
    2009: "2009-03-15",
    2010: "2010-03-14",
    2011: "2011-03-13",
    2012: "2012-03-11",
    2013: "2013-03-17",
    2014: "2014-03-16",
    2015: "2015-03-15",
    2016: "2016-03-13",
    2017: "2017-03-12",
    2018: "2018-03-11",
    2019: "2019-03-17",
    2021: "2021-03-14",
    2022: "2022-03-13",
    2023: "2023-03-12",
    2024: "2024-03-17",
    2025: "2025-03-16",
}
