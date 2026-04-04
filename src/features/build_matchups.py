"""Tournament matchup matrix: static + rolling snapshot + deltas + priors."""

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
    PROJECT_ROOT,
    SELECTION_SUNDAY_DATES,
    TOURNAMENT_SEEDS_PATH,
    TRAINING_YEARS,
)

# Dropped from written matchup matrix (no predictive value or wrong scale).
MATCHUP_DROP_COLS: frozenset[str] = frozenset(
    {
        "t1_fun",
        "t2_fun",
        "delta_fun",
        "t1_con_pf",
        "t2_con_pf",
        "delta_con_pf",
        "t1_con_pa",
        "t2_con_pa",
        "delta_con_pa",
        "t1_con_poss",
        "t2_con_poss",
        "delta_con_poss",
        "ord_date",
        "season",
    }
)
from src.utils.name_normalize import load_team_name_map, torvik_to_canonical

# Do not auto-delta game-log / bookkeeping columns merged from rolling snapshots.
_DELTA_SKIP_BASES: frozenset[str] = frozenset(
    {
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
        "pts_allowed",
        "opp_reb",
        "opp_ast",
        "opp_stl",
        "opp_blk",
        "opp_oreb",
        "opp_dreb",
        "opp_fgm",
        "opp_fga",
        "opp_ftm",
        "opp_fta",
        "opp_three_fgm",
        "opp_three_fga",
        "opp_tov",
        "opp_efg",
        "opp_ft_rate",
        "opp_tov_rate",
        "margin",
        "result",
        "game_num",
        "season_pct_elapsed",
        "is_early_season",
        "conf_game_flag",
        "is_tournament_game",
        "team",
        "game_id",
        "opp_team_norm",
        "result_s",
        "margin_s",
        "pts_scored_s",
        "pts_allowed_s",
        "efg_s",
        "opp_efg_s",
        "tov_rate_s",
        "opp_barthag_s",
        "team_barthag_season",
        "fun",
        "con_pf",
        "con_pa",
        "con_poss",
        "ord_date",
        "season",
    }
)

TOURNEY_PATH = PROJECT_ROOT / "data" / "raw" / "torvik" / "tournament_training_set.parquet"
SEED_PAIR_PATH = DATA_PROCESSED / "tournament_analytics" / "seed_pair_win_rates.parquet"

# Main NCAA bracket game counts (last ``ord_date`` block per season; drops NIT/CBI etc.).
NCAA_BRACKET_GAMES_BY_SEASON: dict[int, int] = {2008: 63, 2009: 64, 2010: 64}
NCAA_BRACKET_GAMES_DEFAULT: int = 67  # 68-team era including First Four


def filter_ncaa_bracket_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to the NCAA tournament bracket only.

    ``conf == 3`` rows include other postseason events earlier in ``ord_date`` order; we keep
    the last *N* games per season (chronological tail), matching 63 / 64 / 67-game brackets.
    """
    if "ord_date" not in df.columns:
        raise ValueError("filter_ncaa_bracket_only requires column ord_date")
    d = df.copy()
    d["ord_date"] = pd.to_numeric(d["ord_date"], errors="coerce")
    d = d.sort_values(["season", "ord_date"], kind="mergesort")
    parts: list[pd.DataFrame] = []
    for season, g in d.groupby("season", sort=True):
        y = int(season)
        n = NCAA_BRACKET_GAMES_BY_SEASON.get(y, NCAA_BRACKET_GAMES_DEFAULT)
        if len(g) < n:
            print(
                f"[build_matchups] NCAA filter season={y}: only {len(g)} rows (< {n}); keeping all"
            )
            parts.append(g)
        else:
            parts.append(g.tail(n))
            print(f"[build_matchups] NCAA filter season={y}: kept {n} / {len(g)} rows (tail)")
    return pd.concat(parts, ignore_index=True)


def apply_random_t1_t2_swap(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Swap team1/team2 (and aligned norms, seeds, result) on ~50% of rows for label balance.

    Torvik often orders the stronger team as team1, skewing ``result`` upward; swapping
    breaks that bias while preserving the true winner in ``winner``.
    """
    out = df.copy()
    rng = np.random.RandomState(seed)
    w = rng.random(len(out)) < 0.5
    if not w.any():
        return out
    t1 = out.loc[w, "team1"].copy()
    out.loc[w, "team1"] = out.loc[w, "team2"].values
    out.loc[w, "team2"] = t1.values
    n1 = out.loc[w, "t1_team_norm"].copy()
    out.loc[w, "t1_team_norm"] = out.loc[w, "t2_team_norm"].values
    out.loc[w, "t2_team_norm"] = n1.values
    if "t1_seed" in out.columns and "t2_seed" in out.columns:
        s1 = out.loc[w, "t1_seed"].copy()
        out.loc[w, "t1_seed"] = out.loc[w, "t2_seed"].values
        out.loc[w, "t2_seed"] = s1.values
    out.loc[w, "result"] = (1 - out.loc[w, "result"].astype(int)).astype(np.int8)
    return out


def load_tournament_base() -> pd.DataFrame:
    """Tournament rows with ``year``, team norms, and binary ``result`` (1 = team1 won)."""
    use_cols = [
        "muid",
        "season",
        "team1",
        "team2",
        "winner",
        "matchup",
        "ord_date",
    ]
    df = pd.read_parquet(TOURNEY_PATH, columns=use_cols)
    df = df.copy()
    df = filter_ncaa_bracket_only(df)
    df["year"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    mapping = load_team_name_map()
    df["t1_team_norm"] = df["team1"].map(lambda x: torvik_to_canonical(str(x), mapping))
    df["t2_team_norm"] = df["team2"].map(lambda x: torvik_to_canonical(str(x), mapping))
    df["result"] = (df["winner"].astype(str) == df["team1"].astype(str)).astype(np.int8)
    df["game_id"] = df["muid"].astype(str)
    if not TOURNAMENT_SEEDS_PATH.exists():
        raise FileNotFoundError(
            f"Official seeds missing: {TOURNAMENT_SEEDS_PATH}. "
            "Run: python -m src.data.ingest_tournament_seeds"
        )
    seeds_tbl = pd.read_parquet(TOURNAMENT_SEEDS_PATH)
    seeds_tbl = seeds_tbl.drop_duplicates(subset=["year", "team_norm"], keep="first")
    seeds_tbl["year"] = pd.to_numeric(seeds_tbl["year"], errors="coerce").astype(int)
    seeds_tbl["team_norm"] = seeds_tbl["team_norm"].astype(str)
    seeds_tbl["official_seed"] = pd.to_numeric(seeds_tbl["official_seed"], errors="coerce").astype(
        int
    )
    seed_pairs: set[tuple[int, str]] = set(
        zip(seeds_tbl["year"].tolist(), seeds_tbl["team_norm"].tolist())
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    n_before = len(df)
    in_field = df.apply(
        lambda r: (int(r["year"]), str(r["t1_team_norm"])) in seed_pairs
        and (int(r["year"]), str(r["t2_team_norm"])) in seed_pairs,
        axis=1,
    )
    df = df.loc[in_field].copy()
    print(
        f"[build_matchups] official NCAA field filter: kept {len(df)} / {n_before} rows "
        f"(dropped {n_before - len(df)} where a team is not in Sports-Reference bracket field)"
    )
    s1 = seeds_tbl.rename(
        columns={"team_norm": "t1_team_norm", "official_seed": "t1_seed"}
    )
    s2 = seeds_tbl.rename(
        columns={"team_norm": "t2_team_norm", "official_seed": "t2_seed"}
    )
    df = df.merge(s1, on=["year", "t1_team_norm"], how="left")
    df = df.merge(s2, on=["year", "t2_team_norm"], how="left")
    t1_null = df["t1_seed"].isna()
    t2_null = df["t2_seed"].isna()
    if t1_null.any():
        bad = df.loc[t1_null, ["year", "team1", "t1_team_norm"]].drop_duplicates()
        print("[build_matchups] missing official seed for team1 (fix team_name_map / re-ingest):")
        print(bad.to_string(index=False))
    if t2_null.any():
        bad = df.loc[t2_null, ["year", "team2", "t2_team_norm"]].drop_duplicates()
        print("[build_matchups] missing official seed for team2 (fix team_name_map / re-ingest):")
        print(bad.to_string(index=False))
    if t1_null.any() or t2_null.any():
        raise ValueError(
            "[build_matchups] official seed join has nulls; update team_name_map.json "
            "sports_ref/canonical/cbbpy and re-run ingest_tournament_seeds."
        )
    df["t1_seed"] = df["t1_seed"].astype(int)
    df["t2_seed"] = df["t2_seed"].astype(int)
    print(
        f"[build_matchups] official seeds joined: t1 null={df['t1_seed'].isna().mean():.4f}, "
        f"t2 null={df['t2_seed'].isna().mean():.4f}"
    )
    df = apply_random_t1_t2_swap(df, seed=42)
    rate = float(df["result"].mean())
    print(f"[build_matchups] after t1/t2 balance swap: result mean={rate:.4f} (target ~0.50)")
    if not (0.48 <= rate <= 0.52):
        print(
            f"[build_matchups] WARN: result mean {rate:.4f} outside [0.48, 0.52] "
            "(check row count / swap logic)"
        )
    return df


def _prefix_static(tab: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Keys ``{prefix}_team_norm`` + ``year``; other columns ``{prefix}_{col}``."""
    out = pd.DataFrame()
    out[f"{prefix}_team_norm"] = tab["team_norm"].values
    out["year"] = tab["year"].values
    for c in tab.columns:
        if c in ("team_norm", "year"):
            continue
        out[f"{prefix}_{c}"] = tab[c].values
    return out


def _prefix_rolling_snap(tab: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rolling snapshot: keys ``{prefix}_team_norm``, ``year``; rest prefixed."""
    skip = {"team_norm", "year"}
    out = pd.DataFrame()
    out[f"{prefix}_team_norm"] = tab["team_norm"].values
    out["year"] = tab["year"].values
    for c in tab.columns:
        if c in skip:
            continue
        out[f"{prefix}_{c}"] = tab[c].values
    return out


def build_rolling_snapshot(rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Last team×game row strictly before Selection Sunday.

    Tournament label ``Y`` matches CBBpy game-log ``year`` (season ending March ``Y``).
    ``game_date`` in ``build_game_log`` is anchored at November ``Y−1`` so rows fall in
    ``Nov (Y−1)``–``Mar Y``; we keep ``rolling.year == tour_y`` and ``game_date`` before
    that year's Selection Sunday.
    """
    parts: list[pd.DataFrame] = []
    rolling = rolling.copy()
    rolling["game_date"] = pd.to_datetime(rolling["game_date"])
    for tour_y in TRAINING_YEARS:
        if tour_y not in SELECTION_SUNDAY_DATES:
            continue
        log_y = tour_y
        cut = pd.Timestamp(SELECTION_SUNDAY_DATES[tour_y])
        sub = rolling[(rolling["year"] == log_y) & (rolling["game_date"] < cut)]
        if sub.empty:
            print(
                f"[build_matchups] rolling snapshot tournament {tour_y} "
                f"(log_year={log_y}): no rows before Selection Sunday"
            )
            continue
        sub = sub.sort_values(["team_norm", "game_date", "game_num"])
        last = sub.groupby("team_norm", as_index=False).tail(1)
        last = last.copy()
        last["year"] = int(tour_y)
        parts.append(last)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def aggregate_seed_priors(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Pool ``seed_pair_win_rates`` across rounds → one row per (seed_low, seed_high)."""
    rows: list[dict] = []
    for (lo, hi), g in seed_df.groupby(["seed_low", "seed_high"]):
        w = g["games_played"].to_numpy(dtype=float)
        if w.sum() <= 0:
            rows.append(
                {
                    "seed_low": lo,
                    "seed_high": hi,
                    "historical_win_rate": np.nan,
                    "last_5yr_win_rate": np.nan,
                    "sample_size": 0,
                }
            )
            continue
        l5 = g["last_5yr_win_rate"].fillna(g["historical_win_rate"]).to_numpy(dtype=float)
        rows.append(
            {
                "seed_low": lo,
                "seed_high": hi,
                "historical_win_rate": float(
                    np.average(g["historical_win_rate"].to_numpy(dtype=float), weights=w)
                ),
                "last_5yr_win_rate": float(np.average(l5, weights=w)),
                "sample_size": int(w.sum()),
            }
        )
    return pd.DataFrame(rows)


def add_delta_columns(mt: pd.DataFrame) -> pd.DataFrame:
    """``delta_{base}`` = ``t1_{base}`` − ``t2_{base}`` for numeric pairs."""
    deltas: dict[str, pd.Series] = {}
    t1cols = [c for c in mt.columns if c.startswith("t1_") and c != "t1_team_norm"]
    for c1 in t1cols:
        base = c1[3:]
        if base in _DELTA_SKIP_BASES:
            continue
        c2 = f"t2_{base}"
        if c2 not in mt.columns:
            continue
        if not pd.api.types.is_numeric_dtype(mt[c1]) or not pd.api.types.is_numeric_dtype(
            mt[c2]
        ):
            continue
        deltas[f"delta_{base}"] = pd.to_numeric(mt[c1], errors="coerce") - pd.to_numeric(
            mt[c2], errors="coerce"
        )
    if not deltas:
        return mt.copy()
    return pd.concat([mt, pd.DataFrame(deltas)], axis=1)


def add_priority_cross_deltas(mt: pd.DataFrame) -> pd.DataFrame:
    """Cross-component matchup deltas (offense vs opponent defense)."""
    out = mt.copy()
    if "t1_efg_pct" in out.columns and "t2_efg_pct_def" in out.columns:
        out["delta_off_efg_vs_def_efg"] = out["t1_efg_pct"] - out["t2_efg_pct_def"]
    if "t1_to_pct" in out.columns and "t2_to_pct_def" in out.columns:
        out["delta_off_to_vs_def_to"] = out["t1_to_pct"] - out["t2_to_pct_def"]
    if "t1_or_pct" in out.columns and "t2_or_pct" in out.columns:
        out["delta_off_or_vs_def_or"] = out["t1_or_pct"] - (1.0 - out["t2_dr_pct"])
    if "t1_adjoe" in out.columns and "t1_adjde" in out.columns:
        if "t1_adj_em" not in out.columns:
            out["t1_adj_em"] = out["t1_adjoe"] - out["t1_adjde"]
    if "t2_adjoe" in out.columns and "t2_adjde" in out.columns:
        if "t2_adj_em" not in out.columns:
            out["t2_adj_em"] = out["t2_adjoe"] - out["t2_adjde"]
    if "t1_adj_em" in out.columns and "t2_adj_em" in out.columns:
        dem = pd.to_numeric(out["t1_adj_em"], errors="coerce") - pd.to_numeric(
            out["t2_adj_em"], errors="coerce"
        )
        out["delta_adj_em"] = dem.clip(-40.0, 40.0)
    if "t1_seed" in out.columns and "t2_seed" in out.columns:
        out["delta_seed"] = pd.to_numeric(out["t1_seed"], errors="coerce") - pd.to_numeric(
            out["t2_seed"], errors="coerce"
        )
    if "t1_barthag" in out.columns and "t2_barthag" in out.columns:
        out["delta_barthag"] = pd.to_numeric(out["t1_barthag"], errors="coerce") - pd.to_numeric(
            out["t2_barthag"], errors="coerce"
        )
    return out


def add_matchup_derived(mt: pd.DataFrame) -> pd.DataFrame:
    """Pace, flags, bubble year, major/mid-major proxy."""
    out = mt.copy()
    if "t1_adjt" in out.columns and "t2_adjt" in out.columns:
        out["projected_tempo"] = (out["t1_adjt"] + out["t2_adjt"]) / 2.0
        out["pace_variance_flag"] = (
            (out["t1_adjt"] - out["t2_adjt"]).abs() > 8
        ).astype(np.int8)
    else:
        out["projected_tempo"] = np.nan
        out["pace_variance_flag"] = 0
    for c in ("t1_fg3_rate", "t2_fg3_rate"):
        if c not in out.columns:
            out[c] = np.nan
    out["three_pt_reliance_flag"] = (
        (pd.to_numeric(out["t1_fg3_rate"], errors="coerce") > 0.42)
        | (pd.to_numeric(out["t2_fg3_rate"], errors="coerce") > 0.42)
    ).astype(np.int8)
    if "delta_seed" in out.columns:
        out["low_tempo_coin_flip"] = (
            (out["projected_tempo"] < 62) & (out["delta_seed"].abs() <= 4)
        ).astype(np.int8)
    else:
        out["low_tempo_coin_flip"] = 0
    thr = 0.88
    out["t1_is_major_conf"] = (pd.to_numeric(out["t1_barthag"], errors="coerce") >= thr).astype(
        np.int8
    )
    out["t2_is_major_conf"] = (pd.to_numeric(out["t2_barthag"], errors="coerce") >= thr).astype(
        np.int8
    )
    out["midmajor_matchup"] = (
        out["t1_is_major_conf"].astype(int) + out["t2_is_major_conf"].astype(int) == 1
    ).astype(np.int8)
    out["is_bubble_year"] = (out["year"] == 2021).astype(np.int8)
    out["coach_tourn_win_rate"] = np.nan
    for side in ("t1", "t2"):
        c = f"{side}_coach_tourn_appearances"
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            out[f"{side}_coach_is_first_tourn"] = (v <= 0).astype(np.int8)
    return out


def merge_seed_priors(mt: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Join pooled seed-pair historical win rate (better seed = ``seed_low``)."""
    out = mt.copy()
    s1 = pd.to_numeric(out["t1_seed"], errors="coerce")
    s2 = pd.to_numeric(out["t2_seed"], errors="coerce")
    out["_seed_low"] = np.minimum(s1, s2)
    out["_seed_high"] = np.maximum(s1, s2)
    out = out.merge(
        priors.rename(
            columns={
                "historical_win_rate": "seed_prior_historical_win_rate",
                "sample_size": "seed_prior_sample_size",
                "last_5yr_win_rate": "seed_prior_last_5yr_win_rate",
            }
        ),
        left_on=["_seed_low", "_seed_high"],
        right_on=["seed_low", "seed_high"],
        how="left",
    )
    out = out.drop(columns=["seed_low", "seed_high"], errors="ignore")
    return out


def add_historical_win_rate_for_team1(mt: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """
    Join pooled (seed_low, seed_high) prior; expose ``historical_win_rate`` as P(team1 wins)
    and ``sample_size`` as games backing the pair.
    """
    tmp = mt.copy()
    s1 = pd.to_numeric(tmp["t1_seed"], errors="coerce")
    s2 = pd.to_numeric(tmp["t2_seed"], errors="coerce")
    tmp["_lo"] = np.minimum(s1, s2)
    tmp["_hi"] = np.maximum(s1, s2)
    pr = priors[["seed_low", "seed_high", "historical_win_rate", "sample_size"]].drop_duplicates(
        subset=["seed_low", "seed_high"]
    )
    merged = tmp.merge(
        pr.rename(columns={"historical_win_rate": "_phr", "sample_size": "_pss"}),
        left_on=["_lo", "_hi"],
        right_on=["seed_low", "seed_high"],
        how="left",
    )
    merged = merged.drop(columns=["_lo", "_hi", "seed_low", "seed_high"], errors="ignore")
    ms1 = pd.to_numeric(merged["t1_seed"], errors="coerce")
    ms2 = pd.to_numeric(merged["t2_seed"], errors="coerce")
    ph = merged["_phr"].fillna(0.5)
    merged["historical_win_rate"] = np.where(
        ms1.isna() | ms2.isna(),
        np.nan,
        np.where(ms1 == ms2, 0.5, np.where(ms1 < ms2, ph, 1.0 - ph)),
    )
    merged["sample_size"] = merged["_pss"].fillna(0).astype(np.int64)
    merged = merged.drop(columns=["_phr", "_pss"], errors="ignore")
    return merged


def drop_matchup_noise_columns(mt: pd.DataFrame) -> pd.DataFrame:
    """Remove cumulative/noise columns from the matchup matrix."""
    drop = [c for c in mt.columns if c in MATCHUP_DROP_COLS]
    return mt.drop(columns=drop, errors="ignore")


def build_matchups() -> pd.DataFrame:
    """Full matchup feature table."""
    base = load_tournament_base()
    print(f"[build_matchups] tournament shape={base.shape} cols={len(base.columns)}")
    print(f"[build_matchups] result mean={base['result'].mean():.3f}")

    static = pd.read_parquet(DATA_FEATURES / "static_features.parquet")
    static = static.drop_duplicates(subset=["team_norm", "year"], keep="first")

    rolling = pd.read_parquet(DATA_FEATURES / "rolling_features.parquet")
    roll_snap = build_rolling_snapshot(rolling)
    print(f"[build_matchups] rolling snapshot rows={len(roll_snap)}")

    mapping = load_team_name_map()

    mt = base.merge(
        _prefix_static(static, "t1"),
        on=["t1_team_norm", "year"],
        how="left",
    )
    mt = mt.merge(_prefix_static(static, "t2"), on=["t2_team_norm", "year"], how="left")

    r1 = _prefix_rolling_snap(roll_snap, "t1")
    r2 = _prefix_rolling_snap(roll_snap, "t2")
    mt = mt.merge(r1, on=["t1_team_norm", "year"], how="left")
    mt = mt.merge(r2, on=["t2_team_norm", "year"], how="left")

    for c in list(mt.columns):
        if "coach_tourn_appearances" in c or "coach_final_four_count" in c or "coach_champ_count" in c:
            mt[c] = pd.to_numeric(mt[c], errors="coerce").fillna(0.0)

    mt = add_delta_columns(mt)
    mt = add_priority_cross_deltas(mt)
    mt = add_matchup_derived(mt)

    if SEED_PAIR_PATH.exists():
        sp = pd.read_parquet(SEED_PAIR_PATH)
        sp_agg = aggregate_seed_priors(sp)
        mt = merge_seed_priors(mt, sp_agg)
        mt = add_historical_win_rate_for_team1(mt, sp_agg)
    else:
        print("[build_matchups] seed_pair_win_rates missing; skipping prior join")

    mt = mt.drop(columns=["_seed_low", "_seed_high"], errors="ignore")
    mt = drop_matchup_noise_columns(mt)
    return mt


def main() -> None:
    mt = build_matchups()
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    out = DATA_FEATURES / "matchup_features.parquet"
    mt.to_parquet(out, index=False)
    print(f"[build_matchups] wrote {out} shape={mt.shape}")
    print(f"[build_matchups] years: {sorted(mt['year'].dropna().unique().tolist())}")
    print(f"[build_matchups] games_per_year:\n{mt.groupby('year').size().to_string()}")
    nulls = mt.isna().mean().sort_values(ascending=False).head(12)
    print(f"[build_matchups] worst null rates:\n{nulls.to_string()}")


if __name__ == "__main__":
    main()
