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
    SELECTION_SUNDAY_DATES,
    TOURNAMENT_RESULTS_PATH,
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
        "t1_score",
        "t2_score",
    }
)
from src.utils.name_normalize import (
    load_team_name_map,
    school_to_canonical,
    torvik_to_canonical,
)

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
        "score",
    }
)

SEED_PAIR_PATH = DATA_PROCESSED / "tournament_analytics" / "seed_pair_win_rates.parquet"


def apply_random_t1_t2_swap(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Swap team1/team2 (and aligned norms, seeds, result) on ~50% of rows for label balance.

    Uses a fixed-size random subset (exactly ``n//2`` rows) so the swap mask is not
    correlated with row order / outcome (avoids biased ``result`` mean vs i.i.d. Bernoulli).
    """
    out = df.copy()
    rng = np.random.RandomState(seed)
    n = len(out)
    if n == 0:
        return out
    perm = rng.permutation(n)
    w = np.zeros(n, dtype=bool)
    w[perm[: n // 2]] = True
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
    if "t1_score" in out.columns and "t2_score" in out.columns:
        sc1 = out.loc[w, "t1_score"].copy()
        out.loc[w, "t1_score"] = out.loc[w, "t2_score"].values
        out.loc[w, "t2_score"] = sc1.values
    out.loc[w, "result"] = (1 - out.loc[w, "result"].astype(int)).astype(np.int8)
    return out


def load_coach_store_agg(mapping: dict) -> pd.DataFrame:
    """Coach counts keyed by ``(team_norm, year)`` (same aggregation as ``build_static``)."""
    coach_path = DATA_PROCESSED / "coach_store.parquet"
    if not coach_path.exists():
        print(f"[build_matchups] WARN: {coach_path} missing; skipping coach supplement merge")
        return pd.DataFrame()
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
    return cs_agg.drop_duplicates(subset=["team_norm", "year"], keep="first")


def load_player_aggregates_norm(mapping: dict) -> pd.DataFrame:
    """Player aggregate rows with ``team_norm`` (Torvik names → canonical)."""
    pa_path = DATA_PROCESSED / "player_aggregates.parquet"
    if not pa_path.exists():
        print(f"[build_matchups] WARN: {pa_path} missing; skipping player supplement merge")
        return pd.DataFrame()
    pa = pd.read_parquet(pa_path)
    pa["team_norm"] = pa["team"].map(lambda x: torvik_to_canonical(str(x), mapping))
    return pa.drop_duplicates(subset=["team_norm", "year"], keep="first")


def merge_supplement_prefixed(
    mt: pd.DataFrame, tab: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """Left-merge columns from ``tab`` that are not already on ``mt`` (after ``t1_/t2_`` prefix)."""
    if tab.empty:
        return mt
    pre = _prefix_static(tab, prefix)
    merge_on = [f"{prefix}_team_norm", "year"]
    add = [c for c in pre.columns if c not in mt.columns and c not in merge_on]
    if not add:
        return mt
    return mt.merge(pre[merge_on + add], on=merge_on, how="left")


def load_tournament_base() -> pd.DataFrame:
    """
    Full NCAA bracket from Sports-Reference ``tournament_results.parquet``.

    Rows already have ``t1_*`` / ``t2_*`` with better seed as ``t1``; ``round`` 0–6.
    """
    if not TOURNAMENT_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Tournament results missing: {TOURNAMENT_RESULTS_PATH}. "
            "Run: python -m src.data.ingest_tournament_results"
        )
    df = pd.read_parquet(TOURNAMENT_RESULTS_PATH)
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype(int)
    df["t1_team_norm"] = df["t1_team_norm"].astype(str)
    df["t2_team_norm"] = df["t2_team_norm"].astype(str)
    df["t1_seed"] = pd.to_numeric(df["t1_seed"], errors="coerce").astype(int)
    df["t2_seed"] = pd.to_numeric(df["t2_seed"], errors="coerce").astype(int)
    df["result"] = pd.to_numeric(df["result"], errors="coerce").astype(np.int8)
    df["game_id"] = df["game_id"].astype(str)
    df["team1"] = df["t1_team_norm"]
    df["team2"] = df["t2_team_norm"]
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
    coach_agg = load_coach_store_agg(mapping)
    pa_tab = load_player_aggregates_norm(mapping)

    mt = base.merge(
        _prefix_static(static, "t1"),
        on=["t1_team_norm", "year"],
        how="left",
    )
    mt = mt.merge(_prefix_static(static, "t2"), on=["t2_team_norm", "year"], how="left")

    mt = merge_supplement_prefixed(mt, coach_agg, "t1")
    mt = merge_supplement_prefixed(mt, coach_agg, "t2")
    mt = merge_supplement_prefixed(mt, pa_tab, "t1")
    mt = merge_supplement_prefixed(mt, pa_tab, "t2")

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


def print_round_distribution(mt: pd.DataFrame) -> None:
    """Log ``round`` counts across all tournament rows (for experiment log)."""
    if "round" not in mt.columns:
        return
    print("[build_matchups] round distribution (all years):")
    print(mt.groupby("round").size().to_string())


def print_2025_seed_prior_ranking(mt: pd.DataFrame) -> None:
    """
    Sanity: rank 2025 teams by mean pooled seed-prior P(win) when listed as t1 / 1−P as t2.

    Uses ``historical_win_rate`` after seed-pair join (not a fitted model).
    """
    sub = mt[mt["year"] == 2025]
    if sub.empty or "historical_win_rate" not in sub.columns:
        print("[build_matchups] 2025 seed-prior sanity: no rows or missing historical_win_rate")
        return
    rows: list[tuple[str, float]] = []
    for _, r in sub.iterrows():
        p = pd.to_numeric(r["historical_win_rate"], errors="coerce")
        pv = float(p) if pd.notna(p) else 0.5
        rows.append((str(r["t1_team_norm"]), pv))
        rows.append((str(r["t2_team_norm"]), 1.0 - pv))
    s = (
        pd.DataFrame(rows, columns=["team", "p"])
        .groupby("team", as_index=False)["p"]
        .mean()
        .sort_values("p", ascending=False)
    )
    print("[build_matchups] 2025 — top 5 teams by mean seed-prior win proxy:")
    print(s.head(5).to_string(index=False))
    print("[build_matchups] 2025 — bottom 5 teams by mean seed-prior win proxy:")
    print(s.tail(5).to_string(index=False))


def main() -> None:
    mt = build_matchups()
    print_round_distribution(mt)
    print_2025_seed_prior_ranking(mt)
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    out = DATA_FEATURES / "matchup_features.parquet"
    mt.to_parquet(out, index=False)
    print(f"[build_matchups] wrote {out} shape={mt.shape}")
    print(f"[build_matchups] years: {sorted(mt['year'].dropna().unique().tolist())}")
    print(f"[build_matchups] games_per_year:\n{mt.groupby('year').size().to_string()}")
    ds = float(mt["delta_seed"].isna().mean()) if "delta_seed" in mt.columns else float("nan")
    print(f"[build_matchups] delta_seed null rate={ds:.4f}")
    nulls = mt.isna().mean().sort_values(ascending=False).head(12)
    print(f"[build_matchups] worst null rates:\n{nulls.to_string()}")


if __name__ == "__main__":
    main()
