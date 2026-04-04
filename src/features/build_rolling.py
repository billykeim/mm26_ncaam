"""Rolling pre-game features from team×game log (no same-game leakage)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import DATA_FEATURES, DATA_PROCESSED


def _win_loss_streaks_entering(shifted_result: np.ndarray) -> tuple[list[int], list[int]]:
    """Streaks of completed games before current row (``shifted_result`` = prior game outcomes)."""
    w_streak: list[int] = []
    l_streak: list[int] = []
    cw, cl = 0, 0
    for x in shifted_result:
        if pd.isna(x):
            w_streak.append(0)
            l_streak.append(0)
            continue
        if int(x) == 1:
            cw += 1
            cl = 0
            w_streak.append(cw)
            l_streak.append(0)
        else:
            cl += 1
            cw = 0
            w_streak.append(0)
            l_streak.append(cl)
    return w_streak, l_streak


def _max_win_streak_so_far(results_so_far: np.ndarray) -> list[int]:
    """Max win streak seen in completed games up to each step (``results_so_far`` = prior outcomes)."""
    out: list[int] = []
    best = 0
    run = 0
    for x in results_so_far:
        if pd.isna(x):
            out.append(0)
            continue
        if int(x) == 1:
            run += 1
            best = max(best, run)
        else:
            run = 0
        out.append(best)
    return out


def _games_since_last_loss(shifted_result: np.ndarray) -> list[int]:
    """Completed games since last loss before current game."""
    out: list[int] = []
    since = 0
    for x in shifted_result:
        if pd.isna(x):
            out.append(0)
            continue
        if int(x) == 0:
            since = 0
        else:
            since += 1
        out.append(since)
    return out


def _roll10_margin_trend(margins_shifted: pd.Series) -> float:
    """Slope of margin vs index for up to 10 prior values ending at t-1."""
    v = margins_shifted.dropna().tail(10).to_numpy(dtype=float)
    n = len(v)
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, v, 1)[0])


def _recency_weighted_win_rate_at(
    dates: np.ndarray, results: np.ndarray, idx: int, decay: float = 0.02
) -> float:
    """Decay-weighted win rate over completed games before ``idx``."""
    if idx < 1:
        return float("nan")
    t0 = pd.Timestamp(dates[idx])
    num = 0.0
    den = 0.0
    for j in range(idx):
        if pd.isna(results[j]):
            continue
        days = (t0 - pd.Timestamp(dates[j])).days
        if days <= 0:
            continue
        w = float(np.exp(-decay * float(days)))
        den += w
        num += w * float(results[j])
    return num / den if den > 0 else float("nan")


def enrich_with_barthag(gl: pd.DataFrame, static_barthag: pd.DataFrame) -> pd.DataFrame:
    """Attach season barthag for team and opponent."""
    b = static_barthag.rename(columns={"barthag": "team_barthag_season"})
    out = gl.merge(b, on=["team_norm", "year"], how="left")
    bo = b.rename(
        columns={
            "team_norm": "opp_team_norm",
            "team_barthag_season": "opp_barthag_season",
        }
    )
    out = out.merge(bo, on=["opp_team_norm", "year"], how="left")
    return out


def compute_rolling_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """All rolling columns; each uses ``shift(1)`` before rolling windows."""
    df = game_log.sort_values(["team_norm", "year", "game_date"]).reset_index(drop=True)
    gcols = ["team_norm", "year"]

    df["result_s"] = df.groupby(gcols)["result"].shift(1)
    df["margin_s"] = df.groupby(gcols)["margin"].shift(1)
    df["pts_scored_s"] = df.groupby(gcols)["pts_scored"].shift(1)
    df["pts_allowed_s"] = df.groupby(gcols)["pts_allowed"].shift(1)
    df["efg_s"] = df.groupby(gcols)["efg"].shift(1)
    df["opp_efg_s"] = df.groupby(gcols)["opp_efg"].shift(1)
    df["tov_rate_s"] = df.groupby(gcols)["tov_rate"].shift(1)
    df["opp_barthag_s"] = df.groupby(gcols)["opp_barthag_season"].shift(1)

    # Streaks (entering game)
    w_enter: list[int] = []
    l_enter: list[int] = []
    max_w: list[int] = []
    gs_loss: list[int] = []
    for _, grp in df.groupby(gcols, sort=False):
        rs = grp["result"].shift(1).to_numpy()
        w, l = _win_loss_streaks_entering(rs)
        w_enter.extend(w)
        l_enter.extend(l)
        max_w.extend(_max_win_streak_so_far(rs))
        gs_loss.extend(_games_since_last_loss(rs))
    df["current_win_streak"] = w_enter
    df["current_loss_streak"] = l_enter
    df["max_win_streak_season"] = max_w
    df["games_since_last_loss"] = gs_loss

    df["roll3_win_rate"] = (
        df.groupby(gcols)["result_s"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll5_win_rate"] = (
        df.groupby(gcols)["result_s"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll10_win_rate"] = (
        df.groupby(gcols)["result_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["roll5_avg_margin"] = (
        df.groupby(gcols)["margin_s"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll10_avg_margin"] = (
        df.groupby(gcols)["margin_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    trend_vals: list[float] = []
    for _, grp in df.groupby(gcols, sort=False):
        ms = grp["margin_s"]
        for i in range(len(ms)):
            trend_vals.append(_roll10_margin_trend(ms.iloc[: i + 1]))
    df["roll10_margin_trend"] = trend_vals

    df["roll10_pts_scored"] = (
        df.groupby(gcols)["pts_scored_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll10_pts_allowed"] = (
        df.groupby(gcols)["pts_allowed_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll10_off_efg"] = (
        df.groupby(gcols)["efg_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll10_def_efg"] = (
        df.groupby(gcols)["opp_efg_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["roll10_tov_rate"] = (
        df.groupby(gcols)["tov_rate_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["roll10_opp_barthag"] = (
        df.groupby(gcols)["opp_barthag_s"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["sos_adj_roll10_win_rate"] = df["roll10_win_rate"] * df["roll10_opp_barthag"]

    q1_rates: list[float] = []
    for _, grp in df.groupby(gcols, sort=False):
        ob = grp["opp_barthag_season"].to_numpy()
        res = grp["result"].to_numpy()
        for i in range(len(grp)):
            prior_ob = ob[:i]
            prior_res = res[:i]
            m = prior_ob > 0.85
            if not m.any():
                q1_rates.append(float("nan"))
            else:
                sel = prior_res[m][-10:]
                q1_rates.append(float(np.mean(sel)))
    df["roll10_q1_win_rate"] = q1_rates

    recency: list[float] = []
    for _, grp in df.groupby(gcols, sort=False):
        d = grp["game_date"].to_numpy()
        r = grp["result"].to_numpy()
        for i in range(len(grp)):
            recency.append(_recency_weighted_win_rate_at(d, r, i))
    df["recency_wtd_win_rate"] = recency

    late_l: list[float] = []
    early_l: list[float] = []
    delta_l: list[float] = []
    for _, grp in df.groupby(gcols, sort=False):
        sp = grp["season_pct_elapsed"].to_numpy()
        gn = grp["game_num"].to_numpy()
        res = grp["result"].to_numpy()
        for i in range(len(grp)):
            late_vals = [float(res[j]) for j in range(i) if sp[j] > 0.60]
            early_vals = [float(res[j]) for j in range(i) if gn[j] <= 8]
            late = float(np.mean(late_vals)) if late_vals else float("nan")
            early = float(np.mean(early_vals)) if early_vals else float("nan")
            dlt = (
                late - early
                if not (np.isnan(late) or np.isnan(early))
                else float("nan")
            )
            late_l.append(late)
            early_l.append(early)
            delta_l.append(dlt)
    df["late_season_win_rate"] = late_l
    df["early_season_win_rate"] = early_l
    df["early_vs_late_win_delta"] = delta_l

    df["days_since_last_game"] = (
        df.groupby(gcols)["game_date"].diff().dt.days.fillna(0).astype(float)
    )

    return df


def _print_sample(df: pd.DataFrame, team: str, year: int) -> None:
    sub = df[(df["team_norm"] == team) & (df["year"] == year)].tail(5)
    cols = [
        "game_date",
        "game_num",
        "result",
        "roll10_win_rate",
        "margin",
        "recency_wtd_win_rate",
    ]
    cols = [c for c in cols if c in df.columns]
    print(f"[build_rolling] sample {team} {year}:\n{sub[cols].to_string()}")


def main() -> None:
    gl_path = DATA_PROCESSED / "game_log.parquet"
    if not gl_path.exists():
        raise FileNotFoundError(f"Missing {gl_path}; run build_game_log first.")
    gl = pd.read_parquet(gl_path)
    gl["game_date"] = pd.to_datetime(gl["game_date"])

    static = pd.read_parquet(
        DATA_FEATURES / "static_features.parquet",
        columns=["team_norm", "year", "barthag"],
    )
    static = static.drop_duplicates(subset=["team_norm", "year"], keep="first")
    gl = enrich_with_barthag(gl, static)

    out_df = compute_rolling_features(gl)

    key_cols = [
        "roll10_win_rate",
        "roll10_opp_barthag",
        "recency_wtd_win_rate",
        "roll10_margin_trend",
    ]
    print(f"[build_rolling] shape={out_df.shape}")
    for c in key_cols:
        if c in out_df.columns:
            print(f"[build_rolling] null_rate {c}={out_df[c].isna().mean():.4f}")

    _print_sample(out_df, "Kansas", 2019)
    _print_sample(out_df, "Duke", 2015)

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    out_path = DATA_FEATURES / "rolling_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"[build_rolling] wrote {out_path} rows={len(out_df):,}")


if __name__ == "__main__":
    main()
