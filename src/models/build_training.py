"""Leave-one-tournament-out train/test parquet splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import DATA_FEATURES, DATA_TRAINING, TRAINING_YEARS

DROP_COL_SUBSTR = ("_source", "_metadata", "_is_derived")
DROP_COLS = {
    "year",
    "game_id",
    "muid",
    "team1",
    "team2",
    "winner",
    "matchup",
    "t1_team_norm",
    "t2_team_norm",
    "result",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric t1_/t2_/delta_ features only; excludes ids and labels."""
    num = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    out: list[str] = []
    for c in num:
        if c in DROP_COLS:
            continue
        if any(s in c for s in DROP_COL_SUBSTR):
            continue
        if not (c.startswith("t1_") or c.startswith("t2_") or c.startswith("delta_")):
            if c not in (
                "projected_tempo",
                "pace_variance_flag",
                "three_pt_reliance_flag",
                "low_tempo_coin_flip",
                "midmajor_matchup",
                "is_bubble_year",
                "t1_is_major_conf",
                "t2_is_major_conf",
                "seed_prior_historical_win_rate",
                "seed_prior_last_5yr_win_rate",
                "seed_prior_sample_size",
            ):
                continue
        out.append(c)
    return sorted(set(out))


def main() -> None:
    path = DATA_FEATURES / "matchup_features.parquet"
    df = pd.read_parquet(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    feats = feature_columns(df)
    DATA_TRAINING.mkdir(parents=True, exist_ok=True)
    flist_path = DATA_TRAINING / "feature_list_v1.txt"
    flist_path.write_text("\n".join(feats) + "\n", encoding="utf-8")
    print(f"[build_training] feature count={len(feats)} list -> {flist_path}")

    for y in TRAINING_YEARS:
        tr = df[df["year"] != y]
        te = df[df["year"] == y]
        tr_out = DATA_TRAINING / f"train_{y}.parquet"
        te_out = DATA_TRAINING / f"test_{y}.parquet"
        tr.to_parquet(tr_out, index=False)
        te.to_parquet(te_out, index=False)
        xtr = tr[feats]
        xte = te[feats]
        print(
            f"[build_training] Y={y} train_rows={len(tr)} test_rows={len(te)} "
            f"X_cols={xtr.shape[1]} train_result_mean={tr['result'].mean():.3f} "
            f"test_result_mean={te['result'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
