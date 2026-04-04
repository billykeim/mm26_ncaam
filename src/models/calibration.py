"""Platt scaling on XGBoost raw probabilities (LOO-CV)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.models.xgboost_model import (
    load_feature_columns,
    load_loo_split,
    make_classifier,
    prepare_xy,
)
from src.utils.constants import (
    CALIBRATION_PLOT_PATH,
    DATA_OUTPUTS,
    OUTPUTS_PREDICTIONS,
    TRAINING_YEARS,
    XGB_V1_CALIBRATED_PREDICTIONS_PATH,
)


def _reliability_plot(
    y: np.ndarray, p: np.ndarray, out_path: Path, n_bins: int = 10
) -> None:
    """Decile bins on predicted probability; plot mean p vs empirical win rate."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    xs: list[float] = []
    ys: list[float] = []
    for b in range(n_bins):
        m = bin_ids == b
        if not np.any(m):
            xs.append((bins[b] + bins[b + 1]) / 2)
            ys.append(float("nan"))
            continue
        xs.append(float(np.mean(p[m])))
        ys.append(float(np.mean(y[m])))

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(xs, ys, "o-", color="C0", label="Calibrated (LOO test)")
    ax.set_xlabel("Mean predicted P(t1 wins)")
    ax.set_ylabel("Actual win rate (t1)")
    ax.set_title("Reliability diagram — XGB + Platt (pooled LOO test)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_columns()
    fold_before: list[dict[str, float]] = []
    fold_after: list[dict[str, float]] = []
    pred_parts: list[pd.DataFrame] = []
    all_y: list[np.ndarray] = []
    all_p_cal: list[np.ndarray] = []

    for y in TRAINING_YEARS:
        tr, te = load_loo_split(y)
        x_tr, y_tr = prepare_xy(tr, feature_cols)
        x_te, y_te = prepare_xy(te, feature_cols)

        clf = make_classifier()
        clf.fit(x_tr, y_tr)
        raw_tr = clf.predict_proba(x_tr)[:, 1]
        raw_te = clf.predict_proba(x_te)[:, 1]

        platt = LogisticRegression(max_iter=1000, random_state=42)
        platt.fit(raw_tr.reshape(-1, 1), y_tr)
        cal_te = platt.predict_proba(raw_te.reshape(-1, 1))[:, 1]

        b_before = float(brier_score_loss(y_te, raw_te))
        b_after = float(brier_score_loss(y_te, cal_te))
        ll_before = float(log_loss(y_te, raw_te, labels=[0, 1]))
        ll_after = float(log_loss(y_te, cal_te, labels=[0, 1]))

        fold_before.append({"brier": b_before, "log_loss": ll_before})
        fold_after.append({"brier": b_after, "log_loss": ll_after})

        print(
            f"[calibration] test_year={y} brier {b_before:.4f} -> {b_after:.4f} | "
            f"log_loss {ll_before:.4f} -> {ll_after:.4f}"
        )

        part = te[["year", "round", "t1_team_norm", "t2_team_norm", "result"]].copy()
        part["xgb_raw_prob_t1"] = raw_te
        part["calibrated_prob_t1"] = cal_te
        part["predicted_winner"] = (cal_te >= 0.5).astype(np.int64)
        part["test_year"] = y
        pred_parts.append(part)

        all_y.append(y_te)
        all_p_cal.append(cal_te)

    bf = pd.DataFrame(fold_before)
    af = pd.DataFrame(fold_after)
    print(
        "[calibration] mean Brier "
        f"before={bf['brier'].mean():.4f} after={af['brier'].mean():.4f}"
    )
    print(
        "[calibration] mean log_loss "
        f"before={bf['log_loss'].mean():.4f} after={af['log_loss'].mean():.4f}"
    )

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p_cal)
    _reliability_plot(y_all, p_all, CALIBRATION_PLOT_PATH)
    print(f"[calibration] plot -> {CALIBRATION_PLOT_PATH}")

    stacked = pd.concat(pred_parts, ignore_index=True)
    stacked.to_parquet(XGB_V1_CALIBRATED_PREDICTIONS_PATH, index=False)
    print(
        f"[calibration] wrote {len(stacked)} rows -> {XGB_V1_CALIBRATED_PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()
