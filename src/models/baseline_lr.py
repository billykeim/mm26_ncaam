"""Seed-only logistic regression baseline (LOO-CV)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import (
    BASELINE_LR_PREDICTIONS_PATH,
    DATA_TRAINING,
    OUTPUTS_PREDICTIONS,
    TRAINING_YEARS,
)

FEATURE_COL = "delta_seed"


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return ROC-AUC or NaN if only one class."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def load_loo_split(test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test parquet for one LOO fold (test = held-out tournament year)."""
    tr = pd.read_parquet(DATA_TRAINING / f"train_{test_year}.parquet")
    te = pd.read_parquet(DATA_TRAINING / f"test_{test_year}.parquet")
    return tr, te


def evaluate_fold(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit seed-only LR on train; return metrics dict, y_test, proba_test."""
    x_tr = train_df[[FEATURE_COL]].to_numpy(dtype=np.float64)
    y_tr = train_df["result"].to_numpy(dtype=np.int64)
    x_te = test_df[[FEATURE_COL]].to_numpy(dtype=np.float64)
    y_te = test_df["result"].to_numpy(dtype=np.int64)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_tr, y_tr)
    proba = clf.predict_proba(x_te)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "log_loss": float(log_loss(y_te, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_te, proba)),
        "auc_roc": _safe_auc(y_te, proba),
    }
    return metrics, y_te, proba


def main() -> None:
    """Run 16-fold LOO CV, print metrics, save stacked test predictions."""
    OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[dict[str, float]] = []
    all_parts: list[pd.DataFrame] = []

    for y in TRAINING_YEARS:
        tr, te = load_loo_split(y)
        m, _, proba = evaluate_fold(tr, te)
        fold_metrics.append(m)
        print(
            f"[baseline_lr] test_year={y} acc={m['accuracy']:.4f} "
            f"log_loss={m['log_loss']:.4f} brier={m['brier']:.4f} auc={m['auc_roc']:.4f}"
        )

        part = te[
            ["year", "round", "t1_team_norm", "t2_team_norm", "result"]
        ].copy()
        part["predicted_prob_t1"] = proba
        part["predicted_winner"] = (proba >= 0.5).astype(np.int64)
        part["test_year"] = y
        all_parts.append(part)

    fm = pd.DataFrame(fold_metrics)
    mean = fm.mean(numeric_only=True)
    std = fm.std(numeric_only=True)
    auc_mean = float(np.nanmean(fm["auc_roc"]))
    auc_std = float(np.nanstd(fm["auc_roc"]))
    print(
        "[baseline_lr] mean±std "
        f"acc={mean['accuracy']:.4f}±{std['accuracy']:.4f} "
        f"log_loss={mean['log_loss']:.4f}±{std['log_loss']:.4f} "
        f"brier={mean['brier']:.4f}±{std['brier']:.4f} "
        f"auc={auc_mean:.4f}±{auc_std:.4f}"
    )

    out = pd.concat(all_parts, ignore_index=True)
    out.to_parquet(BASELINE_LR_PREDICTIONS_PATH, index=False)
    print(f"[baseline_lr] wrote {len(out)} rows -> {BASELINE_LR_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
