"""XGBoost v1 game-level model with LOO-CV and full-data export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.utils.constants import (
    BASELINE_LR_PREDICTIONS_PATH,
    DATA_FEATURES,
    DATA_TRAINING,
    FEATURE_IMPORTANCE_LOG_DIR,
    OUTPUTS_MODELS,
    OUTPUTS_PREDICTIONS,
    TRAINING_YEARS,
    XGB_V1_FULL_MODEL_PATH,
    XGB_V1_PREDICTIONS_PATH,
)


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def load_feature_columns() -> list[str]:
    """Feature names aligned with build_training (numeric model inputs)."""
    path = DATA_TRAINING / "feature_list_v1.txt"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Subset to columns present in df; return X, y."""
    use = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(use)
    if missing:
        raise KeyError(f"Missing expected feature columns: {sorted(missing)[:10]}...")
    x = df[use]
    y = df["result"].to_numpy(dtype=np.int64)
    return x, y


def load_loo_split(test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_parquet(DATA_TRAINING / f"train_{test_year}.parquet")
    te = pd.read_parquet(DATA_TRAINING / f"test_{test_year}.parquet")
    return tr, te


def make_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )


def fold_importance_df(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Gain-based importance aligned to feature order."""
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    gains = [float(scores.get(f"f{i}", 0.0)) for i in range(len(feature_names))]
    out = pd.DataFrame({"feature": feature_names, "gain": gains})
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def baseline_metrics_by_year() -> dict[int, float]:
    """Per-fold accuracy from saved seed-only predictions."""
    b = pd.read_parquet(BASELINE_LR_PREDICTIONS_PATH)
    acc: dict[int, float] = {}
    for y, g in b.groupby("test_year"):
        acc[int(y)] = float((g["predicted_winner"] == g["result"]).mean())
    return acc


def main() -> None:
    OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_columns()
    fold_rows: list[dict[str, float]] = []
    pred_parts: list[pd.DataFrame] = []
    baseline_acc = baseline_metrics_by_year()

    for y in TRAINING_YEARS:
        tr, te = load_loo_split(y)
        x_tr, y_tr = prepare_xy(tr, feature_cols)
        x_te, y_te = prepare_xy(te, feature_cols)
        cols = list(x_tr.columns)

        clf = make_classifier()
        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_te)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)

        m = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "log_loss": float(log_loss(y_te, proba, labels=[0, 1])),
            "brier": float(brier_score_loss(y_te, proba)),
            "auc_roc": _safe_auc(y_te, proba),
        }
        fold_rows.append(m)

        imp = fold_importance_df(clf, cols)
        imp.to_csv(FEATURE_IMPORTANCE_LOG_DIR / f"{y}_importance.csv", index=False)

        part = te[["year", "round", "t1_team_norm", "t2_team_norm", "result"]].copy()
        part["xgb_raw_prob_t1"] = proba
        part["predicted_winner"] = pred
        part["test_year"] = y
        pred_parts.append(part)

        b_acc = baseline_acc.get(y, float("nan"))
        d = (m["accuracy"] - b_acc) * 100
        print(
            f"[xgb_v1] test_year={y} acc={m['accuracy']:.4f} "
            f"log_loss={m['log_loss']:.4f} brier={m['brier']:.4f} auc={m['auc_roc']:.4f} "
            f"vs_baseline_acc_delta={d:+.2f}%"
        )

    fm = pd.DataFrame(fold_rows)
    mean = fm.mean(numeric_only=True)
    std = fm.std(numeric_only=True)
    auc_mean = float(np.nanmean(fm["auc_roc"]))
    auc_std = float(np.nanstd(fm["auc_roc"]))

    b_list = [baseline_acc[y] for y in TRAINING_YEARS]
    b_mean = float(np.mean(b_list))
    delta_acc_pct = (mean["accuracy"] - b_mean) * 100

    print(
        "[xgb_v1] mean±std "
        f"acc={mean['accuracy']:.4f}±{std['accuracy']:.4f} "
        f"log_loss={mean['log_loss']:.4f}±{std['log_loss']:.4f} "
        f"brier={mean['brier']:.4f}±{std['brier']:.4f} "
        f"auc={auc_mean:.4f}±{auc_std:.4f}"
    )
    print(
        f"[xgb_v1] baseline_lr mean acc={b_mean:.4f} | "
        f"xgb mean acc={mean['accuracy']:.4f} | "
        f"delta={delta_acc_pct:+.2f}%"
    )

    stacked = pd.concat(pred_parts, ignore_index=True)
    stacked.to_parquet(XGB_V1_PREDICTIONS_PATH, index=False)
    print(f"[xgb_v1] wrote {len(stacked)} rows -> {XGB_V1_PREDICTIONS_PATH}")

    full = pd.read_parquet(DATA_FEATURES / "matchup_features.parquet")
    full = full[full["year"].isin(TRAINING_YEARS)]
    x_full, y_full = prepare_xy(full, feature_cols)
    full_clf = make_classifier()
    full_clf.fit(x_full, y_full)
    full_clf.save_model(str(XGB_V1_FULL_MODEL_PATH))
    meta = {
        "feature_columns": list(x_full.columns),
        "n_rows": int(len(full)),
        "training_years": TRAINING_YEARS,
    }
    (OUTPUTS_MODELS / "xgb_v1_full_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[xgb_v1] full model -> {XGB_V1_FULL_MODEL_PATH}")


if __name__ == "__main__":
    main()
