from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost is not installed. Run: pip install xgboost") from e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "creditcard_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"Missing features file: {FEATURES_PATH}. Run: python src/features/build_features.py")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    if "Class" not in df.columns:
        raise SystemExit("Expected 'Class' column in features parquet.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Use stratified split because fraud is rare
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class imbalance handling
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, proba))
    roc_auc = float(roc_auc_score(y_test, proba))

    # default threshold 0.5
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds).tolist()

    report = {
        "model": "xgboost",
        "features_path": str(FEATURES_PATH),
        "test_size": 0.2,
        "random_state": 42,
        "scale_pos_weight": float(scale_pos_weight),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix_threshold_0_5": cm,
        "threshold": 0.5,
        "n_test": int(len(y_test)),
        "positives_test": int(y_test.sum()),
    }

    model_path = MODELS_DIR / "xgb.joblib"
    report_path = REPORTS_DIR / "xgb_metrics.json"

    joblib.dump(model, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"PR-AUC: {pr_auc}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Confusion Matrix @0.5: {cm}")


if __name__ == "__main__":
    main()
