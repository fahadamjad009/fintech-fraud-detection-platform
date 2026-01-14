from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/features/creditcard_features.parquet")
MODEL_PATH = Path("models/baseline_logreg.joblib")
REPORT_PATH = Path("reports/baseline_metrics.json")


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    out = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "threshold": 0.5,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(clf, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(out, indent=2))

    print("Saved model:", MODEL_PATH.resolve())
    print("Saved report:", REPORT_PATH.resolve())
    print("\nPR-AUC:", pr_auc)
    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:", cm)


if __name__ == "__main__":
    main()
