from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/features/creditcard_features.parquet")
MODEL_PATH = Path("models/baseline_logreg.joblib")

FIG_DIR = Path("reports/figures")
HTML_DIR = Path("reports/interactive")
OUT_JSON = Path("reports/baseline_threshold_sweep.json")


def save_both(fig: go.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    fig.write_html(HTML_DIR / f"{stem}.html", include_plotlyjs="cdn")
    fig.write_image(FIG_DIR / f"{stem}.png", scale=2)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Recreate the same split used in training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = joblib.load(MODEL_PATH)
    y_score = clf.predict_proba(X_test)[:, 1]

    # PR + ROC curves
    precision, recall, pr_thresh = precision_recall_curve(y_test, y_score)
    fpr, tpr, roc_thresh = roc_curve(y_test, y_score)

    pr_auc = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
    pr_fig.update_layout(
        title=f"Precision-Recall Curve (PR-AUC={pr_auc:.4f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    save_both(pr_fig, "baseline_pr_curve")

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
    roc_fig.update_layout(
        title=f"ROC Curve (ROC-AUC={roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    save_both(roc_fig, "baseline_roc_curve")

    # Score distribution
    score_df = pd.DataFrame({"score": y_score, "Class": y_test.values})
    hist_fig = px.histogram(
        score_df,
        x="score",
        color="Class",
        nbins=100,
        barmode="overlay",
        title="Model Score Distribution by Class",
    )
    hist_fig.update_layout(xaxis_title="Predicted Fraud Probability", yaxis_title="Count")
    save_both(hist_fig, "baseline_score_distribution")

    # Threshold sweep: precision/recall at multiple thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_test.values == 1)).sum())
        fp = int(((pred == 1) & (y_test.values == 0)).sum())
        fn = int(((pred == 0) & (y_test.values == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({"threshold": float(t), "precision": float(prec), "recall": float(rec), "tp": tp, "fp": fp, "fn": fn})

    sweep = pd.DataFrame(rows)

    sweep_fig = go.Figure()
    sweep_fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], mode="lines", name="Precision"))
    sweep_fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], mode="lines", name="Recall"))
    sweep_fig.update_layout(
        title="Threshold Sweep (Precision & Recall vs Threshold)",
        xaxis_title="Threshold",
        yaxis_title="Metric value",
    )
    save_both(sweep_fig, "baseline_threshold_sweep")

    OUT_JSON.write_text(json.dumps({"pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "sweep": rows}, indent=2))
    print("Saved plots to:", FIG_DIR.resolve(), "and", HTML_DIR.resolve())
    print("Saved threshold sweep:", OUT_JSON.resolve())


if __name__ == "__main__":
    main()
