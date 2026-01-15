import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BASELINE_PATH = REPORTS_DIR / "baseline_metrics.json"
XGB_PATH = REPORTS_DIR / "xgb_metrics.json"


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    baseline = load_metrics(BASELINE_PATH)
    xgb = load_metrics(XGB_PATH)

    df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "PR-AUC": baseline["pr_auc"],
                "ROC-AUC": baseline["roc_auc"],
            },
            {
                "Model": "XGBoost",
                "PR-AUC": xgb["pr_auc"],
                "ROC-AUC": xgb["roc_auc"],
            },
        ]
    )

    print(df)

    # Bar plot comparison
    ax = df.set_index("Model")[["PR-AUC", "ROC-AUC"]].plot(
        kind="bar",
        figsize=(8, 5),
        rot=0,
        title="Model Performance Comparison",
    )
    ax.set_ylabel("Score")
    plt.tight_layout()

    out_path = FIGURES_DIR / "model_comparison_auc.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved comparison plot: {out_path}")


if __name__ == "__main__":
    main()
