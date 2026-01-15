from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    # Cost of missing a real fraud (false negative)
    cost_fn: float = 500.0
    # Cost of investigating a legit payment flagged as fraud (false positive)
    cost_fp: float = 5.0


REPORTS = Path("reports")
FIGS = REPORTS / "figures"
SWEEP_JSON = REPORTS / "baseline_threshold_sweep.json"
OUT_JSON = REPORTS / "cost_threshold_optimum.json"
OUT_PNG = FIGS / "cost_curve_by_threshold.png"


def load_sweep(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text())
    if "sweep" not in obj:
        raise ValueError("baseline_threshold_sweep.json missing key 'sweep'")
    return pd.DataFrame(obj["sweep"])


def expected_cost(df: pd.DataFrame, cfg: CostConfig) -> pd.DataFrame:
    df = df.copy()
    df["expected_cost"] = df["fp"] * cfg.cost_fp + df["fn"] * cfg.cost_fn
    return df


def pick_best(df: pd.DataFrame) -> pd.Series:
    # Lowest expected cost wins; tiebreaker = higher recall
    df2 = df.sort_values(["expected_cost", "recall"], ascending=[True, False])
    return df2.iloc[0]


def save_outputs(best: pd.Series, cfg: CostConfig) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    payload = {
        "cost_config": {"cost_fn": cfg.cost_fn, "cost_fp": cfg.cost_fp},
        "best_threshold": float(best["threshold"]),
        "best_expected_cost": float(best["expected_cost"]),
        "metrics_at_best": {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {OUT_JSON}")


def plot_cost_curve(df: pd.DataFrame) -> None:
    # Matplotlib is used only for a simple proof plot
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df["threshold"], df["expected_cost"])
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title("Expected Cost vs Threshold (baseline LogReg sweep)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()
    print(f"Saved: {OUT_PNG}")


def main() -> None:
    if not SWEEP_JSON.exists():
        raise FileNotFoundError(f"Missing: {SWEEP_JSON}. Run make_baseline_plots.py first.")

    cfg = CostConfig(cost_fn=500.0, cost_fp=5.0)

    df = load_sweep(SWEEP_JSON)
    df = expected_cost(df, cfg)

    best = pick_best(df)

    print("Cost config:", cfg)
    print(
        "Best threshold:",
        float(best["threshold"]),
        "| expected_cost:",
        float(best["expected_cost"]),
        "| precision:",
        float(best["precision"]),
        "| recall:",
        float(best["recall"]),
        "| TP/FP/FN:",
        int(best["tp"]),
        int(best["fp"]),
        int(best["fn"]),
    )

    save_outputs(best, cfg)
    plot_cost_curve(df)


if __name__ == "__main__":
    main()
