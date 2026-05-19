"""
Business-grade metrics for the fraud detection champion model.

Recreates the stratified 80/20 holdout (random_state=42) used by train_xgb.py,
scores it with the saved XGBoost champion, and computes:
  - Tier 1 (business impact): $-weighted catch rate, fraud $ caught/leaked,
    net financial benefit, loss reduction vs no-model, approval/decline rates.
  - Tier 2 (operations): alert volume, analyst capacity, decile lift table.
  - Tier 3 (model quality): Brier score, score KS separation statistic.
  - Tier 5 (segments): performance by transaction-amount bucket and hour-of-day.

Also emits a 101-point threshold sweep (0.00 -> 1.00) with full business metrics
at each step so the dashboard slider is live without loading the model on Cloud.

Outputs:
  reports/business_metrics.json
  reports/decile_lift_table.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats as scstats
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "creditcard_features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULTS = {
    "cost_fp_usd": 5.0,
    "analyst_cases_per_hour": 12,
    "analyst_daily_capacity_per_fte": 96,
    "default_threshold": 0.5,
    "cost_optimum_threshold": 0.97,
    "high_recall_threshold": 0.10,
}


def reconstruct_holdout() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(FEATURES_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def score_holdout(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    proba = model.predict_proba(X_test)[:, 1]
    df = X_test.copy()
    df["score"] = proba
    df["y_true"] = y_test.values
    return df


def metrics_at_threshold(df: pd.DataFrame, threshold: float, cost_fp: float) -> dict:
    pred = (df["score"] >= threshold).astype(int)
    tp_m = (pred == 1) & (df["y_true"] == 1)
    fp_m = (pred == 1) & (df["y_true"] == 0)
    fn_m = (pred == 0) & (df["y_true"] == 1)
    tn_m = (pred == 0) & (df["y_true"] == 0)

    tp, fp, fn, tn = int(tp_m.sum()), int(fp_m.sum()), int(fn_m.sum()), int(tn_m.sum())

    fraud_amt_total = float(df.loc[df["y_true"] == 1, "Amount"].sum())
    fraud_amt_caught = float(df.loc[tp_m, "Amount"].sum())
    fraud_amt_leaked = float(df.loc[fn_m, "Amount"].sum())
    fraud_amt_alerted = float(df.loc[tp_m | fp_m, "Amount"].sum())

    alert_volume = tp + fp
    total_tx = len(df)
    operational_cost = alert_volume * cost_fp
    net_benefit = fraud_amt_caught - operational_cost

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alert_volume": alert_volume,
        "alert_rate_pct": alert_volume / total_tx * 100,
        "approval_rate_pct": (total_tx - alert_volume) / total_tx * 100,
        "catch_rate_count": tp / max(tp + fn, 1),
        "catch_rate_dollar": fraud_amt_caught / max(fraud_amt_total, 1e-9),
        "precision_count": tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0,
        "precision_dollar": fraud_amt_caught / max(fraud_amt_alerted, 1e-9) if fraud_amt_alerted > 0 else 0.0,
        "fraud_amount_total": fraud_amt_total,
        "fraud_amount_caught": fraud_amt_caught,
        "fraud_amount_leaked": fraud_amt_leaked,
        "operational_cost": operational_cost,
        "net_financial_benefit": net_benefit,
        "loss_reduction_pct": fraud_amt_caught / max(fraud_amt_total, 1e-9) * 100,
    }


def threshold_sweep(df: pd.DataFrame, cost_fp: float, n_steps: int = 101) -> list[dict]:
    """Full sweep from 0.00 -> 1.00 with business metrics at every step.
    Enables a live dashboard slider that recomputes net benefit instantly."""
    thresholds = np.linspace(0.0, 1.0, n_steps)
    return [metrics_at_threshold(df, float(t), cost_fp) for t in thresholds]


def decile_lift_table(df: pd.DataFrame) -> list[dict]:
    d = df.sort_values("score", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, 10, labels=range(1, 11)).astype(int)

    total_fraud = int((d["y_true"] == 1).sum())
    total_fraud_amt = float(d.loc[d["y_true"] == 1, "Amount"].sum())
    overall_rate = total_fraud / len(d)

    rows = []
    cum_fraud, cum_amt = 0, 0.0
    for dec in range(1, 11):
        bucket = d[d["decile"] == dec]
        n = len(bucket)
        f = int((bucket["y_true"] == 1).sum())
        amt = float(bucket.loc[bucket["y_true"] == 1, "Amount"].sum())
        cum_fraud += f
        cum_amt += amt
        rows.append({
            "decile": dec,
            "n_transactions": n,
            "min_score": float(bucket["score"].min()),
            "max_score": float(bucket["score"].max()),
            "fraud_count": f,
            "fraud_amount": amt,
            "pct_of_total_fraud_count": f / max(total_fraud, 1) * 100,
            "pct_of_total_fraud_amount": amt / max(total_fraud_amt, 1e-9) * 100,
            "cumulative_fraud_count_pct": cum_fraud / max(total_fraud, 1) * 100,
            "cumulative_fraud_amount_pct": cum_amt / max(total_fraud_amt, 1e-9) * 100,
            "lift": (f / n) / overall_rate if n > 0 and overall_rate > 0 else 0.0,
        })
    return rows


def segment_by_amount(df: pd.DataFrame, threshold: float) -> list[dict]:
    pred = (df["score"] >= threshold).astype(int)
    d = df.copy()
    d["pred"] = pred
    buckets = [
        ("Small (< $10)", d["Amount"] < 10),
        ("Medium ($10-$100)", (d["Amount"] >= 10) & (d["Amount"] < 100)),
        ("Large ($100-$1k)", (d["Amount"] >= 100) & (d["Amount"] < 1000)),
        ("XL (>= $1k)", d["Amount"] >= 1000),
    ]
    out = []
    for name, mask in buckets:
        sub = d[mask]
        if len(sub) == 0:
            continue
        fraud = int((sub["y_true"] == 1).sum())
        tp = int(((sub["pred"] == 1) & (sub["y_true"] == 1)).sum())
        fp = int(((sub["pred"] == 1) & (sub["y_true"] == 0)).sum())
        fn = int(((sub["pred"] == 0) & (sub["y_true"] == 1)).sum())
        out.append({
            "segment": name,
            "n_transactions": int(len(sub)),
            "fraud_count": fraud,
            "fraud_rate_bps": fraud / len(sub) * 10000,
            "recall_at_threshold": tp / max(tp + fn, 1) if fraud > 0 else None,
            "precision_at_threshold": tp / max(tp + fp, 1) if (tp + fp) > 0 else None,
            "fraud_amount_caught": float(sub.loc[(sub["pred"] == 1) & (sub["y_true"] == 1), "Amount"].sum()),
            "fraud_amount_leaked": float(sub.loc[(sub["pred"] == 0) & (sub["y_true"] == 1), "Amount"].sum()),
        })
    return out


def segment_by_hour(df: pd.DataFrame, threshold: float) -> list[dict]:
    pred = (df["score"] >= threshold).astype(int)
    d = df.copy()
    d["pred"] = pred
    d["hour"] = ((d["Time"] // 3600) % 24).astype(int)
    out = []
    for hr in range(24):
        sub = d[d["hour"] == hr]
        if len(sub) == 0:
            continue
        fraud = int((sub["y_true"] == 1).sum())
        tp = int(((sub["pred"] == 1) & (sub["y_true"] == 1)).sum())
        fn = int(((sub["pred"] == 0) & (sub["y_true"] == 1)).sum())
        out.append({
            "hour": hr,
            "n_transactions": int(len(sub)),
            "fraud_count": fraud,
            "fraud_rate_bps": fraud / len(sub) * 10000,
            "recall_at_threshold": tp / max(tp + fn, 1) if fraud > 0 else None,
        })
    return out


def main() -> None:
    print("Loading XGBoost champion model...")
    model = joblib.load(MODEL_PATH)
    X_test, y_test = reconstruct_holdout()
    df = score_holdout(model, X_test, y_test)

    n = len(df)
    n_fraud = int((df["y_true"] == 1).sum())
    fraud_vol = float(df.loc[df["y_true"] == 1, "Amount"].sum())
    total_vol = float(df["Amount"].sum())
    print(f"Holdout: {n:,} txns | {n_fraud} fraud | ${fraud_vol:,.0f} fraud volume of ${total_vol:,.0f} total")

    brier = float(brier_score_loss(df["y_true"], df["score"]))
    ks_stat, ks_p = scstats.ks_2samp(
        df.loc[df["y_true"] == 1, "score"],
        df.loc[df["y_true"] == 0, "score"],
    )

    thresholds = {
        "high_recall_0.10": DEFAULTS["high_recall_threshold"],
        "default_0.50": DEFAULTS["default_threshold"],
        "cost_optimum_0.97": DEFAULTS["cost_optimum_threshold"],
    }
    threshold_metrics = {
        name: metrics_at_threshold(df, thr, DEFAULTS["cost_fp_usd"])
        for name, thr in thresholds.items()
    }

    print("Computing 101-point threshold sweep with business metrics...")
    full_sweep = threshold_sweep(df, DEFAULTS["cost_fp_usd"], n_steps=101)

    decile_table = decile_lift_table(df)
    seg_amount = segment_by_amount(df, DEFAULTS["cost_optimum_threshold"])
    seg_hour = segment_by_hour(df, DEFAULTS["cost_optimum_threshold"])

    output = {
        "generated_from": "xgb.joblib on stratified 80/20 holdout (random_state=42)",
        "holdout_summary": {
            "n_transactions": n,
            "n_fraud": n_fraud,
            "fraud_rate_pct": (n_fraud / n) * 100,
            "fraud_rate_bps": (n_fraud / n) * 10000,
            "total_volume_usd": total_vol,
            "fraud_volume_usd": fraud_vol,
            "fraud_volume_pct_of_total": fraud_vol / total_vol * 100,
            "time_window_hours_approx": 9.6,
        },
        "model_quality": {
            "brier_score": brier,
            "score_ks_statistic": float(ks_stat),
            "score_ks_p_value": float(ks_p),
        },
        "thresholds": threshold_metrics,
        "threshold_sweep": full_sweep,
        "decile_lift_table": decile_table,
        "segment_by_amount_at_optimum": seg_amount,
        "segment_by_hour_at_optimum": seg_hour,
        "assumptions": DEFAULTS,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    out_json = REPORTS_DIR / "business_metrics.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_json} ({out_json.stat().st_size:,} bytes)")

    pd.DataFrame(decile_table).to_csv(REPORTS_DIR / "decile_lift_table.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'decile_lift_table.csv'}")

    cost_opt = threshold_metrics["cost_optimum_0.97"]
    print("\n=== HEADLINE (cost-optimum threshold 0.97) ===")
    print(f"$-weighted catch rate: {cost_opt['catch_rate_dollar']*100:.1f}%")
    print(f"Fraud $ caught:        ${cost_opt['fraud_amount_caught']:,.0f}")
    print(f"Fraud $ leaked:        ${cost_opt['fraud_amount_leaked']:,.0f}")
    print(f"Alert volume:          {cost_opt['alert_volume']:,} ({cost_opt['alert_rate_pct']:.3f}% of test)")
    print(f"Operational cost:      ${cost_opt['operational_cost']:,.0f}")
    print(f"Net benefit:           ${cost_opt['net_financial_benefit']:,.0f}")

    print(f"\nThreshold sweep: {len(full_sweep)} points covering 0.00 -> 1.00")


if __name__ == "__main__":
    main()
