"""
Data Drift Detection (PSI + KS)
------------------------------
Simulates production monitoring by comparing:
- Reference (training-like) data
- Current (incoming-like) data (simulated by perturbation)

Outputs:
- PSI (Population Stability Index)
- KS statistic + p-value
- Drift flags
- CSV + JSON reports

FinTech-style monitoring: detect when feature distributions shift.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ------------------------
# Config
# ------------------------
DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("monitoring/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATA = DATA_DIR / "creditcard.parquet"
CURRENT_DATA = DATA_DIR / "creditcard.parquet"  # simulated from same source

# Monitor the same features your baseline model expects:
# V1..V28 + Time + Amount + Amount_log  => 31 features
FEATURES_TO_MONITOR = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Amount_log"]

N_BINS = 10

# Drift thresholds (common rough defaults)
PSI_DRIFT_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.05

# To keep runtime reasonable, sample rows (set None to use full dataset)
SAMPLE_N = 50_000

RNG = np.random.default_rng(42)


# ------------------------
# Helpers
# ------------------------
def ensure_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Amount_log exists and is consistent with Amount."""
    df = df.copy()
    if "Amount" not in df.columns:
        raise ValueError("Expected column 'Amount' not found in dataframe.")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    return df


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two numeric distributions using reference-based bin edges."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    ref_hist, bin_edges = np.histogram(ref, bins=bins)
    cur_hist, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_hist / (np.sum(ref_hist) + 1e-12)
    cur_pct = cur_hist / (np.sum(cur_hist) + 1e-12)

    eps = 1e-6
    psi = np.sum((ref_pct - cur_pct) * np.log((ref_pct + eps) / (cur_pct + eps)))
    return float(psi)


def maybe_sample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


# ------------------------
# Load Data
# ------------------------
print("Loading reference data...")
df_ref = pd.read_parquet(REFERENCE_DATA)

print("Loading current data (simulated)...")
df_cur = pd.read_parquet(CURRENT_DATA)

# Optional sampling for speed
df_ref = maybe_sample(df_ref, SAMPLE_N)
df_cur = maybe_sample(df_cur, SAMPLE_N)

# Simulate drift (DO THIS BEFORE Amount_log is computed)
df_cur = df_cur.copy()

# Drift simulation: Amount tends to inflate + more variance
df_cur["Amount"] = pd.to_numeric(df_cur["Amount"], errors="coerce")
df_cur["Amount"] = df_cur["Amount"] * RNG.normal(loc=1.05, scale=0.10, size=len(df_cur))

# Optional: simulate subtle drift on a couple PCA features (comment out if you want)
for v in ["V1", "V2"]:
    if v in df_cur.columns:
        df_cur[v] = df_cur[v] + RNG.normal(loc=0.05, scale=0.20, size=len(df_cur))

# Now compute Amount_log consistently on both
df_ref = ensure_amount_log(df_ref)
df_cur = ensure_amount_log(df_cur)

# Safety: ensure required features exist
missing = [c for c in FEATURES_TO_MONITOR if c not in df_ref.columns or c not in df_cur.columns]
if missing:
    raise ValueError(f"Missing required columns for drift monitoring: {missing}")


# ------------------------
# Drift Calculation
# ------------------------
results: list[dict] = []

for feature in FEATURES_TO_MONITOR:
    ref_vals = df_ref[feature].to_numpy()
    cur_vals = df_cur[feature].to_numpy()

    psi = population_stability_index(ref_vals, cur_vals, bins=N_BINS)
    ks_stat, ks_p = ks_2samp(ref_vals[~np.isnan(ref_vals)], cur_vals[~np.isnan(cur_vals)])

    drift_flag = (not np.isnan(psi) and psi > PSI_DRIFT_THRESHOLD) or (ks_p < KS_PVALUE_THRESHOLD)

    results.append(
        {
            "feature": feature,
            "psi": float(psi) if not np.isnan(psi) else None,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "drift_flag": bool(drift_flag),
        }
    )

df_report = pd.DataFrame(results).sort_values(["drift_flag", "psi"], ascending=[False, False])


# ------------------------
# Save Outputs
# ------------------------
csv_path = REPORTS_DIR / "data_drift_report.csv"
json_path = REPORTS_DIR / "data_drift_report.json"

df_report.to_csv(csv_path, index=False)
json_path.write_text(json.dumps(results, indent=2))

print("\nDrift report saved:")
print(f" - {csv_path}")
print(f" - {json_path}")

print("\nTop drifted features:")
print(df_report.head(15).to_string(index=False))
