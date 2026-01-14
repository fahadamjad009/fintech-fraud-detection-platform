from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


RAW_PATH = Path("data/raw/creditcard.csv")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not RAW_PATH.exists():
        fail(f"Missing file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Required columns
    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        fail(f"Missing required columns. Found: {list(df.columns)}")

    # No missing values
    if df.isna().any().any():
        na_counts = df.isna().sum().sort_values(ascending=False).head(10)
        fail(f"Null values detected. Top NA columns:\n{na_counts}")

    # Type checks
    if not pd.api.types.is_numeric_dtype(df["Time"]):
        fail("Time must be numeric")
    if not pd.api.types.is_numeric_dtype(df["Amount"]):
        fail("Amount must be numeric")

    # Label checks
    vc = df["Class"].value_counts(dropna=False)
    if set(vc.index.tolist()) - {0, 1}:
        fail(f"Unexpected Class values: {vc.index.tolist()}")

    fraud_rate = float(vc.get(1, 0)) / float(len(df))
    print("[OK] Raw dataset validated")
    print("Shape:", df.shape)
    print("Class distribution:\n", vc)
    print("Fraud rate:", fraud_rate)


if __name__ == "__main__":
    main()
