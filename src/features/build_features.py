from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


IN_PATH = Path("data/processed/creditcard.parquet")
OUT_PATH = Path("data/features/creditcard_features.parquet")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH)

    # Simple, reproducible engineered feature
    df["Amount_log"] = df["Amount"].clip(lower=0).apply(math.log1p)

    # Features: V1..V28 + Time + Amount + Amount_log
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Time", "Amount", "Amount_log"]

    out = df[feature_cols + ["Class"]].copy()
    out.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH.resolve())
    print("Shape:", out.shape)
    print("Columns:", out.columns.tolist()[:10], "... total:", len(out.columns))


if __name__ == "__main__":
    main()
