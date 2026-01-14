from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/creditcard.csv")
OUT_PATH = Path("data/processed/creditcard.parquet")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df["Class"] = df["Class"].astype("int8")

    df.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH.resolve())
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
