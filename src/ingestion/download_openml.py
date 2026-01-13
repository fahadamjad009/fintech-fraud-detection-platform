from __future__ import annotations

import argparse
import time
from pathlib import Path

import openml


DEFAULT_DATA_ID = 42175  # OpenML: CreditCardFraudDetection


def fetch_with_retries(data_id: int, retries: int = 5, sleep_seconds: int = 10):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            ds = openml.datasets.get_dataset(data_id)
            X, y, _, _ = ds.get_data(
                target=ds.default_target_attribute,
                dataset_format="dataframe",
            )
            return ds, X, y
        except Exception as e:
            last_err = e
            print(f"[Attempt {attempt}/{retries}] Download failed: {type(e).__name__}: {e}")
            if attempt < retries:
                print(f"Retrying in {sleep_seconds}s...\n")
                time.sleep(sleep_seconds)
    raise last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-id", type=int, default=DEFAULT_DATA_ID)
    parser.add_argument("--out-dir", type=str, default="data/raw")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, X, y = fetch_with_retries(args.data_id, retries=args.retries, sleep_seconds=args.sleep)

    df = X.copy()
    if y is not None:
        df[dataset.default_target_attribute] = y

    out_path = out_dir / "creditcard_openml.csv"
    df.to_csv(out_path, index=False)

    fraud_col = dataset.default_target_attribute
    counts = df[fraud_col].value_counts(dropna=False)
    print("\nSaved:", out_path.resolve())
    print("Shape:", df.shape)
    print("Class distribution:\n", counts)
    print("Fraud rate:", float(counts.get(1, 0)) / float(len(df)))


if __name__ == "__main__":
    main()
