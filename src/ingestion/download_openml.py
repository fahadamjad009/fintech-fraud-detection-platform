from __future__ import annotations

import argparse
from pathlib import Path

import openml


DEFAULT_DATA_ID = 42175  # OpenML: CreditCardFraudDetection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-id", type=int, default=DEFAULT_DATA_ID)
    parser.add_argument("--out-dir", type=str, default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = openml.datasets.get_dataset(args.data_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )

    df = X.copy()
    if y is not None:
        df[dataset.default_target_attribute] = y

    out_path = out_dir / "creditcard_openml.csv"
    df.to_csv(out_path, index=False)

    fraud_col = dataset.default_target_attribute
    counts = df[fraud_col].value_counts(dropna=False)
    print("Saved:", out_path.resolve())
    print("Shape:", df.shape)
    print("Class distribution:\n", counts)
    print("Fraud rate:", float(counts.get(1, 0)) / float(len(df)))


if __name__ == "__main__":
    main()
