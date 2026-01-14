from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

# Direct single-file download (more reliable than OpenML MinIO streaming)
DEFAULT_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="data/raw/creditcard.csv")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading:", args.url)
    urllib.request.urlretrieve(args.url, out_path)
    print("Saved:", out_path.resolve())

if __name__ == "__main__":
    main()
