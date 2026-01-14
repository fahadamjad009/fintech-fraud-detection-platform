\# Data Contract — Credit Card Transactions (Bronze)



\## Source

\- Raw CSV: `data/raw/creditcard.csv`



\## Columns

\- `Time` (numeric): seconds since first transaction

\- `V1`..`V28` (numeric): anonymized PCA components

\- `Amount` (numeric): transaction amount

\- `Class` (int): label (0=legit, 1=fraud)



\## Quality Rules

\- No missing values allowed

\- `Class` must be in {0,1}

\- Must include `Time`, `Amount`, `Class`

\- Expected rows: ~284,807 (fraud extremely rare)



\## Processed Output

\- Parquet: `data/processed/creditcard.parquet`



