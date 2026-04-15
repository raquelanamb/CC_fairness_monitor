"""
Credit (Give Me Some Credit) preprocessing script.

Reproduces the preprocessing from ConFair repo (Yang & Meliou, 2024) found in the GMCredit class here:
https://github.com/DataProfilor/ConFair/blob/main/PrepareData.py,
but without the generic column renaming (X1, X2, ...) that ConFair uses for their pipeline.

Citation for dataset:
Kaggle. (2011). Give Me Some Credit.
Source: https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

Source data: data/raw/credit/GiveMeSomeCredit_training.csv
Output:      data/processed/credit.csv

Note: I follow Yang & Meliou's choice to use the training file from the Kaggle dataset.

Protected attribute: age (Minority: age < 35)
Predictive task: serious financial distress in next 2 years

Expected output row count: 120,269 (per Yang & Meliou 2024, Figure 4)
"""

import pandas as pd
from pathlib import Path


# paths relative to project root:
RAW_PATH = Path("data/raw/credit/GiveMeSomeCredit_training.csv")
PROCESSED_PATH = Path("data/processed/credit.csv")

# feature columns, per ConFair PrepareData.py:
FEATURE_COLS = [
    'RevolvingUtilizationOfUnsecuredLines', # total balance on lines of credit divided by sum of credit limits
    'NumberOfTime30-59DaysPastDueNotWorse', # number of times borrower has been 30-59 days past due
    'DebtRatio',                            # monthly debt payments divided by monthly gross income
    'MonthlyIncome',                        # monthly income
    'NumberOfOpenCreditLinesAndLoans',      # number of open loans & lines of credit
    'NumberOfTimes90DaysLate',              # number of times borrower has been 90+ days past due
] # SeriousDlqin2yrs (label) & age (protected attribute) are excluded from features & handled separately below


def preprocess(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # column selection:
    df = df[FEATURE_COLS + ['SeriousDlqin2yrs', 'age']] # ConFair selects exactly these 8 columns first, then encodes age

    # age encoding:
    df['age'] = df['age'].apply(lambda x: int(x >= 35)) 

    # drop NaN rows:
    df = df.dropna() # [ConFair base class Dataset.preprocess() calls df.dropna()]

    # save:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Credit preprocessing complete.")
    print(f"  Rows: {len(df)} (expected: 120,269)")
    print(f"  Saved to: {processed_path}")

    return df


if __name__ == "__main__":
    preprocess()