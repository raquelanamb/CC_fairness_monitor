"""
LSAC preprocessing script.

Reproduces the preprocessing from ConFair repo (Yang & Meliou, 2024) found in the LSAC class here:
https://github.com/DataProfilor/ConFair/blob/main/PrepareData.py,
but without the generic column renaming (X1, X2, ...) that ConFair uses for their pipeline.

Yang & Meliou state that the dataset comes from OmniFair:
H. Zhang, X. Chu, A. Asudeh, and S. B. Navathe, “OmniFair: A Declarative System for Model-Agnostic 
Group Fairness in Machine Learning,” in Proceedings of the 2021 International Conference on Management 
of Data (SIGMOD ’21), 2021, pp. 2076–2088, doi: 10.1145/3448016.3452787.

Citation for dataset: 
Wightman, L. F. (1998). LSAC National Longitudinal Bar Passage Study. Law School Admission Council. 
Source: https://eric.ed.gov/?id=ED469370

Source data: data/raw/lsac/lsac.csv
Output:      data/processed/lsac.csv

Protected attribute: race (Minority: African-American)
Predictive task: passing bar exam

Expected output row count: 24,479 (per Yang & Meliou 2024, Figure 4)
"""

import pandas as pd
from pathlib import Path


# paths relative to project root:
RAW_PATH = Path("data/raw/lsac/lsac.csv")
PROCESSED_PATH = Path("data/processed/lsac.csv")


def preprocess(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    
    # race encoding:
    def group_race(x):
        if x == "Black":
            return 0.0
        elif x == "White":
            return 1.0
        else:
            return -1.0
        
    df['race'] = df['race'].apply(lambda x: group_race(x))

    # only keep Black & white students, following Yang & Meliou and the fairness lit's focus on this 
    # particular disparity in law school outcomes:
    df = df[df["race"] >= 0]

    # sex encoding:
    df["sex"] = df["sex"].replace({"Female": 0.0, "Male": 1.0})

    # pass_bar (pred target) encoding:
    df["pass_bar"] = df["pass_bar"].replace({"Passed": 1.0, "Failed_or_not_attempted": 0.0})

    # part time encoding:
    df["isPartTime"] = df["isPartTime"].replace({"Yes": 1.0, "No": 0.0})

    # column selection: 
    FEATURE_COLS = [
        "zfygpa",             # standardized/normalized first year law school GPA
        "zgpa",               # standardized/normalized cumulative law school GPA
        "DOB_yr",             # year of birth
        "isPartTime",         # part time student (encoded above)
        "sex",                # sex (encoded above)
        "race",               # race (encoded above, protected attribute)
        "cluster_tier",       # law school tier group
        "family_income",      # family income bracket
        "lsat",               # LSAT score
        "ugpa",               # undergraduate GPA (not z-scored)
        "weighted_lsat_ugpa", # weighted combination of LSAT & undergrad GPA
        "pass_bar",           # passed bar exam (label)
    ]
    df = df[FEATURE_COLS]

    # drop NaN rows: 
    df = df.dropna() # [ConFair base class Dataset.preprocess() calls df.dropna()]

    # save:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    # verify length matches Yang and Meliou's count:
    print(f"LSAC preprocessing complete.")
    print(f"  Rows: {len(df)} (expected: 24,479)")
    print(f"  Saved to: {processed_path}")

    return df


if __name__ == "__main__":
    preprocess()