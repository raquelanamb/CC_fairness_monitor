"""
MEPS (Medical Expenditure Panel Survey, Panel 21 / FY2016) preprocessing script.

Reproduces the preprocessing from ConFair repo (Yang & Meliou, 2024) found in the MEPS class here:
https://github.com/DataProfilor/ConFair/blob/main/PrepareData.py,
but without the generic column renaming (X1, X2, ...) that ConFair uses for their pipeline or the 
separation of categorical and numerical columns that they do for CAPUCHIN binning.

Yang & Meliou state that the dataset is from AIF360.
Their preprocessing steps follow AIF360's MEPS Panel 21 preprocessing:
Source: https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/meps_dataset_panel21_fy2016.py

Requires h192.csv, produced by running data/raw/meps/generate_data.R.
The R script downloads the raw MEPS Panel 21 data directly from AHRQ
(https://meps.ahrq.gov) and converts it from SAS transport format to CSV.

For more information on extracting the CSV, see details at 
https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/meps/README.md.

Citation for dataset: Agency for Healthcare Research and Quality (AHRQ).
Medical Expenditure Panel Survey (MEPS), Panel 21, Fiscal Year 2016.
Source: https://meps.ahrq.gov/data_stats/download_data/pufs/h192/h192doc.shtml.

Source data: data/raw/meps/h192.csv
Output:      data/processed/meps.csv

Protected attribute: race (Minority: non-White)
Predictive task: high hospital utilization

Note: Yang & Meliou (2024) drop age. This may be unintentional, as they do age column renaming before leaving it out.
For consistency with their preprocessing, I also omit age, but I acknowledge that this is a break from the steps laid
out by AIF360.

Expected output row count: 15,675 (per Yang & Meliou 2024, Figure 4)
"""


import pandas as pd
from pathlib import Path
 
 
# paths relative to project root:
RAW_PATH = Path("data/raw/meps/h192.csv")
PROCESSED_PATH = Path("data/processed/meps.csv")
 
# feature columns, per ConFair PrepareData.py:
# (ConFair separates these into categorical & numerical for their CAPUCHIN baseline, which bins continuous features. 
# Since I don't use CAPUCHIN, I keep them in one list)
FEATURE_COLS = [
    'REGION',   # census region
    'SEX',      # sex
    'MARRY',    # marital status
    'FTSTU',    # student status if ages 17-23
    'ACTDTY',   # military full-time active duty
    'HONRDC',   # honorably discharged from military
    'RTHLTH',   # perceived health status (self-reported)
    'MNHLTH',   # perceived mental health status (self-reported)
    'HIBPDX',   # high bp diagnosis (>17)
    'CHDDX',    # coronary heart disease diagnosis (>17)
    'ANGIDX',   # angina diagnosis (>17)
    'MIDX',     # heart attack (MI) diagnosis (>17)
    'OHRTDX',   # other heart disease diagnosis (>17)
    'STRKDX',   # stroke diagnosis (>17)
    'EMPHDX',   # emphysema diagnosis (>17)
    'CHBRON',   # chronic bronchitis last 12 months (>17)
    'CHOLDX',   # high cholesterol diagnosis (>17)
    'CANCERDX', # cancer diagnosis (>17)
    'DIABDX',   # diabetes diagnosis (>17)
    'JTPAIN',   # joint pain last 12 months (>17)
    'ARTHDX',   # arthritis diagnosis (>17)
    'ARTHTYPE', # type of arthritis diagnosed (>17)
    'ASTHDX',   # asthma diagnosis
    'ADHDADDX', # adhd diagnosis (5-17)
    'PREGNT',   # pregnant during ref period
    'WLKLIM',   # limitation in physical functioning
    'ACTLIM',   # any limitation work/housework/school
    'SOCLIM',   # social limitations
    'COGLIM',   # cognitive limitations
    'DFHEAR42', # serious difficulty hearing
    'DFSEE42',  # serious difficulty seeing w/ glasses
    'ADSMOK42', # currently smoke
    'PHQ242',   # overall rating of feelings from PHQ-2 depression screening
    'EMPST',    # employment status
    'POVCAT',   # category for family income as % of poverty line
    'INSCOV',   # health insurance coverage indicator, 2016
    'PERWT16F', # person-level sampling weight variable
    'MCS42',    # mental-component score
    'PCS42',    # physical component
    'K6SUM42',  # SAQ 30 days overall rating of feelings
] # UTILIZATION (label) & RACE (protected attribute) are excluded from features & handled separately below

# visit columns used to compute UTILIZATION:
_VISIT_COLS = [
    'OBTOTV16', # office-based provider visit count
    'OPTOTV16', # outpatient dept provider visit count
    'ERTOT16',  # emergency room visits
    'IPNGTD16', # nights in hospital for discharges
    'HHTOTD16'  # home health provider days count
    ]  # column descriptions for columns ending in 16 found here:
       # https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H192&varName=[PASTE COLUMN NAME HERE]
 

def preprocess(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep=',')
 
    # race encoding:
    # (follows AIF360's MEPS Panel 21 preprocessing def; this is the standard def of non-Hispanic White 
    # used in health disparities research & the MEPS literature)
    def group_race(row):
        if row['HISPANX'] == 2 and row['RACEV2X'] == 1:  # non-Hispanic White inviduals marked as White
            return 'White'
        return 'Non-White' # all others as Non-White
 
    
    df['RACEV2X'] = df.apply(lambda row: group_race(row), axis=1)
    df = df.rename(columns={'RACEV2X': 'RACE'})
 
    # panel restriction:
    df = df[df['PANEL'] == 21] # keep only Panel 21 rows to match Yang & Meliou (Panel 21 corresponds to fiscal year 2016 data)
 
    # rename columns (per AIF360 and Yang and Meliou):
    df = df.rename(columns={
        'FTSTU53X': 'FTSTU',   
        'ACTDTY53': 'ACTDTY',
        'HONRDC53': 'HONRDC', 
        'RTHLTH53': 'RTHLTH',  
        'MNHLTH53': 'MNHLTH', 
        'CHBRON53': 'CHBRON',
        'JTPAIN53': 'JTPAIN',
        'PREGNT53': 'PREGNT',
        'WLKLIM53': 'WLKLIM',  
        'ACTLIM53': 'ACTLIM',  
        'SOCLIM53': 'SOCLIM', 
        'COGLIM53': 'COGLIM',
        'EMPST53':  'EMPST',  
        'REGION53': 'REGION',
        'MARRY53X': 'MARRY',
        'AGE53X':   'AGE',
        'POVCAT16': 'POVCAT',
        'INSCOV16': 'INSCOV',
    })
 
    # drop rows w/ invalid/missing values:
    # MEPS uses negative values to encode missing or inapplicable responses
    # (e.g. -1 = inapplicable, -7 = refused, -8 = don't know, -9 = not ascertained)
    df = df[df['REGION'] >= 0]  # remove -1
    df = df[df['AGE'] >= 0]     # remove -1
    df = df[df['MARRY'] >= 0]   # remove -1, -7, -8, -9
    df = df[df['ASTHDX'] >= 0]  # remove -1, -7, -8, -9
 
    # for all remaining categorical features (including EDUCYR and HIDEG, which are used for filtering only and not kept in the 
    # final output), remove values < -1:
    df = df[(df[[
        'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG', 'MIDX', 'OHRTDX', 
        'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX', 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 
        'PREGNT', 'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42','PHQ242', 'EMPST', 'POVCAT', 'INSCOV',
    ]] >= -1).all(axis=1)]
 
    # compute & binarize UTILIZATION:
    # Per Yang and Meliou (2024):
    # UTILIZATION = total healthcare visits across all settings:
    #   OBTOTV16 = office-based visits
    #   OPTOTV16 = outpatient visits
    #   ERTOT16  = emergency room visits
    #   IPNGTD16 = inpatient nights
    #   HHTOTD16 = home health days
    # Binarized at threshold of 10: >= 10 visits --> 1 (high utilization), < 10 visits --> 0 (low utilization).
    # This follows AIF360's binarization of the MEPS utilization label.
    df['TOTEXP16'] = df[_VISIT_COLS].sum(axis=1) # adding up all 5 values for each row horizontally
    df['TOTEXP16'] = df['TOTEXP16'].apply(lambda x: 0.0 if x < 10.0 else 1.0)
    df = df.rename(columns={'TOTEXP16': 'UTILIZATION'})
 
    # column selection:
    # via ConFair __init__(): their final column order is categorical + numerical + UTILIZATION + RACE.
    # EDUCYR & HIDEG were used for row filtering above but are not included in the final output.
    df = df[FEATURE_COLS + ['UTILIZATION', 'RACE']]
 
    # race encoding:
    df['RACE'] = df['RACE'].map({'White': 1, 'Non-White': 0})
 
    # drop NaN rows:
    df = df.dropna() # [ConFair base class Dataset.preprocess() calls df.dropna()]
 
    # save:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
 
    print(f"MEPS preprocessing complete.")
    print(f"  Rows: {len(df)} (expected: 15,675)")
    print(f"  Saved to: {processed_path}")
 
    return df
 
 
if __name__ == "__main__":
    preprocess()