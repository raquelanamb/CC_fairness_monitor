"""
Compute test-set sizes and per-batch row counts for each dataset, under both the 70/30 split (this work) and the 70/15/15 split (ConFair). 
With n_batches batches sampled from the test pool, batch_size = test_size // n_batches.

Prints a table comparing the two split schemes so the batch-size arithmetic in the thesis can be stated exactly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data.loaders.load_lsac import load_lsac
from data.loaders.load_meps import load_meps
from data.loaders.load_credit import load_credit


LOADERS = {"LSAC": load_lsac, "MEPS": load_meps, "Credit": load_credit}
N_BATCHES = 20


# total dataset size = train + test (the bundle holds the split halves):
def total_rows(bundle) -> int:
    return len(bundle.X_train) + len(bundle.X_test)


def main(n_batches: int = N_BATCHES):
    print(f"Per-batch row counts with n_batches = {n_batches}\n")
    header = (f"{'dataset':<8}{'total':>10}{'test(70/30)':>14}{'rows/batch':>12}"
              f"{'test(70/15/15)':>16}{'rows/batch':>12}")
    print(header)
    print("-" * len(header))

    for name, loader in LOADERS.items():
        # load once (any seed; split SIZES are seed-independent for a fixed ratio):
        bundle = loader(seed=42)
        total = total_rows(bundle)

        # this work uses 70/30, so the loaded bundle's test set already is the 70/30 test set:
        test_7030 = len(bundle.X_test)
        batch_7030 = test_7030 // n_batches

        # 70/15/15: ConFair's test set is 15% of the total (the other 15% is the validation split this work does not use):
        test_701515 = int(round(0.15 * total))
        batch_701515 = test_701515 // n_batches

        print(f"{name:<8}{total:>10}{test_7030:>14}{batch_7030:>12}"
              f"{test_701515:>16}{batch_701515:>12}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_BATCHES
    main(n)