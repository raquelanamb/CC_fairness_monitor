"""
drift_breakdown.py

Breaks down signal-fairness correlations by drift MECHANISM (gradual / abrupt /
variance) and by drift MAGNITUDE, dimensions not covered by the headline
group-vs-global analysis in analyze.py.

Uses the same statistic as the rest of Chapter 5: the mean absolute Spearman
correlation (|rho|), just sliced along different columns. Reads the canonical
matrix results CSV and writes breakdown tables to CSV.

Usage:
    python drift_breakdown.py matrix_results_with_kde_subsample.csv
    (defaults to matrix_results_with_kde_subsample.csv in the current directory)
"""

import sys
import pandas as pd

# load results, drop control, & parse the scenario name into mechanism / target / magnitude columns:
def load_and_parse(path):

    df = pd.read_csv(path)

    # drop the no-drift control; only the 18 drift scenarios are analyzed here:
    df = df[df["scenario"] != "no_drift"].copy()

    # scenario strings look like "abrupt_global_mag2.0":
    parts = df["scenario"].str.split("_", expand=True)
    df["mechanism"] = parts[0]                 # gradual / abrupt / variance
    df["target"] = parts[1]                    # global / group
    df["magnitude"] = df["scenario"].str.extract(r"mag([0-9.]+)").astype(float)

    return df

# mean of |Spearman| over the group:
def mean_abs_spearman(group):
    return group["spearman_mean"].abs().mean()

# mean |rho| by drift mechanism, pooled over all signals:
def breakdown_by_mechanism(df):
    out = (
        df.groupby("mechanism")
        .apply(mean_abs_spearman)
        .reset_index(name="mean_abs_spearman")
        .sort_values("mean_abs_spearman", ascending=False)
    )
    return out

# mean |rho| by magnitude:
def breakdown_by_magnitude(df):
    mean_shift = df[df["mechanism"].isin(["gradual", "abrupt"])] # mean-shift (gradual/abrupt) & variance use different magnitude scales,
    variance = df[df["mechanism"] == "variance"]                 # so they are reported separately

    ms = (
        mean_shift.groupby("magnitude")
        .apply(mean_abs_spearman)
        .reset_index(name="mean_abs_spearman")
    )
    ms["drift_type"] = "mean_shift (gradual+abrupt)"

    var = (
        variance.groupby("magnitude")
        .apply(mean_abs_spearman)
        .reset_index(name="mean_abs_spearman")
    )
    var["drift_type"] = "variance (spread scaling)"

    return pd.concat([ms, var], ignore_index=True)[
        ["drift_type", "magnitude", "mean_abs_spearman"]
    ]


# mean |rho| by mechanism for a single named signal (default: the top-ranked signal overall), so the breakdown is not diluted 
# by weaker signals and baselines:
def top_signal_by_mechanism(df, signal="per_subgroup_ewma[min_neg]"):
    sub = df[df["signal"] == signal]
    out = (
        sub.groupby("mechanism")
        .apply(mean_abs_spearman)
        .reset_index(name="mean_abs_spearman")
        .sort_values("mean_abs_spearman", ascending=False)
    )
    out.insert(0, "signal", signal)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "matrix_results_with_kde_subsample.csv"
    df = load_and_parse(path)

    by_mech = breakdown_by_mechanism(df)
    by_mag = breakdown_by_magnitude(df)
    top_mech = top_signal_by_mechanism(df)

    # write artifacts:
    by_mech.to_csv("H_breakdown_by_mechanism.csv", index=False)
    by_mag.to_csv("I_breakdown_by_magnitude.csv", index=False)
    top_mech.to_csv("J_top_signal_by_mechanism.csv", index=False)

    # print to console, rounded to match reporting precision:
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("=== By mechanism (mean |rho|, all signals pooled) ===")
    print(by_mech.to_string(index=False))
    print("\n=== By magnitude (mean |rho|, all signals pooled) ===")
    print(by_mag.to_string(index=False))
    print("\n=== Top signal by mechanism ===")
    print(top_mech.to_string(index=False))
    print("\nWrote: H_breakdown_by_mechanism.csv, "
          "I_breakdown_by_magnitude.csv, J_top_signal_by_mechanism.csv")


if __name__ == "__main__":
    main()