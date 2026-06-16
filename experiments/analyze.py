"""
Aggregate the raw matrix results (matrix.py output) into the summary findings.

Produces four analyses, each mapping to a thesis claim, reporting BOTH Spearman and Pearson 
(their agreement is itself evidence: close => robust monotonic+linear relationship; divergence => nonlinearity worth noting):

  A. Overall signal ranking - which signals track fairness across all drift?
  B. GROUP-drift ranking - do subgroup signals win where localization matters?
  C. GLOBAL-drift ranking - does the ranking flip toward generic detectors?
  D. CC vs KS/KL head-to-head - does the best CC signal beat the baselines? (Obj 7)

Ranking is by MEAN ABSOLUTE correlation (the sign of the fairness change depends on the scenario; strength of relationship is what ranks 
signals). The signed mean is also shown, because direction matters for interpretation.

Scenario groups:
  - group = gradual_group_*, variance_group_*
  - global = gradual_global_*, abrupt_global_*
  - control = no_drift
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# signal taxonomy (for grouping in the output):
GLOBAL_SIGNALS = {"global_mean", "global_ewma", "global_std", "subgroup_violation_diff"}
SUBGROUP_SIGNALS = {
    "max_subgroup_deviation", "assignment_proportion_drift",
}  # plus the per_subgroup_* expansions, detected by prefix
BASELINE_SIGNALS = {"ks_statistic", "kl_divergence"}


def _signal_family(signal: str) -> str:
    if signal in BASELINE_SIGNALS:
        return "baseline"
    if signal.startswith("per_subgroup_") or signal in SUBGROUP_SIGNALS:
        return "subgroup"
    if signal == "subgroup_violation_diff":
        return "subgroup"   # it's a subgroup-derived signal despite the global label
    return "global"


def _scenario_group(scenario: str) -> str:
    if scenario == "no_drift":
        return "control"
    if "group" in scenario:
        return "group"
    if "global" in scenario:
        return "global"
    return "other"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["family"] = df["signal"].map(_signal_family)
    df["scenario_group"] = df["scenario"].map(_scenario_group)
    df["abs_spearman"] = df["spearman_mean"].abs()
    df["abs_pearson"] = df["pearson_mean"].abs()
    return df

# rank signals by mean |Spearman| (with mean |Pearson| and signed means shown). 
# optionally filter to one fairness metric and/or one scenario group:
def rank_signals(df: pd.DataFrame, metric: str = None, scenario_group: str = None) -> pd.DataFrame:
    sub = df.copy()
    if metric is not None:
        sub = sub[sub["metric"] == metric]
    if scenario_group is not None:
        sub = sub[sub["scenario_group"] == scenario_group]

    agg = (sub.groupby(["signal", "family"])
              .agg(abs_spearman=("abs_spearman", "mean"),
                   abs_pearson=("abs_pearson", "mean"),
                   signed_spearman=("spearman_mean", "mean"),
                   signed_pearson=("pearson_mean", "mean"))
              .reset_index()
              .sort_values("abs_spearman", ascending=False))
    return agg


def _print_ranking(title, agg):
    print(f"\n{title}\n")
    print(f"{'signal':<40}{'family':<10}"
          f"{'|rho|':>8}{'|r|':>8}{'rho':>8}{'r':>8}")
    print("-" * 82)
    for _, row in agg.iterrows():
        print(f"{row['signal']:<40}{row['family']:<10}"
              f"{row['abs_spearman']:>8.3f}{row['abs_pearson']:>8.3f}"
              f"{row['signed_spearman']:>8.3f}{row['signed_pearson']:>8.3f}")


# within a scenario group (averaged over fairness metrics), find the best signal in each family. Supports two comparisons:
#   - D1: best CC signal (global or subgroup) vs the KS/KL baselines -> "do CC-based signals beat generic detectors?" (broad contribution)
#   - D2: best subgroup signal vs best global signal -> "does subgroup localization add value over global CC tracking?"
#         (the distinctive novel contribution)
# Both global and subgroup signals are CC-based and are part of the contribution; D2 asks specifically whether the localization machinery 
# earns its place over a plain global CC signal
def best_per_family(df: pd.DataFrame, scenario_group: str) -> dict:
    sub = df[df["scenario_group"] == scenario_group]
    agg = (sub.groupby(["signal", "family"])["abs_spearman"]
              .mean().reset_index())

    def top(family):
        f = agg[agg["family"] == family].sort_values("abs_spearman", ascending=False)
        return (f.iloc[0]["signal"], f.iloc[0]["abs_spearman"]) if len(f) else None

    best_global = top("global")
    best_subgroup = top("subgroup")
    baselines = [(r["signal"], r["abs_spearman"])
                 for _, r in agg[agg["family"] == "baseline"]
                 .sort_values("abs_spearman", ascending=False).iterrows()]

    # best CC overall = max of best_global / best_subgroup:
    cc_candidates = [c for c in (best_global, best_subgroup) if c is not None]
    best_cc = max(cc_candidates, key=lambda c: c[1]) if cc_candidates else None

    return {
        "best_cc": best_cc,
        "best_global": best_global,
        "best_subgroup": best_subgroup,
        "baselines": baselines,
    }


def main(path="experiments/results/matrix_results_quick.csv"):
    df = load(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Datasets: {sorted(df['dataset'].unique())} "
          f"(NOTE: preliminary if Credit / full seeds are absent)")

    # A. overall ranking (all scenarios, all metrics):
    _print_ranking("A. OVERALL SIGNAL RANKING (all drift scenarios, all metrics)",
                   rank_signals(df))

    # B. group-drift ranking:
    _print_ranking("B. GROUP-DRIFT RANKING (gradual_group + variance_group)",
                   rank_signals(df, scenario_group="group"))

    # C. global-drift ranking:
    _print_ranking("C. GLOBAL-DRIFT RANKING (gradual_global + abrupt_global)",
                   rank_signals(df, scenario_group="global"))

    # D. head-to-head comparisons, per scenario group:
    print("D. HEAD-TO-HEAD COMPARISONS:")
    for sg in ("group", "global"):
        res = best_per_family(df, sg)
        print(f"\n  Under {sg.upper()} drift:")
        # D1: CC vs generic baselines
        print("[D1: CC vs generic detectors]")
        if res["best_cc"]:
            print(f"best CC signal: {res['best_cc'][0]:<35} |rho|={res['best_cc'][1]:.3f}")
        for name, val in res["baselines"]:
            print(f"baseline: {name:<35} |rho|={val:.3f}")
        # D2: subgroup vs global (does localization add value?)
        print(f"[D2: subgroup localization vs global CC tracking]")
        if res["best_subgroup"]:
            print(f"best SUBGROUP: {res['best_subgroup'][0]:<35} |rho|={res['best_subgroup'][1]:.3f}")
        if res["best_global"]:
            print(f"best GLOBAL: {res['best_global'][0]:<35} |rho|={res['best_global'][1]:.3f}")

    # control sanity check:
    ctrl = rank_signals(df, scenario_group="control")
    print("\nCONTROL CHECK (no_drift): correlations should be WEAK")
    print(f"mean |rho| over all signals under no_drift: {ctrl['abs_spearman'].mean():.3f}  (want near 0)")


if __name__ == "__main__":
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else "experiments/results/matrix_results_quick.csv"
    main(path)