"""
Aggregate raw matrix results (matrix.py output) into the summary findings.

Each block maps to a thesis claim and reports mean |Spearman|, mean |Pearson|, and the signed means. Ranking is by mean |Spearman| 
(strength of the monotonic relationship; sign depends on the scenario). Signed means are kept because direction matters for interpretation. 
Every ranking also reports coverage (n_cells, n_dsets) so a signal that is only defined on a subset of datasets cannot silently inflate its 
rank.

Blocks:
  A. Overall ranking - which signals track fairness across all drift.
  B. Group-drift ranking - do subgroup signals win where localization matters.
  C. Global-drift ranking - does the ranking shift toward generic detectors.
  D. Head-to-head - best CC signal vs KS/KL, and subgroup vs global CC.
  E. Model-agnosticism - is the signal ranking stable across LR and XGBoost (the monitor reads the data distribution, not the model,
     so a model-dependent ranking would contradict the claim).
  F. Control - correlations under no_drift should be weak.
  G. Subsample sensitivity - does KDE fit-set subsampling change the ranking (run only if a second results file is given).

A cell is one (dataset, model, scenario, metric) combination. Cells with no valid seeds arrive as NaN and are excluded from means; 
coverage columns make that exclusion visible.
"""

import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
 
BASELINE = {"ks_statistic", "kl_divergence"}
GLOBAL = {"global_mean", "global_ewma", "global_std"}
SUBGROUP_NAMED = {
    "max_subgroup_deviation",
    "assignment_proportion_drift",
    "subgroup_violation_diff",  # subgroup-derived
}
 
 
def _family(signal: str) -> str:
    if signal in BASELINE:
        return "baseline"
    if signal in GLOBAL:
        return "global"
    if signal in SUBGROUP_NAMED or signal.startswith("per_subgroup_"):
        return "subgroup"
    return "unknown"  # unrecognized signals surface instead of defaulting to global
 
 
def _scenario_group(scenario: str) -> str:
    if scenario == "no_drift":
        return "control"
    if "group" in scenario:
        return "group"
    if "global" in scenario:
        return "global"
    return "other"
 
 
def load(path: str, min_seeds: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["family"] = df["signal"].map(_family)
    df["scenario_group"] = df["scenario"].map(_scenario_group)

    # treat a cell with fewer than min_seeds valid seeds as missing:
    insufficient = df["n_seeds_valid"] < min_seeds
    df.loc[insufficient, ["spearman_mean", "pearson_mean"]] = np.nan
    df["abs_spearman"] = df["spearman_mean"].abs()
    df["abs_pearson"] = df["pearson_mean"].abs()
    if (df["family"] == "unknown").any():
        unknown = sorted(df.loc[df["family"] == "unknown", "signal"].unique())
        print(f"WARNING unrecognized signals (classified 'unknown'): {unknown}")
    return df
 
 
def rank_signals(df: pd.DataFrame, metric: str = None, scenario_group: str = None, drop_control: bool = True) -> pd.DataFrame:
    sub = df
    if drop_control:
        sub = sub[sub["scenario_group"] != "control"]
    if metric is not None:
        sub = sub[sub["metric"] == metric]
    if scenario_group is not None:
        sub = sub[sub["scenario_group"] == scenario_group]
 
    agg = (sub.groupby(["signal", "family"]).agg(abs_spearman=("abs_spearman", "mean"), abs_pearson=("abs_pearson", "mean"),
                                                 signed_spearman=("spearman_mean", "mean"), signed_pearson=("pearson_mean", "mean"),
                                                 n_cells=("abs_spearman", lambda s: int(s.notna().sum()))).reset_index())
    ndsets = (sub.dropna(subset=["abs_spearman"]).groupby("signal")["dataset"].nunique().rename("n_dsets"))
    agg = agg.merge(ndsets, on="signal", how="left")
    agg["n_dsets"] = agg["n_dsets"].fillna(0).astype(int)
    return agg.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
 
 
def write_csv(df: pd.DataFrame, outdir: Path, name: str, manifest: list) -> None:
    path = outdir / name
    df.to_csv(path, index=False)
    manifest.append((name, len(df)))
 
 
def best_per_family(df: pd.DataFrame, scenario_group: str) -> pd.DataFrame:
    agg = rank_signals(df, scenario_group=scenario_group)
 
    def top(family):
        f = agg[agg["family"] == family]
        return None if f.empty else f.iloc[0]
 
    best_global = top("global")
    best_subgroup = top("subgroup")
    cc = [c for c in (best_global, best_subgroup) if c is not None]
    best_cc = max(cc, key=lambda r: r["abs_spearman"]) if cc else None
    baselines = agg[agg["family"] == "baseline"]
 
    rows = []
    if best_cc is not None:
        rows.append(("best CC", best_cc["signal"], best_cc["abs_spearman"], best_cc["n_dsets"]))
    for _, r in baselines.iterrows():
        rows.append(("baseline", r["signal"], r["abs_spearman"], r["n_dsets"]))
    if best_subgroup is not None:
        rows.append(("best subgroup", best_subgroup["signal"], best_subgroup["abs_spearman"], best_subgroup["n_dsets"]))
    if best_global is not None:
        rows.append(("best global", best_global["signal"], best_global["abs_spearman"], best_global["n_dsets"]))
 
    out = pd.DataFrame(rows, columns=["role", "signal", "abs_spearman", "n_dsets"])
    out.insert(0, "scenario_group", scenario_group)
    return out
 
 
def model_agnosticism(df: pd.DataFrame, scenario_group: str = None):
    sub = df[df["scenario_group"] != "control"]
    if scenario_group is not None:
        sub = sub[sub["scenario_group"] == scenario_group]
    models = sorted(sub["model"].unique())
    per = (sub.groupby(["signal", "model"])["abs_spearman"].mean().unstack("model").reindex(columns=models))
    per["mean"] = per[models].mean(axis=1)
    per["spread"] = per[models].max(axis=1) - per[models].min(axis=1)
    per = per.sort_values("mean", ascending=False).reset_index()
 
    summary = {}
    if len(models) == 2:
        rho = per[models[0]].corr(per[models[1]], method="spearman")
        summary["model_rank_corr"] = round(float(rho), 4)
        summary["model_rank_corr_pair"] = f"{models[0]} vs {models[1]}"
        summary["model_max_spread"] = round(float(per["spread"].max()), 4)
        summary["model_max_spread_signal"] = per.loc[per["spread"].idxmax(), "signal"]
    return per, summary
 
 
def control_check(df: pd.DataFrame) -> dict:
    ctrl = df[df["scenario_group"] == "control"]
    valid = int(ctrl["abs_spearman"].notna().sum())
    total = int(len(ctrl))
    mean_abs = float(ctrl["abs_spearman"].mean())
    return {
        "control_mean_abs_rho": round(mean_abs, 4),
        "control_defined_cells": valid,
        "control_total_cells": total,
    }
 
 
def subsample_sensitivity(canonical: pd.DataFrame, baseline: pd.DataFrame):
    common = sorted(set(canonical["dataset"]) & set(baseline["dataset"]))
    a = (rank_signals(canonical[canonical["dataset"].isin(common)])[["signal", "abs_spearman"]].rename(columns={"abs_spearman": "subsample"}))
    b = (rank_signals(baseline[baseline["dataset"].isin(common)])[["signal", "abs_spearman"]].rename(columns={"abs_spearman": "no_subsample"}))
    merged = a.merge(b, on="signal", how="outer")
    merged["delta"] = merged["subsample"] - merged["no_subsample"]
    merged = merged.reindex(merged["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
 
    rho = merged["subsample"].corr(merged["no_subsample"], method="spearman")
    summary = {
        "subsample_rank_corr": round(float(rho), 4),
        "subsample_max_abs_delta": round(float(merged["delta"].abs().max()), 4),
        "subsample_common_datasets": ";".join(common),
    }
    return merged, summary
 
 
def main(canonical_path: str, baseline_path: str = None, outdir: str = None, min_seeds: int = 1) -> None:
    df = load(canonical_path, min_seeds=min_seeds)
    outdir = Path(outdir) if outdir else Path(canonical_path).parent / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
 
    manifest = []
    summary = {
        "canonical": canonical_path,
        "baseline": baseline_path or "",
        "datasets": ";".join(sorted(df["dataset"].unique())),
        "models": ";".join(sorted(df["model"].unique())),
        "metrics": ";".join(sorted(df["metric"].unique())),
        "min_seeds": min_seeds,
        "n_rows": len(df),
    }
 
    write_csv(rank_signals(df), outdir, "A_overall_ranking.csv", manifest)
    write_csv(rank_signals(df, scenario_group="group"), outdir, "B_group_ranking.csv", manifest)
    write_csv(rank_signals(df, scenario_group="global"), outdir, "C_global_ranking.csv", manifest)
 
    head = pd.concat([best_per_family(df, "group"), best_per_family(df, "global")], ignore_index=True)
    write_csv(head, outdir, "D_headtohead.csv", manifest)
 
    per_model, ma_summary = model_agnosticism(df)
    write_csv(per_model, outdir, "E_model_agnosticism.csv", manifest)
    summary.update(ma_summary)
    summary.update(control_check(df))
 
    if baseline_path:
        base = load(baseline_path, min_seeds=min_seeds)
        merged, ss_summary = subsample_sensitivity(df, base)
        write_csv(merged, outdir, "G_subsample_sensitivity.csv", manifest)
        summary.update(ss_summary)
 
    summary_df = pd.DataFrame(list(summary.items()), columns=["metric", "value"])
    write_csv(summary_df, outdir, "run_summary.csv", manifest)
 
    print(f"wrote {len(manifest)} files to {outdir}")
    for name, n in manifest:
        print(f"  {name} ({n} rows)")
    print("headline:")
    for k in ("control_mean_abs_rho", "model_rank_corr", "subsample_rank_corr", "subsample_max_abs_delta"):
        if k in summary:
            print(f"  {k} = {summary[k]}")
 
 
if __name__ == "__main__":
    canonical = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/matrix_results_with_kde_subsample.csv"
    baseline = sys.argv[2] if len(sys.argv) > 2 else None
    out = sys.argv[3] if len(sys.argv) > 3 else None
    main(canonical, baseline, out)