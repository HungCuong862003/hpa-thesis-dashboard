#!/usr/bin/env python3
"""
verify_numbers.py
==================

Standalone audit script. Walks every data file and asserts the anchor numbers
match what the thesis manuscript cites. Designed to run from cron without any
Streamlit dependency.

Exit codes:
    0  All checks passed.
    1  At least one anchor mismatch.

Usage:
    python audit/verify_numbers.py

The complete list of anchors lives here, in one place. data_loaders.py also
asserts a SUBSET of these (the most load-bearing ones) at import time. Both
files reference the same numbers on purpose: drift fires from two places.

Author: Jimmy
Date: May 2026
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Resolve data dir relative to this file (audit/verify_numbers.py → ../data/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Pretty output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

failures: list[str] = []
passes: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one anchor check. Print result, stash for summary."""
    if condition:
        passes.append(label)
        print(f"  {GREEN}✓{RESET} {label}" + (f"  ({detail})" if detail else ""))
    else:
        failures.append(label)
        print(f"  {RED}✗{RESET} {label}" + (f"  ({detail})" if detail else ""))


# ======================================================================
# 1. comparison_table.csv
# ======================================================================
print(f"\n{YELLOW}[1] comparison_table.csv{RESET}")
df = pd.read_csv(DATA_DIR / "comparison_table.csv")
d10 = df.loc[df.Horizon == "10min", "Δpp (Hetero vs Naive)"].iloc[0]
d120 = df.loc[df.Horizon == "120min", "Δpp (Hetero vs Naive)"].iloc[0]
check("Alibaba 10min Δpp ≈ +0.25", abs(d10 - 0.25) < 0.05, f"got {d10}")
check("Alibaba 120min Δpp ≈ +4.64", abs(d120 - 4.64) < 0.05, f"got {d120}")


# ======================================================================
# 2. bitbrains_summary_corrected.csv
# ======================================================================
print(f"\n{YELLOW}[2] bitbrains_summary_corrected.csv{RESET}")
bb = pd.read_csv(DATA_DIR / "bitbrains_summary_corrected.csv")
d30 = bb.loc[bb.Horizon == "30min", "BB_Delta_pp"].iloc[0]
d120 = bb.loc[bb.Horizon == "120min", "BB_Delta_pp"].iloc[0]
check("Bitbrains Δ@30min == -5.46pp (corrected, sign matters)",
      abs(d30 - (-5.46)) < 0.01, f"got {d30}")
check("Bitbrains Δ@120min ≈ +3.30pp", abs(d120 - 3.30) < 0.05, f"got {d120}")


# ======================================================================
# 3. boundary_condition_table_corrected.csv
# ======================================================================
print(f"\n{YELLOW}[3] boundary_condition_table_corrected.csv{RESET}")
bc = pd.read_csv(DATA_DIR / "boundary_condition_table_corrected.csv")
check("BCT has 'ACF@24h' column (not 'ACF@120min')",
      "ACF@24h" in bc.columns and "ACF@120min" not in bc.columns)
bb_row = bc[bc.dataset.str.contains("Bitbrains", case=False)]
check("BCT Bitbrains row delta@30min contains '-5.46'",
      "-5.46" in bb_row["delta @30min"].iloc[0],
      f"got {bb_row['delta @30min'].iloc[0]}")


# ======================================================================
# 4. results_summary_bytedance.csv
# ======================================================================
print(f"\n{YELLOW}[4] results_summary_bytedance.csv{RESET}")
bd = pd.read_csv(DATA_DIR / "results_summary_bytedance.csv")
et_wins = (bd["extratrees_R2"] >= bd["hetero_ensemble_R2"]).sum()
check("ByteDance ExtraTrees-alone beats Hetero ensemble at 4/4 horizons",
      et_wins == 4, f"got {et_wins}/4")
nv_loss = (bd["hetero_ensemble_R2"] > bd["naive_R2"]).sum()
check("ByteDance Hetero ensemble beats Naive at 4/4 horizons",
      nv_loss == 4, f"got {nv_loss}/4")


# ======================================================================
# 5. hpa_simulation_v2.csv
# ======================================================================
print(f"\n{YELLOW}[5] hpa_simulation_v2.csv{RESET}")
hpa = pd.read_csv(DATA_DIR / "hpa_simulation_v2.csv")
check("HPA row count == 800", len(hpa) == 800, f"got {len(hpa)}")
tu_grid = sorted(hpa.target_util.unique())
check("HPA target_util grid == [0.5, 0.6, 0.7, 0.8] (max is 0.8 NOT 0.9)",
      tu_grid == [0.5, 0.6, 0.7, 0.8], f"got {tu_grid}")
sm_grid = sorted(hpa.safety_margin.unique())
check("HPA safety_margin grid has 10 multiplier values starting at 1.0",
      len(sm_grid) == 10 and abs(sm_grid[0] - 1.0) < 1e-9
      and abs(sm_grid[-1] - 1.6) < 1e-9,
      f"got {sm_grid}")
strat = sorted(hpa.Strategy.unique())
check("HPA Strategy values == ['ML-Proactive', 'Reactive']",
      strat == ["ML-Proactive", "Reactive"], f"got {strat}")
grp = hpa.groupby(["Horizon", "target_util", "safety_margin"]).size()
check("HPA: every (horizon, tu, sm) tuple has exactly 5 rows (1 ML + 4 lag)",
      grp.min() == 5 and grp.max() == 5, f"got [{grp.min()}, {grp.max()}]")
# Material lag-variance check: the 30min/0.7/1.2 cell should differ between
# lag=1 and lag=4 by something material on violation_rate.
cell = hpa[(hpa.Horizon == "30min") & (hpa.target_util == 0.7) &
           (abs(hpa.safety_margin - 1.2) < 1e-9) & (hpa.Strategy == "Reactive")]
if len(cell) == 4:
    spread = cell.violation_rate.max() - cell.violation_rate.min()
    check("HPA: lag=1..4 violation_rate at 30min/0.7/1.2 spans >= 0.005 "
          "(material variance, not pre-aggregated)",
          spread >= 0.005, f"spread = {spread:.4f}")
else:
    check("HPA 30min/0.7/1.2 has 4 reactive lag rows", False,
          f"got {len(cell)}")


# ======================================================================
# 6. chronos_benchmark_results.csv
# ======================================================================
print(f"\n{YELLOW}[6] chronos_benchmark_results.csv{RESET}")
ch = pd.read_csv(DATA_DIR / "chronos_benchmark_results.csv")
check("Chronos benchmark row count == 12", len(ch) == 12, f"got {len(ch)}")
chronos_wins = (ch.chronos_r2 > ch.ensemble_r2).sum()
check("Chronos wins == 10/12 (NOT 11/12)", chronos_wins == 10,
      f"got {chronos_wins}")
ens_winning = ch[ch.ensemble_r2 > ch.chronos_r2][["dataset", "horizon"]]
ens_set = {(r.dataset, r.horizon) for _, r in ens_winning.iterrows()}
expected_ens = {("Alibaba", "10min"), ("Alibaba", "120min")}
check("Ensemble-winning cells == {Alibaba 10min, Alibaba 120min}",
      ens_set == expected_ens, f"got {ens_set}")


# ======================================================================
# 7. cv_stratified_skill.csv
# ======================================================================
print(f"\n{YELLOW}[7] cv_stratified_skill.csv{RESET}")
cvs = pd.read_csv(DATA_DIR / "cv_stratified_skill.csv")
row = cvs[(cvs.Horizon == "30min") & (cvs.CV_bin == "CV 0.3-0.5")]
check("Container win-rate at 30min/CV 0.3-0.5 ≈ 27.89% (best CV bin)",
      len(row) == 1 and abs(row["Container_win_%"].iloc[0] - 27.89) < 0.5,
      f"got {row['Container_win_%'].iloc[0] if len(row) else 'MISSING'}")


# ======================================================================
# 8. stratified_skill.csv (per-horizon, the MAE/R² divergence story)
# ======================================================================
print(f"\n{YELLOW}[8] stratified_skill.csv{RESET}")
ss = pd.read_csv(DATA_DIR / "stratified_skill.csv")
r10 = ss[ss.Horizon == "10min"].iloc[0]
check("MAE/R² divergence: at 10min, ensemble MAE > naive MAE (skill < 0)",
      r10["Skill_score"] < 0, f"skill={r10['Skill_score']:.3f}")
r120 = ss[ss.Horizon == "120min"].iloc[0]
check("At 120min, ensemble MAE < naive MAE (skill > 0)",
      r120["Skill_score"] > 0, f"skill={r120['Skill_score']:.3f}")


# ======================================================================
# 9. base_model_correlations.csv
# ======================================================================
print(f"\n{YELLOW}[9] base_model_correlations.csv{RESET}")
ec = pd.read_csv(DATA_DIR / "base_model_correlations.csv")
ec_min = ec.Error_Corr.min()
check("Min error correlation >= 0.92 (BiLSTM non-diversity finding)",
      ec_min >= 0.92, f"got {ec_min:.4f}")


# ======================================================================
# 10. bcf_pooled_results.json
# ======================================================================
print(f"\n{YELLOW}[10] bcf_pooled_results.json{RESET}")
with open(DATA_DIR / "bcf_pooled_results.json") as f:
    bcf = json.load(f)
check("BCF pooled AUC == 0.80", abs(bcf["pooled_auc"] - 0.80) < 0.005,
      f"got {bcf['pooled_auc']}")
check("BCF n_pooled_pairs == 36 (12 cells × 3 forecasters)",
      bcf["n_pooled_pairs"] == 36, f"got {bcf['n_pooled_pairs']}")
ci_lo = bcf["bca_95_ci_lo"]
ci_hi = bcf["bca_95_ci_hi"]
check("BCF BCa CI ≈ [0.70, 0.88]",
      abs(ci_lo - 0.704) < 0.01 and abs(ci_hi - 0.879) < 0.01,
      f"got [{ci_lo:.3f}, {ci_hi:.3f}]")
check("BCF predicate text mentions 'ACF@24h > 0.2' AND 'horizon >= 30min'",
      ("ACF@24h" in bcf["predicate"] and "0.2" in bcf["predicate"]
       and "30" in bcf["predicate"]),
      bcf["predicate"])


# ======================================================================
# 11. bcf_pairs.csv
# ======================================================================
print(f"\n{YELLOW}[11] bcf_pairs.csv{RESET}")
bp = pd.read_csv(DATA_DIR / "bcf_pairs.csv")
check("BCF pairs total == 48 (4 models × 12 cells)", len(bp) == 48,
      f"got {len(bp)}")
real = bp[bp.model.isin(["NNLS", "Chronos-2", "TimesFM"])]
check("BCF: 36 real-model pairs (excludes Naive)", len(real) == 36,
      f"got {len(real)}")
fp = real[(real.predicate == 1) & (real.ml_wins == 0)]
check("BCF: 0 false positives across 3 real forecasters (the headline)",
      len(fp) == 0, f"got {len(fp)}")
fn = real[(real.predicate == 0) & (real.ml_wins == 1)]
check("BCF: 12 false negatives pooled (predicate fails to fire on 12 wins)",
      len(fn) == 12, f"got {len(fn)}")


# ======================================================================
# 12. bcf_per_model_auc.csv
# ======================================================================
print(f"\n{YELLOW}[12] bcf_per_model_auc.csv{RESET}")
pma = pd.read_csv(DATA_DIR / "bcf_per_model_auc.csv")
nnls = pma.loc[pma.model == "NNLS", "auc"].iloc[0]
ch2 = pma.loc[pma.model == "Chronos-2", "auc"].iloc[0]
tfm = pma.loc[pma.model == "TimesFM", "auc"].iloc[0]
check("NNLS AUC ≈ 0.83", abs(nnls - 0.833) < 0.01, f"got {nnls:.3f}")
check("Chronos-2 AUC ≈ 0.77", abs(ch2 - 0.773) < 0.01, f"got {ch2:.3f}")
check("TimesFM AUC == 0.80", abs(tfm - 0.80) < 0.01, f"got {tfm:.3f}")


# ======================================================================
# Summary
# ======================================================================
print("")
print("=" * 60)
total = len(passes) + len(failures)
if not failures:
    print(f"{GREEN}All {total} checks passed.{RESET}")
    sys.exit(0)
else:
    print(f"{RED}{len(failures)} of {total} checks FAILED:{RESET}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
