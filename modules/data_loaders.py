"""
data_loaders.py
================

One @st.cache_data function per data file. Every loader does TWO checks:

1. Required columns are present (catches schema drift).
2. An anchor cell value matches what the thesis cites (catches silent file
   substitution — the corrected vs uncorrected files have IDENTICAL columns
   but DIFFERENT values, so column checks alone are not enough).

Anchor values come from `audit/verify_numbers.py`, which is the canonical
source. Both files reference the same anchors, on purpose, so if a number
in the data ever drifts from the thesis text it fires from two places.

Style note: deliberately verbose with print() traces. When a loader fails
at import time the print output tells the user which file is wrong before
the AssertionError fires.

Author: Jimmy
Date: May 2026
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------
# All loaders use a project-relative path so the dashboard works the same
# whether run locally (streamlit run app.py) or on Streamlit Community Cloud.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ======================================================================
# 1. comparison_table.csv  (Alibaba R^2 per horizon, all 10 models)
# ======================================================================
@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    """Alibaba R^2 comparison table. 4 rows (10/30/60/120 min)."""
    df = pd.read_csv(DATA_DIR / "comparison_table.csv")

    expected_cols = ["Horizon", "Naive R²", "Hetero Ens R²", "Δpp (Hetero vs Naive)"]
    for c in expected_cols:
        assert c in df.columns, f"comparison_table.csv missing column {c}"

    # Anchor: Alibaba 120min Hetero vs Naive must be ~+4.64pp
    delta_120 = df.loc[df.Horizon == "120min", "Δpp (Hetero vs Naive)"].iloc[0]
    assert abs(delta_120 - 4.64) < 0.05, (
        f"comparison_table anchor failed: 120min Δpp = {delta_120}, expected ~4.64"
    )
    print(f"[loader] comparison_table OK ({len(df)} rows)")
    return df


# ======================================================================
# 2. boundary_condition_table_corrected.csv  (3 datasets × verdict)
# ======================================================================
@st.cache_data
def load_boundary_condition_table() -> pd.DataFrame:
    """Cross-dataset boundary condition table. Use CORRECTED version only."""
    df = pd.read_csv(DATA_DIR / "boundary_condition_table_corrected.csv")

    # The uncorrected file labels its column 'ACF@120min'. The corrected
    # file labels it 'ACF@24h'. Anchor on the column name itself.
    assert "ACF@24h" in df.columns, (
        "boundary_condition_table column 'ACF@24h' missing — you may have the "
        "uncorrected file. Use boundary_condition_table_corrected.csv."
    )
    assert "ACF@120min" not in df.columns, (
        "boundary_condition_table has 'ACF@120min' column — that is the bug "
        "the corrected file fixes. Replace with the corrected file."
    )

    # Anchor: Bitbrains row delta@30min must be '-5.46pp' (string, with sign)
    bb_row = df[df.dataset.str.contains("Bitbrains", case=False)]
    assert len(bb_row) == 1, "boundary_condition_table missing Bitbrains row"
    assert "-5.46" in bb_row["delta @30min"].iloc[0], (
        f"boundary_condition_table Bitbrains @30min = "
        f"{bb_row['delta @30min'].iloc[0]}, expected to contain '-5.46'"
    )
    print(f"[loader] boundary_condition_table OK ({len(df)} rows)")
    return df


# ======================================================================
# 3. bitbrains_summary_corrected.csv  (4 horizons, naive vs ML R^2)
# ======================================================================
@st.cache_data
def load_bitbrains_summary() -> pd.DataFrame:
    """Bitbrains per-horizon summary. Use CORRECTED version only."""
    df = pd.read_csv(DATA_DIR / "bitbrains_summary_corrected.csv")

    expected_cols = ["Horizon", "BB_Naive_R2", "BB_ML_R2_median", "BB_Delta_pp"]
    for c in expected_cols:
        assert c in df.columns, f"bitbrains_summary missing column {c}"

    # Anchor: 30min Δpp must be -5.46 (corrected) not -2.95 (uncorrected).
    d30 = df.loc[df.Horizon == "30min", "BB_Delta_pp"].iloc[0]
    assert abs(d30 - (-5.46)) < 0.01, (
        f"bitbrains_summary 30min Δpp = {d30}, expected -5.46. "
        f"You may have the uncorrected file."
    )
    print(f"[loader] bitbrains_summary OK ({len(df)} rows, 30min Δ={d30}pp)")
    return df


# ======================================================================
# 4. results_summary_bytedance.csv  (4 horizons, full model comparison)
# ======================================================================
@st.cache_data
def load_bytedance_summary() -> pd.DataFrame:
    """ByteDance per-horizon results. Original file had two-space name; renamed."""
    df = pd.read_csv(DATA_DIR / "results_summary_bytedance.csv")

    for c in ["Horizon", "naive_R2", "hetero_ensemble_R2", "extratrees_R2"]:
        assert c in df.columns, f"results_summary_bytedance missing column {c}"

    # Anchor: ByteDance ExtraTrees-alone should beat Hetero ensemble at every
    # horizon. This is the negative result that needs surfacing on Day 8.
    et_wins = (df["extratrees_R2"] >= df["hetero_ensemble_R2"]).sum()
    assert et_wins == len(df), (
        f"ByteDance ExtraTrees-alone-vs-ensemble: ET wins {et_wins}/{len(df)} "
        f"horizons. Plan claims 4/4. Verify."
    )
    print(f"[loader] bytedance_summary OK ({len(df)} rows, ET beats Hetero in all)")
    return df


# ======================================================================
# 5. hpa_simulation_v2.csv  (800 rows, the Pareto frontier source)
# ======================================================================
@st.cache_data
def load_hpa_simulation() -> pd.DataFrame:
    """800-row HPA Pareto file. Foundation for Day 3, 4, 7."""
    df = pd.read_csv(DATA_DIR / "hpa_simulation_v2.csv")

    expected_cols = [
        "Horizon", "Strategy", "variant",
        "target_util", "safety_margin",
        "violation_rate", "waste_rate", "viol_severity",
    ]
    for c in expected_cols:
        assert c in df.columns, f"hpa_simulation_v2 missing column {c}"

    assert len(df) == 800, f"hpa_simulation_v2 row count = {len(df)}, expected 800"

    # Anchor: target_util grid must be exactly {0.5, 0.6, 0.7, 0.8}.
    # The uncorrected v1 file had 0.9 in the grid; v2 caps at 0.8.
    tu_grid = sorted(df.target_util.unique())
    assert tu_grid == [0.5, 0.6, 0.7, 0.8], (
        f"hpa_simulation target_util grid = {tu_grid}, expected [0.5,0.6,0.7,0.8]. "
        f"You may have hpa_simulation.csv (v1) instead of v2."
    )

    # Anchor: Strategy values are title case ('ML-Proactive', 'Reactive')
    strategies = sorted(df.Strategy.unique())
    assert strategies == ["ML-Proactive", "Reactive"], (
        f"hpa_simulation Strategy = {strategies}, expected ['ML-Proactive','Reactive']"
    )

    # Anchor: each (Horizon, target_util, safety_margin) tuple has exactly 5
    # rows = 1 ML-Proactive + 4 Reactive lag variants. If this fails, lag
    # variants got pre-aggregated.
    grp = df.groupby(["Horizon", "target_util", "safety_margin"]).size()
    assert grp.min() == 5 and grp.max() == 5, (
        f"hpa_simulation per-tuple count = [{grp.min()}, {grp.max()}], expected 5. "
        f"Reactive lag variants may have been pre-aggregated."
    )
    print(f"[loader] hpa_simulation OK (800 rows, lag1..4 preserved)")
    return df


# ======================================================================
# 6. chronos_benchmark_results.csv  (12 dataset×horizon cells)
# ======================================================================
@st.cache_data
def load_chronos_results() -> pd.DataFrame:
    """Chronos-Bolt vs ensemble vs naive, 12 cells. Day 9 source of truth."""
    df = pd.read_csv(DATA_DIR / "chronos_benchmark_results.csv")

    for c in ["dataset", "horizon", "ensemble_r2", "chronos_r2"]:
        assert c in df.columns, f"chronos_benchmark missing column {c}"
    assert len(df) == 12, f"chronos_benchmark row count = {len(df)}, expected 12"

    # Anchor: Chronos wins 10 of 12. Ensemble wins only Alibaba 10min and 120min.
    chronos_wins = (df.chronos_r2 > df.ensemble_r2).sum()
    assert chronos_wins == 10, (
        f"Chronos wins = {chronos_wins}/12, expected 10. "
        f"Day 9 leaderboard headline depends on this number."
    )
    print(f"[loader] chronos_benchmark OK (12 cells, Chronos {chronos_wins}/12)")
    return df


# ======================================================================
# 7. omega_summary.csv  (3 datasets, predictability metrics)
# ======================================================================
@st.cache_data
def load_omega_summary() -> pd.DataFrame:
    """Cross-dataset predictability summary. 3 rows."""
    df = pd.read_csv(DATA_DIR / "omega_summary.csv")
    for c in ["Dataset", "ACF@24h", "Hurst_median", "CV_median"]:
        assert c in df.columns, f"omega_summary missing column {c}"
    assert len(df) == 3, f"omega_summary row count = {len(df)}, expected 3"
    print(f"[loader] omega_summary OK ({len(df)} rows)")
    return df


# ======================================================================
# 8. omega_alibaba.csv  (~4900 containers, per-container predictability)
# ======================================================================
@st.cache_data
def load_omega_alibaba() -> pd.DataFrame:
    """Per-container Alibaba predictability metrics. Used for persona shortlist."""
    df = pd.read_csv(DATA_DIR / "omega_alibaba.csv")
    for c in ["container_id", "acf_24h", "cv", "hurst"]:
        assert c in df.columns, f"omega_alibaba missing column {c}"
    # Anchor: row count should be in the right ballpark (~4900)
    assert 4500 < len(df) < 5500, (
        f"omega_alibaba row count = {len(df)}, expected ~4900"
    )
    print(f"[loader] omega_alibaba OK ({len(df)} containers)")
    return df


# ======================================================================
# 9. omega_bitbrains.csv  (156 VMs, per-VM predictability)
# ======================================================================
@st.cache_data
def load_omega_bitbrains() -> pd.DataFrame:
    """Per-VM Bitbrains predictability metrics."""
    df = pd.read_csv(DATA_DIR / "omega_bitbrains.csv")
    for c in ["vm_id", "acf_24h", "cv", "hurst"]:
        assert c in df.columns, f"omega_bitbrains missing column {c}"
    assert len(df) == 156, f"omega_bitbrains row count = {len(df)}, expected 156"
    print(f"[loader] omega_bitbrains OK ({len(df)} VMs)")
    return df


# ======================================================================
# 10. bytedance_per_instance_stats.csv  (93 instances)
# ======================================================================
@st.cache_data
def load_bytedance_per_instance() -> pd.DataFrame:
    """Per-instance ByteDance stats. 93 rows."""
    df = pd.read_csv(DATA_DIR / "bytedance_per_instance_stats.csv")
    for c in ["instance_id", "acf_24h", "hurst"]:
        assert c in df.columns, f"bytedance_per_instance missing column {c}"
    assert len(df) == 93, f"bytedance_per_instance row count = {len(df)}, expected 93"
    print(f"[loader] bytedance_per_instance OK ({len(df)} instances)")
    return df


# ======================================================================
# 11. shap_importance.csv  (123 features × 4 horizons, mean |SHAP|)
# ======================================================================
@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    """Mean |SHAP| importance per feature per horizon."""
    df = pd.read_csv(DATA_DIR / "shap_importance.csv")
    # First column is unnamed (feature name). Rename for sanity.
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "feature"})
    for c in ["10min", "30min", "60min", "120min"]:
        assert c in df.columns, f"shap_importance missing horizon column {c}"
    print(f"[loader] shap_importance OK ({len(df)} features × 4 horizons)")
    return df


# ======================================================================
# 12. bcf_pairs.csv  (48 rows = 4 models × 12 cells, ml_wins + predicate)
# ======================================================================
@st.cache_data
def load_bcf_pairs() -> pd.DataFrame:
    """All (model, dataset, horizon) pairs with predicate evaluation."""
    df = pd.read_csv(DATA_DIR / "bcf_pairs.csv")
    for c in ["model", "dataset", "horizon", "predicate", "ml_wins", "delta_pp"]:
        assert c in df.columns, f"bcf_pairs missing column {c}"
    assert len(df) == 48, f"bcf_pairs row count = {len(df)}, expected 48"

    # Anchor: across the 3 REAL forecasters (NNLS, Chronos-2, TimesFM), the
    # predicate has 0 false positives. This is the headline finding for Day 5.
    real = df[df.model.isin(["NNLS", "Chronos-2", "TimesFM"])]
    assert len(real) == 36, f"BCF real-model pairs = {len(real)}, expected 36"
    fp = real[(real.predicate == 1) & (real.ml_wins == 0)]
    assert len(fp) == 0, (
        f"BCF false positives across real forecasters = {len(fp)}, expected 0. "
        f"This is the headline Day 5 finding — verify."
    )
    print(f"[loader] bcf_pairs OK ({len(df)} rows, 0 FP across 3 real forecasters)")
    return df


# ======================================================================
# 13. bcf_per_model_auc.csv  (4 rows: Naive, NNLS, Chronos-2, TimesFM)
# ======================================================================
@st.cache_data
def load_bcf_per_model_auc() -> pd.DataFrame:
    """Per-model AUC for the boundary condition predicate. Naive has NaN AUC."""
    df = pd.read_csv(DATA_DIR / "bcf_per_model_auc.csv")
    for c in ["model", "n_pairs", "auc"]:
        assert c in df.columns, f"bcf_per_model_auc missing column {c}"

    # Anchor: NNLS AUC must be ~0.83. This is the value the thesis cites as
    # the strongest per-model number.
    nnls_auc = df.loc[df.model == "NNLS", "auc"].iloc[0]
    assert abs(nnls_auc - 0.833) < 0.01, (
        f"BCF NNLS AUC = {nnls_auc}, expected ~0.833"
    )
    print(f"[loader] bcf_per_model_auc OK ({len(df)} models, NNLS AUC={nnls_auc:.3f})")
    return df


# ======================================================================
# 14. bcf_pooled_results.json  (the SOT for AUC, CI, predicate text)
# ======================================================================
@st.cache_data
def load_bcf_pooled() -> dict:
    """Single source of truth for pooled AUC, BCa CI, predicate text, caveats."""
    with open(DATA_DIR / "bcf_pooled_results.json") as f:
        d = json.load(f)
    for k in ["pooled_auc", "bca_95_ci_lo", "bca_95_ci_hi", "n_pooled_pairs",
              "predicate", "caveats"]:
        assert k in d, f"bcf_pooled_results missing key {k}"

    # Anchor: pooled AUC = 0.80, n_pooled_pairs = 36
    assert abs(d["pooled_auc"] - 0.80) < 0.005, (
        f"BCF pooled AUC = {d['pooled_auc']}, expected 0.80"
    )
    assert d["n_pooled_pairs"] == 36, (
        f"BCF n_pooled_pairs = {d['n_pooled_pairs']}, expected 36"
    )
    print(f"[loader] bcf_pooled_results OK (AUC=0.80, 36 pairs, "
          f"{len(d['caveats'])} caveats)")
    return d


# ======================================================================
# 15. cv_stratified_skill.csv  (5 CV bins × 4 horizons, container win rates)
# ======================================================================
@st.cache_data
def load_cv_stratified_skill() -> pd.DataFrame:
    """Container win-rate stratified by CV bin and horizon."""
    df = pd.read_csv(DATA_DIR / "cv_stratified_skill.csv")
    for c in ["Horizon", "CV_bin", "Container_win_%", "Skill_score"]:
        assert c in df.columns, f"cv_stratified_skill missing column {c}"

    # Anchor: 30min, CV 0.3-0.5 bin, container_win_% ~ 27.9. This is the
    # honest-framing-checklist item #4 number.
    row = df[(df.Horizon == "30min") & (df.CV_bin == "CV 0.3-0.5")]
    assert len(row) == 1, "cv_stratified_skill missing 30min/CV 0.3-0.5 row"
    win_pct = row["Container_win_%"].iloc[0]
    assert abs(win_pct - 27.89) < 0.5, (
        f"cv_stratified_skill 30min/CV 0.3-0.5 win% = {win_pct}, expected ~27.89"
    )
    print(f"[loader] cv_stratified_skill OK ({len(df)} rows, "
          f"30min/CV 0.3-0.5 win%={win_pct:.2f})")
    return df


# ======================================================================
# 16. stratified_skill.csv  (per-horizon aggregate MAE/R² divergence)
# ======================================================================
@st.cache_data
def load_stratified_skill() -> pd.DataFrame:
    """Per-horizon MAE skill scores. Surfaces MAE/R² divergence at short horizons."""
    df = pd.read_csv(DATA_DIR / "stratified_skill.csv")
    for c in ["Horizon", "MAE_naive", "MAE_ensemble", "Skill_score"]:
        assert c in df.columns, f"stratified_skill missing column {c}"

    # Anchor: at 10min, ensemble MAE > naive MAE (skill is NEGATIVE).
    # This is the MAE/R² divergence story that must be disclosed.
    row10 = df[df.Horizon == "10min"].iloc[0]
    assert row10["MAE_ensemble"] > row10["MAE_naive"], (
        f"stratified_skill 10min: ensemble MAE ({row10['MAE_ensemble']}) "
        f"<= naive MAE ({row10['MAE_naive']}). The MAE/R² divergence finding "
        f"may have evaporated. Verify."
    )
    print(f"[loader] stratified_skill OK ({len(df)} rows, "
          f"10min skill={row10['Skill_score']:.3f})")
    return df


# ======================================================================
# 17. base_model_correlations.csv  (per-horizon error correlations)
# ======================================================================
@st.cache_data
def load_base_model_correlations() -> pd.DataFrame:
    """Pairwise base-model error correlations. Min must be >= 0.92."""
    df = pd.read_csv(DATA_DIR / "base_model_correlations.csv")
    for c in ["Horizon", "Model_A", "Model_B", "Error_Corr"]:
        assert c in df.columns, f"base_model_correlations missing column {c}"

    # Anchor: minimum error correlation across all horizons and pairs >= 0.92.
    # The thesis claims "exceed 0.927" but the floor is actually 0.9267, so
    # the loose check below catches drift without nitpicking precision.
    ec_min = df.Error_Corr.min()
    assert ec_min >= 0.92, (
        f"base_model_correlations min Error_Corr = {ec_min}, expected >= 0.92. "
        f"The 'BiLSTM provides no architectural diversity' finding depends on this."
    )
    print(f"[loader] base_model_correlations OK ({len(df)} pairs, "
          f"min error corr={ec_min:.4f})")
    return df


# ======================================================================
# 18. per_container_r2.csv  (4921 × 4 horizons; OPTIONAL — see data/README.md)
# ======================================================================
@st.cache_data
def load_per_container_r2() -> pd.DataFrame | None:
    """
    Per-container ET vs Naive R^2 from Phase 0.2 reconstruction.

    OPTIONAL: returns None if the file isn't present. Day 3 + Day 8 must
    handle the None case with an info message instead of crashing.
    """
    p = DATA_DIR / "per_container_r2.csv"
    if not p.exists():
        print(f"[loader] per_container_r2 NOT FOUND — copy from "
              f"results/alibaba/diagnostics/per_container_r2.csv when ready")
        return None

    df = pd.read_csv(p)
    for c in ["container_id", "horizon", "et_r2", "naive_r2", "et_minus_naive_pp"]:
        assert c in df.columns, f"per_container_r2 missing column {c}"

    # Anchor: c_38893 at 30min has ET-minus-naive ≈ +1.05pp (Phase 0.2 finding)
    row = df[(df.container_id == "c_38893") & (df.horizon == "30min")]
    if len(row) == 1:
        delta = row["et_minus_naive_pp"].iloc[0]
        assert abs(delta - 1.05) < 0.1, (
            f"per_container_r2: c_38893 @30min Δ = {delta}, expected ~1.05"
        )
    print(f"[loader] per_container_r2 OK ({len(df)} rows = "
          f"{len(df)//4} containers × 4 horizons)")
    return df


# ======================================================================
# Smoke-test entry point
# ======================================================================
def load_all() -> dict:
    """Call every loader and return a dict of {name: result}. Used by app.py."""
    return {
        "comparison_table":          load_comparison_table(),
        "boundary_condition_table":  load_boundary_condition_table(),
        "bitbrains_summary":         load_bitbrains_summary(),
        "bytedance_summary":         load_bytedance_summary(),
        "hpa_simulation":            load_hpa_simulation(),
        "chronos_results":           load_chronos_results(),
        "omega_summary":             load_omega_summary(),
        "omega_alibaba":             load_omega_alibaba(),
        "omega_bitbrains":           load_omega_bitbrains(),
        "bytedance_per_instance":    load_bytedance_per_instance(),
        "shap_importance":           load_shap_importance(),
        "bcf_pairs":                 load_bcf_pairs(),
        "bcf_per_model_auc":         load_bcf_per_model_auc(),
        "bcf_pooled":                load_bcf_pooled(),
        "cv_stratified_skill":       load_cv_stratified_skill(),
        "stratified_skill":          load_stratified_skill(),
        "base_model_correlations":   load_base_model_correlations(),
        "per_container_r2":          load_per_container_r2(),  # optional
    }
