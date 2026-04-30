"""
HPA Thesis Dashboard — entry point.

Day 1 deliverable: a stub homepage that calls every data loader and
shows a green check mark per file. If anything fails to load, the page
reports which file and why.

Pages get added under pages/ in Day 3 onward; Streamlit auto-discovers
files matching pages/N_*.py and builds the sidebar.

Run locally:
    streamlit run app.py

Author: Jimmy
"""

from __future__ import annotations

import sys
import traceback

import streamlit as st

# Make the modules/ folder importable
sys.path.insert(0, ".")
from modules import data_loaders  # noqa: E402

st.set_page_config(
    page_title="HPA Thesis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HPA Autoscaling Thesis Dashboard")
st.caption(
    "Companion artefact for *Hybrid Ensemble Learning for Proactive Resource "
    "Prediction in Kubernetes* — Jimmy, IU SCSE, 2026. "
    "Frozen at commit `<TBD-after-Day-14-tag>`. All numbers verified against "
    "thesis manuscript via `audit/verify_numbers.py`."
)

st.markdown("---")
st.subheader("Day 1 smoke test")
st.write(
    "Calls every cached loader, surfaces success/failure per file. If "
    "anything is red, the dashboard cannot render — fix the data folder "
    "before continuing."
)

# Map of loader name -> callable
LOADER_MAP = {
    "comparison_table.csv":             data_loaders.load_comparison_table,
    "boundary_condition_table_corrected.csv": data_loaders.load_boundary_condition_table,
    "bitbrains_summary_corrected.csv":  data_loaders.load_bitbrains_summary,
    "results_summary_bytedance.csv":    data_loaders.load_bytedance_summary,
    "hpa_simulation_v2.csv":            data_loaders.load_hpa_simulation,
    "chronos_benchmark_results.csv":    data_loaders.load_chronos_results,
    "omega_summary.csv":                data_loaders.load_omega_summary,
    "omega_alibaba.csv":                data_loaders.load_omega_alibaba,
    "omega_bitbrains.csv":              data_loaders.load_omega_bitbrains,
    "bytedance_per_instance_stats.csv": data_loaders.load_bytedance_per_instance,
    "shap_importance.csv":              data_loaders.load_shap_importance,
    "bcf_pairs.csv":                    data_loaders.load_bcf_pairs,
    "bcf_per_model_auc.csv":            data_loaders.load_bcf_per_model_auc,
    "bcf_pooled_results.json":          data_loaders.load_bcf_pooled,
    "cv_stratified_skill.csv":          data_loaders.load_cv_stratified_skill,
    "stratified_skill.csv":             data_loaders.load_stratified_skill,
    "base_model_correlations.csv":      data_loaders.load_base_model_correlations,
    "per_container_r2.csv (optional)":  data_loaders.load_per_container_r2,
}

cols = st.columns(2)
results = []
for i, (name, fn) in enumerate(LOADER_MAP.items()):
    target = cols[i % 2]
    try:
        out = fn()
        if out is None:
            # The optional per_container_r2 case
            target.warning(f"⚠ {name} — file not present (optional, see data/README.md)")
            results.append((name, "missing-optional"))
        else:
            n = len(out) if hasattr(out, "__len__") else "?"
            target.success(f"✓ {name} — {n} {'rows' if isinstance(n, int) else 'items'}")
            results.append((name, "ok"))
    except AssertionError as e:
        target.error(f"✗ {name} — anchor failed: {e}")
        results.append((name, f"anchor-failed: {e}"))
    except FileNotFoundError as e:
        target.error(f"✗ {name} — file not found: {e}")
        results.append((name, "missing"))
    except Exception as e:
        target.error(f"✗ {name} — {type(e).__name__}: {e}")
        target.code(traceback.format_exc())
        results.append((name, f"error: {e}"))

# Summary line
st.markdown("---")
ok_count = sum(1 for _, s in results if s == "ok")
warn_count = sum(1 for _, s in results if s == "missing-optional")
fail_count = len(results) - ok_count - warn_count
if fail_count == 0:
    st.success(
        f"All required loaders OK ({ok_count} green, {warn_count} optional missing). "
        f"Day 1 smoke test passed."
    )
else:
    st.error(
        f"{fail_count} loader(s) failed. Fix the data folder before continuing."
    )

st.markdown("---")
st.markdown(
    "**Pages will be added Day 3 onward.** See "
    "`docs/dashboard_implementation_plan_final.md` for the full schedule."
)
