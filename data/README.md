# Data folder — provenance and freeze policy

These CSVs and the JSON file are frozen artefacts copied from the thesis repo
(`kubernetes-cpu-ensemble-thesis/`). The dashboard reads them as ground truth.
No retraining, no live cluster, no recomputation. Numbers must match what the
thesis manuscript cites.

## File-by-file source map

| Dashboard filename                          | Source path in thesis repo                                                  |
|---------------------------------------------|-----------------------------------------------------------------------------|
| `comparison_table.csv`                      | `reports/tables/comparison_table.csv`                                       |
| `boundary_condition_table_corrected.csv`    | `reports/tables/boundary_condition_table_corrected.csv` (NOT the uncorrected version) |
| `bitbrains_summary_corrected.csv`           | `results/bitbrains/bitbrains_summary_corrected.csv` (NOT the uncorrected version) |
| `results_summary_bytedance.csv`             | `results/bytedance/results_summary.csv` (renamed; original had 2 spaces)    |
| `hpa_simulation_v2.csv`                     | `results/hpa_simulation/hpa_simulation_v2.csv`                              |
| `chronos_benchmark_results.csv`             | `reports/tables/chronos_benchmark_results.csv`                              |
| `omega_summary.csv`                         | `results/omega_summary.csv`                                                 |
| `omega_alibaba.csv`                         | `results/alibaba/omega_alibaba.csv`                                         |
| `omega_bitbrains.csv`                       | `results/bitbrains/omega_bitbrains.csv`                                     |
| `bytedance_per_instance_stats.csv`          | `results/bytedance/bytedance_per_instance_stats.csv`                        |
| `shap_importance.csv`                       | `reports/tables/shap_importance.csv`                                        |
| `bcf_pairs.csv`                             | `results/bcf/bcf_pairs.csv`                                                 |
| `bcf_per_model_auc.csv`                     | `results/bcf/bcf_per_model_auc.csv`                                         |
| `bcf_pooled_results.json`                   | `results/bcf/bcf_pooled_results.json`                                       |
| `cv_stratified_skill.csv`                   | `reports/tables/cv_stratified_skill.csv`                                    |
| `stratified_skill.csv`                      | `reports/tables/stratified_skill.csv`                                       |
| `base_model_correlations.csv`               | `reports/tables/base_model_correlations.csv`                                |
| `per_container_r2.csv` *(see below)*        | `results/alibaba/diagnostics/per_container_r2.csv`                          |

## Critical: corrected versions only

Two files in the thesis repo have an uncorrected sibling that LOOKS identical
but is wrong. Always use the `_corrected.csv` version:

- **`boundary_condition_table_corrected.csv`** — the uncorrected version had
  the `ACF@120min` column relabeled as `ACF@24h`. The relabel is the bug.
- **`bitbrains_summary_corrected.csv`** — the uncorrected version had a sign
  error on Δ@30min. Corrected: −5.46pp. Uncorrected: −2.95pp (or whatever).

The data loaders in `modules/data_loaders.py` assert anchor cell values that
are unique to the corrected files. If the uncorrected file is dropped in by
mistake, the loader fails at import time, not at render time.

## per_container_r2.csv — must be copied separately

This file is too large to live alongside the others without thinking about it
carefully (1.4 MB, 19,684 rows). Copy it from your Drive:

    cp /content/drive/MyDrive/kubernetes-cpu-ensemble-thesis/results/alibaba/diagnostics/per_container_r2.csv data/

Until you do, the loader for it returns `None` and Day 3's per-container
panel renders as a "data unavailable" info message. Other pages do not depend
on this file.

## Figure PDFs (data/figures/)

Two pre-rendered figures live here for the SHAP page (Day 10):

- `shap_heatmap.pdf`        — from `reports/figures/shap_heatmap.pdf`
- `shap_group_bars_v2.pdf`  — from `reports/figures/shap_group_bars_v2.pdf`

The Day 10 page renders these as static images inside an expander. Copy them
in before you start Day 10 work.

## Authoritative source for BCF numbers

`bcf_pooled_results.json` is the SINGLE source of truth for:

- pooled AUC = 0.80
- BCa 95% CI = [0.704, 0.879]
- n_pooled_pairs = 36 (12 cells × 3 forecasters)
- predicate text: `(ACF@24h > 0.2) AND (horizon >= 30min)`
- caveat list (must surface verbatim on Day 5)

`bcf_per_model_auc.csv` is the per-model breakdown. It has 4 rows including
`Naive` (which has `auc = NaN` because Naive never wins by construction —
filter it out before display).
