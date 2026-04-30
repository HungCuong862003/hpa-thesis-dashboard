"""
verdict_templates.py — Day 2 deliverable
==========================================

Three category-level verdict templates rendered by the recommendation page.
Templates encode the honest-framing constraint structurally:

  - Every verdict is one of the three constants below, period.
  - Every verdict carries the negative findings the thesis surfaces:
      * Chronos-Bolt wins 10/12 dataset×horizon comparisons.
      * On ByteDance, ExtraTrees-alone outperforms the hetero ensemble at
        all four horizons by 5.5–9.6pp R² — "ensemble" is a thin layer.
      * Base-model error correlations exceed 0.927 (architectural diversity
        is illusory).
      * BCF pooled AUC = 0.80 [0.70, 0.88] across 36 pairs (non-iid caveat).
      * Lag-sensitivity: at lag=1 (production-realistic) proactive dominates
        59.4% of cells; at lag=4, 98.1%. The verdict is sensitive to the
        reactive baseline you compare against.

Anchor numbers below are computed against hpa_simulation_v2.csv
(800 rows = 160 unique (Horizon, target_util, safety_margin) cells × 5
strategy rows each = 1 ML-Proactive + 4 Reactive lag variants). Verified
by audit/verify_numbers.py.

Style: deliberately verbose, factual, no marketing language. Render as markdown.

Author: Jimmy
Date: May 2026
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# Anchor constants — change these in ONE place if the simulation rerun
# produces different numbers. Verified by audit/verify_numbers.py.
# ----------------------------------------------------------------------

# Cell-level dominance counts (out of 160 cells per lag), computed against
# hpa_simulation_v2.csv with weak Pareto dominance (EPS=1e-9).
LAG_DOMINANCE = {
    1: {"proactive": 95,  "reactive": 2, "neither": 63, "total": 160},
    2: {"proactive": 132, "reactive": 0, "neither": 28, "total": 160},
    3: {"proactive": 149, "reactive": 0, "neither": 11, "total": 160},
    4: {"proactive": 157, "reactive": 0, "neither":  3, "total": 160},
}

# Per-dataset ACF@24h > 0.2 fractions (predicate eligibility).
ACF_FRACTIONS = {
    "Alibaba":   {"n_pos": 2535, "n_total": 4902, "pct": 51.7},
    "Bitbrains": {"n_pos":   61, "n_total":  156, "pct": 39.1},
    "ByteDance": {"n_pos":   67, "n_total":   93, "pct": 72.0},
}


# ----------------------------------------------------------------------
# Shared footnotes — every verdict carries these. Concentrating them in
# one constant is what makes "soften the verdict" hard later: there is no
# single sentence to delete, only this whole block, which would also
# delete equally-true context for the alternate verdicts.
# ----------------------------------------------------------------------

_NEGATIVE_FINDINGS = """
**What this verdict does not say.** This recommendation reflects forecasting
accuracy from the NNLS hetero ensemble that produced the simulation. Three
items the thesis discloses, included here so the verdict is read in context:

- **Chronos-Bolt outperforms our ensemble on 10 of 12 dataset×horizon cells.**
  The ensemble wins only at Alibaba 10min and Alibaba 120min. A foundation
  model with no domain tuning would, on most cells, give better point
  forecasts than the ensemble that drives the proactive strategy here.
- **On ByteDance, ExtraTrees-alone outperforms the hetero ensemble at every
  horizon by 5.5–9.6pp R².** The "hybrid" framing is technically accurate
  (the pipeline does combine four base learners via NNLS) but empirically
  the ensemble's predictions are dominated by a single tree model.
- **Base-model error correlations span 0.9267 to 0.9968 (mean 0.981).**
  BiLSTM contributes weight at longer horizons because it is locally more
  accurate, not because it decorrelates errors. Architectural diversity is
  the design intent; numerically the diversity is small.
""".strip()


_LAG_DISCLOSURE = """
**Lag sensitivity.** Across the 160-cell simulation grid (4 horizons × 4
target utilizations × 10 safety margins), how often ML-Proactive weakly
Pareto-dominates the Reactive baseline depends materially on the reactive
controller's reaction lag:

| Reactive lag | Proactive dominates | Reactive dominates | Neither |
|---|---|---|---|
| lag = 1 (production-realistic) | 95/160 (59.4%) | 2/160 (1.2%) | 63/160 (39.4%) |
| lag = 2 | 132/160 (82.5%) | 0/160 (0.0%) | 28/160 (17.5%) |
| lag = 3 | 149/160 (93.1%) | 0/160 (0.0%) | 11/160 (6.9%) |
| lag = 4 | 157/160 (98.1%) | 0/160 (0.0%) | 3/160 (1.9%) |

The thesis defaults the comparison to lag = 1 because that is what a tuned
production reactive HPA achieves. Higher lags inflate proactive's apparent
advantage. The two cells where reactive weakly dominates at lag = 1 are
both at 120min horizon × 0.8 target utilization — long-horizon ML
predictions degrade enough that a 1-step-lag reactive controller comes out
ahead.
""".strip()


_BCF_DISCLOSURE = """
**Predicate eligibility for your workload.** The thesis's boundary
condition predicate is: *ACF@24h > 0.2 AND horizon ≥ 30min*. Across 36
pairs (12 dataset×horizon cells × 3 real forecasters: NNLS, Chronos-2,
TimesFM), the predicate has 0 false positives and 12 false negatives.
Pooled AUC = 0.80, BCa 95% CI [0.70, 0.88]. Caveat: the 36 pairs are not
iid, so the CI overstates precision relative to cluster-aware inference at
n = 12 cells.

Workload-level eligibility varies by dataset:

- **Bitbrains**: 39.1% of VMs (61 / 156) have ACF@24h > 0.2.
- **Alibaba**: 51.7% of containers (2,535 / 4,902) have ACF@24h > 0.2.
- **ByteDance**: 72.0% of instances (67 / 93) have ACF@24h > 0.2.

Predicate eligibility you would expect to encounter therefore depends on
which production environment your workload most resembles, not on a single
"average" answer.
""".strip()


def _common_footnotes() -> str:
    """All three verdicts append the same footnotes block."""
    return "\n\n".join([_LAG_DISCLOSURE, _BCF_DISCLOSURE, _NEGATIVE_FINDINGS])


# ----------------------------------------------------------------------
# The three verdict templates
# ----------------------------------------------------------------------

def _format_pareto_block(ctx: dict) -> str:
    """Format the user's chosen-config Pareto comparison as a markdown table."""
    p_sev = ctx.get('proactive_severity', 0.0)
    r_sev = ctx.get('reactive_severity', 0.0)
    return (
        f"| Strategy | violation_rate | waste_rate | viol_severity |\n"
        f"|---|---|---|---|\n"
        f"| ML-Proactive | {ctx['proactive_violation']:.4f} | "
        f"{ctx['proactive_waste']:.4f} | {p_sev:.4f} |\n"
        f"| Reactive (lag={ctx['reactive_lag']}) | "
        f"{ctx['reactive_violation']:.4f} | "
        f"{ctx['reactive_waste']:.4f} | {r_sev:.4f} |"
    )


def _format_config(ctx: dict) -> str:
    return (
        f"horizon={ctx['horizon']}, target_util={ctx['target_util']:.1f}, "
        f"safety_margin={ctx['safety_margin']:.3f}×, "
        f"reactive_lag={ctx['reactive_lag']}"
    )


def _proactive_template(ctx: dict) -> str:
    """Verdict: ML-Proactive weakly Pareto-dominates Reactive at the chosen config."""
    n_p = LAG_DOMINANCE[ctx['reactive_lag']]['proactive']
    return f"""
### Verdict: proactive autoscaling

At the chosen configuration ({_format_config(ctx)}), ML-Proactive weakly
Pareto-dominates Reactive: it is no worse on either axis (violation rate,
waste rate) and strictly better on at least one.

{_format_pareto_block(ctx)}

This cell is one of **{n_p} out of 160** (lag={ctx['reactive_lag']}) where
proactive dominates in the simulation grid. For context on how that count
shifts with the reactive baseline, see *Lag sensitivity* below.

{_common_footnotes()}
""".strip()


def _reactive_template(ctx: dict) -> str:
    """Verdict: Reactive weakly Pareto-dominates ML-Proactive at the chosen config."""
    n_r = LAG_DOMINANCE[ctx['reactive_lag']]['reactive']
    n_total = LAG_DOMINANCE[ctx['reactive_lag']]['total']
    return f"""
### Verdict: reactive HPA

At the chosen configuration ({_format_config(ctx)}), the Reactive baseline
weakly Pareto-dominates ML-Proactive: it is no worse on either axis and
strictly better on at least one.

{_format_pareto_block(ctx)}

This is one of **{n_r} out of {n_total}** cells (lag={ctx['reactive_lag']})
where reactive dominates. At lag = 1, the only cells where this happens are
at the 120min horizon with 0.8 target utilization — long-horizon ML
predictions degrade enough that even a 1-step-lag reactive controller wins
on both axes. At lag ≥ 2, reactive never weakly dominates.

The thesis's broader claim is that ML adds value when the workload has a
strong diurnal cycle and the prediction horizon is long enough for the
forecaster to outpace lag — neither condition is being met here.

{_common_footnotes()}
""".strip()


def _neither_template(ctx: dict) -> str:
    """Verdict: neither weakly dominates — the user faces a Pareto trade-off."""
    n_n = LAG_DOMINANCE[ctx['reactive_lag']]['neither']
    n_total = LAG_DOMINANCE[ctx['reactive_lag']]['total']
    d_v = ctx['proactive_violation'] - ctx['reactive_violation']
    d_w = ctx['proactive_waste']     - ctx['reactive_waste']
    # Describe the trade-off direction
    if d_v < 0 and d_w > 0:
        tradeoff = (
            "ML-Proactive has the lower violation rate; "
            "Reactive has the lower waste rate."
        )
    elif d_v > 0 and d_w < 0:
        tradeoff = (
            "Reactive has the lower violation rate; "
            "ML-Proactive has the lower waste rate."
        )
    else:
        tradeoff = (
            "The two strategies are within numerical tolerance on at least "
            "one axis; neither is uniformly better."
        )
    return f"""
### Verdict: neither dominates — Pareto trade-off

At the chosen configuration ({_format_config(ctx)}), neither strategy
weakly Pareto-dominates the other. {tradeoff}

{_format_pareto_block(ctx)}

ML-Proactive minus Reactive: Δviolation = {d_v:+.4f},
Δwaste = {d_w:+.4f}.

This is one of **{n_n} out of {n_total}** cells (lag={ctx['reactive_lag']})
with this pattern. At lag = 1 the trade-off cells cluster at the horizon
extremes (10min, 120min); at the middle horizons (30min, 60min) the verdict
resolves cleanly to proactive in >90% of cells.

Pick the strategy whose dominant axis matches your operational priority.
If your SLO penalizes violations far more than overprovisioning, pick
proactive; if waste is the binding constraint, pick reactive.

{_common_footnotes()}
""".strip()


# ----------------------------------------------------------------------
# Public entrypoint — recommend.py calls this with a verdict kind + context
# ----------------------------------------------------------------------

_DISPATCH = {
    "PROACTIVE_DOMINATES": _proactive_template,
    "REACTIVE_DOMINATES":  _reactive_template,
    "NEITHER_DOMINATES":   _neither_template,
}


def render(verdict_kind: str, ctx: dict) -> str:
    """
    Render one of the three templates with values filled in from ctx.

    Required ctx keys:
      horizon: str            ('10min', '30min', '60min', '120min')
      target_util: float      (∈ {0.5, 0.6, 0.7, 0.8})
      safety_margin: float    (∈ {1.0, 1.067, ..., 1.6})
      reactive_lag: int       (∈ {1, 2, 3, 4})
      proactive_violation: float
      proactive_waste: float
      reactive_violation: float
      reactive_waste: float
    Optional:
      proactive_severity: float
      reactive_severity: float

    Returns the formatted markdown string for the verdict expander.
    """
    if verdict_kind not in _DISPATCH:
        raise ValueError(
            f"Unknown verdict_kind={verdict_kind!r}. "
            f"Must be one of {list(_DISPATCH.keys())}"
        )
    if ctx['reactive_lag'] not in (1, 2, 3, 4):
        raise ValueError(
            f"reactive_lag={ctx['reactive_lag']} not in {{1,2,3,4}}"
        )
    return _DISPATCH[verdict_kind](ctx)


# Constants exposed for tests / framing audit
PROACTIVE_DOMINATES = "PROACTIVE_DOMINATES"
REACTIVE_DOMINATES  = "REACTIVE_DOMINATES"
NEITHER_DOMINATES   = "NEITHER_DOMINATES"
