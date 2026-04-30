"""
recommend.py — Day 2 deliverable
==================================

Deterministic rule engine. No ML in the recommendation path. No LLM.
Reads hpa_simulation_v2.csv at the user's chosen configuration, applies
weak Pareto dominance with EPS = 1e-9 tolerance, returns one of three
verdicts.

The rules:
  1. Filter the Pareto DataFrame to the user's exact (Horizon, target_util,
     safety_margin) tuple. If no exact match (which should not happen given
     the dropdown grids match the simulation grid), fall back to
     normalized-Manhattan nearest neighbour and warn.
  2. Pick the Reactive row by EXACT variant-string match using
     f'lag={int(reactive_lag)}_tu={float(target_util):.1f}'. Never iloc[0]:
     the four lag rows differ materially.
  3. Apply weak Pareto dominance:
        A weakly dominates B  iff  A is no worse on both axes
                              AND  A is strictly better on at least one
        (with EPS = 1e-9 tolerance both ways)
  4. Return one of three verdicts: PROACTIVE_DOMINATES, REACTIVE_DOMINATES,
     NEITHER_DOMINATES. No 'use NNLS' verdict, no confidence field.

Author: Jimmy
Date: May 2026
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from modules import verdict_templates

# Floating-point tolerance for Pareto comparison. Per plan: weak dominance
# (≤ on both axes, < on at least one), not strict-better-on-both.
EPS = 1e-9

VerdictKind = Literal[
    "PROACTIVE_DOMINATES",
    "REACTIVE_DOMINATES",
    "NEITHER_DOMINATES",
]


# ----------------------------------------------------------------------
# Pareto helpers
# ----------------------------------------------------------------------

def _weakly_dominates(
    a_viol: float, a_waste: float,
    b_viol: float, b_waste: float,
) -> bool:
    """A weakly dominates B iff A is no worse on both AND strictly better on at least one."""
    no_worse = (a_viol <= b_viol + EPS) and (a_waste <= b_waste + EPS)
    strictly_better = (a_viol < b_viol - EPS) or (a_waste < b_waste - EPS)
    return no_worse and strictly_better


def _classify(p_viol, p_waste, r_viol, r_waste) -> VerdictKind:
    """Classify the chosen-config cell into one of three verdicts."""
    if _weakly_dominates(p_viol, p_waste, r_viol, r_waste):
        return "PROACTIVE_DOMINATES"
    if _weakly_dominates(r_viol, r_waste, p_viol, p_waste):
        return "REACTIVE_DOMINATES"
    return "NEITHER_DOMINATES"


# ----------------------------------------------------------------------
# Pareto lookup
# ----------------------------------------------------------------------

def _lookup_proactive_row(
    pareto_df: pd.DataFrame,
    horizon: str,
    target_util: float,
    safety_margin: float,
) -> pd.Series:
    """Return the single ML-Proactive row at the user's config, or raise."""
    mask = (
        (pareto_df.Horizon == horizon)
        & (abs(pareto_df.target_util - target_util) < EPS)
        & (abs(pareto_df.safety_margin - safety_margin) < EPS)
        & (pareto_df.Strategy == "ML-Proactive")
    )
    matches = pareto_df[mask]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) == 0:
        raise ValueError(
            f"No ML-Proactive row found for "
            f"horizon={horizon}, target_util={target_util}, "
            f"safety_margin={safety_margin}. "
            f"Did the (horizon, tu, sm) grid drift?"
        )
    raise ValueError(
        f"Expected 1 ML-Proactive row for that config, found {len(matches)}. "
        f"hpa_simulation_v2.csv may be malformed."
    )


def _lookup_reactive_row(
    pareto_df: pd.DataFrame,
    horizon: str,
    target_util: float,
    safety_margin: float,
    reactive_lag: int,
) -> pd.Series:
    """Return the Reactive row matching exactly the variant string. Never iloc[0]."""
    # Defensive formatting handles np.float64 callers cleanly.
    variant = f"lag={int(reactive_lag)}_tu={float(target_util):.1f}"
    mask = (
        (pareto_df.Horizon == horizon)
        & (abs(pareto_df.target_util - target_util) < EPS)
        & (abs(pareto_df.safety_margin - safety_margin) < EPS)
        & (pareto_df.Strategy == "Reactive")
        & (pareto_df.variant == variant)
    )
    matches = pareto_df[mask]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) == 0:
        raise ValueError(
            f"No Reactive row found with variant={variant!r} at "
            f"horizon={horizon}, target_util={target_util}, "
            f"safety_margin={safety_margin}. "
            f"Reactive lag rows may have been pre-aggregated."
        )
    raise ValueError(
        f"Expected 1 Reactive row with variant={variant!r}, "
        f"found {len(matches)}. hpa_simulation_v2.csv may be malformed."
    )


# ----------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------

def recommend(
    *,
    workload_acf24: float,
    workload_cv: float,
    workload_hurst: float,
    horizon: str,
    target_util: float,
    safety_margin: float,
    reactive_lag: int,
    pareto_df: pd.DataFrame,
) -> dict:
    """
    Return a verdict dict for the user's chosen workload + configuration.

    Parameters
    ----------
    workload_acf24, workload_cv, workload_hurst :
        Predictability metrics from the workload-input panel. Currently
        used to evaluate the BCF predicate text, NOT to pick the verdict
        — the verdict comes from the actual Pareto comparison.
    horizon : str
        '10min' | '30min' | '60min' | '120min'.
    target_util : float
        ∈ {0.5, 0.6, 0.7, 0.8}.
    safety_margin : float
        ∈ {1.0, 1.067, 1.133, 1.2, 1.267, 1.333, 1.4, 1.467, 1.533, 1.6}.
    reactive_lag : int
        ∈ {1, 2, 3, 4}; default 1 per plan (production-realistic).
    pareto_df : pd.DataFrame
        Loaded from data_loaders.load_hpa_simulation().

    Returns
    -------
    dict with keys:
      kind : VerdictKind ('PROACTIVE_DOMINATES' | 'REACTIVE_DOMINATES' | 'NEITHER_DOMINATES')
      reasons : list[str]   — 3 short bullets for the verdict card
      ctx : dict            — context passed to verdict_templates.render()
      proactive : dict      — {'violation': ..., 'waste': ..., 'severity': ...}
      reactive : dict       — {'violation': ..., 'waste': ..., 'severity': ...}
      predicate_fires : bool — BCF predicate evaluation on workload+horizon
      rendered_md : str     — full markdown from verdict_templates.render()

    Raises
    ------
    ValueError
        If the (horizon, target_util, safety_margin) tuple has no matching
        row, or if the Reactive variant string has no match. The page
        wraps these in st.error() + st.stop() per plan — never with
        Python assert (asserts leak tracebacks mid-defense).
    """
    # 1. Validate inputs upfront — fail loudly, not after a partial computation.
    if reactive_lag not in (1, 2, 3, 4):
        raise ValueError(f"reactive_lag must be in {{1,2,3,4}}, got {reactive_lag}")
    if horizon not in ("10min", "30min", "60min", "120min"):
        raise ValueError(f"horizon must be a known cell, got {horizon!r}")

    # 2. Look up both rows. These raise with diagnostic messages if the
    #    grid drifted or variant strings are wrong.
    p_row = _lookup_proactive_row(pareto_df, horizon, target_util, safety_margin)
    r_row = _lookup_reactive_row(
        pareto_df, horizon, target_util, safety_margin, reactive_lag,
    )

    # 3. Classify with weak Pareto dominance.
    kind = _classify(
        p_row.violation_rate, p_row.waste_rate,
        r_row.violation_rate, r_row.waste_rate,
    )

    # 4. Build the context dict used both for verdict_templates.render()
    #    AND for the page's metric cards / Pareto plot annotations.
    ctx = {
        "horizon": horizon,
        "target_util": float(target_util),
        "safety_margin": float(safety_margin),
        "reactive_lag": int(reactive_lag),
        "proactive_violation": float(p_row.violation_rate),
        "proactive_waste":     float(p_row.waste_rate),
        "proactive_severity":  float(p_row.viol_severity),
        "reactive_violation":  float(r_row.violation_rate),
        "reactive_waste":      float(r_row.waste_rate),
        "reactive_severity":   float(r_row.viol_severity),
    }

    # 5. BCF predicate evaluation on the user's workload+horizon. Used by
    #    Day 5 page; surfaced here so the recommendation page can show a
    #    one-line "predicate fires / does not fire" badge.
    horizon_minutes = int(horizon.replace("min", ""))
    predicate_fires = (workload_acf24 > 0.2) and (horizon_minutes >= 30)

    # 6. Build the three short bullet reasons for the verdict card.
    #    These are headlines; the full justification lives in rendered_md.
    reasons = _build_reasons(kind, ctx, predicate_fires)

    # 7. Render the full markdown body via the template module.
    rendered_md = verdict_templates.render(kind, ctx)

    return {
        "kind": kind,
        "reasons": reasons,
        "ctx": ctx,
        "proactive": {
            "violation": ctx["proactive_violation"],
            "waste":     ctx["proactive_waste"],
            "severity":  ctx["proactive_severity"],
        },
        "reactive": {
            "violation": ctx["reactive_violation"],
            "waste":     ctx["reactive_waste"],
            "severity":  ctx["reactive_severity"],
        },
        "predicate_fires": predicate_fires,
        "rendered_md": rendered_md,
    }


def _build_reasons(
    kind: VerdictKind, ctx: dict, predicate_fires: bool,
) -> list[str]:
    """Three short bullet reasons that go in the verdict card on Day 3."""
    h = ctx["horizon"]
    lag = ctx["reactive_lag"]

    if kind == "PROACTIVE_DOMINATES":
        return [
            f"ML-Proactive Pareto-dominates Reactive(lag={lag}) at this config: "
            f"violation {ctx['proactive_violation']:.4f} vs "
            f"{ctx['reactive_violation']:.4f}, waste "
            f"{ctx['proactive_waste']:.4f} vs {ctx['reactive_waste']:.4f}.",
            f"At lag={lag}, proactive dominates in "
            f"{verdict_templates.LAG_DOMINANCE[lag]['proactive']}/160 grid cells "
            f"({100 * verdict_templates.LAG_DOMINANCE[lag]['proactive'] / 160:.1f}%).",
            "BCF predicate (ACF@24h > 0.2 AND horizon ≥ 30min) "
            + ("fires for this workload+horizon."
               if predicate_fires else
               "does NOT fire for this workload+horizon — proactive wins here "
               "even though the predicate would have predicted otherwise."),
        ]

    if kind == "REACTIVE_DOMINATES":
        return [
            f"Reactive(lag={lag}) Pareto-dominates ML-Proactive at this config: "
            f"violation {ctx['reactive_violation']:.4f} vs "
            f"{ctx['proactive_violation']:.4f}, waste "
            f"{ctx['reactive_waste']:.4f} vs {ctx['proactive_waste']:.4f}.",
            f"At lag={lag} this happens in "
            f"{verdict_templates.LAG_DOMINANCE[lag]['reactive']}/160 cells "
            f"({100 * verdict_templates.LAG_DOMINANCE[lag]['reactive'] / 160:.1f}%); "
            "all such cells are at horizon=120min × target_util=0.8.",
            "BCF predicate "
            + ("fires here — and the predicate is wrong on this cell. "
               "0/36 false positives across the 36-pair pooled set, but the "
               "predicate is at the boundary on long-horizon high-utilization "
               "configs."
               if predicate_fires else
               "does not fire — consistent with the verdict that ML adds no "
               "value here."),
        ]

    # NEITHER_DOMINATES
    d_v = ctx['proactive_violation'] - ctx['reactive_violation']
    d_w = ctx['proactive_waste']     - ctx['reactive_waste']
    return [
        f"Neither weakly dominates: ML-Proactive minus Reactive(lag={lag}) "
        f"gives Δviolation={d_v:+.4f}, Δwaste={d_w:+.4f}. Pareto trade-off.",
        f"At lag={lag}, "
        f"{verdict_templates.LAG_DOMINANCE[lag]['neither']}/160 cells have "
        f"this trade-off pattern "
        f"({100 * verdict_templates.LAG_DOMINANCE[lag]['neither'] / 160:.1f}%).",
        "Pick the strategy whose dominant axis matches your operational "
        "priority: violations vs waste.",
    ]
