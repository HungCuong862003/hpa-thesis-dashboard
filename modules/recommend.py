"""
recommend.py — Day 2 deliverable (PLACEHOLDER)
================================================

Deterministic rule engine. No ML. No LLM. Reads hpa_simulation_v2.csv
at the user's chosen configuration, applies weak Pareto dominance with
EPS = 1e-9 tolerance, returns one of three verdicts.

Signature is locked in so the recommendation page can be sketched against
this interface on Day 3 even if the body is filled in late on Day 2.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# Floating-point tolerance for Pareto comparison. Per plan: weak dominance
# (≤ on both axes, < on at least one), not strict-better-on-both.
EPS = 1e-9

VerdictKind = Literal["PROACTIVE_DOMINATES", "REACTIVE_DOMINATES", "NEITHER_DOMINATES"]


def recommend(
    *,
    workload_acf24: float,
    workload_cv: float,
    workload_hurst: float,
    horizon: str,           # '10min', '30min', '60min', '120min'
    target_util: float,     # ∈ {0.5, 0.6, 0.7, 0.8}
    safety_margin: float,   # ∈ {1.0, 1.067, ..., 1.6}
    reactive_lag: int,      # ∈ {1, 2, 3, 4}; default 1 per plan
    pareto_df: pd.DataFrame,
) -> dict:
    """
    Return a verdict dict.

    Schema:
      {
        "kind": VerdictKind,
        "reasons": list[str],           # 3 bullet points for the verdict card
        "ctx": dict,                    # passed to verdict_templates.render()
        "proactive": {"violation": float, "waste": float, "severity": float},
        "reactive": {"violation": float, "waste": float, "severity": float},
      }

    Implementation notes (Day 2):
      1. Filter pareto_df to (Horizon=horizon, target_util, safety_margin).
         Use a normalized-Manhattan-distance fallback if no exact match.
      2. Pick the Reactive row by EXACT variant-string match:
            f'lag={int(reactive_lag)}_tu={float(target_util):.1f}'
         NEVER iloc[0] — the four lag rows differ materially.
      3. Apply weak Pareto dominance with EPS tolerance.
      4. On any failure, raise ValueError with a useful message; the page
         translates it to st.error()+st.stop(), NOT a Python assert (asserts
         leak tracebacks mid-defense).
      5. NO confidence field. AUC is population-level.

    Day 2 status: NOT IMPLEMENTED. Stub raises so any accidental Day 1+
    caller fails loudly.
    """
    raise NotImplementedError("Day 2 task — see plan §'Day 2'")
