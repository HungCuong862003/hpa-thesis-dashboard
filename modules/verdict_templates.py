"""
verdict_templates.py — Day 2 deliverable (PLACEHOLDER)
=======================================================

Three category-level verdict templates rendered by the recommendation page.
Templates encode the honest-framing constraint structurally: every verdict
is one of the three constants below, and all downstream pages import from
here so a "soften this verdict" instinct has nowhere to land.

DAY 2 TASK:
  Fill in PROACTIVE_DOMINATES, REACTIVE_DOMINATES, NEITHER_DOMINATES with
  the actual template strings per the plan section "Day 2 — Verdict
  templates and `recommend()` function".

Each template must:
  - Report actual Pareto comparison values (violation, waste) at the user's
    chosen (horizon, target_util, safety_margin, reactive_lag) tuple.
  - Inline the negative findings: Chronos 10/12, ET-alone-on-ByteDance,
    BCF pooled AUC 0.80 [0.70, 0.88] across 36 pairs.
  - Disclose lag-sensitivity (lag=1 ~59% of cells; lag=4 ~98%).
  - Use per-VM/per-instance distributions (39%/52%/72% have ACF@24h>0.2),
    not aggregate framing.

DO NOT:
  - Add a "use NNLS" or "use our model" verdict. Verdicts are category-level.
  - Add a confidence field. AUC is a population-level metric, not per-instance
    posterior; showing it as confidence is a category error (per plan).
"""

PROACTIVE_DOMINATES = """
TODO Day 2: fill in template. See plan §"Day 2".
""".strip()

REACTIVE_DOMINATES = """
TODO Day 2: fill in template. See plan §"Day 2".
""".strip()

NEITHER_DOMINATES = """
TODO Day 2: fill in template. See plan §"Day 2".
""".strip()


def render(verdict_kind: str, ctx: dict) -> str:
    """
    Render one of the three templates with values filled in from ctx.

    ctx keys (filled by recommend.py):
        horizon: str            ('30min', etc.)
        target_util: float      (0.5..0.8)
        safety_margin: float    (1.0..1.6 multiplier)
        reactive_lag: int       (1..4)
        proactive_violation: float
        proactive_waste: float
        reactive_violation: float
        reactive_waste: float

    Returns the formatted markdown string for st.expander("Why this recommendation?").
    """
    raise NotImplementedError("Day 2 task")
