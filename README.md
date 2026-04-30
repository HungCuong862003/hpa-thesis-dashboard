# HPA Thesis Dashboard

Companion artefact for *Hybrid Ensemble Learning for Proactive Resource
Prediction in Kubernetes* (Jimmy, IU SCSE, 2026, supervisor: Dr. Ho Long Van).

## What this is

A Streamlit dashboard that reads frozen result files from the thesis pipeline
and renders them as an interactive exploration tool for defense. Eight pages
covering recommendation, configurator, boundary-condition predictor, Pareto
explorer, per-dataset comparison, Chronos leaderboard, SHAP, and methodology.

No live cluster. No retraining. No LLMs in the recommendation path. The
dashboard either reflects the thesis honestly or it contradicts the thesis;
there is no third option.

## Quick start

```bash
git clone <this repo>
cd hpa-thesis-dashboard
pip install -r requirements.txt
python audit/verify_numbers.py    # gate: every anchor number must pass
streamlit run app.py              # open http://localhost:8501
```

`verify_numbers.py` is the canonical guard. If it fails, the thesis-vs-data
contract is broken and the dashboard cannot be deployed.

## Project layout

```
hpa-thesis-dashboard/
├── app.py                    # Streamlit entry point
├── requirements.txt          # pinned dependencies
├── .streamlit/config.toml    # theme, server config
├── data/                     # 16 frozen CSVs + 1 JSON; see data/README.md
├── modules/
│   ├── data_loaders.py       # 18 cached loaders with anchor-value asserts
│   ├── verdict_templates.py  # Day 2 — three verdict templates
│   └── recommend.py          # Day 2 — deterministic rule engine
├── pages/                    # multipage app pages (Day 3 onward)
├── audit/
│   └── verify_numbers.py     # 32 anchor checks; standalone, no streamlit
└── docs/
    └── dashboard_implementation_plan_final.md
```

## Build status

| Phase | Status |
|---|---|
| Phase 0: Prereq + dept policy | ✓ done |
| Phase 1 Day 1: data layer + verify_numbers | ✓ done |
| Phase 1 Day 2: verdict templates + recommend()  | pending |
| Phase 2: Load-bearing pages | pending |
| Phase 3: Supporting pages | pending |
| Phase 4: Honest-framing audit | pending |
| Phase 5: Polish + deploy | pending |

## Data freeze policy

After the freeze commit (`defense-v1.0`), do not push to `main`. If a critical
bug surfaces, branch from the tag, fix, tag a new release. See the deploy plan.

## License

MIT (or whatever the thesis policy mandates — TBD).
