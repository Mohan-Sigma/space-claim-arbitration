# Space-Claim Arbitration Benchmark

Code, data, and run outputs for:

> M. R. Manoharan, "AI Should Formulate, Not Negotiate: LLM-Formulated Target Cascading for Space-Claim Arbitration in Compact Product Design," submitted to *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 2026.

The benchmark models eight engineering disciplines (battery, board, structure, thermal, antenna, acoustics, haptics, NFC) competing for the interior of a commercially available 47 mm rugged smartwatch: 15 design variables, 15 hard constraints, 4 objectives, and physics-based evaluators, including a lumped acoustic circuit and the ANSI S3.5 Speech Intelligibility Index. The reference product's name is withheld; see `paper/supplementary.pdf` for all values and their provenance.

## Coordination methods

| ID | Method | Entry point |
|---|---|---|
| A1 | Sequential waterfall | `arm1_sweep.py` |
| A1-PM | Waterfall with program-manager escalation | `arm1_pm.py` |
| A2 | Monolithic NSGA-III | `arm2v7_run.py` |
| A3 | Hand-formulated analytical target cascading (ATC) | `arm3_sweep.py` |
| A4 | LLM-agent negotiation | `arm4v2_run.py`, `arm4v2_run_rep.py` |
| A4-PM | Negotiation with an LLM program-manager agent | `arm4pm_run.py` |
| A5 | LLM-formulated ATC (probe, formulate, solve, explain) | `arm5_final.py`, `arm5_final2.py`, `arm5_xmodel.py` |

## Repository layout

- `space_claim_schema_v0.7.json`: benchmark schema (claimants, pools, couplings, provenance)
- `evaluators.py`, `acoustic_circuit.py`, `voice_sii.py`, `driver_18x13.py`: domain evaluators
- `arm2v5_problem.py`, `arm2v6_problem.py`: the benchmark problem (main case = v6)
- `arm*_*.py`: the coordination methods listed above
- `arm_compare2.py`, `arm_xmodel_compare.py`, `make_figs.py`: tables and figures in the paper
- `*.json`: all run outputs used in the paper, including negotiation transcripts and decision records
- `paper/supplementary.pdf`: equations, constants, prompts, per-run results

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The deterministic methods (A1, A1-PM, A2, A3) and all analysis scripts run without an API key. The LLM methods (A4, A4-PM, A5) call the Anthropic API: put your key in a local file and point `ANTHROPIC_KEY_FILE` at it. Never commit the key.

## Reproduce the paper's tables and figures

```bash
python arm_xmodel_compare.py   # hypervolume shares, feasibility, grounding (Table IV)
python make_figs.py            # Figs. 2-4
```

Full re-runs: see Section S9 of `paper/supplementary.pdf`. A1, A1-PM and A3 reproduce exactly; A2 per seed; LLM methods vary run to run (provider default sampling).

## License

Code: MIT (see `LICENSE`). Data and run outputs (`*.json`): CC BY 4.0.

## Citation

See `CITATION.cff`. Until publication, please cite the manuscript as above.
