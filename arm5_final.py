"""
Arm 5 final pipeline (A5d): probe -> grounded formulation -> ATC (local siren floor + restoration) -> rationale.

Step 0 PROBE  : 5 cheap ATC solves with a neutral formulation (equal weights, then 0.7 emphasis on each
               objective), restored to feasibility. Gives achievable ranges WITHOUT any LLM.
Step 1 FORMULATE: the LLM gets the requirements, the catalog (formulation fields only - solver parameters
               are engine-owned and no longer offered) and the probe results, and writes the spec.
Step 2 SOLVE  : 35-weight sweep + 10 program directions (LLM re-formulates each from the base spec).
Step 3 EXPLAIN: decision trace per direction, numbers checked against the data.
Run:  python3 arm5_final.py <rep_tag>
"""
import sys, json, time, numpy as np
from pymoo.util.ref_dirs import get_reference_directions
import arm5_engine2 as E2, arm5_engine as E1, arm5_llm as L
from arm4_run import DIRECTIONS

PROBE_W = {"equal": [0.25, 0.25, 0.25, 0.25], "runtime": [0.7, 0.1, 0.1, 0.1], "siren": [0.1, 0.7, 0.1, 0.1],
           "voice": [0.1, 0.1, 0.7, 0.1], "l5": [0.1, 0.1, 0.1, 0.7]}

CATALOG = L.CATALOG.split('  "coordination"')[0] + (
    '  "weights":       {obj: w} default priorities (they will be normalised).\n'
    '  "justification": {field: one sentence citing the requirement or probe result that drove it}\n'
    'Solver parameters (penalty weights, tolerances, iteration limits) are owned by the solver and are not yours to set.\n')
FORMAT = 'Return ONLY one fenced ```json block with keys normalisation, scalarisation, aspiration, objective_floors, weights, justification.'


def probe():
    out = {}
    for nm, w in PROBE_W.items():
        r = E2.run(E1.ARM3_SPEC, w, floor_mode="local")
        out[nm] = dict(objectives=dict(zip(E1.OBJ, [round(v, 3) for v in r["f"]])), feasible=r["feasible"])
    F = np.array([[v["objectives"][k] for k in E1.OBJ] for v in out.values()])
    summary = dict(probes=out, achieved_range={k: [float(F[:, i].min()), float(F[:, i].max())] for i, k in enumerate(E1.OBJ)},
                   note="Five neutral probe solves (weighted sum, no floors) of the actual design problem; values are achieved, feasible designs. "
                        "Ranges are indicative, not exhaustive.")
    return summary


def formulate(llm, probe_summary, direction_note="", base_spec=None):
    system = "You are a senior systems engineer setting up the arbiter of a multidisciplinary design optimisation."
    user = (f"REQUIREMENTS (as given to the engineering leads):\n{L.REQUIREMENTS}\n\n"
            f"PROBE RESULTS (what this design problem actually achieves):\n```json\n{json.dumps(probe_summary)}\n```\n\n")
    if direction_note:
        user += f"PROGRAM DIRECTION FOR THIS STUDY: {direction_note}\n\n"
    if base_spec:
        user += ("BASELINE FORMULATION (already reviewed; change only what the program direction requires, and say why):\n"
                 f"```json\n{json.dumps(base_spec)}\n```\n\n")
    user += CATALOG + "\nSet aspirations and normalisation from the probe results so that every aspiration is reachable or only "\
                      "slightly beyond what was achieved; an unreachable aspiration makes the weights ineffective.\n" + FORMAT
    text = llm.ask(system, user)
    spec = L._json(text) or {}
    errs = L.validate(spec)
    if errs:
        text = llm.ask(system, user + "\n\nYOUR PREVIOUS ANSWER FAILED VALIDATION: " + "; ".join(errs) + "\nPrevious answer:\n" + text[:3000])
        spec = L._json(text) or {}; errs = L.validate(spec)
    spec.pop("coordination", None)
    spec["_errors"] = errs
    for k in E1.OBJ:
        spec["normalisation"][k] = [float(v) for v in spec["normalisation"][k]]
    return spec


def aspiration_reachability(spec, probe_summary):
    if spec.get("scalarisation") != "achievement":
        return None
    rng = probe_summary["achieved_range"]
    return {k: round(float(spec["aspiration"][k]) - rng[k][1], 3) for k in E1.OBJ}   # > 0 means beyond best probe


if __name__ == "__main__":
    rep = sys.argv[1]
    t0 = time.time()
    pr = json.load(open("arm5_probe.json")) if len(sys.argv) > 2 else probe()
    json.dump(pr, open("arm5_probe.json", "w"), indent=1)
    llm = L.LLM()
    base = formulate(llm, pr)
    base_clean = {k: v for k, v in base.items() if not k.startswith("_")}
    print(f"[{rep}] base spec: scal={base.get('scalarisation')} floors={base.get('objective_floors')} asp={base.get('aspiration')} "
          f"beyond-best-probe={aspiration_reachability(base, pr)}", flush=True)
    W = get_reference_directions("das-dennis", 4, n_partitions=4)
    sweep = []
    for w in W:
        r = E2.run(base, w, floor_mode="local"); sweep.append(r)
    print(f"[{rep}] sweep: ATC-strict {sum(r['atc_strict'] for r in sweep)}/35, after restoration {sum(r['feasible'] for r in sweep)}/35, "
          f"mean outer {np.mean([r['outer'] for r in sweep]):.1f}, max move {100*max(r['move_max'] for r in sweep):.2f}%", flush=True)
    dirs = []
    for nm, note in DIRECTIONS.items():
        sp = formulate(llm, pr, note, base_spec=base_clean) if note else dict(base)
        r = E2.run(sp, floor_mode="local")
        trace, data = L.rationale(llm, sp, r, note)
        r.update(direction=nm, spec=sp, trace=trace, grounding=L.grounding(trace, data),
                 beyond_best_probe=aspiration_reachability(sp, pr))
        dirs.append(r)
        print(f"[{rep}] {nm:<14} feas={r['feasible']} atc_strict={r['atc_strict']} f={np.round(r['f'],3)} "
              f"w={[round(sp['weights'][k],2) for k in E1.OBJ]} grounding={r['grounding']['grounded']}/{r['grounding']['numbers']}", flush=True)
    json.dump(dict(rep=rep, probe=pr, base_spec=base, sweep=sweep, directions=dirs, usage=llm.usage, cost=llm.cost(),
                   wall_s=time.time() - t0), open(f"arm5d_{rep}.json", "w"), indent=1)
    print(f"[{rep}] DONE cost ${llm.cost():.3f} wall {time.time()-t0:.0f}s", flush=True)
