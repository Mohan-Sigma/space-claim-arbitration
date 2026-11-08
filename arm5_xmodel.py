"""
Cross-model check of the final Arm 5 pipeline (A5e).
Identical to the Sonnet A5e runs: same probe results (arm5_probe.json), same requirements text and catalog,
same engine (two-sided siren floor, <=2% repair limit), same rationale prompt and grounding check.
Only the LLM changes, in both the formulation and the rationale layers.
Usage: python3 arm5_xmodel.py <model> <rep_tag>
"""
import sys, json, time, numpy as np
from pymoo.util.ref_dirs import get_reference_directions
import arm5_llm as L, arm5_engine2 as E2, arm5_final as F
from arm4_run import DIRECTIONS

PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (5.0, 25.0), "claude-haiku-5-5": (0.10, 0.50)}  # $/M in, out
model, rep = sys.argv[1], sys.argv[2]
L.MODEL = model
pr = json.load(open("arm5_probe.json"))
llm = L.LLM(); t0 = time.time()
base = F.formulate(llm, pr)
base_clean = {k: v for k, v in base.items() if not k.startswith("_")}
print(f"[{model} {rep}] base: scal={base.get('scalarisation')} floors={base.get('objective_floors')} asp={base.get('aspiration')} "
      f"beyond-best-probe={F.aspiration_reachability(base, pr)} errors={base.get('_errors')}", flush=True)
W = get_reference_directions("das-dennis", 4, n_partitions=4)
sweep = [E2.run2(base, w) for w in W]
print(f"[{model} {rep}] sweep: ATC-strict {sum(r['atc_strict'] for r in sweep)}/35, feasible(<=2%) {sum(r['feasible'] for r in sweep)}/35, "
      f"median repair {100*np.median([r['move_max'] for r in sweep]):.2f}%", flush=True)
dirs = []
for nm, note in DIRECTIONS.items():
    sp = F.formulate(llm, pr, note, base_spec=base_clean) if note else dict(base)
    r = E2.run2(sp)
    trace, data = L.rationale(llm, sp, r, note)
    r.update(direction=nm, spec=sp, trace=trace, grounding=L.grounding(trace, data), beyond_best_probe=F.aspiration_reachability(sp, pr))
    dirs.append(r)
    print(f"[{model} {rep}] {nm:<14} feas={r['feasible']} f={np.round(r['f'],3)} w={[round(sp['weights'][k],2) for k in L.E.OBJ]} "
          f"grounding={r['grounding']['grounded']}/{r['grounding']['numbers']}", flush=True)
pi, po = PRICE[model]
cost = (llm.usage["input"] * pi + llm.usage["output"] * po) / 1e6
json.dump(dict(model=model, rep=rep, probe=pr, base_spec=base, sweep=sweep, directions=dirs, usage=llm.usage, cost=cost,
               wall_s=time.time() - t0), open(f"arm5x_{model}_{rep}.json", "w"), indent=1)
print(f"[{model} {rep}] DONE usage={llm.usage} est cost ${cost:.3f} {time.time()-t0:.0f}s", flush=True)
