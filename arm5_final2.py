"""A5e: identical LLM formulations as A5d (saved specs), engine with two-sided siren floor + <=2% repair limit."""
import sys, json, time, numpy as np
from pymoo.util.ref_dirs import get_reference_directions
import arm5_engine2 as E2, arm5_llm as L
rep = sys.argv[1]; d = json.load(open(f"arm5d_{rep}.json")); t0 = time.time()
llm = L.LLM(); W = get_reference_directions("das-dennis", 4, n_partitions=4)
sweep = [E2.run2(d["base_spec"], w) for w in W]
print(f"[{rep}] sweep: ATC-strict {sum(r['atc_strict'] for r in sweep)}/35, feasible (<=2% repair) {sum(r['feasible'] for r in sweep)}/35, "
      f"any repair {sum(r['feasible_any_repair'] for r in sweep)}/35, median repair {100*np.median([r['move_max'] for r in sweep]):.2f}%, "
      f"mean outer {np.mean([r['outer'] for r in sweep]):.1f}", flush=True)
dirs = []
for old in d["directions"]:
    sp = old["spec"]; r = E2.run2(sp)
    trace, data = L.rationale(llm, sp, r, "")
    r.update(direction=old["direction"], spec=sp, trace=trace, grounding=L.grounding(trace, data)); dirs.append(r)
    print(f"[{rep}] {r['direction']:<14} feas={r['feasible']} atc_strict={r['atc_strict']} repair={100*r['move_max']:.2f}% f={np.round(r['f'],3)}", flush=True)
json.dump(dict(rep=rep, probe=d["probe"], base_spec=d["base_spec"], sweep=sweep, directions=dirs, usage=llm.usage,
               cost=llm.cost() + d["cost"], cost_rationale_only=llm.cost(), wall_s=time.time() - t0), open(f"arm5e_{rep}.json", "w"), indent=1)
print(f"[{rep}] DONE rationale cost ${llm.cost():.3f} {time.time()-t0:.0f}s", flush=True)
