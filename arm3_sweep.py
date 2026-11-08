import sys, json, time, numpy as np
from pymoo.util.ref_dirs import get_reference_directions
import arm3_atc as A
part = int(sys.argv[1])
W = get_reference_directions("das-dennis", 4, n_partitions=4)       # 35 weight vectors
out = []
for i in range(part, len(W), 2):
    t0 = time.time(); r = A.run(W[i]); r["wall_s"] = time.time() - t0; out.append(r)
    print(f"w={np.round(W[i],2)} outer={r['outer']} max_c={r['max_c']:.1e} feas={r['feasible']}/{r['feasible_tol']} "
          f"f={np.round(r['f'],3)} evals={r['circuit_evals']} {r['wall_s']:.0f}s", flush=True)
json.dump(out, open(f"arm3_part{part}.json", "w"))
