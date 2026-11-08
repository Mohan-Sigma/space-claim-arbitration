import sys, json, time, numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import arm2v6_problem as P
seeds = [int(s) for s in sys.argv[1].split(",")]
class Mono(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=len(P.XL), n_obj=4, n_ieq_constr=len(P.CONSTRAINTS), xl=P.XL, xu=P.XU)
    def _evaluate(self, x, out, *a, **k):
        f, g, _ = P.evaluate(x); out["F"] = -f; out["G"] = g
rd = get_reference_directions("das-dennis", 4, n_partitions=7)
for seed in seeds:
    t0 = time.time()
    res = minimize(Mono(), NSGA3(pop_size=120, ref_dirs=rd), ("n_gen", 1500), seed=seed, verbose=False)
    out = dict(seed=seed, X=[] if res.F is None else np.atleast_2d(res.X).tolist(),
               F=[] if res.F is None else (-np.atleast_2d(res.F)).tolist())
    json.dump(out, open(f"arm2v7_seed{seed}.json", "w"))
    print(f"seed {seed}: {len(out['F'])} feasible ND ({time.time()-t0:.0f}s)", flush=True)
