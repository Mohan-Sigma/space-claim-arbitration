import json, numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import arm2v6_problem as P

R = json.load(open("arm3_part0.json")) + json.load(open("arm3_part1.json"))
strict = [r for r in R if r["feasible"]]; tolr = [r for r in R if r["feasible_tol"]]
F3s = np.array([r["f"] for r in strict]); F3t = np.array([r["f"] for r in tolr]); X3s = np.array([r["x"] for r in strict])
a1 = json.load(open("arm1_results.json")); F1 = np.array(a1["F"])
a2 = json.load(open("arm2v7_main_front.json")); F2 = np.array(a2["F"]); X2 = np.array(a2["X"])
print(f"Arm 3: {len(R)} weight vectors | strictly feasible {len(strict)} | feasible within 5e-3 {len(tolr)}")
print(f"outer iterations: mean {np.mean([r['outer'] for r in R]):.1f}, max {max(r['outer'] for r in R)}; "
      f"non-converged (40 iters) {sum(r['outer']==40 for r in R)}")
ev3 = np.array([r["circuit_evals"] + r["system_evals"] for r in R])
print(f"evaluations: per run median {np.median(ev3):.0f}, sweep total {ev3.sum()} (circuit {sum(r['circuit_evals'] for r in R)}); "
      f"wall {np.sum([r['wall_s'] for r in R]):.0f} s")
viol = {}
for r in R:
    if not r["feasible"]:
        for n, g in zip(P.CONSTRAINTS, r["g"]):
            if g > 1e-6: viol.setdefault(n, []).append(g)
print("strict violations:", {k: (len(v), f"max {max(v):.4f}") for k, v in viol.items()})

allF = np.vstack([F2, F1, F3t]); lo, hi = allF.min(0), allF.max(0)
hv = HV(ref_point=np.full(4, 1.1)); H = lambda F: hv(1 - (F - lo) / (hi - lo)) if len(F) else 0.0
h2, h1, h3s, h3t = H(F2), H(F1), H(F3s), H(F3t)
print(f"\nhypervolume (common normalisation over all arms): Arm 2 {h2:.3f} | Arm 1 {h1:.3f} ({100*h1/h2:.0f}%) | "
      f"Arm 3 strict {h3s:.3f} ({100*h3s/h2:.0f}%) | Arm 3 tol {h3t:.3f} ({100*h3t/h2:.0f}%)")
U = np.vstack([F2, F1, F3s]); nd = NonDominatedSorting().do(-U, only_non_dominated_front=True)
src = np.array(["A2"] * len(F2) + ["A1"] * len(F1) + ["A3"] * len(F3s))
print(f"union HV {H(U):.3f}; union front {len(nd)} designs: " + ", ".join(f"{s} {np.sum(src[nd]==s)}" for s in ("A2", "A1", "A3")))
dom = lambda a, B: np.any(np.all(B >= a, axis=1) & np.any(B > a, axis=1))
print(f"Arm 3 (strict) dominated by Arm 2: {sum(dom(f, F2) for f in F3s)}/{len(F3s)} | by Arm 1: {sum(dom(f, F1) for f in F3s)}/{len(F3s)}")
print(f"Arm 2 dominated by Arm 3: {sum(dom(f, F3s) for f in F2)}/{len(F2)} | Arm 1 dominated by Arm 3: {sum(dom(f, F3s) for f in F1)}/{len(F1)}")

print("\nobjective ranges   Arm 1            Arm 2            Arm 3 (strict)")
for j, o in enumerate(P.OBJECTIVES):
    print(f"  {o:<18} {F1[:,j].min():6.2f}..{F1[:,j].max():6.2f}   {F2[:,j].min():6.2f}..{F2[:,j].max():6.2f}   {F3s[:,j].min():6.2f}..{F3s[:,j].max():6.2f}")
for nm, F in (("A1", F1), ("A2", F2), ("A3", F3s)):
    print(f"  {nm}: siren>=3h {np.sum(F[:,1]>=3)}, siren>=3h & runtime>=59.8h {np.sum((F[:,1]>=3)&(F[:,0]>=59.8))}")
print("\nbinned (runtime within +/-1.2 h), best: siren A1/A2/A3 | voice A1/A2/A3 | L5 A1/A2/A3")
for rt in (42, 46.4, 50.9, 55.3, 59.8, 62):
    m = [np.abs(F[:,0]-rt) < 1.2 for F in (F1, F2, F3s)]
    def b(F, mm, j): return f"{F[mm,j].max():.2f}" if mm.any() else "  -  "
    print(f"  ~{rt:>4} h: " + "/".join(b(F, mm, 1) for F, mm in zip((F1,F2,F3s), m)) + " | " +
          "/".join(b(F, mm, 2) for F, mm in zip((F1,F2,F3s), m)) + " | " + "/".join(b(F, mm, 3) for F, mm in zip((F1,F2,F3s), m)))
info3 = [P.evaluate(np.array(x))[2] for x in X3s]; p3 = np.array([i["p_siren"] for i in info3])
info2 = [P.evaluate(x)[2] for x in X2]; p2 = np.array([i["p_siren"] for i in info2])
print(f"\nsiren tone power median: Arm 3 {np.median(p3):.3f} W (range {p3.min():.3f}..{p3.max():.3f}) | Arm 2 {np.median(p2):.3f} | Arm 1 0.994")
j = P.NAMES.index
for n in ("V_acou", "split", "port", "t_wall", "V_ant", "V_struct", "A_board", "V_hap", "V_loop", "V_ferr"):
    print(f"  {n:<9} median Arm 3 {np.median(X3s[:,j(n)]):8.2f}   Arm 2 {np.median(X2[:,j(n)]):8.2f}")
hi62 = [r for r in strict if r["f"][0] > 61.5]
if hi62:
    b = max(hi62, key=lambda r: r["f"][1]); print("\nbest high-runtime ATC design:", np.round(b["f"], 3), "w", np.round(b["w"], 2),
          "| V_batt", round(b["x"][0]), "V_ant", round(b["x"][5]), "V_acou", round(b["x"][6]), "port", round(b["x"][7], 2))
json.dump(dict(F=F3s.tolist(), X=X3s.tolist(), F_tol=F3t.tolist()), open("arm3_results.json", "w"))
