import json, numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import arm2v6_problem as P

W = json.load(open("arm1_part0.json")) + json.load(open("arm1_part1.json"))
ok = [w for w in W if w["ok"]]
Fw = np.array([w["f"] for w in ok]); Xw = np.array([w["x"] for w in ok])
loops = np.array([w["loops"] for w in ok]); evals = np.array([w["acoustic_evals"] + w["system_evals"] for w in ok])
a2 = json.load(open("arm2v7_main_front.json")); F2 = np.array(a2["F"]); X2 = np.array(a2["X"])
print(f"Arm 1: {len(ok)}/{len(W)} runs feasible; Arm 2 front: {len(F2)} designs")

allF = np.vstack([F2, Fw]); lo, hi = allF.min(0), allF.max(0)
hv = HV(ref_point=np.full(4, 1.1)); H = lambda F: hv(1 - (F - lo) / (hi - lo))
ndw = NonDominatedSorting().do(-Fw, only_non_dominated_front=True)
nd_all = NonDominatedSorting().do(-allF, only_non_dominated_front=True)
print(f"hypervolume (common normalisation): Arm 2 {H(F2):.3f} | Arm 1 {H(Fw):.3f} ({100*H(Fw)/H(F2):.0f}% of Arm 2) | union {H(allF):.3f}")
print(f"Arm 1 non-dominated among itself: {len(ndw)}/{len(Fw)}")
print(f"union front: {len(nd_all)} designs, {np.sum(nd_all < len(F2))} from Arm 2, {np.sum(nd_all >= len(F2))} from Arm 1")

def dominated_by(a, B):
    return np.any(np.all(B >= a, axis=1) & np.any(B > a, axis=1))
dw = np.array([dominated_by(f, F2) for f in Fw]); d2 = np.array([dominated_by(f, Fw) for f in F2])
print(f"Arm 1 designs dominated by some Arm 2 design: {dw.sum()}/{len(Fw)}; Arm 2 designs dominated by Arm 1: {d2.sum()}/{len(F2)}")

print("\nobjective ranges   Arm 1                Arm 2")
for j, o in enumerate(P.OBJECTIVES):
    print(f"  {o:<18} {Fw[:,j].min():6.2f} .. {Fw[:,j].max():6.2f}    {F2[:,j].min():6.2f} .. {F2[:,j].max():6.2f}")
print(f"siren >= 3 h: Arm 1 {np.sum(Fw[:,1]>=3)} | Arm 2 {np.sum(F2[:,1]>=3)}")
print(f"siren >= 3 h AND runtime >= 59.8 h: Arm 1 {np.sum((Fw[:,1]>=3)&(Fw[:,0]>=59.8))} | Arm 2 {np.sum((F2[:,1]>=3)&(F2[:,0]>=59.8))}")

print("\nmatched-runtime comparison (best value at runtime >= bin):")
print("  runtime >=   siren A1/A2      voice A1/A2      L5 A1/A2")
for rt in (42, 48, 54, 58, 60):
    m1, m2 = Fw[:,0] >= rt - 0.05, F2[:,0] >= rt - 0.05
    if m1.any() and m2.any():
        print(f"  {rt:>5} h     {Fw[m1,1].max():4.2f} / {F2[m2,1].max():4.2f}     {Fw[m1,2].max():5.1f} / {F2[m2,2].max():5.1f}     {Fw[m1,3].max():.3f} / {F2[m2,3].max():.3f}")

info_w = [P.evaluate(x)[2] for x in Xw]; info_2 = [P.evaluate(x)[2] for x in X2]
pw = np.array([i["p_siren"] for i in info_w]); p2 = np.array([i["p_siren"] for i in info_2])
print(f"\nsiren tone power median: Arm 1 {np.median(pw):.3f} W (range {pw.min():.3f}..{pw.max():.3f}) | Arm 2 {np.median(p2):.3f} W")
j = P.NAMES.index
for n in ("V_acou", "split", "port", "t_wall", "V_ant", "V_struct", "A_board", "V_hap"):
    print(f"  {n:<9} median Arm 1 {np.median(Xw[:,j(n)]):8.2f}   Arm 2 {np.median(X2[:,j(n)]):8.2f}")
used = Xw[:, j("V_acou")]
slack = np.array([P.base.POOL_R1_NOM - P.base.DV_DT*(x[j("t_wall")]-1.2) - (x[0] + x[2]*P.base.T_BOARD + x[3] + x[5] + x[6] + x[11] + x[12]*P.base.T_SPREADER + x[13] + x[14]) for x in Xw])
print(f"  unused interior volume (slack): Arm 1 median {np.median(slack):.0f} mm3 (max {slack.max():.0f})")

print(f"\nrework loops: mean {loops.mean():.1f}, median {np.median(loops):.0f}, max {loops.max()}, runs with 0 loops {np.sum(loops==0)}")
rt_spec = np.array([w["v_batt_spec"]*600/1e6/0.038 for w in ok])
for lo_, hi_ in ((41, 47), (47, 54), (54, 63)):
    m = (rt_spec >= lo_) & (rt_spec < hi_)
    print(f"  runtime spec {lo_}-{hi_} h: mean loops {loops[m].mean():.1f}")
from collections import Counter
c = Counter(r for w in ok for r in w["log"]); print("  rework types:", dict(c))
print(f"evaluations: per waterfall run median {np.median(evals):.0f}; whole 50-run sweep {evals.sum()}; Arm 2 per seed 180000 (x5 seeds = 900000)")
batt_eroded = np.array([w["x"][0] < w["v_batt_spec"] - 1 for w in ok]).sum()
print(f"runs where the battery spec was eroded by rework: {batt_eroded}")
json.dump(dict(F=Fw.tolist(), X=Xw.tolist(), loops=loops.tolist()), open("arm1_results.json", "w"))
