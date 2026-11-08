import json, glob, numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from scipy.stats import spearmanr
import arm2v6_problem as P

runs = [json.load(open(f)) for f in sorted(glob.glob("arm2v7_seed*.json"))]
feas = [r for r in runs if r["F"]]
X = np.vstack([r["X"] for r in feas]); F = np.vstack([r["F"] for r in feas])
nd = NonDominatedSorting().do(-F, only_non_dominated_front=True); X, F = X[nd], F[nd]
ev_ = [P.evaluate(x) for x in X]; infos = [e[2] for e in ev_]; gs = np.array([e[1] for e in ev_])
I = lambda k: np.array([q[k] for q in infos])
print(f"v0.7 main case: {len(feas)}/{len(runs)} seeds feasible, {len(F)} designs on union front")
for j, o in enumerate(P.OBJECTIVES):
    print(f"  {o:<18} {F[:,j].min():7.3f} .. {F[:,j].max():7.3f}")
print(f"  siren >= 3 h: {np.sum(F[:,1]>=3)} | runtime >= 59.8 h: {np.sum(F[:,0]>=59.8)} | both: {np.sum((F[:,1]>=3)&(F[:,0]>=59.8))}")
tol = np.array([.02,.02,.02,.02,.02,.02,.01,.05,.01,.01,.01,.005,.05,.02,.02])
print("  binding:", ", ".join(f"{n} {100*np.mean(c>-t):.0f}%" for n, c, t in zip(P.CONSTRAINTS, gs.T, tol) if np.mean(c>-t) > 0.05))
for k in ("p_siren", "q_ss", "f_res", "vfc", "vb", "mass", "pmarg"):
    print(f"    {k:<8} median {np.median(I(k)):9.3f}  range {I(k).min():9.3f} .. {I(k).max():9.3f}")
for n in ("V_batt", "V_acou", "port", "split", "h_out", "r_mesh", "t_wall", "V_ant"):
    j = P.NAMES.index(n); print(f"    {n:<8} median {np.median(X[:,j]):9.2f}  range {X[:,j].min():9.2f} .. {X[:,j].max():9.2f}")
lo, hi = F.min(0), F.max(0); k = int(np.argmax(((F-lo)/np.maximum(hi-lo,1e-9)).min(1)))
print("  BALANCED: " + ", ".join(f"{o} {v:.2f}" for o, v in zip(P.OBJECTIVES, F[k])) +
      f" | P {infos[k]['p_siren']:.3f} W, Q {infos[k]['q_ss']:.2f}, f_res {infos[k]['f_res']:.0f}, vfc {infos[k]['vfc']:.0f}, vb {infos[k]['vb']:.0f}, split {X[k,8]:.2f}, V_batt {X[k,0]:.0f}, V_acou {X[k,6]:.0f}, mass {infos[k]['mass']:.1f}")
for a, b in (("split","voice_score"),("split","siren_endurance_h"),("V_acou","voice_score"),("V_batt","siren_endurance_h"),("V_batt","voice_score")):
    xa = X[:,P.NAMES.index(a)]; yb = F[:,P.OBJECTIVES.index(b)]
    print(f"    rho({a},{b}) {spearmanr(xa,yb)[0]:+.2f}")
print("  rho(siren_end, voice) on front", f"{spearmanr(F[:,1],F[:,2])[0]:+.2f}")
hv = HV(ref_point=np.full(4,1.1)); s=[hv(1-(np.array(r['F'])-lo)/(hi-lo)) for r in feas]
print(f"  HV per seed {np.round(s,3)} mean {np.mean(s):.3f} +/- {np.std(s):.3f}; union {hv(1-(F-lo)/(hi-lo)):.3f}")
# old v0.6 front re-scored with SII, for comparison
old = json.load(open("arm2v6_main95_front.json")); Xo = np.array(old["X"])
Fo = np.array([P.evaluate(x)[0] for x in Xo])
print(f"\n  v0.6 front re-scored with SII: voice {Fo[:,2].min():.1f}..{Fo[:,2].max():.1f}, split median {np.median(Xo[:,8]):.2f}")
allF = np.vstack([F, Fo]); nd2 = NonDominatedSorting().do(-allF, only_non_dominated_front=True)
print(f"  of the combined non-dominated set ({len(nd2)}), {np.sum(nd2 < len(F))} come from the v0.7 run, {np.sum(nd2 >= len(F))} from v0.6")
json.dump(dict(X=X.tolist(), F=F.tolist(), balanced=k), open("arm2v7_main_front.json", "w"))
