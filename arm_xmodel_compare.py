"""Cross-model comparison of the final Arm 5 pipeline: Sonnet (arm5e_r*), Opus and Haiku (arm5x_*)."""
import json, glob, runpy, io, contextlib, numpy as np
from pymoo.indicators.hv import HV
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path("./arm_compare2.py")
groups, a5_group = G["groups"], G["a5_group"]
groups["A5e Opus 5.5"] = a5_group("arm5x_claude-opus-5-5_r*.json")
groups["A5e Haiku 5.5"] = a5_group("arm5x_claude-haiku-5-5_r*.json")
groups["A5e Sonnet 5.5"] = groups.pop("A5e grounded, 2-sided floor")
_pm = json.load(open("./arm1pm_part0.json")) + json.load(open("./arm1pm_part1.json"))
groups["A1 + PM escalation"] = [dict(F=np.array([r["f"] for r in _pm if r["ok"]]).reshape(-1, 4))]
_a4 = [json.load(open(f)) for f in sorted(glob.glob("./arm4pm_*.json")) if "ckpt" not in f]
if _a4:
    groups["A4 + PM agent"] = [dict(F=np.array([r["f"] for r in _a4 if r["feasible"]]).reshape(-1, 4))]
for _nm, _key in (("A5e Sonnet 5.5", "sonnet"), ("A5e Haiku 5.5", "haiku"), ("A5e Opus 5.5", "opus")):
    for _k, _r in enumerate(groups[_nm], 1):
        _rec = json.load(open(f"arm5rec_{_key}_r{_k}.json"))["records"]
        _r["grounding"] = (sum(x["grounding"]["grounded"] for x in _rec), sum(x["grounding"]["numbers"] for x in _rec))
allF = np.vstack([r["F"] for reps in groups.values() for r in reps if len(r["F"])])
lo, hi = allF.min(0), allF.max(0); hv = HV(ref_point=np.full(4, 1.1))
H = lambda F: float(hv(1 - (F - lo) / (hi - lo))) if len(F) else 0.0
comp = lambda F: F[F[:, 1] >= 3.0] if len(F) else F
Hu, Huc = H(allF), H(comp(allF))
print(f"union {len(allF)} designs, HV {Hu:.3f}, compliant HV {Huc:.3f}\n")
print(f"{'model':<16} {'compliant HV % union':>22} {'feasible(<=2%) /45':>20} {'ATC-strict':>11} {'priorities /4':>14} {'grounding':>10} {'$/rep':>7}")
for nm in ("A5e Haiku 5.5", "A5e Sonnet 5.5", "A5e Opus 5.5"):
    reps = groups[nm]
    if not reps: print(nm, "no data"); continue
    hc = [100 * H(comp(r["F"])) / Huc for r in reps]
    print(f"{nm:<16} {np.mean(hc):10.0f}% ± {np.std(hc):3.0f}%   {np.mean([r['feasible'] for r in reps]):9.1f} ± {np.std([r['feasible'] for r in reps]):3.1f} "
          f"{np.mean([r['atc_strict'] for r in reps]):9.1f}  {[r['priority'] for r in reps]!s:>13} "
          f"{100*sum(r['grounding'][0] for r in reps)/max(1,sum(r['grounding'][1] for r in reps)):8.1f}%  "
          f"{np.mean([r['cost'] for r in reps]):6.2f}")
print("\nbase formulations:")
for f in sorted(glob.glob("./arm5e_r*.json")) + sorted(glob.glob("./arm5x_*.json")):
    d = json.load(open(f)); b = d["base_spec"]
    print(f"  {f.split('/')[-1]:<34} scal={b.get('scalarisation'):<13} floors={b.get('objective_floors')} asp={b.get('aspiration')} errors={b.get('_errors')}")
for nm in ("A5e Haiku 5.5", "A5e Sonnet 5.5", "A5e Opus 5.5"):
    hs = [100 * H(r["F"]) / Hu for r in groups[nm]]
    if hs: print(f"{nm}: full HV {np.mean(hs):.0f}% ± {np.std(hs):.0f}% of union")
print("\nall arms on the extended union:")
for nm, reps in groups.items():
    hc=[100*H(comp(r["F"]))/Huc for r in reps]; both=[int(np.sum((r["F"][:,1]>=3)&(r["F"][:,0]>=59.8))) if len(r["F"]) else 0 for r in reps]
    mv=[100*r["move_med"] for r in reps if "move_med" in r]
    print(f"  {nm:<30} compliant {np.mean(hc):4.0f}% ± {np.std(hc):3.0f}  siren>=3&rt>=59.8 {np.mean(both):4.1f}  median repair {np.mean(mv) if mv else float('nan'):.2f}%")
