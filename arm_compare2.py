"""Final cross-arm comparison with repeats (main case, schema v0.7)."""
import json, glob, os, numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

D = "./"
OBJ = ["runtime", "siren", "voice", "l5"]
EMPH = {"battery_first": "runtime", "siren_first": "siren", "voice_first": "voice", "gnss_first": "l5"}
J = lambda f: json.load(open(D + f))

groups = {}          # arm -> list of reps, each rep = dict(F=feasible objective array, extra metrics)
groups["A1 waterfall"] = [dict(F=np.array(J("arm1_results.json")["F"]))]
seeds = []
for f in sorted(glob.glob(D + "arm7_seed*.json")) or sorted(glob.glob(D + "arm2v7_seed*.json")):
    r = json.load(open(f))
    if r["F"]:
        seeds.append(dict(F=np.array(r["F"])))
groups["A2 monolithic MDO (per seed)"] = seeds
A3r = J("arm3_restored.json")
R3 = J("arm3_part0.json") + J("arm3_part1.json")
groups["A3 hand ATC, strict"] = [dict(F=np.array([r["f"] for r in R3 if r["feasible"]]))]
groups["A3 hand ATC, restored"] = [dict(F=np.array([r["f"] for r in A3r if r["feasible"]]))]

a4 = []
for rep in ("", "_r2", "_r3"):
    fs = [f for f in glob.glob(D + f"arm4v2_*{rep}.json") if (rep or not any(s in f for s in ("_r2", "_r3")))]
    R = [json.load(open(f)) for f in fs]
    if not R:
        continue
    pri = None
    by = {r["direction"]: r for r in R}
    if "balanced" in by:
        pri = sum(1 for dd, k in EMPH.items() if dd in by and by[dd]["f"][OBJ.index(k)] > by["balanced"]["f"][OBJ.index(k)])
    a4.append(dict(F=np.array([r["f"] for r in R if r["feasible"]]).reshape(-1, 4), n=len(R),
                   confirmed=sum(r["confirmed"] for r in R), feasible=sum(r["feasible"] for r in R),
                   cost=sum(r["est_cost_usd"] for r in R), priority=pri,
                   siren_fail=sum("siren_driver_power_1W" in r["violated"] for r in R)))
groups["A4 LLM negotiation"] = a4

def a5_group(pattern, limit=0.02):
    out = []
    for f in sorted(glob.glob(D + pattern)):
        d = json.load(open(f)); runs = d["sweep"] + d["directions"]
        ok = lambda r: bool(r.get("feasible_any_repair", r["feasible"]) and r["move_max"] <= limit)
        by = {r["direction"]: r for r in d["directions"]}
        pri = sum(1 for dd, k in EMPH.items() if by[dd]["f"][OBJ.index(k)] > by["balanced"]["f"][OBJ.index(k)] + 1e-6)
        g = sum(r["grounding"]["grounded"] for r in d["directions"]); n = sum(r["grounding"]["numbers"] for r in d["directions"])
        out.append(dict(F=np.array([r["f"] for r in runs if ok(r)]).reshape(-1, 4), n=len(runs),
                        atc_strict=sum(r["atc_strict"] for r in runs), feasible=sum(ok(r) for r in runs),
                        move_med=float(np.median([r["move_max"] for r in runs])), outer=float(np.mean([r["outer"] for r in runs])),
                        cost=d["cost"], priority=pri, grounding=(g, n),
                        base=dict(scal=d["base_spec"].get("scalarisation"), floors=d["base_spec"].get("objective_floors"),
                                  asp=d["base_spec"].get("aspiration"))))
    return out
groups["A5d grounded, 1-sided floor"] = a5_group("arm5d_r*.json")
groups["A5e grounded, 2-sided floor"] = a5_group("arm5e_r*.json")
a5 = groups["A5e grounded, 2-sided floor"] or groups["A5d grounded, 1-sided floor"]


allF = np.vstack([r["F"] for reps in groups.values() for r in reps if len(r["F"])])
lo, hi = allF.min(0), allF.max(0); hv = HV(ref_point=np.full(4, 1.1))
H = lambda F: float(hv(1 - (F - lo) / (hi - lo))) if len(F) else 0.0
comp = lambda F: F[F[:, 1] >= 3.0] if len(F) else F
Hu, Huc = H(allF), H(comp(allF))
ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
print(f"union of all feasible designs: {len(allF)}, HV {Hu:.3f}, compliant HV {Huc:.3f}\n")
print(f"{'arm':<30} {'reps':>4} {'HV (% union)':>22} {'compliant HV (% union)':>26} {'siren>=3&rt>=59.8':>18}")
summary = {}
for k, reps in groups.items():
    hs = [H(r["F"]) for r in reps]; hc = [H(comp(r["F"])) for r in reps]
    both = [int(np.sum((r["F"][:, 1] >= 3) & (r["F"][:, 0] >= 59.8))) if len(r["F"]) else 0 for r in reps]
    print(f"{k:<30} {len(reps):>4} {ms(hs):>14} ({100*np.mean(hs)/Hu:3.0f}%) {ms(hc):>18} ({100*np.mean(hc)/Huc:3.0f}%) {np.mean(both):>12.1f} ± {np.std(both):.1f}")
    summary[k] = dict(reps=len(reps), hv=hs, compliant_hv=hc, both=both)
print("\nA4 per repeat:", [(r["confirmed"], r["feasible"], r["siren_fail"], r["priority"], round(r["cost"], 2)) for r in a4],
      "(confirmed, feasible, siren failing, priorities honoured of 4, $)")
for nm in ("A5d grounded, 1-sided floor", "A5e grounded, 2-sided floor"):
  print(nm, "per repeat:", [(r["atc_strict"], r["feasible"], r["n"], round(100 * r["move_med"], 2), round(r["outer"], 1), r["priority"],
                          f"{r['grounding'][0]}/{r['grounding'][1]}", round(r["cost"], 3)) for r in groups[nm]],
      "(ATC-strict, feasible with <=2% repair, runs, median repair %, mean outer, priorities honoured of 4, grounding, $)")
for r in a5:
    print("   base spec:", r["base"])
json.dump(dict(union_hv=Hu, union_compliant_hv=Huc, arms=summary), open(D + "arm_compare2.json", "w"), indent=1)
