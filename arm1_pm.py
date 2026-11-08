"""
A1 + program-manager escalation (deterministic).

The waterfall of arm1_waterfall.py runs as before. When the integrated design meets every
hard constraint but misses the system requirement (siren endurance >= 3 h), the program
manager holds an escalation review. Each review is one further design spin and uses only the
information a PM receives from the leads in a review:
  - acoustics reports its siren tone power P;
  - battery reports the energy needed for 3 h at that power, E_req = 3 (delta P / eta + P_sys).
PM levers, in the order a program would normally use them:
  E1  direct acoustics to minimise siren power (keeping voice >= 70) instead of maximising voice;
  E2  raise the battery specification to the reported E_req (+2% margin) and re-run the waterfall,
      so upstream claimants absorb the volume through the normal rework rules;
  E3  once the battery can no longer grow (cell geometry limit), approve the largest speaker port
      (11 mm^2) in the interface specification; structure thickens the wall to keep the margin;
  E4  reserve acoustic volume (1300 mm^3, then +100 mm^3 per further review); upstream claimants,
      and finally the battery, give it up through the normal rework rules.
At most 6 reviews. Usage: python3 arm1_pm.py <part 0|1>
"""
import sys, json, time, numpy as np
from scipy.optimize import differential_evolution
import arm1_waterfall as W, arm2v6_problem as P
base = P.base
MODE = {"m": "voice"}
VOICE_FLOOR = 70.0

_orig = W.acoustics_local
def acoustics_pm(v_alloc, port_cap, seed):
    if MODE["m"] == "voice":
        return _orig(v_alloc, port_cap, seed)
    if v_alloc < 900:
        return None
    hi_v = max(v_alloc, 900.0 + 1e-6)
    def obj(z):
        W.EVALS["acoustic"] += 1
        v, split, port, h, mesh = z
        free = v - base.V_DRIVER
        s, q, fres, vs, _, _ = base._acoustics(free * split, free * (1 - split), port, h, mesh)
        pen = max(0.0, q - 15.0) + max(0.0, VOICE_FLOOR - vs)
        if s["P"] > base.P_DRIVER_MAX or pen > 0:
            return 1000.0 + s["P"] + pen
        return s["P"]
    res = differential_evolution(obj, [(900.0, hi_v), (0.30, 0.95), (3.0, port_cap), (0.3, 3.5), (20.0, 200.0)],
                                 popsize=10, maxiter=25, tol=1e-6, seed=seed, polish=False)
    v, split, port, h, mesh = res.x
    free = v - base.V_DRIVER
    s, q, fres, vs, _, _ = base._acoustics(free * split, free * (1 - split), port, h, mesh)
    return dict(v=v, split=split, port=port, h=h, mesh=mesh, P=s["P"], voice=vs, ok=res.fun < 1000)
W.acoustics_local = acoustics_pm

def program(vb, va, max_reviews=6):
    MODE["m"] = "voice"; W.ICD_PORT = 9.0; W.MIN_AC = 0.0; spins = 0; reviews = []; spec = vb; last_vb = None
    r = W.run(spec, va); spins += r["loops"]
    for k in range(max_reviews + 1):
        if not r["ok"]:
            return dict(ok=False, reached=False, reviews=reviews, spins=spins, r=r)
        f, g, info = P.evaluate(np.array(r["x"]))
        if f[1] >= 3.0 - 1e-6 or k == max_reviews:
            return dict(ok=True, reached=bool(f[1] >= 3.0 - 1e-6), reviews=reviews, spins=spins + len(reviews), r=r, f=f.tolist(),
                        x=r["x"], p_siren=float(info["p_siren"]))
        p = float(info["p_siren"])
        e_req = 3.0 * (P.SIREN_DUTY * p / base.AMP_EFF + 0.15)
        if MODE["m"] == "voice":
            MODE["m"] = "power"; reviews.append(dict(lever="E1 acoustics minimise power", siren=float(f[1]), p=p))
        elif (last_vb is None or r["x"][0] > last_vb * 1.005) and W.ICD_PORT < 11.0:
            last_vb = r["x"][0]
            v_req = e_req / 600.0 * 1e6 * 1.02
            spec = max(spec, v_req, r["x"][0] * 1.02)
            reviews.append(dict(lever="E2 raise battery spec", siren=float(f[1]), p=p, new_spec=spec))
        elif W.ICD_PORT < 11.0:
            W.ICD_PORT = 11.0
            reviews.append(dict(lever="E3 approve 11 mm2 port", siren=float(f[1]), p=p))
        else:
            W.MIN_AC = 1300.0 if W.MIN_AC == 0 else W.MIN_AC + 100.0
            reviews.append(dict(lever=f"E4 reserve acoustic volume {W.MIN_AC:.0f}", siren=float(f[1]), p=p))
        r = W.run(spec, va); spins += r["loops"]

if __name__ == "__main__":
    part = int(sys.argv[1])
    runtimes = np.linspace(42, 62, 10)
    v_batts = [rt * 0.038 / 600 * 1e6 for rt in runtimes]
    v_ants = [900, 1200, 1500, 1800, 2100]
    jobs = [(vb, va) for vb in v_batts for va in v_ants][part::2]
    out = []
    for vb, va in jobs:
        t0 = time.time(); res = program(vb, va)
        res.update(spec_runtime=vb / 0.038 * 600 / 1e6, v_ant=va, wall_s=time.time() - t0)
        res.pop("r", None); out.append(res)
        print(f"spec {res['spec_runtime']:.1f} h ant {va}: reached={res['reached']} reviews={len(res['reviews'])} spins={res['spins']}"
              + (f" f={np.round(res['f'],3).tolist()}" if res.get("f") else ""), flush=True)
    json.dump(out, open(f"arm1pm_part{part}.json", "w"), indent=1)
