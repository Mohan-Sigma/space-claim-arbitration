"""
Arm 5 engine - the Arm 3 ATC machinery driven by a FORMULATION SPEC instead of hand-set choices.

What the spec controls (everything that was a hand decision in Arm 3 and is not a domain's own model):
  normalisation      : {objective: [worst, best]}          -> how objectives are scaled
  scalarisation      : "weighted_sum" | "achievement"      -> how the arbiter trades objectives
  aspiration         : {objective: value}                  -> reference point for "achievement" (ASF)
  weights            : {objective: w}                      -> priorities (or supplied by a sweep)
  objective_floors   : {objective: min value}              -> program statements promoted to hard system constraints
  coordination       : {beta, gamma, w0, tol, max_outer}
What stays fixed (owned by the domains, identical to Arm 3): the 14 linking targets, which domain
holds which copy, every local constraint set, and every evaluator.

Returns the final design plus the coordination record (multipliers = the "price" of each coupling)
used by the rationale layer.
"""
import numpy as np
from scipy.optimize import minimize
import arm3_atc as A3

OBJ = ["runtime", "siren", "voice", "l5"]
P, base = A3.P, A3.base


def _norm(f, spec):
    return np.array([(f[i] - spec["normalisation"][k][0]) / (spec["normalisation"][k][1] - spec["normalisation"][k][0])
                     for i, k in enumerate(OBJ)])


def _scalar(n, spec, wts):
    if spec.get("scalarisation", "weighted_sum") == "achievement":
        asp = _norm([spec["aspiration"][k] for k in OBJ], spec)
        short = wts * (asp - n)
        kappa, rho = 40.0, 0.05
        m = short.max()
        return float(m + np.log(np.sum(np.exp(kappa * (short - m)))) / kappa + rho * short.sum())
    return -float(np.dot(wts, n))


def _floors(T, spec):
    f = A3.sys_objectives(T)
    out = []
    for i, k in enumerate(OBJ):
        lo = (spec.get("objective_floors") or {}).get(k)
        if lo is not None:
            rng = spec["normalisation"][k][1] - spec["normalisation"][k][0]
            out.append((f[i] - lo) / rng)
    return np.array(out)


def solve_system(T0, X, v, w, wts, spec):
    TL, SC, CI = A3.TL, A3.SC, A3.CI

    def obj(Tn):
        T = TL + Tn * SC
        A3.EV["system"] += 1
        c = (T[CI] - X) / SC[CI]
        return _scalar(_norm(A3.sys_objectives(T), spec), spec, wts) + A3.penalty(c, v, w)

    cons = [{"type": "ineq", "fun": lambda Tn: A3.sys_constraints(TL + Tn * SC)}]
    if spec.get("objective_floors"):
        cons.append({"type": "ineq", "fun": lambda Tn: _floors(TL + Tn * SC, spec)})
    r = minimize(obj, (T0 - TL) / SC, method="SLSQP", bounds=[(0, 1)] * len(A3.TN),
                 constraints=cons, options=dict(maxiter=200, ftol=1e-10))
    return TL + np.clip(r.x, 0, 1) * SC


def run(spec, wts=None):
    wts = np.asarray(wts if wts is not None else [spec["weights"][k] for k in OBJ], float)
    wts = wts / max(wts.sum(), 1e-12)
    co = dict(beta=2.2, gamma=0.4, w0=1.0, tol=1e-3, max_outer=40); co.update(spec.get("coordination") or {})
    A3.EV["circuit"] = A3.EV["system"] = 0
    I, CI, SC, TN = A3.I, A3.CI, A3.SC, A3.TN
    T = np.array([P.TARGET[P.NAMES.index(n)] if n in P.NAMES else 0 for n in TN], float)
    T[I["V_acou"]], T[I["P_siren"]], T[I["voice"]] = 1460, 0.9, 78.0
    X = T[CI].copy()
    v = np.zeros(len(A3.COPIES)); w = np.full(len(A3.COPIES), float(co["w0"]))
    z = np.array([1460, 9.0, 0.5, 2.0, 30.0])
    c_prev, hist = None, []
    for k in range(int(co["max_outer"])):
        T = solve_system(T, X, v, w, wts, spec)
        X = X.copy()
        X[[0, 1]] = A3.sp_battery(T, v, w, X[[0, 1]])
        X[[2]] = A3.sp_board(T, v, w, X[[2]])
        X[[3, 4, 5]] = A3.sp_structure(T, v, w, X[[3, 4, 5]])
        X[[6, 7]] = A3.sp_thermal(T, v, w, X[[6, 7]])
        X[[8]] = A3.sp_antenna(T, v, w, X[[8]])
        z, resp = A3.sp_acoustics(T, v, w, z, first=(k == 0))
        X[[9, 10, 11, 12]] = resp
        X[[13]] = A3.sp_haptics(T, v, w, X[[13]])
        X[[14, 15]] = A3.sp_nfc(T, v, w, X[[14, 15]])
        c = (T[CI] - X) / SC[CI]
        hist.append(float(np.max(np.abs(c))))
        if hist[-1] < co["tol"]:
            break
        v = v + 2 * w ** 2 * c
        if c_prev is not None:
            w = np.where(np.abs(c) > co["gamma"] * np.abs(c_prev), co["beta"] * w, w)
        c_prev = c
    d = dict(V_batt=X[0], A_batt=X[1], A_board=X[2], V_struct=X[3], t_wall=X[4], V_ant=X[8],
             V_acou=z[0], port=z[1], split=z[2], h_out=z[3], r_mesh=z[4], V_hap=X[13],
             A_spr=X[6], V_loop=X[14], V_ferr=X[15])
    x = np.array([d[n] for n in P.NAMES])
    f, g, info = P.evaluate(x)
    prices = {f"{sp}:{n}": float(v[i]) for i, (sp, n) in enumerate(A3.COPIES)}
    return dict(w=wts.tolist(), x=x.tolist(), f=f.tolist(), g=g.tolist(), outer=k + 1, max_c=hist[-1], hist=hist,
                circuit_evals=A3.EV["circuit"], system_evals=A3.EV["system"],
                feasible=bool(np.all(g <= 1e-6)), feasible_tol=bool(np.all(g <= 5e-3)),
                targets={n: float(T[i]) for i, n in enumerate(TN)}, prices=prices,
                info={k: float(val) for k, val in info.items()})


ARM3_SPEC = dict(normalisation=dict(runtime=[40.0, 63.0], siren=[1.5, 3.5], voice=[70.0, 85.0], l5=[0.05, 0.30]),
                 scalarisation="weighted_sum", objective_floors={}, coordination={})
