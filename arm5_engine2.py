"""
Arm 5 engine v2 - two fixes for the floor/response coupling found in Arm 5 v1-v3.

FIX 1 (ATC-standard reformulation): a floor on siren endurance couples a battery quantity (energy)
with an acoustic RESPONSE (siren power). Enforcing it at the arbiter on the response TARGET made the
arbiter ask for a siren power the acoustics subproblem could not deliver, and coordination stalled at
~0.2% residual. Here the floor is moved into the acoustics subproblem as a LOCAL constraint on its own
response, parameterised by the arbiter's battery target:
        endurance(V_batt_target, P) >= S   <=>   P <= P_cap = (E(V_batt_target)/S - 0.15) * eta_amp / duty
The arbiter no longer carries the siren floor; floors on quantities the arbiter computes directly from
its own targets (e.g. runtime from battery volume) stay at the arbiter.

FIX 2 (feasibility restoration): after ATC, if the assembled design violates any constraint (residual
inconsistency between subproblem copies), solve the minimum-change problem
        min sum(((x - x_atc) / range)^2)  s.t. all constraints and floors, full model
The size of the move is reported for every design - it is the honest cost of the repair.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import evaluators as ev
import arm3_atc as A3
import arm5_engine as E1

P, base = A3.P, A3.base
OBJ = E1.OBJ


def p_cap(v_batt, floor_h):
    e = ev.battery_energy_wh(v_batt)
    return min(base.P_DRIVER_MAX, (e / floor_h - 0.15) * base.AMP_EFF / P.SIREN_DUTY)


def sp_acoustics_local_floor(T, v, w, z0, first, floor_h):
    ks = [9, 10, 11, 12]
    names = ["V_acou", "port", "P_siren", "voice"]
    sc = np.array([A3.SC[A3.I[n]] for n in names]); tgt = np.array([T[A3.I[n]] for n in names])
    lo = np.array([900, 3.0, 0.30, 0.3, 20.0]); hi = np.array([2400, 11.0, 0.95, 3.5, 200.0])
    cap = p_cap(T[A3.I["V_batt"]], floor_h) if floor_h else base.P_DRIVER_MAX

    def obj(z):
        z = np.clip(z, lo, hi)
        p, vs, q = A3.acoustics_eval(z)
        resp = np.array([z[0], z[1], p, vs])
        viol = max(0.0, p - cap) + max(0.0, q - 15.0) / 15.0 + max(0.0, 840.0 - z[0]) / 100.0
        return A3.penalty((tgt - resp) / sc, v[ks], w[ks]) + 1e3 * viol ** 2 + (10.0 * viol if viol > 0 else 0.0)

    if first:
        r = differential_evolution(obj, list(zip(lo, hi)), popsize=10, maxiter=30, seed=0, polish=False, tol=1e-8)
        z = r.x
    else:
        r = minimize(obj, z0, method="Nelder-Mead", options=dict(maxfev=400, xatol=1e-4, fatol=1e-9))
        z = np.clip(r.x, lo, hi)
        if obj(z) > obj(np.clip(z0, lo, hi)):
            z = np.clip(z0, lo, hi)
    p, vs, q = A3.acoustics_eval(z)
    return z, np.array([z[0], z[1], p, vs])


def floors_ok(f, floors, tol=0.0):
    return all(f[OBJ.index(k)] >= v - tol for k, v in (floors or {}).items())


def restore(x0, floors=None):
    """Minimum-change feasibility restoration with the full model."""
    x0 = np.asarray(x0, float); rng = P.XU - P.XL
    f0, g0, _ = P.evaluate(x0)
    if np.all(g0 <= 0) and floors_ok(f0, floors):
        return dict(x=x0, restored=False, ok=True, move_max=0.0, move_rms=0.0, moved=[])
    fl = list((floors or {}).items())

    def cons(xn):
        x = P.XL + xn * rng
        f, g, _ = P.evaluate(x)
        c = list(-g)
        for k, v in fl:
            c.append((f[OBJ.index(k)] - v) / max(abs(v), 1e-6))
        return np.array(c)
    xn0 = (x0 - P.XL) / rng
    r = minimize(lambda xn: float(np.sum((xn - xn0) ** 2)), xn0, method="SLSQP", bounds=[(0, 1)] * len(xn0),
                 constraints=[{"type": "ineq", "fun": cons}], options=dict(maxiter=300, ftol=1e-12))
    x = P.XL + np.clip(r.x, 0, 1) * rng
    f, g, _ = P.evaluate(x)
    ok = bool(np.all(g <= 1e-9) and floors_ok(f, floors, tol=1e-9))
    rel = (x - x0) / rng
    moved = sorted(((P.NAMES[i], float(x0[i]), float(x[i])) for i in range(len(x)) if abs(rel[i]) > 1e-3),
                   key=lambda t: -abs(t[2] - t[1]) / rng[P.NAMES.index(t[0])])
    return dict(x=x, restored=True, ok=ok, move_max=float(np.max(np.abs(rel))), move_rms=float(np.sqrt(np.mean(rel ** 2))), moved=moved)


def run(spec, wts=None, floor_mode="local", do_restore=True):
    wts = np.asarray(wts if wts is not None else [spec["weights"][k] for k in OBJ], float)
    wts = wts / max(wts.sum(), 1e-12)
    co = dict(beta=2.2, gamma=0.4, w0=1.0, tol=1e-3, max_outer=40)   # solver parameters locked (engine-owned)
    floors = dict(spec.get("objective_floors") or {})
    siren_floor = floors.get("siren") if floor_mode == "local" else None
    sys_spec = dict(spec)
    if floor_mode == "local":
        sys_spec["objective_floors"] = {k: v for k, v in floors.items() if k != "siren"}
    A3.EV["circuit"] = A3.EV["system"] = 0
    I, CI, SC, TN = A3.I, A3.CI, A3.SC, A3.TN
    T = np.array([P.TARGET[P.NAMES.index(n)] if n in P.NAMES else 0 for n in TN], float)
    T[I["V_acou"]], T[I["P_siren"]], T[I["voice"]] = 1460, 0.9, 78.0
    X = T[CI].copy()
    v = np.zeros(len(A3.COPIES)); w = np.full(len(A3.COPIES), co["w0"])
    z = np.array([1460, 9.0, 0.5, 2.0, 30.0]); c_prev, hist = None, []
    for k in range(co["max_outer"]):
        T = E1.solve_system(T, X, v, w, wts, sys_spec)
        X = X.copy()
        X[[0, 1]] = A3.sp_battery(T, v, w, X[[0, 1]])
        X[[2]] = A3.sp_board(T, v, w, X[[2]])
        X[[3, 4, 5]] = A3.sp_structure(T, v, w, X[[3, 4, 5]])
        X[[6, 7]] = A3.sp_thermal(T, v, w, X[[6, 7]])
        X[[8]] = A3.sp_antenna(T, v, w, X[[8]])
        if siren_floor:
            z, resp = sp_acoustics_local_floor(T, v, w, z, k == 0, siren_floor)
        else:
            z, resp = A3.sp_acoustics(T, v, w, z, first=(k == 0))
        X[[9, 10, 11, 12]] = resp
        X[[13]] = A3.sp_haptics(T, v, w, X[[13]])
        X[[14, 15]] = A3.sp_nfc(T, v, w, X[[14, 15]])
        c = (T[CI] - X) / SC[CI]; hist.append(float(np.max(np.abs(c))))
        if hist[-1] < co["tol"]:
            break
        v = v + 2 * w ** 2 * c
        if c_prev is not None:
            w = np.where(np.abs(c) > co["gamma"] * np.abs(c_prev), co["beta"] * w, w)
        c_prev = c
    d = dict(V_batt=X[0], A_batt=X[1], A_board=X[2], V_struct=X[3], t_wall=X[4], V_ant=X[8], V_acou=z[0], port=z[1],
             split=z[2], h_out=z[3], r_mesh=z[4], V_hap=X[13], A_spr=X[6], V_loop=X[14], V_ferr=X[15])
    x_atc = np.array([d[n] for n in P.NAMES])
    f_atc, g_atc, _ = P.evaluate(x_atc)
    atc_strict = bool(np.all(g_atc <= 1e-6) and floors_ok(f_atc, floors, 1e-6))
    rs = restore(x_atc, floors) if do_restore else dict(x=x_atc, restored=False, ok=atc_strict, move_max=0, move_rms=0, moved=[])
    x = rs["x"]; f, g, info = P.evaluate(x)
    prices = {f"{sp}:{n}": float(v[i]) for i, (sp, n) in enumerate(A3.COPIES)}
    return dict(w=wts.tolist(), x=x.tolist(), f=f.tolist(), g=g.tolist(), x_atc=x_atc.tolist(), f_atc=f_atc.tolist(),
                outer=k + 1, max_c=hist[-1], atc_strict=atc_strict,
                feasible=bool(np.all(g <= 1e-6) and floors_ok(f, floors, 1e-6)),
                feasible_tol=bool(np.all(g_atc <= 5e-3)),
                restored=rs["restored"], restore_ok=rs["ok"], move_max=rs["move_max"], move_rms=rs["move_rms"], moved=rs["moved"][:4],
                circuit_evals=A3.EV["circuit"], system_evals=A3.EV["system"],
                targets={n: float(T[i]) for i, n in enumerate(TN)}, prices=prices,
                info={kk: float(val) for kk, val in info.items()})


# ---------------------------------------------------------------------------------------------
# FIX 1b: two-sided local floor. Endurance(V_batt, P) >= S couples battery and acoustics, so each
# side gets the constraint on its OWN variable with the OTHER side's quantity taken from the
# arbiter's target. At consistency (copies == targets) both hold, so the floor holds.
#   battery   : E(V_batt) >= S * (duty * P_siren_target / eta_amp + 0.15)
#   acoustics : P_siren   <= (E(V_batt_target) / S - 0.15) * eta_amp / duty
# ---------------------------------------------------------------------------------------------
def sp_battery_floor(T, v, w, x0, floor_h):
    pt = T[A3.I["P_siren"]]
    def cf(x):
        th = x[0] / x[1]
        endur = ev.battery_energy_wh(x[0]) / (P.SIREN_DUTY * pt / base.AMP_EFF + 0.15)
        return np.array([th - 2.6, 4.8 - th, base.Z_AVAIL - base.T_BOARD - th, (endur - floor_h) / floor_h])
    return A3._solve_sub([0, 1], ["V_batt", "A_batt"], T, v, w, x0, cf, [(2600, 5200), (520, 960)])


def run2(spec, wts=None, do_restore=True, repair_limit=0.02):
    """As run(), with the siren floor split two-sided between battery and acoustics."""
    wts = np.asarray(wts if wts is not None else [spec["weights"][k] for k in OBJ], float)
    wts = wts / max(wts.sum(), 1e-12)
    co = dict(beta=2.2, gamma=0.4, w0=1.0, tol=1e-3, max_outer=40)
    floors = dict(spec.get("objective_floors") or {})
    S = floors.get("siren")
    sys_spec = dict(spec); sys_spec["objective_floors"] = {k: v for k, v in floors.items() if k != "siren"}
    A3.EV["circuit"] = A3.EV["system"] = 0
    I, CI, SC, TN = A3.I, A3.CI, A3.SC, A3.TN
    T = np.array([P.TARGET[P.NAMES.index(n)] if n in P.NAMES else 0 for n in TN], float)
    T[I["V_acou"]], T[I["P_siren"]], T[I["voice"]] = 1460, 0.9, 78.0
    X = T[CI].copy()
    v = np.zeros(len(A3.COPIES)); w = np.full(len(A3.COPIES), co["w0"])
    z = np.array([1460, 9.0, 0.5, 2.0, 30.0]); c_prev, hist = None, []
    for k in range(co["max_outer"]):
        T = E1.solve_system(T, X, v, w, wts, sys_spec)
        X = X.copy()
        X[[0, 1]] = sp_battery_floor(T, v, w, X[[0, 1]], S) if S else A3.sp_battery(T, v, w, X[[0, 1]])
        X[[2]] = A3.sp_board(T, v, w, X[[2]])
        X[[3, 4, 5]] = A3.sp_structure(T, v, w, X[[3, 4, 5]])
        X[[6, 7]] = A3.sp_thermal(T, v, w, X[[6, 7]])
        X[[8]] = A3.sp_antenna(T, v, w, X[[8]])
        z, resp = sp_acoustics_local_floor(T, v, w, z, k == 0, S) if S else A3.sp_acoustics(T, v, w, z, first=(k == 0))
        X[[9, 10, 11, 12]] = resp
        X[[13]] = A3.sp_haptics(T, v, w, X[[13]])
        X[[14, 15]] = A3.sp_nfc(T, v, w, X[[14, 15]])
        c = (T[CI] - X) / SC[CI]; hist.append(float(np.max(np.abs(c))))
        if hist[-1] < co["tol"]:
            break
        v = v + 2 * w ** 2 * c
        if c_prev is not None:
            w = np.where(np.abs(c) > co["gamma"] * np.abs(c_prev), co["beta"] * w, w)
        c_prev = c
    d = dict(V_batt=X[0], A_batt=X[1], A_board=X[2], V_struct=X[3], t_wall=X[4], V_ant=X[8], V_acou=z[0], port=z[1],
             split=z[2], h_out=z[3], r_mesh=z[4], V_hap=X[13], A_spr=X[6], V_loop=X[14], V_ferr=X[15])
    x_atc = np.array([d[n] for n in P.NAMES]); f_atc, g_atc, _ = P.evaluate(x_atc)
    atc_strict = bool(np.all(g_atc <= 1e-6) and floors_ok(f_atc, floors, 1e-6))
    rs = restore(x_atc, floors) if do_restore else dict(x=x_atc, restored=False, ok=atc_strict, move_max=0, move_rms=0, moved=[])
    x = rs["x"]; f, g, info = P.evaluate(x)
    strict_after = bool(np.all(g <= 1e-6) and floors_ok(f, floors, 1e-6))
    prices = {f"{sp}:{n}": float(v[i]) for i, (sp, n) in enumerate(A3.COPIES)}
    return dict(w=wts.tolist(), x=x.tolist(), f=f.tolist(), g=g.tolist(), x_atc=x_atc.tolist(), f_atc=f_atc.tolist(),
                outer=k + 1, max_c=hist[-1], atc_strict=atc_strict,
                feasible=bool(strict_after and rs["move_max"] <= repair_limit),     # polishing only
                feasible_any_repair=strict_after, restored=rs["restored"], move_max=rs["move_max"],
                move_rms=rs["move_rms"], moved=rs["moved"][:4], circuit_evals=A3.EV["circuit"], system_evals=A3.EV["system"],
                targets={n: float(T[i]) for i, n in enumerate(TN)}, prices=prices,
                info={kk: float(val) for kk, val in info.items()})
