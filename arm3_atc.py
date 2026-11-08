"""
Arm 3 - hand-formulated Analytical Target Cascading (ATC), augmented-Lagrangian
coordination (Tosserams, Etman, Papalambros & Rooda 2006/2008; Kim et al. 2003).
Main benchmark case, schema v0.7.

System (arbiter) level
  targets T (14): V_batt, A_batt, A_board, V_struct, t_wall, V_ant, V_acou, port, V_hap,
                  A_spr, V_loop, V_ferr, P_siren (response), voice (response)
  objective     : weighted sum of normalised system objectives
                  runtime(V_batt), siren endurance(V_batt, P_siren), voice, L5(V_ant, t_wall)
  constraints   : interior volume pool (depends on t_wall), R6 perimeter, R2 caseback, R7 port pool

Subproblems (each holds its own constraints, matches the targets it is given)
  battery   (V_batt, A_batt)          2.6 <= V/A <= 4.8, Z-stack
  board     (A_board)                 routability >= 0.95
  structure (t_wall, V_struct, port)  pressure margin >= 1.5
  thermal   (A_spr, t_wall)           skin <= 43 C
  antenna   (V_ant)                   bounds
  acoustics (V_acou, port, split, h_out, r_mesh) -> responses P_siren, voice
                                      P_siren <= 1 W, Q <= 15
  haptics   (V_hap)                   force >= 0.9 N
  nfc       (V_loop, V_ferr)          read range >= 10 mm

Coordination: alternate system solve / subproblem solves; after each pass
  v <- v + 2 w^2 c ;  w <- beta w where |c| did not shrink below gamma |c_prev|
  beta 2.2, gamma 0.4, w0 = 1, v0 = 0 (Tosserams et al. recommended values)
Stop when max |c| < 1e-3 (normalised) or 40 outer iterations.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import evaluators as ev
import arm2v6_problem as P
base = P.base

TN = ["V_batt", "A_batt", "A_board", "V_struct", "t_wall", "V_ant", "V_acou", "port",
      "V_hap", "A_spr", "V_loop", "V_ferr", "P_siren", "voice"]
TL = np.array([2600, 520, 950, 1400, 1.0, 900, 900, 3.0, 200, 180, 180, 110, 0.05, 0.0])
TU = np.array([5200, 960, 1500, 2400, 1.8, 2400, 2400, 11.0, 620, 520, 520, 300, 1.0, 100.0])
SC = TU - TL
I = {n: i for i, n in enumerate(TN)}

# copies held by subproblems: (subproblem, target name)
COPIES = [("bat", "V_batt"), ("bat", "A_batt"), ("board", "A_board"),
          ("str", "V_struct"), ("str", "t_wall"), ("str", "port"),
          ("th", "A_spr"), ("th", "t_wall"), ("ant", "V_ant"),
          ("ac", "V_acou"), ("ac", "port"), ("ac", "P_siren"), ("ac", "voice"),
          ("hap", "V_hap"), ("nfc", "V_loop"), ("nfc", "V_ferr")]
CI = np.array([I[n] for _, n in COPIES])

# fixed a-priori normalisation of system objectives (hand-set formulation choice)
NORM = dict(runtime=(40.0, 63.0), siren=(1.5, 3.5), voice=(70.0, 85.0), l5=(0.05, 0.30))
EV = {"circuit": 0, "system": 0}


def sys_objectives(T):
    vb, t = T[I["V_batt"]], T[I["t_wall"]]
    runtime = ev.battery_runtime_h(vb)
    p_avg = P.SIREN_DUTY * T[I["P_siren"]] / base.AMP_EFF
    siren = ev.battery_energy_wh(vb) / (p_avg + 0.15)
    cond = T[I["V_ant"]] / (167.0 * 6.0)
    l5 = ev.antenna_efficiency(T[I["V_ant"]], cond, base.L5_F, base.L5_BW) * (0.8 / t) ** 0.5
    return np.array([runtime, siren, T[I["voice"]], l5])


def _n(f):
    keys = ("runtime", "siren", "voice", "l5")
    return np.array([(f[i] - NORM[k][0]) / (NORM[k][1] - NORM[k][0]) for i, k in enumerate(keys)])


def penalty(c, v, w):
    return float(np.dot(v, c) + np.sum((w * c) ** 2))


def sys_constraints(T):
    t = T[I["t_wall"]]
    pool = base.POOL_R1_NOM - base.DV_DT * (t - 1.2)
    vol = (T[I["V_batt"]] + T[I["A_board"]] * base.T_BOARD + T[I["V_struct"]] + T[I["V_ant"]]
           + T[I["V_acou"]] + T[I["V_hap"]] + T[I["A_spr"]] * base.T_SPREADER + T[I["V_loop"]] + T[I["V_ferr"]])
    return np.array([(pool - vol) / 1000.0,
                     (base.R6_POOL - T[I["V_ant"]] - T[I["V_loop"]]) / 100.0,
                     (base.R2_POOL - T[I["A_batt"]] - T[I["A_spr"]]) / 100.0,
                     (base.R7_POOL - base.MIC_PORT - T[I["port"]])])


def solve_system(T0, X, v, w, wts):
    """X: current subproblem copy values (len(COPIES)), in physical units."""
    def obj(Tn):
        T = TL + Tn * SC
        EV["system"] += 1
        c = (T[CI] - X) / SC[CI]
        return -float(np.dot(wts, _n(sys_objectives(T)))) + penalty(c, v, w)
    cons = [{"type": "ineq", "fun": lambda Tn: sys_constraints(TL + Tn * SC)}]
    r = minimize(obj, (T0 - TL) / SC, method="SLSQP", bounds=[(0, 1)] * len(TN),
                 constraints=cons, options=dict(maxiter=200, ftol=1e-10))
    return TL + np.clip(r.x, 0, 1) * SC


# ---------------- subproblems (each: argmin sum penalty on its copies s.t. local constraints)
def _sp_generic(names, x0, T, v, w, cons_fun, bounds):
    idx = [k for k, (_, n) in enumerate(COPIES) if (_, n) in [(s, m) for s, m in COPIES] and n in names]
    ks = [k for k, (sp, n) in enumerate(COPIES) if n in names]
    sc = np.array([SC[I[n]] for n in names])
    tgt = np.array([T[I[n]] for n in names])

    def obj(xn):
        c = (tgt - xn * sc) / sc
        return penalty(c, v[ks], w[ks])
    cons = [] if cons_fun is None else [{"type": "ineq", "fun": lambda xn: cons_fun(xn * sc)}]
    bnd = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, sc)]
    r = minimize(obj, np.clip(x0, [b[0] * s for b, s in zip(bounds, sc)], [b[1] * s for b, s in zip(bounds, sc)]) / sc,
                 method="SLSQP", bounds=bnd, constraints=cons, options=dict(maxiter=200, ftol=1e-12))
    return r.x * sc


def sp_battery(T, v, w, x0):
    ks = [0, 1]
    def cf(x):
        th = x[0] / x[1]
        return np.array([th - 2.6, 4.8 - th, base.Z_AVAIL - base.T_BOARD - th])
    return _solve_sub(ks, ["V_batt", "A_batt"], T, v, w, x0, cf, [(2600, 5200), (520, 960)])


def sp_board(T, v, w, x0):
    return _solve_sub([2], ["A_board"], T, v, w, x0, lambda x: np.array([ev.board_routability(x[0]) - 0.95]), [(950, 1500)])


def sp_structure(T, v, w, x0):
    return _solve_sub([3, 4, 5], ["V_struct", "t_wall", "port"], T, v, w, x0,
                      lambda x: np.array([ev.pressure_margin(x[1], x[2], x[0]) - 1.5]),
                      [(1400, 2400), (1.0, 1.8), (3.0, 11.0)])


def sp_thermal(T, v, w, x0):
    return _solve_sub([6, 7], ["A_spr", "t_wall"], T, v, w, x0,
                      lambda x: np.array([43.0 - ev.skin_temp_c(x[0], 0.9, x[1], r_case=base.R_CASE_CAL)]),
                      [(180, 520), (1.0, 1.8)])


def sp_antenna(T, v, w, x0):
    return _solve_sub([8], ["V_ant"], T, v, w, x0, None, [(900, 2400)])


def sp_haptics(T, v, w, x0):
    return _solve_sub([13], ["V_hap"], T, v, w, x0,
                      lambda x: np.array([ev.haptic_force_n(x[0], base.HAPTIC_STROKE) - 0.9]), [(200, 620)])


def sp_nfc(T, v, w, x0):
    return _solve_sub([14, 15], ["V_loop", "V_ferr"], T, v, w, x0,
                      lambda x: np.array([ev.nfc_read_range_mm(400 * x[0] / 320, x[1] / 900, 1.0) - 10.0]),
                      [(180, 520), (110, 300)])


def _solve_sub(ks, names, T, v, w, x0, cf, bounds):
    sc = np.array([SC[I[n]] for n in names]); tgt = np.array([T[I[n]] for n in names])
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    def obj(xn):
        return penalty((tgt - xn * sc) / sc, v[ks], w[ks])
    cons = [] if cf is None else [{"type": "ineq", "fun": lambda xn: cf(xn * sc)}]
    r = minimize(obj, np.clip(x0, lo, hi) / sc, method="SLSQP", bounds=list(zip(lo / sc, hi / sc)),
                 constraints=cons, options=dict(maxiter=200, ftol=1e-12))
    x = r.x * sc
    if cf is not None and np.any(cf(x) < -1e-6):          # SLSQP failed: fall back to a feasible start
        r = minimize(obj, np.clip(x0, lo, hi) / sc, method="trust-constr", bounds=list(zip(lo / sc, hi / sc)),
                     constraints=cons)
        x = r.x * sc
    return x


def acoustics_eval(z):
    v_acou, port, split, h, mesh = z
    free = v_acou - base.V_DRIVER
    EV["circuit"] += 1
    s, q, fres, vs, _, _ = base._acoustics(free * split, free * (1 - split), port, h, mesh)
    return s["P"], vs, q


def sp_acoustics(T, v, w, z0, first):
    ks = [9, 10, 11, 12]
    names = ["V_acou", "port", "P_siren", "voice"]
    sc = np.array([SC[I[n]] for n in names]); tgt = np.array([T[I[n]] for n in names])
    lo = np.array([900, 3.0, 0.30, 0.3, 20.0]); hi = np.array([2400, 11.0, 0.95, 3.5, 200.0])

    def obj(z):
        z = np.clip(z, lo, hi)
        p, vs, q = acoustics_eval(z)
        resp = np.array([z[0], z[1], p, vs])
        viol = max(0.0, p - base.P_DRIVER_MAX) + max(0.0, q - 15.0) / 15.0 + max(0.0, 840.0 - z[0]) / 100.0
        return penalty((tgt - resp) / sc, v[ks], w[ks]) + 1e3 * viol ** 2 + (10.0 * viol if viol > 0 else 0.0)

    if first:
        r = differential_evolution(obj, list(zip(lo, hi)), popsize=10, maxiter=30, seed=0, polish=False, tol=1e-8)
        z = r.x
    else:
        r = minimize(obj, z0, method="Nelder-Mead",
                     options=dict(maxfev=400, xatol=1e-4, fatol=1e-9, initial_simplex=None))
        z = np.clip(r.x, lo, hi)
        if obj(z) > obj(np.clip(z0, lo, hi)):
            z = np.clip(z0, lo, hi)
    p, vs, q = acoustics_eval(z)
    return z, np.array([z[0], z[1], p, vs])


def run(wts, max_outer=40, tol=1e-3, beta=2.2, gamma=0.4):
    wts = np.asarray(wts, float)
    EV["circuit"] = EV["system"] = 0
    T = np.array([P.TARGET[P.NAMES.index(n)] if n in P.NAMES else 0 for n in TN], float)
    T[I["V_acou"]], T[I["P_siren"]], T[I["voice"]] = 1460, 0.9, 78.0
    X = T[CI].copy()
    v = np.zeros(len(COPIES)); w = np.ones(len(COPIES))
    z = np.array([1460, 9.0, 0.5, 2.0, 30.0])
    c_prev = None; hist = []
    for k in range(max_outer):
        T = solve_system(T, X, v, w, wts)
        X = X.copy()
        X[[0, 1]] = sp_battery(T, v, w, X[[0, 1]])
        X[[2]] = sp_board(T, v, w, X[[2]])
        X[[3, 4, 5]] = sp_structure(T, v, w, X[[3, 4, 5]])
        X[[6, 7]] = sp_thermal(T, v, w, X[[6, 7]])
        X[[8]] = sp_antenna(T, v, w, X[[8]])
        z, resp = sp_acoustics(T, v, w, z, first=(k == 0))
        X[[9, 10, 11, 12]] = resp
        X[[13]] = sp_haptics(T, v, w, X[[13]])
        X[[14, 15]] = sp_nfc(T, v, w, X[[14, 15]])
        c = (T[CI] - X) / SC[CI]
        hist.append(float(np.max(np.abs(c))))
        if hist[-1] < tol:
            break
        v = v + 2 * w ** 2 * c
        if c_prev is not None:
            w = np.where(np.abs(c) > gamma * np.abs(c_prev), beta * w, w)
        c_prev = c
    x = np.array([X[0], X[1], X[2], X[3], X[4], z[0], z[1], z[2], z[3], z[4], X[13], X[6], X[14], X[15]])
    # assemble in P.NAMES order
    d = dict(V_batt=X[0], A_batt=X[1], A_board=X[2], V_struct=X[3], t_wall=X[4], V_ant=X[8],
             V_acou=z[0], port=z[1], split=z[2], h_out=z[3], r_mesh=z[4], V_hap=X[13],
             A_spr=X[6], V_loop=X[14], V_ferr=X[15])
    x = np.array([d[n] for n in P.NAMES])
    f, g, info = P.evaluate(x)
    return dict(w=wts.tolist(), x=x.tolist(), f=f.tolist(), g=g.tolist(), outer=k + 1,
                max_c=hist[-1], hist=hist, circuit_evals=EV["circuit"], system_evals=EV["system"],
                feasible=bool(np.all(g <= 1e-6)), feasible_tol=bool(np.all(g <= 5e-3)),
                port_struct=float(X[5]), t_thermal=float(X[7]))
