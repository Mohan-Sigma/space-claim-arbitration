"""
Arm 1 - sequential waterfall with program-manager rework loops.
Main benchmark case (schema v0.7): open cavity, 86 dB @ 1 m, 18x13 driver, 50% siren duty, SII voice.

Program order (each domain sees only upstream decisions, claims its own target, never
plans for downstream):
  1 EE board        A_board = target 1250 mm^2
  2 Battery         V_batt from the runtime spec; footprint 800 mm^2 preferred, thickest allowed if needed
  3 Antenna         V_ant = program allocation
  4 Thermal         A_spr = minimum meeting 43 C skin at the current wall thickness
  5 Structure       V_struct = target 1800; wall = minimum meeting 1.5x margin at the port in the ICD
                    (acoustics' declared target port, 9.0 mm^2)
  6 Haptics         V_hap = target 350
  7 NFC             loop 320, ferrite 180
  8 Acoustics       takes the remaining volume (900..2400 mm^3), port <= ICD cap; optimises its own
                    split / port / outlet / mesh: meet siren <= 1 W first, then maximise voice

Rework rules on failure (each pass through the rules = one rework loop = one design spin):
  R-port   siren fails although acoustics has its full volume or is volume-starved at max port
           -> structure reopens the ICD: +1 mm^2 port cap (wall recomputed), max 11 mm^2
  R-vol    acoustics remainder < its minimum or siren fails while port is at the cap
           -> every upstream non-battery claimant cuts 10% of (target - own minimum)
  R-batt   all non-battery claimants already at their minimums -> battery -2% volume
Loop limit 60.
"""
import numpy as np
from scipy.optimize import differential_evolution
import evaluators as ev
import arm2v6_problem as P
base = P.base

TGT = dict(A_board=1250, V_struct=1800, V_hap=350, V_loop=320, V_ferr=180)
MINS = dict(A_board=1197.0, V_struct=1400.0, V_hap=295.0, V_loop=180.0, V_ferr=110.0, V_ant=900.0)
ICD_PORT = 9.0
MIN_AC = 0.0          # acoustic volume reserved by a program decision (0 = none; used by arm1_pm.py)
EVALS = {"acoustic": 0}


def wall_for(port, v_struct):
    lo, hi = 1.0, 1.8
    if ev.pressure_margin(hi, port, v_struct) < 1.5:
        return None
    if ev.pressure_margin(lo, port, v_struct) >= 1.5:
        return lo
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if ev.pressure_margin(m, port, v_struct) < 1.5 else (lo, m)
    return hi


def spreader_for(t):
    return 1280.0 / (20.0 - 0.9 * (5.0 + base.R_CASE_CAL + 0.15 * t)) * 0.9 if False else \
        1280.0 / ((43.0 - 25.0) / 0.9 - 5.0 - base.R_CASE_CAL - 0.15 * t)


def acoustics_local(v_alloc, port_cap, seed):
    """Acoustics' own optimisation inside its allocation: it may use LESS than it was given
    (unused volume stays as slack - upstream claims are already locked).
    Variables: own volume, split, port, outlet height, mesh. Meet siren <= 1 W first, then maximise voice."""
    if v_alloc < 900:
        return None
    hi_v = max(v_alloc, 900.0 + 1e-6)

    def unpack(z):
        v, split, port, h, mesh = z
        free = v - base.V_DRIVER
        return v, free, split, port, h, mesh

    def obj(z):
        EVALS["acoustic"] += 1
        v, free, split, port, h, mesh = unpack(z)
        s, q, fres, vs, _, _ = base._acoustics(free * split, free * (1 - split), port, h, mesh)
        pen = max(0.0, q - 15.0)
        if s["P"] > base.P_DRIVER_MAX or pen > 0:
            return 1000.0 + s["P"] + pen
        return -vs

    res = differential_evolution(obj, [(900.0, hi_v), (0.30, 0.95), (3.0, port_cap), (0.3, 3.5), (20.0, 200.0)],
                                 popsize=10, maxiter=25, tol=1e-6, seed=seed, polish=False)
    v, free, split, port, h, mesh = unpack(res.x)
    s, q, fres, vs, _, _ = base._acoustics(free * split, free * (1 - split), port, h, mesh)
    return dict(v=v, split=split, port=port, h=h, mesh=mesh, P=s["P"], voice=vs, ok=res.fun < 1000)


def run(v_batt_spec, v_ant, seed=0, max_loops=60):
    c = dict(TGT, V_ant=v_ant)
    v_batt = float(v_batt_spec)
    port_cap = ICD_PORT
    log = []
    for loop in range(max_loops + 1):
        t = wall_for(port_cap, c["V_struct"])
        if t is None:                       # wall cannot carry this port
            port_cap -= 1.0
            t = wall_for(port_cap, c["V_struct"])
        a_spr = spreader_for(t)
        a_batt = float(np.clip(800.0, v_batt / 4.8, base.R2_POOL - a_spr))
        if v_batt / a_batt > 4.8:           # battery cannot fit its footprint -> forced shrink
            v_batt = 4.8 * a_batt
        pool = base.POOL_R1_NOM - base.DV_DT * (t - 1.2)
        upstream = (v_batt + c["A_board"] * base.T_BOARD + c["V_ant"] + a_spr * base.T_SPREADER
                    + c["V_struct"] + c["V_hap"] + c["V_loop"] + c["V_ferr"])
        remainder = pool - upstream
        v_acou = min(remainder, 2400.0)
        ac = acoustics_local(v_acou, min(port_cap, 11.0), seed) if v_acou >= max(900, MIN_AC) else None

        x = None
        if ac is not None and ac["ok"]:
            x = np.array([v_batt, a_batt, c["A_board"], c["V_struct"], t, c["V_ant"], ac["v"],
                          ac["port"], ac["split"], ac["h"], ac["mesh"], c["V_hap"], a_spr,
                          c["V_loop"], c["V_ferr"]])
            f, g, info = P.evaluate(x)
            if np.all(g <= 1e-6):
                return dict(x=x.tolist(), f=f.tolist(), loops=loop, ok=True, log=log,
                            v_batt_spec=v_batt_spec, v_ant_spec=v_ant)
        # ---- rework decision
        if ac is not None and not ac["ok"] and port_cap < 11.0 and v_acou >= 2400.0 - 1e-6:
            port_cap += 1.0; log.append("R-port")
        elif ac is not None and not ac["ok"] and port_cap < 11.0 and remainder >= 900:
            port_cap += 1.0; log.append("R-port")
        else:
            cut_any = False
            for k in ("A_board", "V_ant", "V_struct", "V_hap", "V_loop", "V_ferr"):
                tgt = TGT.get(k, v_ant)
                if c[k] > MINS[k] + 1e-9:
                    c[k] = max(MINS[k], c[k] - 0.10 * (tgt - MINS[k])); cut_any = True
            if cut_any:
                log.append("R-vol")
            else:
                v_batt *= 0.98; log.append("R-batt")
    return dict(x=None, f=None, loops=max_loops, ok=False, log=log,
                v_batt_spec=v_batt_spec, v_ant_spec=v_ant)
