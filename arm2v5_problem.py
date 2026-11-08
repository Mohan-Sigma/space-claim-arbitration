"""
Arm 2, schema v0.5: physical acoustic circuit replaces the damping index.

Scenarios (set via configure()):
  arch = "membrane": GAW346-class ePTFE membrane over the speaker port (vent area = port area)
  arch = "open":     sealed waterproof driver, open water-tolerant front cavity, hydrophobic mesh >= 20 rayl
  r_ref = 1.0 or 0.1 m   (siren SPL reference distance; recommendation 3 runs both)

Siren endurance is an objective only (no >= 3 h floor) so the front shows what is achievable.
"""
import numpy as np
import evaluators as ev
import acoustic_circuit as AC
from arm2_problem import (POOL_R1_NOM, DV_DT, R6_POOL, R2_POOL, R7_POOL, MIC_PORT,
                          Z_AVAIL, T_BOARD, T_SPREADER, R_CASE_CAL, L5_F, L5_BW,
                          HAPTIC_STROKE, MASS_FIXED)

V_DRIVER = 450.0
P_DRIVER_MAX = 1.0          # W, datasheet max for 15x11 class (recommendation 3)
AMP_EFF = 0.7
CFG = dict(arch="membrane", r_ref=1.0)

VARS = [
    ("V_batt",    2600,  5200,  3786),
    ("A_batt",     520,   960,   800),
    ("A_board",    950,  1500,  1250),
    ("V_struct",  1400,  2400,  1800),
    ("t_wall",     1.0,   1.8,   1.2),
    ("V_ant",      900,  2400,  1500),
    ("V_acou",     850,  2200,  1370),
    ("port",       3.0,  11.0,   9.0),
    ("split",     0.30,  0.95,  0.78),
    ("h_out",      0.3,   3.5,   1.0),    # outlet slit height (mm); >= sqrt(port) means square/round
    ("r_mesh",     0.0,   200,    40),    # specific resistance of any mesh in the port (MKS rayl)
    ("V_hap",      200,   620,   350),
    ("A_spr",      180,   520,   320),
    ("V_loop",     180,   520,   320),
    ("V_ferr",     110,   300,   180),
]
NAMES = [v[0] for v in VARS]
XL = np.array([v[1] for v in VARS], float)
XU = np.array([v[2] for v in VARS], float)
TARGET = np.array([v[3] for v in VARS], float)

CONSTRAINTS = [
    "C1_volume", "C11_R6_perimeter", "C2_caseback", "Z_stack", "batt_t_max", "batt_t_min",
    "O6_pressure", "O5_skin_temp", "siren_driver_power_1W", "O8_nfc_range", "haptic_force",
    "board_routability", "R7_port_pool", "acoustic_free_volume", "Q_guard",
]
OBJECTIVES = ["runtime_h", "siren_endurance_h", "voice_score", "l5_efficiency"]


def configure(arch, r_ref):
    CFG.update(arch=arch, r_ref=r_ref)
    j = NAMES.index("r_mesh")
    XL[j] = 0.0 if arch == "membrane" else 20.0


def _acoustics(vb, vfc, port_mm2, h_mm, r_mesh):
    mem = CFG["arch"] == "membrane"
    a, Vb, Vf, h = port_mm2 * 1e-6, vb * 1e-9, vfc * 1e-9, h_mm * 1e-3
    s = _siren(Vb, Vf, a, h, r_mesh, mem)
    q, fres = _q(Vb, Vf, a, h, r_mesh, mem)
    v, flo, rip = _voice(Vb, Vf, a, h, r_mesh, mem)
    return s, q, fres, v, flo, rip


def _siren(Vb, Vf, a, h, r_mesh, mem):
    f = np.linspace(3000, 3400, 5)
    unl = None
    for _ in range(6):
        r = AC.speaker_response(f, Vb, Vf, a, h, r_mesh, volts=1.0, u_out_nl=unl,
                                membrane=mem, r_m=CFG["r_ref"])
        k = int(np.argmax(r["spl"] - 10 * np.log10(np.maximum(r["P"], 1e-12))))
        sc = 10 ** ((86.0 - r["spl"][k]) / 20)
        un = abs(r["Uout"][k]) * sc
        if unl is not None and abs(un - unl) < 2e-3 * un:
            break
        unl = un if unl is None else 0.5 * (unl + un)
    return dict(P=float(r["P"][k] * sc ** 2), f=float(f[k]), x=float(r["x"][k] * sc),
                v_port=float(r["part_vel"][k] * sc))


def _q(Vb, Vf, a, h, r_mesh, mem):
    f = np.geomspace(1500, 14000, 120)
    s = AC.speaker_response(f, Vb, Vf, a, h, r_mesh, membrane=mem)["spl"]
    i = int(np.argmax(s))
    if i in (0, len(f) - 1):
        return 0.5, float(f[i])
    pk = s[i] - 3.0
    if s[:i + 1].min() >= pk or s[i:].min() >= pk:
        return 0.5, float(f[i])
    lo = np.interp(pk, s[:i + 1], f[:i + 1])
    hi = np.interp(pk, s[i:][::-1], f[i:][::-1])
    return float(f[i] / (hi - lo)), float(f[i])


def _voice(Vb, Vf, a, h, r_mesh, mem):
    f = np.geomspace(150, 8000, 60)
    s = AC.speaker_response(f, Vb, Vf, a, h, r_mesh, membrane=mem)["spl"]
    ref = np.mean(s[(f >= 1000) & (f <= 2000)])
    below = np.where(s[f < 1000] < ref - 6)[0]
    f_lo = f[below[-1]] if len(below) else f[0]
    band = (f >= max(f_lo, 300)) & (f <= 4000)
    rip = s[band].max() - s[band].min() if band.sum() > 1 else 40.0
    return float(np.clip(100 - 25 * max(0.0, (f_lo - 500) / 500) - 2 * rip, 0, 100)), float(f_lo), float(rip)


def evaluate(x):
    d = dict(zip(NAMES, x))
    t = d["t_wall"]
    v_board, v_therm = d["A_board"] * T_BOARD, d["A_spr"] * T_SPREADER
    t_batt = d["V_batt"] / d["A_batt"]
    pool = POOL_R1_NOM - DV_DT * (t - 1.2)
    v_total = (d["V_batt"] + v_board + d["V_struct"] + d["V_ant"] + d["V_acou"]
               + d["V_hap"] + v_therm + d["V_loop"] + d["V_ferr"])
    mass = (MASS_FIXED + 3.04e-3 * d["V_batt"] + 24.0 + 6.6 * (t - 1.2)
            + 2.7e-3 * (d["V_struct"] - 1800) + 5.14e-3 * d["V_hap"] + 1.12e-3 * d["V_acou"]
            + 5.0e-3 * d["V_ferr"] + 4.0 * d["A_board"] / 1250.0)

    free = max(d["V_acou"] - V_DRIVER, 1.0)
    vb, vfc = free * d["split"], free * (1 - d["split"])
    s, q, fres, voice, flo, rip = _acoustics(vb, vfc, d["port"], d["h_out"], d["r_mesh"])
    siren_end = ev.siren_endurance_h(d["V_batt"], s["P"] / AMP_EFF)
    q_cap = 8.0 if CFG["arch"] == "membrane" else 15.0

    cond_dist = d["V_ant"] / (167.0 * 6.0)
    l5 = ev.antenna_efficiency(d["V_ant"], cond_dist, L5_F, L5_BW) * (0.8 / t) ** 0.5
    nfc = ev.nfc_read_range_mm(400.0 * d["V_loop"] / 320.0, d["V_ferr"] / 900.0, 1.0)
    runtime = ev.battery_runtime_h(d["V_batt"])
    skin = ev.skin_temp_c(d["A_spr"], 0.9, t, r_case=R_CASE_CAL)
    pmarg = ev.pressure_margin(t, d["port"], d["V_struct"])
    force = ev.haptic_force_n(d["V_hap"], HAPTIC_STROKE)
    route = ev.board_routability(d["A_board"])

    g = np.array([
        (v_total - pool) / 1000.0, (d["V_ant"] + d["V_loop"] - R6_POOL) / 100.0,
        (d["A_batt"] + d["A_spr"] - R2_POOL) / 100.0, (t_batt + T_BOARD - Z_AVAIL),
        (t_batt - 4.8), (2.6 - t_batt), (1.5 - pmarg), (skin - 43.0),
        np.log10(max(s["P"], 1e-9) / P_DRIVER_MAX),   # log-scaled: P spans decades
        (10.0 - nfc) / 10.0, (0.9 - force), (0.95 - route),
        (d["port"] + MIC_PORT - R7_POOL), (300.0 - free) / 100.0, (q - q_cap) / q_cap,
    ])
    f = np.array([runtime, siren_end, voice, l5])
    info = dict(mass=mass, p_siren=s["P"], f_siren=s["f"], x_um=s["x"] * 1e6, v_port=s["v_port"],
                q_ss=q, f_res=fres, f_lo=flo, ripple=rip, pmarg=pmarg, vb=vb, vfc=vfc)
    return f, g, info
