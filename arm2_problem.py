"""
Arm 2 - monolithic all-at-once MDO baseline.
[withheld] space-claim benchmark, evaluators from run 01 (schema v0.3 + fixes).

Every claimant's allocation and the acoustic internal trades are optimised
simultaneously by one optimiser with full knowledge of every model.
This is the 'optimal but expensive to formulate' upper bound the other arms
are measured against.
"""
import numpy as np
import evaluators as ev

# ------------------------------------------------------------ fixed parameters
POOL_R1_NOM = 11200.0   # mm^3 contested interior at 1.2 mm wall
DV_DT = 1500.0          # mm^3 interior lost per mm of extra wall (perimeter ~167 mm x ~9 mm band)
R6_POOL = 2400.0        # mm^3 peripheral non-conductive keepout
R2_POOL = 1143.0        # mm^2 caseback plane net of Qi exclusion disc
R7_POOL = 12.0          # mm^2 total through-wall opening
MIC_PORT = 1.0          # mm^2 microphone inlet share of R7
Z_AVAIL = 7.1           # mm between sensor stack top (2.8) and display (9.9)
T_BOARD = 2.24          # mm board + component stack (2800 mm^3 / 1250 mm^2, 10 layers fixed)
T_SPREADER = 0.78       # mm effective graphite + TIM thickness (250 / 320)
MASS_CAP = 60.5         # g, shipped weight - treated as an ID/comfort cap
SPL_TARGET = 86.0       # dB @ 1 m  (PENDING USER CONFIRMATION)
DRIVERS = 2             # PENDING USER CONFIRMATION
P_AMP_MAX = 1.2         # W amplifier ceiling
R_CASE_CAL = 10.82      # K/W, calibrated so the shipped spreader (320 mm^2) sits at 43.0 C
L5_F, L5_BW = 1176.45, 24.0
HAPTIC_STROKE = 0.35
MASS_FIXED = 15.9       # g display, sapphire, sensor, coil, buttons, bezel, band lugs

# ------------------------------------------------------------ design variables
VARS = [
    # name,        lo,    hi,    target (claimant's v0.3 wish)
    ("V_batt",    2600,  5200,  3786),
    ("A_batt",     520,   960,   800),
    ("A_board",    950,  1500,  1250),
    ("V_struct",  1400,  2400,  1800),
    ("t_wall",     1.0,   1.8,   1.2),
    ("V_ant",      900,  2400,  1500),
    ("V_acou",    1400,  3200,  2140),
    ("port",       3.0,  11.0,  11.0),
    ("split",     0.50,  0.95,  0.78),
    ("damp",        50,  1200,   300),
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
    "C1_volume", "C11_R6_perimeter", "C2_caseback", "Z_stack",
    "batt_t_max", "batt_t_min", "C6_mass", "O6_pressure", "O5_skin_temp",
    "siren_amp_power", "O9_siren_endurance", "O8_nfc_range", "haptic_force",
    "board_routability", "R7_port_pool",
]
OBJECTIVES = ["runtime_h", "siren_endurance_h", "voice_score", "l5_efficiency"]


def voice_score_vec(vb, port, vfc, duct, damp, band=(300.0, 4000.0)):
    """Vectorised copy of evaluators.voice_score (identical formula)."""
    fc = ev.sealed_box_fc(vb)
    f_h = ev.helmholtz_frequency(port, vfc, duct)
    q = ev.duct_q(damp)
    lo = max(fc, band[0])
    hi = max(lo * 1.5, band[1])
    f = np.geomspace(lo, hi, 96)
    resp = 1.0 + 0.5 * (ev.resonator_gain(f, f_h, q) - 1.0)
    resp_db = 20 * np.log10(np.maximum(resp, 0.25))
    ripple = resp_db.max() - resp_db.min()
    score = 100.0 - 25.0 * max(0.0, (fc - 500.0) / 500.0) - 2.0 * ripple
    return float(np.clip(score, 0.0, 100.0))


def evaluate(x):
    d = dict(zip(NAMES, x))
    t = d["t_wall"]

    # derived geometry
    v_board = d["A_board"] * T_BOARD
    v_therm = d["A_spr"] * T_SPREADER
    t_batt = d["V_batt"] / d["A_batt"]
    pool = POOL_R1_NOM - DV_DT * (t - 1.2)
    v_total = (d["V_batt"] + v_board + d["V_struct"] + d["V_ant"] + d["V_acou"]
               + d["V_hap"] + v_therm + d["V_loop"] + d["V_ferr"])

    # mass
    mass = (MASS_FIXED + 3.04e-3 * d["V_batt"]
            + 24.0 + 6.6 * (t - 1.2) + 2.7e-3 * (d["V_struct"] - 1800)
            + 5.14e-3 * d["V_hap"] + 1.12e-3 * d["V_acou"]
            + 5.0e-3 * d["V_ferr"] + 4.0 * d["A_board"] / 1250.0)

    # acoustics (nested trades are just variables here - monolithic)
    vb = d["V_acou"] * d["split"]
    vfc = d["V_acou"] * (1 - d["split"])
    p_siren = ev.siren_required_power_w(SPL_TARGET, vb, d["port"], vfc, 3.0,
                                        d["damp"], DRIVERS)
    siren_end = ev.siren_endurance_h(d["V_batt"], p_siren)
    voice = voice_score_vec(vb, d["port"], vfc, 3.0, d["damp"])

    # rf
    cond_dist = d["V_ant"] / (167.0 * 6.0)
    l5 = ev.antenna_efficiency(d["V_ant"], cond_dist, L5_F, L5_BW) * (0.8 / t) ** 0.5
    nfc = ev.nfc_read_range_mm(400.0 * d["V_loop"] / 320.0, d["V_ferr"] / 900.0, 1.0)

    # others
    runtime = ev.battery_runtime_h(d["V_batt"])
    skin = ev.skin_temp_c(d["A_spr"], 0.9, t, r_case=R_CASE_CAL)
    pmarg = ev.pressure_margin(t, d["port"], d["V_struct"])
    force = ev.haptic_force_n(d["V_hap"], HAPTIC_STROKE)
    route = ev.board_routability(d["A_board"])

    g = np.array([
        (v_total - pool) / 1000.0,
        (d["V_ant"] + d["V_loop"] - R6_POOL) / 100.0,
        (d["A_batt"] + d["A_spr"] - R2_POOL) / 100.0,
        (t_batt + T_BOARD - Z_AVAIL),
        (t_batt - 4.8),
        (2.6 - t_batt),
        (mass - MASS_CAP),
        (1.5 - pmarg),
        (skin - 43.0),
        (p_siren - P_AMP_MAX),
        (3.0 - siren_end),
        (10.0 - nfc) / 10.0,
        (0.9 - force),
        (0.95 - route),
        (d["port"] + MIC_PORT - R7_POOL),
    ])
    f = np.array([runtime, siren_end, voice, l5])
    info = dict(v_total=v_total, pool=pool, mass=mass, p_siren=p_siren,
                pmarg=pmarg, skin=skin, nfc=nfc, force=force, route=route,
                t_batt=t_batt)
    return f, g, info
