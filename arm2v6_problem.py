"""
Schema v0.6 MAIN BENCHMARK CASE (user decision 2026-11-01):
  - open, water-tolerant speaker cavity (sealed waterproof driver, hydrophobic mesh >= 20 rayl)
  - 86 dB referenced at 1 m
  - larger driver: 18x13x2.5 mm class (Same Sky CMS-18132 / CMS-18138A), ~540 mm^3 envelope
  - gapped siren: duty cycle 0.5 (average electrical power = duty x tone power)
Everything else as v0.5. Driver tone power <= 1.0 W (rated 1 W).
"""
import numpy as np
import evaluators as ev
import acoustic_circuit as AC
import arm2v5_problem as base
import driver_18x13

SIREN_DUTY = 0.5
SENSITIVITY_DB_10CM = 95.0

base.configure("open", 1.0)
base.V_DRIVER = 540.0
j = base.NAMES.index("V_acou")
base.XL[j], base.XU[j], base.TARGET[j] = 900.0, 2400.0, 1460.0
NAMES, XL, XU, TARGET = base.NAMES, base.XL, base.XU, base.TARGET
CONSTRAINTS, OBJECTIVES = base.CONSTRAINTS, base.OBJECTIVES


def set_driver(sens_db=SENSITIVITY_DB_10CM):
    AC.DRIVER.clear()
    AC.DRIVER.update(driver_18x13.make(sens_db))


set_driver()


def evaluate(x):
    f, g, info = base.evaluate(x)
    p_avg = SIREN_DUTY * info["p_siren"] / base.AMP_EFF
    f = f.copy()
    f[1] = ev.battery_energy_wh(x[0]) / (p_avg + 0.15)
    info["p_siren_avg_battery"] = p_avg
    return f, g, info


# ---------------- v0.7: voice = 100 x ANSI S3.5 SII (replaces saturating f_lo/ripple proxy)
import voice_sii as _V

def _voice_sii(Vb, Vf, a, h, r_mesh, mem):
    s, _ = _V.voice_sii(Vb, Vf, a, h, r_mesh, membrane=mem)
    return s, float("nan"), float("nan")

base._voice = _voice_sii
