"""
Physical lumped-element model of the watch speaker path.
Replaces the uncalibrated damping index (Q = 120/d + 0.7).

Topology (acoustic impedance analogy, rms quantities):

  V --Re+jwLe--[Bl gyrator]-- driver (Mms, Rms, Cms)/Sd^2 -- back cavity C_b
                                   |
                              front cavity C_f (with thermal loss)  ||  outlet branch:
                                   R_duct(LRF) + R_rad + R_mesh + R_nl + R_m
                                   + jw(M_port + M_m) + 1/(jw C_m)
  far field (half space): |p| = rho * w * |U_out| / (2 pi r)

Sources for parameters: Same Sky CMS-15113/15118 datasheets, TI SLAA857 (iPhone 7
speaker), Ole Wolff WP03; Gore acoustic design guide (GAW346 IL 3.5 dB @ 1 kHz,
2.0 mm ID, 33 um); loss formulas per Stinson/Kampinga wide-duct limit.
Membrane Q_mem and the nonlinear orifice coefficient are engineering estimates.
"""
import numpy as np

RHO, C0 = 1.2, 343.0
NU = 1.5e-5                 # m^2/s
PR = 0.71
GAMMA = 1.4
RHOC2 = RHO * C0**2

# ---------------- driver: 15x11 mm class (nominal from datasheet ranges)
DRIVER = dict(Re=7.4, Le=60e-6, Bl=0.86, Mms=0.098e-3, Rms=0.28,
              Cms=5.2e-4, Sd=110e-6)      # fs free air ~ 705 Hz

# ---------------- membrane: GAW346-class
IL_1K_DB = 3.5              # Gore design guide, 2.0 mm ID
V_RIG = 2.0e-9              # m^3, Gore IL rig volume (assumed 1-3 mm^3)
RHO_S = 0.020               # kg/m^2 areal density (10-30 g/m^2)
Q_MEM = 10.0                # membrane loss Q (5-20, estimate)
ID_REF = 2.0e-3

C_NL = 1.0                  # Ingard-type nonlinear orifice coefficient (0.5-1.5)
H_CAV = 1.0e-3              # front cavity effective height for thermal loss


def _membrane_cm0():
    """Solve C_m at 2.0 mm ID so the Gore rig shows IL(1 kHz) = 3.5 dB,
    given M_m and R_m (R_m depends on C_m through the membrane resonance)."""
    A = np.pi * (ID_REF / 2) ** 2
    Mm = (4 / 3) * RHO_S / A
    Crig = V_RIG / RHOC2
    w = 2 * np.pi * 1000.0
    target = 10 ** (IL_1K_DB / 20)
    lo, hi = 1e-17, 1e-11
    for _ in range(200):
        Cm = np.sqrt(lo * hi)
        Rm = np.sqrt(Mm / Cm) / Q_MEM
        Zm = Rm + 1j * w * Mm + 1 / (1j * w * Cm)
        Zc = 1 / (1j * w * Crig)
        il = abs((Zc + Zm) / Zc)
        if il > target:      # too stiff -> raise compliance
            lo = Cm
        else:
            hi = Cm
    return Cm


CM0 = _membrane_cm0()


def membrane_rmc(area_m2):
    """C_m ~ a^4, M_m ~ 1/a^2, R_m from Q_mem at the membrane's own resonance."""
    a_ratio = np.sqrt(area_m2 / np.pi) / (ID_REF / 2)
    Cm = CM0 * a_ratio ** 4
    Mm = (4 / 3) * RHO_S / area_m2
    Rm = np.sqrt(Mm / Cm) / Q_MEM
    return Rm, Mm, Cm


def gore_rig_il(f, id_m=ID_REF):
    """Insertion loss the Gore MEMS-mic rig would report for this membrane model."""
    A = np.pi * (id_m / 2) ** 2
    Rm, Mm, Cm = membrane_rmc(A)
    w = 2 * np.pi * np.asarray(f, float)
    Zm = Rm + 1j * w * Mm + 1 / (1j * w * Cm)
    Zc = 1 / (1j * w * (V_RIG / RHOC2))
    return 20 * np.log10(np.abs((Zc + Zm) / Zc))


def outlet_geometry(area_m2, h_m, length_m):
    """Rectangular outlet w x h (h <= w); returns inertance, viscous R per unit omega-delta."""
    h = min(h_m, np.sqrt(area_m2))
    w = area_m2 / h
    perim = 2 * (w + h)
    dh = 2 * w * h / (w + h)
    l_eff = length_m + 0.85 * dh                  # Jiang et al. 2023 end correction
    l_r = length_m + 0.5 * dh                     # viscous end allowance
    return dict(h=h, w=w, perim=perim, dh=dh, l_eff=l_eff, l_r=l_r)


def speaker_response(f, Vb_m3, Vf_m3, area_m2, h_m, r_mesh, length_m=3.0e-3,
                     volts=1.0, u_out_nl=None, drv=DRIVER, membrane=True, r_m=1.0):
    """Complex quantities at frequencies f for drive `volts` rms.
    u_out_nl: rms volume velocity used to set the nonlinear orifice resistance."""
    f = np.atleast_1d(np.asarray(f, float))
    w = 2 * np.pi * f
    d = drv
    Ze = d["Re"] + 1j * w * d["Le"]
    Sd = d["Sd"]
    Zsrc = d["Bl"] ** 2 / (Ze * Sd ** 2)
    Zad = (d["Rms"] + 1j * w * d["Mms"] + 1 / (1j * w * d["Cms"])) / Sd ** 2
    Zb = 1 / (1j * w * Vb_m3 / RHOC2)
    ps = volts * d["Bl"] / (Ze * Sd)

    g = outlet_geometry(area_m2, h_m, length_m)
    delta = np.sqrt(2 * NU / w)
    R_duct = RHO * w * delta * g["l_r"] * g["perim"] / (2 * area_m2 ** 2)
    M_port = RHO * g["l_eff"] / area_m2
    R_rad = RHO * w ** 2 / (2 * np.pi * C0)
    R_mesh = r_mesh / area_m2
    R_nl = 0.0 if u_out_nl is None else C_NL * RHO * u_out_nl / area_m2 ** 2
    if membrane:
        Rm, Mm, Cm = membrane_rmc(area_m2)
        Zmem = Rm + 1j * w * Mm + 1 / (1j * w * Cm)
    else:
        Zmem = 0.0
    Zbr = R_duct + R_rad + R_mesh + R_nl + 1j * w * M_port + Zmem

    Cf = Vf_m3 / RHOC2
    q_th = H_CAV / ((GAMMA - 1) * (delta / np.sqrt(PR)))
    Yf = 1j * w * Cf + w * Cf / q_th
    Zfront = 1 / (Yf + 1 / Zbr)

    Ud = ps / (Zsrc + Zad + Zb + Zfront)
    Uout = Ud * Zfront / Zbr
    p = RHO * w * np.abs(Uout) / (2 * np.pi * r_m)
    u = Ud / Sd
    I = (volts - d["Bl"] * u) / Ze
    P = np.real(volts * np.conj(I))
    x = np.abs(u) / w
    return dict(f=f, spl=20 * np.log10(np.maximum(p, 1e-12) / 2e-5), Uout=Uout,
                P=P, x=x, part_vel=np.abs(Uout) / area_m2)


def resonance_q(Vb, Vf, area, h, r_mesh):
    """Small-signal Q of the port resonance from the -3 dB bandwidth of |U_out/U_d|-like SPL peak."""
    f = np.geomspace(1500, 14000, 160)
    r = speaker_response(f, Vb, Vf, area, h, r_mesh)
    s = r["spl"]
    # remove the mass-controlled driver slope (+0 dB/oct in SPL above fs) by referencing the peak
    i = int(np.argmax(s))
    if i == 0 or i == len(f) - 1:
        return 0.5, f[i]
    pk = s[i] - 3.0
    lo = np.interp(pk, s[:i + 1], f[:i + 1]) if s[:i + 1].min() < pk else np.nan
    hi_side = s[i:][::-1]; f_hi = f[i:][::-1]
    hi = np.interp(pk, hi_side, f_hi) if s[i:].min() < pk else np.nan
    if np.isnan(lo) or np.isnan(hi):
        return 0.5, f[i]           # no defined peak -> heavily damped
    return float(f[i] / (hi - lo)), float(f[i])


def siren(Vb, Vf, area, h, r_mesh, spl_target=86.0, r_m=1.0, band=(3000, 3400)):
    """Electrical power (W) to reach the target at the best tone in the band,
    with the nonlinear orifice resistance iterated at the operating level."""
    f = np.linspace(band[0], band[1], 5)
    unl = None
    for _ in range(6):
        r = speaker_response(f, Vb, Vf, area, h, r_mesh, volts=1.0, u_out_nl=unl, r_m=r_m)
        k = int(np.argmax(r["spl"] - 10 * np.log10(np.maximum(r["P"], 1e-12))))
        scale = 10 ** ((spl_target - r["spl"][k]) / 20)       # linear in volts
        unl_new = np.abs(r["Uout"][k]) * scale
        if unl is not None and abs(unl_new - unl) < 1e-3 * unl_new:
            break
        unl = unl_new if unl is None else 0.5 * (unl + unl_new)
    P = r["P"][k] * scale ** 2
    return dict(P=float(P), f=float(f[k]), volts=float(scale), x=float(r["x"][k] * scale),
                part_vel=float(r["part_vel"][k] * scale))


def voice(Vb, Vf, area, h, r_mesh):
    """Voice score from the small-signal response (same scoring rule as before):
    100 - 25*max(0,(f_lo-500)/500) - 2*ripple(300..4000 above f_lo)."""
    f = np.geomspace(150, 8000, 72)
    s = speaker_response(f, Vb, Vf, area, h, r_mesh)["spl"]
    ref = np.mean(s[(f >= 1000) & (f <= 2000)])
    below = np.where(s[f < 1000] < ref - 6)[0]
    f_lo = f[below[-1]] if len(below) else f[0]
    band = (f >= max(f_lo, 300)) & (f <= 4000)
    ripple = s[band].max() - s[band].min() if band.sum() > 1 else 40.0
    score = 100 - 25 * max(0.0, (f_lo - 500) / 500) - 2 * ripple
    return float(np.clip(score, 0, 100)), float(f_lo), float(ripple)
