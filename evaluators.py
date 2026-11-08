"""
Space-claim arbitration benchmark - evaluator layer.
[withheld] reference platform, schema v0.3.

Every evaluator is closed-form or a small lumped model. Design target is
microseconds per call so that thousands of negotiation rounds are tractable
and the benchmark is reproducible without a commercial solver licence.
"""
import numpy as np

# ---------------------------------------------------------------- constants
RHO0 = 1.2          # kg/m^3, air density
C_AIR = 343.0       # m/s
P_REF = 2e-5        # Pa, SPL reference

# ---------------------------------------------------------------- battery
def battery_energy_wh(volume_mm3, rho_v_wh_per_l=600.0):
    """Volumetric energy density model. 600 Wh/L for a curved watch pouch."""
    return rho_v_wh_per_l * volume_mm3 / 1e6


def battery_runtime_h(volume_mm3, p_avg_w=0.038, rho_v=600.0):
    return battery_energy_wh(volume_mm3, rho_v) / p_avg_w


# ---------------------------------------------------------------- acoustics
def sealed_box_fc(back_volume_mm3, fs_hz=850.0, vas_mm3=1800.0):
    """Sealed-box corner frequency. Larger back volume -> lower fc -> better voice."""
    return fs_hz * np.sqrt(1.0 + vas_mm3 / max(back_volume_mm3, 1.0))


def helmholtz_frequency(port_area_mm2, front_cavity_mm3, duct_length_mm):
    """
    Helmholtz resonance of the outlet duct + front cavity.
    Note the direction of the port-area dependence: f_H rises with area.
    Tuning DOWN to the siren band therefore requires a SMALLER port,
    which fights the raw-SPL preference for a large one.
    """
    a_p = max(port_area_mm2, 1e-6) * 1e-6          # m^2
    v_fc = max(front_cavity_mm3, 1.0) * 1e-9       # m^3
    radius = np.sqrt(a_p / np.pi)
    l_eff = duct_length_mm * 1e-3 + 2 * 0.85 * radius
    return (C_AIR / (2 * np.pi)) * np.sqrt(a_p / (v_fc * l_eff))


def duct_q(damping_rayl):
    """
    Mesh acoustic resistance -> resonator Q. High damping kills the peak.
    Calibrated to realistic damped-duct values (Q ~ 0.8 to 3), not the
    Q ~ 7 an undamped textbook resonator would give.
    """
    return 120.0 / max(damping_rayl, 1.0) + 0.7


def resonator_gain(f_hz, f_h, q):
    """Second-order resonance magnitude. Peak gain = q at f = f_h."""
    r = f_hz / max(f_h, 1.0)
    return 1.0 / np.sqrt((1.0 - r**2) ** 2 + (r / max(q, 1e-3)) ** 2)


def acoustic_efficiency(back_volume_mm3, eta_base=0.003, vb_ref_mm3=1000.0):
    """
    Baseline electroacoustic efficiency, mildly improved by back volume
    (less air-spring stiffness loading the diaphragm).
    """
    return eta_base * (back_volume_mm3 / vb_ref_mm3) ** 0.15


def spl_from_acoustic_power(p_ac_w, r_m=1.0):
    """Half-space radiation."""
    area = 2 * np.pi * r_m**2
    intensity = max(p_ac_w, 1e-15) / area
    p_rms = np.sqrt(intensity * RHO0 * C_AIR)
    return 20 * np.log10(p_rms / P_REF)


def acoustic_power_for_spl(spl_db, r_m=1.0):
    """Inverse of the above - how much acoustic watt a target SPL demands."""
    p_rms = P_REF * 10 ** (spl_db / 20.0)
    intensity = p_rms**2 / (RHO0 * C_AIR)
    return intensity * 2 * np.pi * r_m**2


def siren_effective_efficiency(back_volume_mm3, port_area_mm2, front_cavity_mm3,
                               duct_length_mm, damping_rayl, driver_count=2,
                               f_siren=3200.0, driver_separation_mm=8.0):
    """
    Effective electroacoustic efficiency in the siren band for a given design.

    Two competing effects on port area, which is the source of the bimodality:
      - a SMALL port tunes f_H down into the siren band -> resonant gain
      - a SMALL port also chokes radiation -> throttling loss
    """
    f_h = helmholtz_frequency(port_area_mm2, front_cavity_mm3, duct_length_mm)
    q = duct_q(damping_rayl)
    gain = resonator_gain(f_siren, f_h, q)          # amplitude gain

    port_factor = min(1.0, port_area_mm2 / 9.0) ** 0.5   # radiation throttling

    wavelength_mm = C_AIR / f_siren * 1000.0
    coherence = np.cos(np.pi * driver_separation_mm / wavelength_mm) ** 2
    n_gain = 1.0 + (driver_count - 1) * (1.0 + coherence)

    eta = acoustic_efficiency(back_volume_mm3) * gain**2 * port_factor * n_gain
    return float(eta)


def siren_required_power_w(spl_target_db, back_volume_mm3, port_area_mm2,
                           front_cavity_mm3, duct_length_mm, damping_rayl,
                           driver_count=2, amp_efficiency=0.7):
    """
    Electrical power needed to hit a target SPL with this design.
    Causally the right direction: the SPL is a requirement, the design
    determines what it costs to meet it.
    """
    p_ac = acoustic_power_for_spl(spl_target_db, r_m=1.0)
    eta = siren_effective_efficiency(back_volume_mm3, port_area_mm2,
                                     front_cavity_mm3, duct_length_mm,
                                     damping_rayl, driver_count)
    return float(p_ac / max(eta, 1e-9) / amp_efficiency)


def siren_max_spl(back_volume_mm3, port_area_mm2, front_cavity_mm3,
                  duct_length_mm, damping_rayl, driver_count=2,
                  p_amp_max_w=1.2, amp_efficiency=0.7):
    """Ceiling SPL given the amplifier's power limit."""
    eta = siren_effective_efficiency(back_volume_mm3, port_area_mm2,
                                     front_cavity_mm3, duct_length_mm,
                                     damping_rayl, driver_count)
    return spl_from_acoustic_power(p_amp_max_w * amp_efficiency * eta, r_m=1.0)


def voice_score(back_volume_mm3, port_area_mm2, front_cavity_mm3,
                duct_length_mm, damping_rayl, band=(300.0, 4000.0)):
    """
    Intelligibility proxy: penalise a high corner frequency and penalise
    in-band ripple caused by the Helmholtz peak. 100 is best.
    """
    fc = sealed_box_fc(back_volume_mm3)
    f_h = helmholtz_frequency(port_area_mm2, front_cavity_mm3, duct_length_mm)
    q = duct_q(damping_rayl)

    # Response is direct radiation PLUS a port contribution, not port alone.
    # Ripple is measured over the speech-critical band and above fc, since
    # the low-end rolloff is already penalised through fc.
    lo = max(fc, band[0])
    hi = max(lo * 1.5, band[1])
    freqs = np.geomspace(lo, hi, 96)
    port_coupling = 0.5
    resp = np.array([1.0 + port_coupling * (resonator_gain(f, f_h, q) - 1.0)
                     for f in freqs])
    resp = np.maximum(resp, 0.25)
    resp_db = 20 * np.log10(resp)
    ripple_db = resp_db.max() - resp_db.min()

    score = 100.0 - 25.0 * max(0.0, (fc - 500.0) / 500.0) - 2.0 * ripple_db
    return float(np.clip(score, 0.0, 100.0))


def siren_endurance_h(battery_volume_mm3, p_elec_w,
                      p_system_w=0.15, rho_v=600.0):
    """
    THE key use-case coupling (C13). Takes the SAME electrical power that
    siren_required_power_w computed, so SPL and endurance stay consistent.
    Louder siren drains the beacon faster, and the volume that buys
    loudness is taken from the cell that powers it.
    """
    energy = battery_energy_wh(battery_volume_mm3, rho_v)
    return float(energy / (p_elec_w + p_system_w))


# ---------------------------------------------------------------- antenna
def chu_q_min(clearance_volume_mm3, f_mhz):
    """
    Chu-Harrington fundamental bound. Gives a physics-grounded floor on the
    volume the RF claimant needs - the arbiter cannot negotiate below it.
    """
    a_m = (3.0 * clearance_volume_mm3 * 1e-9 / (4 * np.pi)) ** (1.0 / 3.0)
    k = 2 * np.pi * f_mhz * 1e6 / 3e8      # free-space wavenumber
    ka = k * a_m
    return 1.0 / ka**3 + 1.0 / ka


def antenna_efficiency(clearance_volume_mm3, conductor_distance_mm, f_mhz,
                       required_bw_mhz):
    q_min = chu_q_min(clearance_volume_mm3, f_mhz)
    bw_max = f_mhz / q_min
    bw_ratio = np.clip(bw_max / max(required_bw_mhz, 1e-3), 0.0, 1.0)
    prox = np.clip(conductor_distance_mm / 1.5, 0.0, 1.0)
    return float(bw_ratio * prox)


# ---------------------------------------------------------------- nfc
def nfc_read_range_mm(loop_area_mm2, ferrite_thickness_mm, distance_to_ti_mm,
                      turns=4, mu_r=120.0):
    """Ferrite restores the flux path that titanium eddy currents destroy."""
    shielding = mu_r * ferrite_thickness_mm
    eddy_loss = 1.0 / (1.0 + shielding / 12.0) * np.exp(-distance_to_ti_mm / 2.0)
    q_eff = 1.0 - 0.85 * eddy_loss
    coupling = np.sqrt(loop_area_mm2 / 400.0) * (turns / 4.0) ** 0.5
    return float(22.0 * coupling * q_eff)


# ---------------------------------------------------------------- haptics
def haptic_force_n(volume_mm3, stroke_mm, f_r_hz=170.0,
                   rho_g_cm3=17.0, mass_fraction=0.45):
    mass_kg = rho_g_cm3 * 1e3 * (volume_mm3 * mass_fraction * 1e-9)
    omega = 2 * np.pi * f_r_hz
    return float(mass_kg * omega**2 * stroke_mm * 1e-3)


# ---------------------------------------------------------------- thermal
def skin_temp_c(spreader_area_mm2, power_w, wall_thickness_mm,
                t_amb=25.0, r_jc=5.0, r_case=12.0):
    r_spread = 1280.0 / max(spreader_area_mm2, 1.0)
    r_wall = 0.15 * wall_thickness_mm
    return t_amb + power_w * (r_jc + r_spread + r_case + r_wall)


# ---------------------------------------------------------------- structure
def pressure_margin(wall_thickness_mm, port_area_mm2, rib_volume_mm3):
    """Ports are stress raisers. Margin must stay >= 1.5 for 10ATM."""
    # calibrated so that the schema's structure.max_mm2 = 8.0 gives exactly
    # the 1.5x margin the 10ATM requirement demands, at nominal wall and ribs
    base = 1.55 * (wall_thickness_mm / 1.2) ** 1.5
    port_penalty = 0.0625 * port_area_mm2
    rib_bonus = 0.15 * (rib_volume_mm3 / 1800.0)
    return float(base + rib_bonus - port_penalty + 0.30)


def board_routability(area_mm2, pin_count=1400, layers=10):
    area_per_pin = area_mm2 / pin_count
    return float(np.clip(area_per_pin / 0.9 * (layers / 10.0), 0.0, 1.5))
