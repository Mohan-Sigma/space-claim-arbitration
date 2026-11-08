"""18x13x2.5 mm driver class (Same Sky CMS-18132-18SP / CMS-18138A-SP).
fs 500 Hz free / 900-1000 Hz in 1 cc, Qms ~2.4, rated 1 W, 95 dB @ 1 W / 10 cm typ (92-98).
Cms from the box-shift ratio, Mms from fs, Rms from Qms, Bl fitted to the sensitivity spec."""
import numpy as np
import acoustic_circuit as AC

def make(target_db_10cm=95.0):
    Sd = 157e-6
    Cmb = 1e-6 / (AC.RHOC2 * Sd ** 2)
    Cms = np.mean([3.0 * Cmb, 2.24 * Cmb])
    ws = 2 * np.pi * 500
    Mms = 1 / (ws ** 2 * Cms)
    d = dict(Re=7.2, Le=70e-6, Bl=0.73, Mms=Mms, Rms=ws * Mms / 2.4, Cms=Cms, Sd=Sd)
    for _ in range(30):
        r = AC.speaker_response([2000.0], 1e-6, 1e-12, Sd, 10e-3, 0, length_m=1e-6,
                                membrane=False, drv=d)
        s = r["spl"][0] - 10 * np.log10(r["P"][0]) + 20
        d["Bl"] *= 10 ** ((target_db_10cm - s) / 40)
    return d

if __name__ == "__main__":
    for tgt in (92, 95, 98):
        d = make(tgt); ws = 2 * np.pi * 500
        print(f"target {tgt} dB: Bl {d['Bl']:.3f} T.m, Qes {ws*d['Mms']*d['Re']/d['Bl']**2:.2f}, "
              f"Mms {d['Mms']*1e6:.0f} mg, Rms {d['Rms']:.3f}")
