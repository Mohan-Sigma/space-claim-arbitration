"""
Voice metric: ANSI S3.5-1997 Speech Intelligibility Index (third-octave procedure),
via the MOSQITO implementation (mosqito 1.2.1, _main_sii).

Use case: far-end speech replayed through the watch speaker in speakerphone use.
  - talker: ANSI standard speech spectrum, normal vocal effort (shape only)
  - drive: speech-shaped electrical signal, 0.1 W rms total (~10 dB crest -> 1 W peaks)
  - listener at 0.3 m (wrist raised toward the face)
  - ambient: speech-shaped noise, 55 dB SPL overall (moderate office / street)
  - normal hearing (threshold = 0)
Voice score = 100 x SII (0..100). Replaces the f_lo/ripple proxy, which saturated
whenever the low-frequency corner exceeded 1 kHz.
"""
import numpy as np
from mosqito.sq_metrics.speech_intelligibility.sii_ansi._main_sii import _main_sii
from mosqito.sq_metrics.speech_intelligibility.sii_ansi._band_procedure_data import (
    _get_third_octave_band_data)
import acoustic_circuit as AC

FC, F_LO, F_HI, _, _, _, E_SPEECH = _get_third_octave_band_data()
FC = FC.astype(float)
BW = (F_HI - F_LO).astype(float)
P_VOICE = 0.1
R_LISTEN = 0.3
NOISE_OVERALL = 55.0
SPEECH_OVERALL = 62.35            # ANSI normal effort overall level
NOISE_SPEC = E_SPEECH + (NOISE_OVERALL - SPEECH_OVERALL)

_band_w = 10 ** (E_SPEECH / 10) * BW
P_BAND = P_VOICE * _band_w / _band_w.sum()     # electrical power per band (W)


def sii_from_sensitivity(sens_db_per_w):
    """sens_db_per_w: SPL at R_LISTEN for 1 W electrical, at each band centre."""
    band_spl = 10 * np.log10(P_BAND) + sens_db_per_w
    speech_spec = band_spl - 10 * np.log10(BW)
    sii, specific, _ = _main_sii("third_octave", speech_spec, NOISE_SPEC.copy(), None)
    return float(sii), speech_spec


def voice_sii(Vb, Vf, area, h, r_mesh, membrane=False, drv=None):
    kw = {} if drv is None else dict(drv=drv)
    r = AC.speaker_response(FC, Vb, Vf, area, h, r_mesh, membrane=membrane,
                            r_m=R_LISTEN, **kw)
    sens = r["spl"] - 10 * np.log10(np.maximum(r["P"], 1e-12))
    sii, _ = sii_from_sensitivity(sens)
    return 100.0 * sii, sens
