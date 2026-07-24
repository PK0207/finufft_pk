"""Correctness check against a pre-computed abacus solution (see data/generate_reference.py).

Deliberately simple: one stored fixture, one comparison. No live abacusutils calls here.
"""

import numpy as np
import pytest

from finufft_pk.power_spectrum import FinufftPowerSpectrum


@pytest.mark.xfail(
    reason=(
        "Known issue: the half-plane phase-shift trick in FinufftPowerSpectrum "
        "(power_spectrum.py, shift = nmesh_half // 2, ported verbatim from "
        "fft_benchmark/finufft_halfplane.ipynb) misaligns which array index corresponds to "
        "kz=0. Power matches abacus to ~1% for k above ~0.3 h/Mpc but is off by 0.4x-1.5x in "
        "the first several low-k bins on the stored reference catalog -- consistent with a "
        "kz=0 indexing bug rather than sample variance or Nyquist-edge systematics. Needs the "
        "shift constant re-derived against the actual FINUFFT modeord convention."
    ),
    strict=True,
)
def test_matches_stored_abacus_power(abacus_reference):
    meta = abacus_reference["meta"]
    pos = abacus_reference["pos"]

    fps = FinufftPowerSpectrum(Lbox=meta["L"], nmesh=meta["ngen"])
    fps.set_positions(pos)
    P_finufft = fps.compute_power()

    k = np.asarray(P_finufft["k_mid"])
    pk_finufft = np.asarray(P_finufft["power"])
    pk_abacus = abacus_reference["power"]

    assert np.allclose(k, abacus_reference["k_mid"])

    # Skip the DC bin and the last couple of bins near Nyquist, where FINUFFT (no aliasing)
    # and TSC-compensated (aliasing-corrected but imperfectly) are expected to diverge a bit.
    well_sampled = slice(1, -2)
    rel_diff = np.abs(pk_finufft[well_sampled] - pk_abacus[well_sampled]) / pk_abacus[well_sampled]
    assert np.nanmax(rel_diff) < 0.05
