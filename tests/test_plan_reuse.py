import numpy as np

from finufft_pk.power_spectrum import FinufftPowerSpectrum


def test_reexecute_with_new_weights_without_resetting_positions(rng):
    """compute_field(weights) with new weights, same positions, should equal a fresh
    FinufftPowerSpectrum computed with those weights directly (sanity check that weight-only
    reuse doesn't need set_positions() again)."""
    Lbox, nmesh, N = 64.0, 16, 2000
    pos = rng.uniform(0, Lbox, size=(N, 3)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=N).astype(np.float32)

    fps = FinufftPowerSpectrum(Lbox=Lbox, nmesh=nmesh)
    fps.set_positions(pos)
    fps.compute_field()  # unit weights first
    reused = fps.compute_field(weights=w).copy()

    fresh = FinufftPowerSpectrum(Lbox=Lbox, nmesh=nmesh)
    fresh.set_positions(pos)
    direct = fresh.compute_field(weights=w)

    np.testing.assert_allclose(reused, direct, rtol=1e-4, atol=1e-4)
