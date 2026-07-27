"""Tests for finufft_pk.finufft_powerspectrum.FinufftPowerSpectrum."""

import numpy as np
import pytest

from finufft_pk.finufft_powerspectrum import FinufftPowerSpectrum


def _make_catalog(rng, boxsize=64.0, n=2000):
    return rng.uniform(0, boxsize, size=(3, n)).astype(np.float32)


def test_rejects_unsupported_dtype():
    with pytest.raises(AssertionError):
        FinufftPowerSpectrum(nmesh=(16, 16, 16), boxsize=64.0, dtype=np.float32)


def test_plan_shape_is_half_plane_on_last_axis():
    fps = FinufftPowerSpectrum(nmesh=(16, 16, 16), boxsize=64.0)
    assert fps._plan_shape() == (16, 16, 8)


def test_set_positions_and_compute_field_default_weights(rng):
    nmesh = (16, 16, 16)
    fps = FinufftPowerSpectrum(nmesh=nmesh, boxsize=64.0)
    pos = _make_catalog(rng)

    fps.set_positions(pos)
    field = fps.compute_field()

    assert field.shape == fps._plan_shape()
    assert field.dtype == np.complex64
    assert np.all(np.isfinite(field))


def test_compute_field_with_explicit_weights(rng):
    nmesh = (16, 16, 16)
    fps = FinufftPowerSpectrum(nmesh=nmesh, boxsize=64.0)
    pos = _make_catalog(rng)
    w = rng.uniform(0.5, 1.5, size=pos.shape[-1]).astype(np.complex64)

    fps.set_positions(pos)
    field = fps.compute_field(weights=w)
    assert field.shape == fps._plan_shape()
    assert np.all(np.isfinite(field))


def test_compute_bandpower_recovers_shot_noise_level(abacus_reference):
    """A weakly-clustered catalog (small P0) should have P(k) close to the analytic shot-noise
    level 1/nbar away from the DC/Nyquist bins. Compared against the known analytic shot noise
    rather than the stored abacus fixture directly, since this class's bin count/edges (derived
    from nmesh alone) don't match abacus's calc_power binning.
    """
    meta = abacus_reference["meta"]
    pos = abacus_reference["pos"].T.astype(np.float32)  # stored as (N, 3); this class wants (D, N)
    N = pos.shape[-1]
    L = meta["L"]

    fps = FinufftPowerSpectrum(nmesh=(meta["ngen"],) * 3, boxsize=L)
    fps.set_positions(pos)
    field = fps.compute_field()
    counts, weighted_counts = fps.compute_bandpower(field)

    counts, weighted_counts = counts[:, 0], weighted_counts[:, 0]
    power = weighted_counts / counts * L**3 / N**2

    shot_noise = L**3 / N
    well_sampled = power[2:-2]  # skip the DC bin and the last couple bins near Nyquist
    assert np.all(np.isfinite(well_sampled))
    assert np.median(well_sampled) == pytest.approx(shot_noise, rel=0.5)
