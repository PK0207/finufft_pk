"""Tests for finufft_pk.power.FinufftPk."""

import numpy as np
import pytest
from finufft import nufft3d1

from finufft_pk.power import FinufftPk


def _make_catalog(rng, boxsize=64.0, n=2000):
    return rng.uniform(0, boxsize, size=(3, n)).astype(np.float32)


def test_rejects_unsupported_dtype():
    with pytest.raises(AssertionError):
        FinufftPk(nmesh=(16, 16, 16), boxsize=64.0, dtype=np.float32)


def test_rejects_modeord_1():
    with pytest.raises(AssertionError):
        FinufftPk(nmesh=(16, 16, 16), boxsize=64.0, modeord=1)


def test_plan_shape_is_half_plane_on_last_axis():
    fpk = FinufftPk(nmesh=(16, 16, 16), boxsize=64.0)
    assert fpk._plan_shape() == (16, 16, 9)


def test_set_positions_and_compute_field_default_weights(rng):
    nmesh = (16, 16, 16)
    fpk = FinufftPk(nmesh=nmesh, boxsize=64.0)
    pos = _make_catalog(rng)

    fpk.set_positions(pos)
    field = fpk.compute_field()

    assert field.shape == fpk._plan_shape()
    assert field.dtype == np.complex64
    assert np.all(np.isfinite(field))


def test_compute_field_with_explicit_weights(rng):
    nmesh = (16, 16, 16)
    fpk = FinufftPk(nmesh=nmesh, boxsize=64.0)
    pos = _make_catalog(rng)
    w = rng.uniform(0.5, 1.5, size=pos.shape[-1]).astype(np.complex64)

    fpk.set_positions(pos)
    field = fpk.compute_field(weights=w)
    assert field.shape == fpk._plan_shape()
    assert np.all(np.isfinite(field))


def test_bandpower_matches_stored_abacus_power_pointwise(abacus_reference):
    """Bin-for-bin comparison against the stored abacus solution (not another FINUFFT call),
    using the same number of k bins so the two are directly comparable."""
    meta = abacus_reference["meta"]
    pos = abacus_reference["pos"].T.astype(np.float32)  # stored as (N, 3); this class wants (D, N)
    N = pos.shape[-1]
    L = meta["L"]

    fpk = FinufftPk(nmesh=(meta["ngen"],) * 3, boxsize=L)
    fpk.set_positions(pos)
    field = fpk.compute_field()

    delta = (field / N).astype(np.complex64)
    delta[0, 0, 0] = 0.0  # DC mode

    k_binc, _counts, _weighted_counts, bandpower = fpk.compute_bandpower(
        delta, kbins=meta["ngen"], mubins=1, nthread=1,
    )

    assert np.allclose(k_binc, abacus_reference["k_mid"])

    # skip the DC bin (identically ~0 in both, comparing ratios there is meaningless)
    rel_diff = np.abs(bandpower[1:] - abacus_reference["power"][1:]) / abacus_reference["power"][1:]
    assert np.max(rel_diff) < 0.05


def test_half_plane_field_matches_full_cube_modewise(rng):
    """Mode-by-mode comparison of the half-plane field against a directly-computed full cube
    on the same rescaled positions (the kz=0..N/2-1 modes should agree to FINUFFT's eps; the
    Nyquist mode, kz=N/2, is excluded since it isn't recoverable by simply slicing the full
    cube -- the NUFFT sum at k=-N/2 and k=+N/2 aren't the same value in general)."""
    L, nmesh, N = 64.0, 16, 2000
    pos = _make_catalog(rng, boxsize=L, n=N)

    fpk = FinufftPk(nmesh=(nmesh, nmesh, nmesh), boxsize=L)
    fpk.set_positions(pos)
    half = fpk.compute_field().copy()

    pos_rescaled = (pos * (2 * np.pi / L)).astype(np.float32)
    full = nufft3d1(
        x=pos_rescaled[0], y=pos_rescaled[1], z=pos_rescaled[2],
        c=np.ones(N, dtype=np.complex64), n_modes=(nmesh, nmesh, nmesh),
        isign=-1, modeord=0, eps=1e-4, upsampfac=1.25, fftw=0,
    )

    start = nmesh // 2  # CMCL order: index nmesh//2 is k=0
    full_kz_0_to_half = full[:, :, start:]
    half_kz_0_to_half = half[:, :, :-1]  # drop the Nyquist mode

    np.testing.assert_allclose(half_kz_0_to_half, full_kz_0_to_half, rtol=1e-2, atol=1e-2)
