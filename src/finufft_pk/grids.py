"""Synthetic catalog generation with a known analytic power spectrum.

Moved and generalized from ``fft_benchmark/generate_grids.py``: builds a unit-amplitude
Gaussian random field with P(k) = (k/k0)^n_index, then Poisson-samples particle positions
from it. Used as ground truth for correctness testing of power spectrum estimators.
"""

import numpy as np


def make_delta_unit(ngen, L, k0, n_index, rng, dtype=np.float32):
    """Unit-amplitude (P0=1) Gaussian random field with P(k) = (k/k0)^n_index."""
    kf = 2 * np.pi / L
    kx = np.fft.fftfreq(ngen, d=1.0 / ngen) * kf
    kz = np.fft.rfftfreq(ngen, d=1.0 / ngen) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    shape = np.zeros_like(Kmag)
    nz = Kmag > 0
    shape[nz] = (Kmag[nz] / k0) ** n_index  # DC mode left at 0 -> <delta> = 0

    Vcell = (L / ngen) ** 3
    white_k = np.fft.rfftn(rng.standard_normal((ngen,) * 3))
    delta = np.fft.irfftn(white_k * np.sqrt(shape / Vcell), s=(ngen,) * 3, axes=(0, 1, 2))
    return delta.astype(dtype)


def sample_positions(delta_unit, P0, nbar, L, ngen, rng, cell_idx=None, dtype=np.float32):
    """Poisson-sample particle positions from n(x) = nbar*(1 + sqrt(P0)*delta_unit)."""
    delta = np.sqrt(P0) * delta_unit
    Vcell, cell = (L / ngen) ** 3, L / ngen
    if cell_idx is None:
        cell_idx = np.arange(ngen**3)

    lam = nbar * Vcell * np.clip(1.0 + delta, 0.0, None)
    counts = rng.poisson(lam)
    N_red = int(counts.sum())

    flat = np.repeat(cell_idx, counts.ravel())
    ix, iy, iz = np.unravel_index(flat, (ngen,) * 3)
    pos = np.empty((N_red, 3), dtype=dtype)
    pos[:, 0] = (ix + rng.random(N_red)) * cell
    pos[:, 1] = (iy + rng.random(N_red)) * cell
    pos[:, 2] = (iz + rng.random(N_red)) * cell
    return pos


def make_synthetic_catalog(ngen, L, k0, n_index, P0, nbar, rng, dtype=np.float32):
    """Convenience wrapper: build a delta field and sample a catalog from it in one call.

    Returns (positions, meta) where meta carries the parameters needed to reconstruct
    the analytic input P(k) = P0*(k/k0)^n_index and shot noise 1/nbar_eff.
    """
    delta_unit = make_delta_unit(ngen, L, k0, n_index, rng, dtype=dtype)
    pos = sample_positions(delta_unit, P0, nbar, L, ngen, rng, dtype=dtype)
    meta = dict(k0=k0, P0=P0, nbar=nbar, n_index=n_index, L=L, ngen=ngen, N=len(pos))
    return pos, meta
