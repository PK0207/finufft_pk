"""FINUFFT-based power spectrum estimator.

Reuses the half-plane (Hermitian) trick prototyped in ``fft_benchmark/finufft_halfplane.ipynb``:
FINUFFT is run onto only ``nmesh//2 + 1`` z-modes (instead of the full ``nmesh``), with source
strengths pre-multiplied by ``exp(-1j * shift * z)`` (shift = nmesh_half // 2) so the resulting
half-plane field lines up with ``np.fft.rfftn``/abacus conventions (``modeord=0``) and can be fed
straight into ``calc_pk_from_deltak``. This halves the field's memory footprint for free.
"""

import os

import numpy as np
from finufft import Plan, nufft3d1

from .binning import bandpower_from_field, make_k_mu_edges


def _rescale(positions, Lbox):
    """[0, Lbox) -> [-pi, pi), returned as contiguous per-axis float32 arrays."""
    rescaled = (positions / Lbox) * 2 * np.pi - np.pi
    x = np.ascontiguousarray(rescaled[:, 0], dtype=np.float32)
    y = np.ascontiguousarray(rescaled[:, 1], dtype=np.float32)
    z = np.ascontiguousarray(rescaled[:, 2], dtype=np.float32)
    return x, y, z


class FinufftPowerSpectrum:
    """Compute galaxy/particle power spectra with FINUFFT as the field-estimation engine.

    Holds a single, persistent ``finufft.Plan`` (the guru interface) so that repeated calls
    with the same particle positions but different weights only pay for ``execute``, not for
    re-sorting the points (``setpts``) — the expensive step for MCMC-style reuse.
    """

    def __init__(self, Lbox, nmesh, eps=1e-6, upsampfac=2.0, fftw=64,
                 dtype='complex64', nthreads=None, kbins=None, mubins=1):
        self.Lbox = Lbox
        self.nmesh = nmesh
        self.nmesh_half = nmesh // 2 + 1
        self.shift = self.nmesh_half // 2
        self.eps = eps
        self.nthreads = nthreads or os.cpu_count()
        self._finufft_kwargs = dict(eps=eps, upsampfac=upsampfac, fftw=fftw)

        self._plan = Plan(
            nufft_type=1,
            n_modes_or_dim=(nmesh, nmesh, self.nmesh_half),
            n_trans=1,
            isign=-1,
            dtype=dtype,
            modeord=0,
            nthreads=self.nthreads,
            **self._finufft_kwargs,
        )

        self.k_bin_edges, self.mu_bin_edges = make_k_mu_edges(Lbox, nmesh, kbins, mubins)

        self._x = self._y = self._z = None
        self._phase = None
        self._N = None
        self.field = None

    def _phase_shift(self, z):
        return np.exp(-1j * self.shift * z).astype(np.complex64)

    def set_positions(self, positions):
        """Register a new particle catalog: rescale, re-sort (setpts). Expensive; call once
        per distinct set of positions."""
        self._x, self._y, self._z = _rescale(positions, self.Lbox)
        self._phase = self._phase_shift(self._z)
        self._N = positions.shape[0]
        self._plan.setpts(self._x, self._y, self._z)

    def compute_field(self, weights=None, out=None):
        """Execute the persistent plan against the current positions. Cheap: no re-sorting."""
        if self._x is None:
            raise RuntimeError('call set_positions() before compute_field()')
        c = np.ones(self._N, dtype=np.complex64) if weights is None else \
            np.ascontiguousarray(weights, dtype=np.complex64)
        self.field = self._plan.execute(c * self._phase, out=out)
        return self.field

    def compute_power(self, weights=None):
        """Convenience: compute_field() + bin into P(k) with abacus's calc_pk_from_deltak."""
        field = self.compute_field(weights)
        delta = (field / self._N).astype(np.complex64)
        delta[0, 0, 0] = 0.0  # DC mode
        result = bandpower_from_field(delta, self.Lbox, self.k_bin_edges, self.mu_bin_edges,
                                       nthread=self.nthreads)
        # bin midpoints from the shared edges, so results are directly comparable to
        # abacus's calc_power (whose 'k_mid' is defined the same way)
        result['k_mid'] = 0.5 * (self.k_bin_edges[:-1] + self.k_bin_edges[1:])
        return result

    def replace_positions(self, new_positions, weights=None):
        """Whole catalog redrawn but grid/config unchanged: re-setpts + re-execute on the same
        persistent plan. No delta-trick benefit when (almost) everything changes."""
        self.set_positions(new_positions)
        return self.compute_field(weights)

    def update_positions_subset(self, old_positions, new_positions, weights=None):
        """A small subset of particles moved; everything else is unchanged.

        Exploits linearity of the type-1 NUFFT: the field contributed by any fixed set of
        particles simply changes by -NUFFT(old subset) + NUFFT(new subset). Runs a small,
        one-off NUFFT sized to just the changed subset (not the persistent plan, since the
        subset size varies call to call) and adds the delta into the cached ``self.field``.
        """
        if self.field is None:
            raise RuntimeError('call compute_field()/replace_positions() before a partial update')

        n_sub = old_positions.shape[0]
        c = np.ones(n_sub, dtype=np.complex64) if weights is None else \
            np.ascontiguousarray(weights, dtype=np.complex64)

        xo, yo, zo = _rescale(old_positions, self.Lbox)
        xn, yn, zn = _rescale(new_positions, self.Lbox)

        n_modes = (self.nmesh, self.nmesh, self.nmesh_half)
        removed = nufft3d1(x=xo, y=yo, z=zo, c=c * self._phase_shift(zo),
                            n_modes=n_modes, isign=-1, modeord=0,
                            nthreads=self.nthreads, **self._finufft_kwargs)
        added = nufft3d1(x=xn, y=yn, z=zn, c=c * self._phase_shift(zn),
                          n_modes=n_modes, isign=-1, modeord=0,
                          nthreads=self.nthreads, **self._finufft_kwargs)

        self.field += added - removed
        return self.field

    def compute_field_streaming(self, position_batches, weight_batches=None, out=None):
        """Accumulate the field over particle batches without holding the full catalog in
        memory at once (type-1 NUFFT is linear in the source strengths, so batches just sum)."""
        n_modes = (self.nmesh, self.nmesh, self.nmesh_half)
        field = np.zeros(n_modes, dtype='complex64') if out is None else out
        field[...] = 0
        weight_batches = weight_batches or ([None] * len(position_batches))

        total_N = 0
        for batch_pos, batch_w in zip(position_batches, weight_batches):
            x, y, z = _rescale(batch_pos, self.Lbox)
            n = batch_pos.shape[0]
            c = np.ones(n, dtype=np.complex64) if batch_w is None else \
                np.ascontiguousarray(batch_w, dtype=np.complex64)
            field += nufft3d1(x=x, y=y, z=z, c=c * self._phase_shift(z),
                               n_modes=n_modes, isign=-1, modeord=0,
                               nthreads=self.nthreads, **self._finufft_kwargs)
            total_N += n

        self.field = field
        self._N = total_N
        return field
