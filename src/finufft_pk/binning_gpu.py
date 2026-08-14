import cupy as cp
import numpy as np

def folded_sq(idx, n):
    """Squared centered mode index for a CMCL-ordered axis of length n."""
    # CMCL order is already centered (idx=0 -> k=-N/2), so no wraparound needed
    return (idx - n // 2) ** 2


def _kmu_binning(chunk_idx, raw_power, fold_shape, len_z, kedges2, muedges2, Nk, Nmu):
    """
    Accumulate counts and weighted power into (k, mu) bins for a chunk
    of perpendicular-axis indices; run as a worker in the multiprocessing pool.
    """
    # Code inherited and refactored from abacus
    counts = cp.zeros((Nk, Nmu))
    weighted_counts = cp.zeros((Nk, Nmu))

    for perp_idx in chunk_idx:
        perp2 = sum(folded_sq(i, n) for i, n in zip(perp_idx, fold_shape))
        for k in range(len_z):
            bk, bmu = 0, 0  # k-mode counter, so we don't add counts the zero mode
            k2 = k**2
            mag = perp2 + k2

            if mag > 0:
                invkmag2 = mag**-1
                mu2 = k2 * invkmag2
            else:
                mu2 = 0

            if mag < kedges2[0]:  # if it goes below the initial mode ignore it
                continue
            if mag >= kedges2[-1]:  # if it reaches kmax, we're done
                break
            while mag > kedges2[bk + 1]:
                bk += 1  # don't double count zeroth order
            while mu2 > muedges2[bmu + 1]:
                bmu += 1

            weight = 1 if k == 0 else 2
            counts[bk, bmu] += weight
            weighted_counts[bk, bmu] += weight * raw_power[perp_idx + (k,)]
    return counts, weighted_counts



def make_k_mu_edges(boxsize, nmesh, kbins=None, mubins=1):
    """Build the same k/mu bin edges abacus's calc_power uses internally, once."""
    fold_shape = nmesh[:-1]
    len_z = nmesh[-1]
    # cp.linspace's `num` must be a plain int, not a 0-d cupy array
    n_min = min(list(fold_shape) + [2 * len_z])
    k_max = np.pi * n_min / boxsize
    if not kbins:
        k_bins = cp.linspace(0, k_max, n_min // 2 + 1)
    else:
        k_bins = cp.linspace(0, k_max, int(kbins) + 1)

    mu_bins = cp.linspace(0, 1, int(mubins) + 1)
    return k_bins, mu_bins


def bandpower_from_field_gpu(field_fft, boxsize, kbins, mubins):
    """
    Bin a half-plane complex field into P(k) with vectorized CuPy ops (no
    host-side loop, no multiprocessing -- everything stays on the GPU).
    See ``bandpower_from_field_cpp`` for the C++-accelerated CPU version.

    field_fft: complex ndarray, shape (n0, ..., n_{d-2}, nz//2 + 1)
        Field in ``rfftn``-style layout: DC mode at [0, 0, 0], leading axes
        in CMCL (centered) order, last axis half-length (real-to-complex).
        Must already be normalized as delta_k (divided by N_particles, DC
        mode zeroed) the way ``calc_pk_from_deltak`` expects.
    boxsize: float
        Physical size of the box the field was computed in.
    kbins: int or None
        Number of k bins; if None, defaults to min(full_nmesh) // 2 + 1.
    mubins: int
        Number of mu (angle-cosine) bins.
    nthread: int or None
        Unused; kept for call-site compatibility with the CPU binner.

    Returns
    -------
    k_binc: ndarray, shape (kbins,)
        Bin-center k values.
    counts: ndarray, shape (kbins, mubins)
        Mode counts per (k, mu) bin.
    weighted_counts: ndarray, shape (kbins, mubins)
        Mean power per (k, mu) bin.
    bandpower: ndarray, shape (kbins * mubins,)
        Flattened band power, ``weighted_counts * boxsize**3``.
    """
    raw_power = cp.abs(field_fft) ** 2
    nmesh = field_fft.shape
    fold_shape = nmesh[:-1]
    len_z = nmesh[-1]
    dk = 2 * cp.pi / boxsize
    k_bins, mu_bins = make_k_mu_edges(boxsize, nmesh, kbins, mubins)
    kedges2 = (k_bins / dk) ** 2
    muedges2 = mu_bins**2
    Nk, Nmu = len(k_bins) - 1, len(mu_bins) - 1
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5

    # perp2: squared centered mode index (CMCL order, idx=0 -> k=-N/2), summed
    # across the leading axes via broadcasting instead of a per-point loop
    perp2 = cp.zeros(fold_shape if fold_shape else (), dtype=cp.float64)
    for axis, n in enumerate(fold_shape):
        idx2 = (cp.arange(n) - n // 2) ** 2
        shape = [1] * len(fold_shape)
        shape[axis] = n
        perp2 = perp2 + idx2.reshape(shape)

    # last axis is the real-transform half-plane (0..N/2): no folding needed
    k_last = cp.arange(len_z)
    k2_last = k_last**2
    mag2 = perp2[..., None] + k2_last  # shape == raw_power.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        invmag2 = cp.where(mag2 > 0, 1.0 / mag2, 0.0)
    mu2 = k2_last * invmag2

    # weight 2x everywhere except the k=0 plane, whose conjugate-symmetric
    # half isn't separately stored
    weight = cp.where(k_last == 0, 1.0, 2.0)
    weight = cp.broadcast_to(weight, mag2.shape)

    k_idx = cp.searchsorted(kedges2, mag2.ravel(), side="left").reshape(mag2.shape) - 1
    mu_idx = cp.searchsorted(muedges2, mu2.ravel(), side="left").reshape(mu2.shape) - 1

    valid = (mag2 >= kedges2[0]) & (mag2 < kedges2[-1])
    valid &= (k_idx >= 0) & (k_idx < Nk) & (mu_idx >= 0) & (mu_idx < Nmu)

    lin_idx = (k_idx * Nmu + mu_idx)[valid]
    w = weight[valid]
    p = raw_power[valid]

    counts = cp.bincount(lin_idx, weights=w, minlength=Nk * Nmu).reshape(Nk, Nmu)
    weighted_counts = cp.bincount(lin_idx, weights=w * p, minlength=Nk * Nmu).reshape(
        Nk, Nmu
    )
    weighted_counts = cp.where(counts > 0, weighted_counts / counts, 0.0)

    bandpower = (weighted_counts * boxsize**3).flatten()
    return k_binc, counts, weighted_counts, bandpower
