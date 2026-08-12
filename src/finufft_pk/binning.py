import numpy as np
import multiprocessing as mp
import os
from ._binning import *
import cupy as cp


def make_k_mu_edges(boxsize, nmesh, kbins=None, mubins=1):
    """Build the same k/mu bin edges abacus's calc_power uses internally, once."""
    fold_shape = nmesh[:-1]
    len_z = nmesh[-1]
    full_n = np.array(list(fold_shape) + [2 * len_z])
    k_max = np.pi * full_n.min() / boxsize
    if not kbins:
        k_bins = np.linspace(0, k_max, full_n.min() // 2 + 1)
    else:
        k_bins = np.linspace(0, k_max, kbins + 1)

    mu_bins = np.linspace(0, 1, mubins + 1)
    return k_bins, mu_bins


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
    counts = np.zeros((Nk, Nmu))
    weighted_counts = np.zeros((Nk, Nmu))

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


def bandpower_from_field(field_fft, boxsize, kbins, mubins, nthread: int = None):
    """
    Bin a half-plane complex field into P(k) using a pure-Python/multiprocessing
    implementation (see ``bandpower_from_field_cpp`` for the C++-accelerated version).

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
        Number of worker processes; defaults to mp.cpu_count().

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
    raw_power = np.abs(field_fft) ** 2
    nmesh = field_fft.shape
    fold_shape = nmesh[:-1]
    len_z = nmesh[-1]
    dk = 2 * np.pi / boxsize
    k_bins, mu_bins = make_k_mu_edges(boxsize, nmesh, kbins, mubins)
    kedges2 = (k_bins / dk) ** 2
    muedges2 = mu_bins**2
    Nk, Nmu = len(k_bins) - 1, len(mu_bins) - 1
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5

    if nthread is None:
        nthread = mp.cpu_count()

    # simple chunks distributed across CPUs, totalled at the end
    all_indices = list(np.ndindex(*fold_shape))
    chunk_size = max(1, len(all_indices) // nthread)
    chunks = [
        all_indices[i : i + chunk_size] for i in range(0, len(all_indices), chunk_size)
    ]
    # raw_power, fold_shape, len_z, kedges2, muedges2, Nk, Nmu
    args = [
        (chunk, raw_power, fold_shape, len_z, kedges2, muedges2, Nk, Nmu)
        for chunk in chunks
    ]
    with mp.Pool(nthread) as pool:
        results = pool.starmap(_kmu_binning, args)

    counts = sum(r[0] for r in results)
    weighted_counts = sum(r[1] for r in results)
    for i in range(Nk):
        for j in range(Nmu):
            if counts[i, j] > 0:
                weighted_counts[i, j] /= counts[i, j]

    bandpower = (weighted_counts * boxsize**3).flatten()
    return k_binc, counts, weighted_counts, bandpower


def bandpower_from_field_cpp(field_fft, boxsize, kbins, mubins, nthread: int = None):
    """
    Bin a half-plane complex field into P(k), dispatching the inner
    (k, mu)-binning loop to a C++ implementation (f32/f64) for speed.

    field_fft: complex ndarray, shape (n0, ..., n_{d-2}, nz//2 + 1)
        Field in ``rfftn``-style layout: DC mode at [0, 0, 0], leading axes
        in CMCL (centered) order, last axis half-length (real-to-complex).
        Must already be normalized as delta_k (divided by N_particles, DC
        mode zeroed) the way ``calc_pk_from_deltak`` expects. dtype must be
        complex64 or complex128 so the derived real dtype is float32/float64.
    boxsize: float
        Physical size of the box the field was computed in.
    kbins: int or None
        Number of k bins; if None, defaults to min(full_nmesh) // 2 + 1.
    mubins: int
        Number of mu (angle-cosine) bins.
    nthread: int or None
        Number of threads passed to the C++ binning kernel.

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
    raw_power = np.abs(field_fft) ** 2  # Could put this in C++
    nmesh = field_fft.shape
    dk = 2 * np.pi / boxsize
    k_bins, mu_bins = make_k_mu_edges(boxsize, nmesh, kbins, mubins)
    kedges2 = (k_bins / dk) ** 2
    muedges2 = mu_bins**2
    Nk, Nmu = len(k_bins) - 1, len(mu_bins) - 1
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5

    compute_dtype = raw_power.dtype
    kedges2 = kedges2.astype(compute_dtype)
    muedges2 = muedges2.astype(compute_dtype)
    counts = np.zeros((Nk, Nmu), dtype=compute_dtype)
    weighted_counts = np.zeros((Nk, Nmu), dtype=compute_dtype)

    if compute_dtype == np.float32:
        binning_func = kmu_binning_cpp_f32
    elif compute_dtype == np.float64:
        binning_func = kmu_binning_cpp_f64
    else:
        raise ValueError(
            f"Unsupported dtype {compute_dtype}, expected float32 or float64"
        )

    binning_func(kedges2, muedges2, raw_power, counts, weighted_counts, nthread)

    for i in range(Nk):
        for j in range(Nmu):
            if counts[i, j] > 0:
                weighted_counts[i, j] /= counts[i, j]

    bandpower = (weighted_counts * boxsize**3).flatten()
    return k_binc, counts, weighted_counts, bandpower
