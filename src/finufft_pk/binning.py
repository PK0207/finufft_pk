import numpy as np
import multiprocessing as mp
import os
from ._binning import kmu_binning_cpp_f32, kmu_binning_cpp_f64

def make_k_mu_edges(boxsize, nmesh, kbins=None, mubins=1):
    """Build the same k/mu bin edges abacus's calc_power uses internally, once."""
    fold_shape = nmesh[:-1]
    len_z = nmesh[-1]
    full_n = np.array(list(fold_shape) + [2 * len_z])
    k_max = np.pi * full_n.min() / boxsize
    if not kbins:
        k_bins = np.linspace(0, k_max, full_n.min() // 2 + 1)
    else:
        k_bins = np.linspace(0, k_max, kbins+1)
    
    mu_bins = np.linspace(0, 1, mubins + 1)
    return k_bins, mu_bins

def folded_sq(idx, n):
            # CMCL order is already centered (idx=0 -> k=-N/2), so no wraparound needed
            return (idx - n // 2) ** 2

def _kmu_binning(chunk_idx, raw_power, fold_shape, len_z, kedges2, muedges2, Nk, Nmu):
    #Code inherited and refactored from abacus
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

def bandpower_from_field(field_fft, boxsize, kbins, mubins, nthread=None):
    """Bin a (possibly half-plane) complex field into P(k) using abacus's binner.

    ``field_fft`` must already be in ``rfftn``-style layout (DC mode at [0, 0, 0], last
    axis of length nmesh//2 + 1) and normalized the way ``calc_pk_from_deltak`` expects
    (delta_k, i.e. divided by N_particles and with the DC mode zeroed).
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
        
    #simple chunks distributed across CPUs, totalled at the end
    all_indices = list(np.ndindex(*fold_shape))
    chunk_size = max(1, len(all_indices) // nthread)
    chunks = [all_indices[i:i + chunk_size] for i in range(0, len(all_indices), chunk_size)]
    #raw_power, fold_shape, len_z, kedges2, muedges2, Nk, Nmu
    args = [(chunk, raw_power, fold_shape, len_z, kedges2, muedges2, Nk, Nmu)
            for chunk in chunks]
    with mp.Pool(nthread) as pool:
        results = pool.starmap(_kmu_binning, args)

    counts         = sum(r[0] for r in results)
    weighted_counts = sum(r[1] for r in results)
    for i in range(Nk):
        for j in range(Nmu):
            if counts[i, j] > 0:
                weighted_counts[i, j] /= counts[i, j]

    bandpower = (weighted_counts * boxsize**3).flatten()
    return k_binc, counts, weighted_counts, bandpower

def bandpower_from_field_cpp(field_fft, boxsize, kbins, mubins, nthread=None):
    """Bin a (possibly half-plane) complex field into P(k) using abacus's binner.

    ``field_fft`` must already be in ``rfftn``-style layout (DC mode at [0, 0, 0], last
    axis of length nmesh//2 + 1) and normalized the way ``calc_pk_from_deltak`` expects
    (delta_k, i.e. divided by N_particles and with the DC mode zeroed).
    """
    raw_power = np.abs(field_fft) ** 2 # Could put this in C++
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

    if compute_dtype==np.float32:
        binning_func = kmu_binning_cpp_f32
    elif compute_dtype == np.float64:
        binning_func = kmu_binning_cpp_f64
    else:
        raise ValueError(f"Unsupported dtype {compute_dtype}, expected float32 or float64")
    if nthread is None:
        nthread = os.cpu_count()

    binning_func(kedges2, muedges2, raw_power, counts, weighted_counts, nthread)
    
    for i in range(Nk):
        for j in range(Nmu):
            if counts[i, j] > 0:
                weighted_counts[i, j] /= counts[i, j]

    bandpower = (weighted_counts * boxsize**3).flatten()
    return k_binc, counts, weighted_counts, bandpower
