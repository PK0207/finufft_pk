"""Thin wrapper around abacusnbody's k/mu binning, reused so FINUFFT fields are binned
identically to abacus's own ``calc_power`` output.
"""

import os

import numpy as np
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak, get_k_mu_edges


def make_k_mu_edges(Lbox, nmesh, kbins=None, mubins=1, logk=False):
    """Build the same k/mu bin edges abacus's calc_power uses internally, once."""
    k_max = np.pi * nmesh / Lbox
    if kbins is None:
        kbins = nmesh
    return get_k_mu_edges(Lbox, k_max, kbins, mubins, logk)


def bandpower_from_field(field_fft, Lbox, k_bin_edges, mu_bin_edges, nthread=None):
    """Bin a (possibly half-plane) complex field into P(k) using abacus's binner.

    ``field_fft`` must already be in ``rfftn``-style layout (DC mode at [0, 0, 0], last
    axis of length nmesh//2 + 1) and normalized the way ``calc_pk_from_deltak`` expects
    (delta_k, i.e. divided by N_particles and with the DC mode zeroed).
    """
    nthread = nthread or os.cpu_count()
    return calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, nthread=nthread)