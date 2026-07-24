import numpy as np

from finufft_pk.power_spectrum import FinufftPowerSpectrum


def test_update_subset_matches_full_recompute(rng):
    """Moving a handful of particles via update_positions_subset() should give the same
    field as recomputing from scratch on the final positions."""
    Lbox, nmesh, N, n_moved = 64.0, 16, 2000, 25
    pos = rng.uniform(0, Lbox, size=(N, 3)).astype(np.float32)

    fps = FinufftPowerSpectrum(Lbox=Lbox, nmesh=nmesh)
    fps.set_positions(pos)
    fps.compute_field()

    moved_idx = rng.choice(N, size=n_moved, replace=False)
    old_sub = pos[moved_idx].copy()
    new_pos = pos.copy()
    new_pos[moved_idx] = rng.uniform(0, Lbox, size=(n_moved, 3)).astype(np.float32)

    fps.update_positions_subset(old_sub, new_pos[moved_idx])

    fresh = FinufftPowerSpectrum(Lbox=Lbox, nmesh=nmesh)
    fresh.set_positions(new_pos)
    expected = fresh.compute_field()

    np.testing.assert_allclose(fps.field, expected, rtol=1e-3, atol=1e-3)


def test_replace_positions_matches_full_recompute(rng):
    """Whole-catalog redraw with the same grid config should match a from-scratch object."""
    Lbox, nmesh, N = 64.0, 16, 2000
    pos_a = rng.uniform(0, Lbox, size=(N, 3)).astype(np.float32)
    pos_b = rng.uniform(0, Lbox, size=(N, 3)).astype(np.float32)

    fps = FinufftPowerSpectrum(Lbox=Lbox, nmesh=nmesh)
    fps.set_positions(pos_a)
    fps.compute_field()
    reused = fps.replace_positions(pos_b).copy()

    fresh = FinufftPowerSpectrum(Lbox=Lbox, nmesh=nmesh)
    fresh.set_positions(pos_b)
    expected = fresh.compute_field()

    np.testing.assert_allclose(reused, expected, rtol=1e-4, atol=1e-4)
