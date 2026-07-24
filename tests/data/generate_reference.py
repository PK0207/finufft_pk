"""One-off script to (re)generate tests/data/reference_small.pkl.

Not run by pytest. Re-run manually (inside an env with abacusutils installed, e.g.
../fft_benchmark/astrovenv) only if the reference catalog/parameters need to change:

    astrovenv/bin/python3 tests/data/generate_reference.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finufft_pk.grids import make_synthetic_catalog  # noqa: E402
from abacusnbody.analysis.power_spectrum import calc_power  # noqa: E402

L = 64.0       # Mpc/h, small box to keep the catalog small
NGEN = 32      # mesh resolution used both to draw the field and as nmesh for calc_power
NBAR = 1.0     # (Mpc/h)^-3
P0 = 0.05
K0 = 1.0
N_INDEX = -2.0
SEED = 0

rng = np.random.default_rng(SEED)
pos, meta = make_synthetic_catalog(NGEN, L, K0, N_INDEX, P0, NBAR, rng)

P_abacus = calc_power(
    pos, Lbox=L, nmesh=NGEN, paste='TSC', compensated=True, interlaced=False, nthread=1,
)

out = {
    'pos': pos,
    'meta': meta,
    'k_mid': np.asarray(P_abacus['k_mid']),
    'power': np.asarray(P_abacus['power']),
    'N_mode': np.asarray(P_abacus['N_mode']),
}

outfile = Path(__file__).resolve().parent / 'reference_small.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(out, f)
print(f'wrote {outfile} ({pos.nbytes / 1e6:.1f} MB positions, N={len(pos)})')
