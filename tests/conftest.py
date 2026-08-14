import pickle
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def abacus_reference():
    """Pre-computed small catalog + abacus calc_power result (see data/generate_reference.py).

    Stored on disk rather than calling abacusutils at test time.
    """
    with open(DATA_DIR / "reference_small.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
