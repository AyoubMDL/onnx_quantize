import numpy as np
import pytest


@pytest.fixture
def rng():
    seed = 42
    return np.random.default_rng(seed)
