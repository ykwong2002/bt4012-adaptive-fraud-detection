import numpy as np

from camo.drift.ks import ks
from camo.drift.psi import psi


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20000), rng.normal(size=20000)
    assert psi(a, b) < 0.01


def test_psi_and_ks_detect_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(loc=1.0, size=5000)
    assert psi(a, b) > 0.2
    stat, p = ks(a, b)
    assert stat > 0.3 and p < 0.01


def test_constant_feature_is_stable():
    assert psi(np.ones(100), np.ones(100)) == 0.0
