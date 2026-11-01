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


def test_calibrated_monitor_ignores_noise_and_flags_real_shift():
    """Review issue #10: with thousands of rows per window, p-values flag pure noise. The calibrated
    effect-size monitor must stay quiet on stationary windows and fire on a genuine shift."""
    import pandas as pd

    from camo.drift.monitor import calibrate, flag_windows, window_signals

    rng = np.random.default_rng(0)
    n_per, n_w = 5000, 8
    windows = np.repeat(np.arange(n_w), n_per)
    x = rng.normal(size=(n_w * n_per, 3))
    x[windows >= 6, 0] += 0.8                      # real shift in the last two windows
    sig = window_signals(windows, [0, 1, 2], {"f": (x, ["a", "b", "c"])}, None, None)
    thr = calibrate(sig, margin=1.25, min_psi=0.1, min_ks=0.05)
    flags = flag_windows(sig, thr, ["f"], k=2)
    assert isinstance(flags, pd.DataFrame)
    assert not flags.loc[flags["window"] < 6, "flag"].any()
    assert flags.loc[flags["window"] >= 6, "flag"].all()
    assert flags["alarm"].tolist() == [False, False, False, False, True]
