import numpy as np

from camo.data.features_pit import FEATURES, compute_pit_features


def test_no_future_leakage(raw_mini):
    full = compute_pit_features(raw_mini)
    cutoff = raw_mini["date"].quantile(0.5)
    past = raw_mini[raw_mini["date"] <= cutoff]
    part = compute_pit_features(past)
    a = full.loc[past.index].to_numpy()
    b = part.to_numpy()
    assert np.allclose(a, b, equal_nan=True), "features changed when future reviews were added"


def test_feature_columns_and_finite(raw_mini):
    f = compute_pit_features(raw_mini)
    assert list(f.columns) == FEATURES
    assert np.isfinite(f.to_numpy()).all()
    assert (f["u_first_review"].isin([0, 1])).all()
    assert (f["u_n_prev"] >= 0).all() and (f["p_n_prev"] >= 0).all()
