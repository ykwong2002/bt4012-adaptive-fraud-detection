"""Review issue #5: injection must never delete products or touch the training period."""
import numpy as np

from camo.adversarial.inject import inject_reviews
from camo.data.graph_build import filter_products


def test_injection_respects_product_cap_and_training_period(raw_mini):
    raw = filter_products(raw_mini, 60).copy()
    cap = 60
    train_end = raw["date"].quantile(0.5)
    later = raw[raw["date"] > train_end]
    windows = [(later["date"].min(), later["date"].max(), len(later))]
    aug, inj = inject_reviews(raw, train_end, windows, level=0.5, seed=0, max_reviews_per_product=cap)
    assert len(inj) > 0
    assert aug.groupby("product_id").size().max() <= cap
    assert (inj["date"] > train_end).all()
    # every original product survives the graph builder's filter
    assert set(filter_products(aug, cap)["product_id"]) == set(raw["product_id"])
    assert np.isin(raw["review_id"], filter_products(aug, cap)["review_id"]).all()
