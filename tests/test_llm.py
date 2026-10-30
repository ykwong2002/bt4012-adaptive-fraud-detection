import numpy as np
import pandas as pd

from camo.llm.batch_runner import estimate_cost, make_request
from camo.llm.cache import ParquetCache
from camo.llm.mock import mock_pair_score, mock_review_fields
from camo.llm.prompts import REVIEW_FIELDS, REVIEW_SCHEMA, REVIEW_SYSTEM
from camo.llm.score_reviews import fields_matrix, row_from_result


def test_request_shape_and_cost():
    r = make_request("rev-1", "claude-haiku-5-5", REVIEW_SYSTEM, "hello", REVIEW_SCHEMA)
    assert r["params"]["output_config"]["format"]["type"] == "json_schema"
    assert r["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    c = estimate_cost([r] * 1000)
    assert 0 < c["usd"] < 1.0


def test_cache_roundtrip(tmp_path):
    c = ParquetCache(tmp_path / "x.parquet", ["review_id"])
    assert c.missing([(1,), (2,)]) == [(1,), (2,)]
    c.add([{"review_id": 1, "status": "ok", "specificity": 0.5}])
    c.save()
    c2 = ParquetCache(tmp_path / "x.parquet", ["review_id"])
    assert c2.missing([(1,), (2,)]) == [(2,)]


def test_fields_matrix_handles_unscored():
    nt = pd.DataFrame({"review_id": [10, 11, 12]})
    rows = [row_from_result("rev-10", mock_review_fields("great great great!!"), "ok", "mock"),
            row_from_result("rev-11", None, "refusal", "mock")]
    x, names = fields_matrix(nt, pd.DataFrame(rows))
    assert x.shape == (3, len(REVIEW_FIELDS) + 1) and names[-1] == "llm_unscored"
    assert x[1, -1] == 1.0 and x[2, -1] == 1.0 and x[0, -1] == 0.0
    assert np.isfinite(x).all()


def test_mock_pair_score_in_range():
    rng = np.random.default_rng(0)
    for rel in ("rur", "rsr", "rtr"):
        s = mock_pair_score("the lamb was good", "amazing best ever", rel, rng)
        assert 0 <= s["relative_fit"] <= 1
