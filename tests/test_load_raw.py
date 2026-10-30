"""Review issue #4: the text join must never silently shift texts onto the wrong review."""
import pytest

from camo.data.load_raw import load_raw


def _write(d, meta_lines, text_lines):
    (d / "metadata").write_text("".join(l + "\n" for l in meta_lines))
    (d / "reviewContent").write_text("".join(l + "\n" for l in text_lines))


def test_tab_inside_text_is_kept(tmp_path):
    _write(tmp_path, ["u1\tp1\t5\t1\t2010-01-01", "u2\tp1\t1\t-1\t2010-01-02", "u3\tp2\t4\t1\t2010-01-03"],
           ["u1\tp1\t2010-01-01\tfirst\twith a tab", "u2\tp1\t2010-01-02\tsecond", "u3\tp2\t2010-01-03\tthird"])
    df = load_raw(tmp_path)
    assert df["text"].tolist() == ["first\twith a tab", "second", "third"]
    assert df.attrs["text_join"]["method"] == "row_order_verified"
    assert df["label"].tolist() == [0, 1, 0]


def test_reordered_and_duplicate_keys_join_by_occurrence(tmp_path):
    # same (user, product, date) twice: matched by occurrence order, not by line position
    _write(tmp_path, ["u1\tp1\t5\t1\t2010-01-01", "u1\tp1\t3\t1\t2010-01-01", "u2\tp2\t4\t1\t2010-02-01"],
           ["u2\tp2\t2010-02-01\tthird", "u1\tp1\t2010-01-01\tfirst", "u1\tp1\t2010-01-01\tsecond"])
    df = load_raw(tmp_path)
    assert df["text"].tolist() == ["first", "second", "third"]
    assert df.attrs["text_join"]["method"] == "key_occurrence"


def test_missing_text_raises(tmp_path):
    _write(tmp_path, ["u1\tp1\t5\t1\t2010-01-01", "u2\tp1\t1\t-1\t2010-01-02"],
           ["u1\tp1\t2010-01-01\tfirst"])
    with pytest.raises(ValueError, match="no text"):
        load_raw(tmp_path)


def test_malformed_line_raises_with_line_number(tmp_path):
    _write(tmp_path, ["u1\tp1\t5\t1\t2010-01-01"], ["u1 p1 2010-01-01 no tabs at all"])
    with pytest.raises(ValueError, match="line 1"):
        load_raw(tmp_path)
