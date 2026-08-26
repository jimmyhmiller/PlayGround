from mill.merge import combine, merge_ratio


def test_combine_later_wins():
    out = combine({"a": "1", "b": "2"}, {"b": " 3 "}, ["clean"])
    assert out == {"a": "1", "b": "3"}


def test_combine_applies_rules():
    out = combine({"a": " x "}, {}, ["clean", "upper"])
    assert out == {"a": "X"}


def test_merge_ratio():
    assert merge_ratio(1, 2) == 50
    assert merge_ratio(0, 0) == 0
