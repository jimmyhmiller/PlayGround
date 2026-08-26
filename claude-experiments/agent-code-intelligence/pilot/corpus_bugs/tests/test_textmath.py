from mill.mathutil import checked_div, clamp
from mill.textutil import clean, splay


def test_clean():
    assert clean("  a   b ") == "a b"


def test_splay():
    assert splay(" a , b ,, c ") == ["a", "b", "c"]


def test_checked_div():
    assert checked_div(7, 2) == 3


def test_clamp():
    assert clamp(99, 1, 10) == 10
