"""
Parity test for the Mojo BPE tokenizer vs the upstream chatterbox tokenizer.

Expected (from `~/.cache/...tokenizers/` for text =
"Hello, this is a Mojo-based Chatterbox vocoder running on AMD."):
  ids = [284, 18, 84, 28, 7, 2, 147, 2, 54, 2, 14, 2, 289, 28, 23, 28, 8, 15,
         55, 49, 2, 279, 21, 48, 114, 165, 37, 2, 35, 28, 174, 234, 2, 31, 97,
         27, 52, 2, 47, 2, 277, 289, 280, 9]
  len = 44
"""
from std.testing import TestSuite, assert_equal
from bpe_tokenizer import load_tokenizer, tokenize


def test_tokenize_hello() raises:
    var tok = load_tokenizer(
        "tests/fixtures/tokenizer/vocab.txt",
        "tests/fixtures/tokenizer/merges.txt",
    )
    var text = "Hello, this is a Mojo-based Chatterbox vocoder running on AMD."
    var ids = tokenize(text, tok)
    print("Mojo ids:", ids)
    print("len:", len(ids))
    var expected = List[Int]()
    expected.append(284); expected.append(18); expected.append(84); expected.append(28)
    expected.append(7); expected.append(2); expected.append(147); expected.append(2)
    expected.append(54); expected.append(2); expected.append(14); expected.append(2)
    expected.append(289); expected.append(28); expected.append(23); expected.append(28)
    expected.append(8); expected.append(15); expected.append(55); expected.append(49)
    expected.append(2); expected.append(279); expected.append(21); expected.append(48)
    expected.append(114); expected.append(165); expected.append(37); expected.append(2)
    expected.append(35); expected.append(28); expected.append(174); expected.append(234)
    expected.append(2); expected.append(31); expected.append(97); expected.append(27)
    expected.append(52); expected.append(2); expected.append(47); expected.append(2)
    expected.append(277); expected.append(289); expected.append(280); expected.append(9)
    assert_equal(len(ids), len(expected))
    for i in range(len(expected)):
        assert_equal(ids[i], expected[i])
    print("BPE tokenizer parity: PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
