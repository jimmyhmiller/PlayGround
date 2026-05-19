"""Smoke test: BPE tokenizer produces IDs for a simple text."""
from std.testing import TestSuite, assert_true

from bpe_tokenizer import load_tokenizer, tokenize


def test_bpe_basic() raises:
    var tok = load_tokenizer(
        "../mojo-t3/tests/fixtures/tokenizer/vocab.txt",
        "../mojo-t3/tests/fixtures/tokenizer/merges.txt",
    )
    var ids = tokenize("hello world", tok)
    print("[bpe] 'hello world' →", len(ids), "ids:")
    for i in range(len(ids)):
        print("  ", ids[i])
    assert_true(len(ids) > 0, "should produce some ids")

    var ids2 = tokenize("the quick brown fox", tok)
    print("[bpe] 'the quick brown fox' →", len(ids2), "ids:")
    for i in range(len(ids2)):
        print("  ", ids2[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
