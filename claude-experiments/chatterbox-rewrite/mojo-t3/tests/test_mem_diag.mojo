"""
Diagnostic: load file once, look at baseline RSS.
"""
from fixture import Tensor, load_fp32

def main() raises:
    var t = load_fp32("tests/fixtures/forward/layer0/down_w_fp32.bin")
    print("loaded shape", t.shape[0], "x", t.shape[1])
