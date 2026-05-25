"""Quick check: what alpha values are loaded into hift.resblocks[0].activations1[0]?"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from fixture import load_fp32
from weights import load_hift_generator


def test_alpha_check() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var raw = load_fp32("weights/s3gen/mel2wav/resblocks/0/activations1/0/alpha.bin")
    print("disk alpha[0..7]:", raw.data[0], raw.data[1], raw.data[2], raw.data[3],
          raw.data[4], raw.data[5], raw.data[6], raw.data[7])
    print("disk alpha shape len:", len(raw.data))

    var alpha = hift.resblocks[0].activations1[0].alpha
    print("Mojo struct alpha size:", alpha.unsafe_ptr())
    with alpha.map_to_host() as h:
        print("Mojo alpha[0..7]:", h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
